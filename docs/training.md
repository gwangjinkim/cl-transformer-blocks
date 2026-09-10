# Native training and PEFT interchange

For a frozen teacher guiding a separate student, see [native distillation](distillation.md). Its temperature-scaled KL and optional supervised loss use the same full-model/LoRA optimizers, clipping and checkpoint APIs described here.

The Lisp runtime supports standard LoRA and rank-stabilized LoRA on registered Llama, Qwen2, BERT masked-LM, and `tb_composed` linear projections. It imports Python-produced adapters, trains native adapter parameters with a frozen base, exports files loaded by ordinary PEFT, and merges adapters into a Transformers checkpoint. Independent PEFT 0.20.0 tests cover both scaling rules on Llama, BERT, and composed models, plus import/export/merge on Qwen2. The Lisp-defined [composed architecture](components.md) requires its exported Python source files and `trust_remote_code=True` when loading its base in Python.

```lisp
(load "scripts/load.lisp")
(tb:with-resource (model (tb:from-pretrained ".build/fixtures/tiny/" :device :gpu))
  (tb:make-lora model :rank 2 :alpha 4 :targets '("q_proj" "v_proj") :seed 17)
  ;; Alternatively: (tb:load-adapter model "python-adapter/")
  (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001
                                            :weight-decay 0.01))
    (dotimes (step 10)
      (format t "~D: ~F~%" step
              (tb:train-step model optimizer #2A((3 4 5 6))
                             :labels #2A((-100 -100 5 6))
                             :max-grad-norm 1.0))))
  (tb:save-adapter model ".build/my-adapter/")
  (tb:merge-adapter model)
  (tb:save-pretrained model ".build/my-merged-model/"))
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained(".build/fixtures/tiny")
adapted = PeftModel.from_pretrained(base, ".build/my-adapter")
merged = AutoModelForCausalLM.from_pretrained(".build/my-merged-model")
```

`save-adapter` writes `adapter_config.json` and `adapter_model.safetensors` using PEFT's canonical names. It builds both files in a sibling stage and atomically publishes the complete adapter directory. Its checkpoint omits the runtime adapter name (such as `default`), following the [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint). Standard scaling is alpha/r; rank-stabilized scaling is alpha/sqrt(r), following [PEFT's LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora). A is `(rank,input)`, B is `(output,rank)`, and the forward delta is computed as two small projections, without materializing a full delta matrix on every call.

## Resume native training

`save-training-checkpoint` writes standard Hugging Face model files for full-model training or standard PEFT files for adapter training. It adds `training_state.json` and `optimizer.safetensors` for the native SGD/AdamW step count and momentum/moment tensors. All standard artifacts and native sidecars are staged and published together, so a path-based reader cannot observe weights from one training step with optimizer state from another. Python Transformers or PEFT can load the standard files without this Lisp package and ignore the sidecar.

```lisp
;; Save after one or more successful updates.
(tb:save-training-checkpoint model optimizer ".build/checkpoint/")

;; Full model: reload the model from the checkpoint itself.
(tb:with-resource (resumed (tb:from-pretrained ".build/checkpoint/" :device :gpu))
  (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
    (format t "resuming after step ~D~%"
            (tb:restore-training-checkpoint resumed optimizer ".build/checkpoint/"))
    (tb:train-step resumed optimizer ids :labels labels)))
```

For an adapter checkpoint, reload the unchanged base model, call `load-adapter` with the checkpoint directory, construct the same optimizer, then restore. Restoration requires a fresh model state and unbound optimizer, and validates the format version, native implementation, model/adapter kind, model type, complete ordered trainable-parameter names, optimizer options, state slots, tensor coverage, tensor shapes, finite values, and checkpoint source directory before taking ownership of any state.

Native training currently requires zero configured dropout and uses no runtime RNG, so the manifest records that no RNG state exists. The worker checkpoint format remains separate because it also owns Torch scheduler, GradScaler, CPU/device RNG, and mixed-precision state. Neither sidecar changes the standard model or PEFT artifact contract.

## Supported adapter contract

- One active adapter on Llama, Qwen2, BERT masked LM, or `tb_composed`. Llama/Qwen2/composed projection suffixes are q/k/v/o and gate/up/down; the verified BERT contract uses `query` and `value`.
- Uniform positive rank and alpha; optional rank-stabilized scaling; zero dropout and no bias adaptation.
- Complete safetensors coverage and shape validation. DoRA, regex targets, rank/alpha patterns, layer selection/replication, embedding adapters, modules_to_save, and other variants are rejected.
- Base identity and revision must match. For a pinned Hub checkpoint loaded from a local directory, pass `:model-id` and `:revision` to `from-pretrained`. For local checkpoints the adapter records that directory; distribute the corresponding base separately and update its reference deliberately if relocated. This checks declared identity, not a cryptographic fingerprint of all base weights.
- Modified base weights must be saved/reloaded before attaching another adapter, so an adapter is never silently associated with the original base after base training/merging.
- Newly constructed composed models must also be saved and reloaded before attaching adapters, to replace their synthetic source identity with a reusable base reference. Both tied and untied composed heads are qualified.
- Attaching, removing, merging, or updating adapters invalidates existing caches. `save-pretrained` rejects active adapters; choose `save-adapter` or explicitly merge first.
- `make-lora` uses a local seeded generator for A and zeros for B, so initial outputs equal the base exactly. It does not reproduce PyTorch's RNG sequence.

## Training controls and ownership

`trainable-parameters` returns borrowed canonical tensors; `loss-and-gradients` returns a scalar loss and owned gradient pairs. Dispose returned gradients after use. Without an adapter all unique base parameters are trainable, including one canonical tied embedding parameter. With an adapter only its A/B parameters are differentiated and updated.

Labels have the input shape and use `-100` for unsupervised targets. Decoder language models follow next-token shifting: the first label has no preceding prediction, and right-padding mask zeros also exclude padded targets from the mean. BERT masked LM uses labels at the same token positions without shifting; its explicit labels alone select the loss positions. All-ignored batches raise `shape-error` before tracing. Attention and loss masks are distinct: masking a label does not hide the corresponding input token from attention.

`make-sgd` supports momentum and coupled weight decay. `make-adamw` supports betas, epsilon and decoupled weight decay. `train-step` optionally clips the global gradient norm before optimizer weight decay, matching the tested PyTorch settings. The native engine computes the norm; complete gradients are not copied to Lisp just for clipping.

An optimizer belongs to one model/version after its first successful step. External updates, adapter changes or merges make it stale. Dispose it and create a fresh optimizer when changing the trainable parameter set. Parameter and optimizer-state replacements are evaluated and checked for finite values before committing. A failed update leaves the prior model and optimizer state in place.

Optimizers own native state and require disposal. The tests check stable live tensor handles, stable MLX active allocation after warmup, frozen base weights, and complete disposal after repeated steps. Allocator cache/peak bytes are reported separately from live allocation. This is a bounded regression check, not a proof for arbitrary model sizes or unlimited training duration.

Native distributed training, mixed precision, stochastic dropout, gradient accumulation and scheduling remain future work. Native parameters and optimizer state are FP32. The explicit Python worker separately provides accumulated/scheduled autocast training and resumable Torch state for arbitrary installed AutoModel classes; see [python-worker.md](python-worker.md).

## Reproduce

```sh
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
uv run --no-sync python scripts/run-training-tests.py --device cpu
uv run --no-sync python scripts/run-training-tests.py --device gpu
uv run --no-sync python scripts/run-bert-tests.py --device cpu
uv run --no-sync python scripts/run-bert-tests.py --device gpu
uv run --no-sync python scripts/run-component-lora-tests.py --device cpu
uv run --no-sync python scripts/run-component-lora-tests.py --device gpu
```

The independent fixtures exercise nonzero A and B, prompt-masked labels, padding, three SGD/AdamW steps, clipping, decay, merged checkpoints, and a native-created adapter trained for repeated steps before Python reload. The existing inference/round-trip suite remains a regression gate.
