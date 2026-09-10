# Reuse pretrained blocks in a different stack

Version 0.31 adds `save-recomposed-pretrained`. Select blocks from one native `tb_composed` model by zero-based index, in any order, including repeated indices. Export the resulting architecture and current weights, reload it, and train it from Lisp or Python. This makes layer removal, reordering and depth expansion explicit model-building operations in Common Lisp.

## Select, save and reload

First [convert a supported pretrained Llama](pretrained-components.md) with `save-composed-pretrained`, or create a model with the [component API](components.md). Then select its blocks:

```lisp
(load "scripts/load.lisp")

(tb:with-resource (source (tb:from-pretrained ".build/my-pretrained-blocks/" :device :gpu))
  ;; New layer 0 uses source layer 1; new layer 1 uses source layer 0;
  ;; new layer 2 starts as another copy of source layer 1.
  (tb:save-recomposed-pretrained source ".build/edited-blocks/" #(1 0 1)
                                :max-shard-size 16777216))

(tb:with-resource (edited (tb:from-pretrained ".build/edited-blocks/" :device :gpu))
  (format t "~D blocks~%" (length (tb:transformer-blocks edited)))
  ;; Full-model TRAIN-STEP, native optimizer checkpoints and cached generation
  ;; use the edited stack. An adapter can also be created and trained here.
  (tb:make-lora edited :rank 8 :alpha 16 :targets '("q_proj" "v_proj")))
```

The last line creates an identity adapter; learning requires supervised data and optimizer updates. See [training.md](training.md) and [composed-lora.lisp](../examples/composed-lora.lisp) for actual training. The selection can be a proper list or vector; it must be nonempty and every index must exist in the source.

Each selected occurrence preserves that block's attention heads, RoPE base, feed-forward width, normalization and residual topology. The source's embeddings, final normalization, vocabulary head, context limit, tokenizer and embedding/head tying are retained. Repeated blocks have separate trainable parameters after reload: they start equal, but their updates can differ. Parameter names are renumbered to match their new layer positions.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    ".build/edited-blocks", trust_remote_code=True, local_files_only=True,
)
assert model.config.num_hidden_layers == 3
# Train with ordinary PyTorch or PEFT, then model.save_pretrained(...).
# Native Lisp can load that Python-trained model again.
```

Python uses the four emitted architecture files and standard safetensors. As with other composed exports, the Python generation path is uncached; native Lisp supports cached greedy generation.

## What changes, and what is preserved

Selecting a different stack changes the model's function. Removing half the layers does not establish a useful smaller model; evaluate on the intended task and expect to retrain or distill. Version 0.32 supplies [native teacher/student distillation](distillation.md) as one training path. This API does not combine multiple source models, resize projections, replace a selected block's mathematics or automatically restore a standard Llama architecture identity. It operates on the exact native composed class; other pretrained families retain their existing adapters.

The source remains unchanged. Current full-model-updated base weights are accepted. An active adapter must be explicitly merged or removed before selection. All source parameter names and shapes are checked, including unselected blocks, and mutated configurations and unknown subclasses are rejected.

Export borrows the source's native tensors synchronously, without extra initialization or weight copies through Lisp arrays. Keep the source alive and avoid concurrent mutation until export returns. Every repeated occurrence is serialized under its own parameter names, so repeated layers increase checkpoint size. Reload supplies independent ownership. Sharding and atomic directory publication follow ordinary export: failed publication preserves the previous complete destination, and overwriting the source directory is rejected.

`recomposition.json` records the immediate source ID/revision, source layer count, whether the resident base was updated, and selected indices. This is provenance, not a content fingerprint or complete edit history. A local source ID can be an absolute path; all weights needed to run the result are included in the export. Native model saves and training checkpoints preserve the sidecar. Ordinary Python `save_pretrained` does not copy arbitrary sidecars: copy this file separately if you want to retain it after Python training. Execution depends on the exported config, weights and code, not this provenance file.

## Run the real-model experiment

With the pinned SmolLM2 checkpoint cached as described in [pretrained-components.md](pretrained-components.md):

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/recompose-pretrained.lisp
```

The [example](../examples/recompose-pretrained.lisp) converts the pretrained model, prints its original generation, selects layers `0, 2, …, 28`, and generates from the resulting 15-layer model. In the local Metal run, the original model produced a readable continuation, while the reduced model repeated punctuation. That observed degradation is why this is an architecture/training experiment, not a qualified text model or a quality-preserving compression recipe.

## Reproduce the acceptance checks

```sh
uv run --no-sync python scripts/run-recomposition-tests.py --device cpu
uv run --no-sync python scripts/run-recomposition-tests.py --device gpu
uv run --no-sync python scripts/run-recomposition-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-recomposition-tests.py --device gpu --real --local-files-only
```

Tiny tests cover identity, removal, reversal and repetition of a heterogeneous sequential/parallel model with tied and untied heads. Python builds its reference by independently selecting and copying the original source modules. Tests compare complete weights, logits, all gradients, SGD updates, greedy generation, native optimizer continuation, native-trained PEFT adapters and Python training followed by Lisp reload. Repeated blocks must become unequal after native SGD, demonstrating independent training. Invalid selections, source immutability, failed publication and tensor cleanup are also checked.

The real suite includes the original pretrained conversion gate, then selects 15 of SmolLM2's 30 blocks. It verifies all 137 resulting canonical tensors exactly, logits, tokenizer assets, greedy tokens and a native AdamW LoRA update reloaded in Python PEFT. CPU and Metal evidence is in [validation.md](validation.md); reports are `.build/recomposition-DEVICE/validation.json` and `real-validation.json`. CUDA is prepared in the external workflow and remains unqualified.
