# Build Transformers from Lisp components

Version 0.28 adds a public component-construction API. Describe an ordered stack of blocks in Common Lisp, initialize its weights, train it with native MLX differentiation, and export it for ordinary Transformers AutoClass loading. The architecture is represented by the versioned `tb_composed` format; it is not labeled as Llama or another pretrained family.

## Construct a model

```lisp
(load "scripts/load.lisp")

(let ((blocks
        (list
         (tb:make-transformer-block
          :attention (tb:make-rotary-attention :heads 4 :kv-heads 2)
          :feed-forward (tb:make-swiglu :intermediate-size 28)
          :residual :sequential)
         (tb:make-transformer-block
          :attention (tb:make-rotary-attention :heads 2 :kv-heads 1)
          :feed-forward (tb:make-swiglu :intermediate-size 36)
          :residual :parallel))))
  (tb:with-resource
      (model (tb:make-transformer
              :vocab-size 32 :hidden-size 16 :blocks blocks
              :context-length 64 :seed 17 :device :gpu))
    (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
      (tb:train-step model optimizer #2A((3 4 5 6))))
    (tb:save-pretrained model ".build/composed-example/")))
```

This creates a randomly initialized model, not a pretrained language model. The vocabulary size determines embedding rows; no tokenizer is implicitly selected. The existing `attach-tokenizer` API is available when you have matching tokenizer assets. Meaningful language behavior requires suitable weights and training data.

Run the complete [example](../examples/composed-transformer.lisp), which performs three training steps and exports the result:

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/composed-transformer.lisp
```

CPU is the example's default when `TB_DEVICE` is omitted. After the usual locked dependency sync and bootstrap, no Python interpreter is needed during the Lisp model's construction, numerical execution, differentiation, or export.

## What the components mean

| Public operation | Contract |
|---|---|
| `make-rotary-attention :heads H :kv-heads K :rope-theta R` | Bias-free causal attention with full-head, unscaled, split-half RoPE. K defaults to H; R defaults to 10000. H must be divisible by K. Supports MHA, GQA, and MQA through head counts. |
| `make-swiglu :intermediate-size F` | Bias-free `down(silu(gate(x)) * up(x))` with intermediate width F. |
| `make-transformer-block :attention A :feed-forward F :norm-epsilon E :residual R` | Pre-RMSNorm block. E defaults to 1e-5; R defaults to `:sequential`. |
| `make-transformer :vocab-size V :hidden-size D :blocks B :context-length T ...` | Snapshots an ordered list/vector of descriptions, validates all block dimensions, creates the canonical parameter tree, and initializes it deterministically on the requested device. |
| `component-config` | Generic function returning a fresh JSON object describing a component. User descriptor classes may implement this generic to produce supported block descriptions. |
| `transformer-blocks` | Returns fresh component descriptions for a composed model, suitable for inspection or constructing a new model. These descriptions own no tensors. |

Sequential and parallel blocks compute different functions:

```text
Sequential:
  residual = x + attention(norm1(x))
  output   = residual + swiglu(norm2(residual))

Parallel:
  normalized = norm(x)
  output     = x + attention(normalized) + swiglu(normalized)
```

The sequential block owns two normalization weight vectors. The parallel block owns one shared normalization vector. The final model adds token embeddings, a stack of these blocks, a final RMSNorm, and a vocabulary projection.

The model-wide hidden dimension must divide evenly into each layer's head count, with an even head dimension for RoPE. Head counts, KV head counts, RoPE bases, feed-forward widths, normalization epsilons, and residual topology can differ by layer. Hidden width stays constant across the stack. Reusing the same descriptor in several positions creates separate parameter tensors; it does not tie layer weights. `:tie-word-embeddings t` explicitly ties the final vocabulary projection to the embedding table.

Model-wide options include `:norm-epsilon` for the final normalization, `:initializer-range`, `:bos-token-id`, `:eos-token-id`, `:pad-token-id`, `:seed`, and `:device`. Initialization uses the existing local seeded normal generator and unit normalization weights; it is reproducible within its documented native setup, not a claim to reproduce PyTorch's random sequence.

## CLOS extensibility with a defined export boundary

The descriptors are CLOS objects. A user-defined descriptor can implement `component-config` to lower an application-specific description into the supported block vocabulary. The constructor validates and snapshots that description. The runtime then uses CLOS dispatch at block, attention, and feed-forward boundaries while numerical work remains in MLX tensors.

This version deliberately supports a finite set of numerical components: rotary attention, SwiGLU, RMSNorm, and the two residual topologies. Adding arbitrary Lisp functions as layers is not part of this public export contract. A new numerical component needs a versioned description, parameter schema, native graph implementation, matching Python implementation, and independent numerical tests. Redefining internal computation methods does not automatically update the emitted Python code.

Treat `model-config` as read-only. A composed model retains a separate construction snapshot and rejects altered configuration during graph construction and export, so changing a JSON field cannot silently relabel a different runtime graph. Obtain new descriptions through `transformer-blocks` and construct a new model to change its architecture. Native model/backend/cache ownership and concurrency restrictions continue to apply.

## Exchange with Python

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    ".build/composed-example",
    trust_remote_code=True,
    local_files_only=True,
)
# Train with ordinary PyTorch/Transformers, then save for native Lisp reload.
model.save_pretrained(".build/python-composed")
```

```lisp
(tb:with-resource
    (model (tb:from-pretrained ".build/python-composed/" :device :gpu))
  (tb:with-resource (logits (tb:forward model #2A((3 4 5 6))))
    (format t "~S~%" (tb:tensor-shape logits))))
```

Export writes standard safetensors, `config.json` with explicit per-block configuration and `auto_map`, and reviewed Python source files. Two composed-model files reuse two emitted support files from the existing parallel decoder. Python requires `trust_remote_code=True` because the architecture is custom. Native reload uses its registered Lisp implementation and does not execute those Python files.

This is a matching Python implementation of the supported composition vocabulary, not an arbitrary Lisp-to-Python compiler. A standard Hugging Face Llama checkpoint still loads through the existing Llama adapter. Version 0.30 adds [faithful pretrained conversion](pretrained-components.md) through `save-composed-pretrained`: export its current weights as an equivalent sequential composed stack. Version 0.31 adds [layer selection, reordering and repetition](layer-reuse.md) from one composed source through `save-recomposed-pretrained`. Combining different models, transplanting weights into unrelated blocks, or converting additional pretrained architectures remains a separate task.

Full-model SGD/AdamW training, clipping, native optimizer checkpoints, single/sharded export, native cached greedy generation, and right-padded batches use the existing public APIs. Python generation currently recomputes complete prefixes (`use_cache=False`); Python KV caching is unavailable for this format.

## Train a portable LoRA adapter

Version 0.29 adds standard LoRA and rsLoRA for composed models, using the same [native training API](training.md). All seven projection suffixes are supported: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. The default targets are query and value. Each factor's shape comes from its actual layer weight, so different feed-forward widths and attention head layouts work in the same stack. Tied and untied vocabulary heads are supported; adapting the embeddings or vocabulary head is outside this contract.

Save the base and reload it before creating or loading an adapter. A newly constructed model has only a synthetic source identity, which cannot identify the weights another Lisp or Python process needs. After full-model training or merging, save and reload again before attaching a new adapter. An adapter contains only its small A/B factors: distribute the exact base checkpoint and its four Python source files separately. Local base references are paths, not weight fingerprints; update references deliberately when relocating artifacts.

```lisp
;; First create .build/composed-example/ using the example above.
(tb:with-resource (model (tb:from-pretrained ".build/composed-example/" :device :gpu))
  (tb:make-lora model :rank 3 :alpha 6 :rslora t
                :targets '("q_proj" "v_proj" "down_proj"))
  (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
    (tb:train-step model optimizer #2A((3 4 5 6))
                   :labels #2A((-100 -100 5 6)))
    (tb:save-training-checkpoint model optimizer ".build/composed-lora-checkpoint/"))
  (tb:save-adapter model ".build/composed-lora-adapter/")
  (tb:merge-adapter model)
  (tb:save-pretrained model ".build/composed-lora-merged/"))
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    ".build/composed-example", trust_remote_code=True, local_files_only=True,
)
adapted = PeftModel.from_pretrained(base, ".build/composed-lora-adapter", is_trainable=True)
# Continue training in Python, then save an adapter that Lisp can load.
adapted.save_pretrained(".build/python-composed-adapter")
merged = AutoModelForCausalLM.from_pretrained(
    ".build/composed-lora-merged", trust_remote_code=True, local_files_only=True,
)
```

To bring a Python adapter back, reload the unchanged base in Lisp and call `load-adapter`. To resume native optimizer state, load the adapter checkpoint and call `restore-training-checkpoint` with the same optimizer options. The native checkpoint's optimizer sidecars do not prevent ordinary PEFT loading.

The complete [composed LoRA example](../examples/composed-lora.lisp) builds a tiny base, trains it, reloads it, trains an rsLoRA adapter, and writes adapter/checkpoint/merged exports:

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/composed-lora.lisp
```

As with the construction example, this demonstrates mechanics with token IDs and random initialization. It is not a useful pretrained language model. Lisp defines and executes the architecture; MLX performs numerical kernels and differentiation; Python is needed only for the independent interoperability checks or subsequent Python use.

## Verification

```sh
uv run --no-sync python scripts/run-component-tests.py --device cpu
uv run --no-sync python scripts/run-component-tests.py --device gpu
uv run --no-sync python scripts/run-component-lora-tests.py --device cpu
uv run --no-sync python scripts/run-component-lora-tests.py --device gpu
```

The suite constructs a mixed model in Lisp, checks descriptor ownership and configuration rejection, and compares against a separate functional Torch oracle. That oracle uses complex-number RoPE and Torch SDPA, independently of the exported implementation's rotate-half trigonometry and explicit attention calculation. Checks cover all logits, task loss, every parameter gradient, SGD updates, tied embeddings, causal isolation, heterogeneous-head cached chunks, sharded exports, fresh Python AutoClass loading, Python training/resave, native reload, greedy generation, and exact AdamW checkpoint continuation.

Observed errors on the local Mac with MLX 0.32.2 and the locked Torch/Transformers versions:

| Maximum absolute difference from the independent oracle | CPU | Metal |
|---|---:|---:|
| Logits | 5.96e-8 | 4.47e-8 |
| Full-model gradients | 1.19e-7 | 1.19e-7 |
| SGD-updated parameters | 3.73e-9 | 3.73e-9 |
| Tied-model gradients | 5.96e-8 | 1.19e-7 |

The mixed test has 20 canonical parameter tensors; its tied counterpart has 19. Tests use `atol=2e-6, rtol=3e-4`. Each run writes `.build/components-created-DEVICE/validation.json`. Tiny-model parity qualifies the component math and interchange; it does not establish useful pretrained behavior, large-model performance, low-precision execution, or CUDA qualification.

The adapter runner first executes the component suite, then uses ordinary PEFT with nonzero A/B factors for both standard LoRA and rsLoRA, on tied and untied versions of the mixed model. It checks all 28 factor gradients, three clipped/decayed momentum-SGD and AdamW updates, bit-exact native checkpoint continuation, frozen base weights, identity initialization, cache invalidation and chunked cache parity, exact unchanged adapter transport, all merged weights, and native-created adapter logits/greedy tokens in fresh Python. Python then trains those Lisp-created adapters further, saves them, and a fresh Lisp process reloads them and checks logits/loss and differentiation. All owned tensor handles in the main native adapter gate are released. Results are written to `.build/components-created-DEVICE/lora-validation.json`; CPU and Metal evidence is recorded in [validation.md](validation.md).
