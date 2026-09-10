# GPT-2 architecture contract

The native GPT-2 adapter loads and exports standard Hugging Face GPT-2 language-model checkpoints, including DistilGPT-2. It runs inference, cached greedy decoding, and tokenization on CPU/Metal. Tiny zero-dropout configurations also pass full gradient and SGD-update comparisons against Transformers.

```lisp
(load "scripts/load.lisp")
(tb:with-resource (model (tb:from-pretrained ".build/models/distilgpt2/" :device :gpu))
  (let ((tokens (tb:generate model "Common Lisp is" :max-new-tokens 20)))
    (format t "~A~%" (tb:decode-tokens model tokens)))
  (tb:save-pretrained model ".build/my-gpt2-export/"))
```

The implementation follows the [Transformers GPT-2 model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) and its [configuration contract](https://huggingface.co/docs/transformers/en/model_doc/gpt2), with independent fixtures produced by the pinned Transformers version.

## Supported behavior

| Feature | Contract |
|---|---|
| Positions | Learned absolute embeddings; cache offsets respect configured `n_positions`. |
| Attention | Packed QKV, multihead causal attention, optional head-dimension and inverse-layer scaling. |
| Projections | GPT-2 Conv1D `(input,output)` weight layout and learned biases. |
| Normalization | Pre-LayerNorm blocks plus final LayerNorm, with configured epsilon, weight and bias. |
| Activation | `gelu_new` tanh approximation or exact `gelu` using erf. |
| Embeddings | Tied by default, or a separate `lm_head.weight` when configured. |
| Checkpoints | Single/sharded safetensors import and export with exact learned parameter coverage. |
| Legacy buffers | Serialized `attn.bias` buffers accepted only after every causal triangle entry is validated; derived buffers are omitted on export. |
| Masks/cache | Shared absolute-position/right-padding contract; token-wise and chunk-capable native cache. |
| Tokenization | Native Tokenizers pipeline; real DistilGPT-2 byte-level BPE is verified. |
| Training | FP32 deterministic full-model loss/gradients and optimizers for zero-dropout configurations. |

Configured dropout is ignored during evaluation, matching Transformers `eval()`. The training API rejects nonzero GPT-2 dropout until native stochastic dropout is implemented. Cross-attention, reordered/upcast attention, arbitrary activations, left padding, token-type embeddings, custom position IDs, and GPT-2 PEFT adapters are not supported yet. Existing LoRA support remains specific to Llama projections. Generation remains the documented greedy API; it does not execute arbitrary saved generation policies.

## Reproduce

```sh
uv run --no-sync python scripts/bootstrap.py
uv run --no-sync python scripts/run-gpt2-tests.py --device cpu
uv run --no-sync python scripts/run-gpt2-tests.py --device gpu
uv run --no-sync python scripts/run-gpt2-tests.py --device cpu --real
uv run --no-sync python scripts/run-gpt2-tests.py --device gpu --real
```

The real fixture pins `distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa` and downloads about 350 MB of model weights plus tokenizer assets. It uses local safetensors; no Python pickle model is executed by Lisp. Python is needed for downloading and independent verification, not native model execution.

Tiny fixtures cover standard/tied, untied, alternative scaling, exact GELU, sharded serialization, and valid/corrupted legacy buffers. Every parameter gradient and the exported SGD update is compared with Python. Fresh Transformers processes reload original and updated native exports. The real checkpoint additionally verifies its tokenizer, generation, cache, and exported learned weights.

## Extension structure

The loader resolves `model_type` through a CLOS class registry. Architecture-specific generic methods define validation, parameter schemas, default tying, canonical embedding names, legacy-buffer sanitization, graph construction, and training constraints. Common resource ownership, safetensors I/O, generation, cache handling, and optimizers are reused. These architecture implementation methods are provisional extension points; new adapters need the same independent numerical and round-trip gates before being advertised.
