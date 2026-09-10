# Lisp-defined portable models

`TB:MAKE-PORTABLE-CAUSAL-LM` creates a Transformer architecture in Common Lisp without starting from a Python model class. Its standard export contains the matching Python `PretrainedConfig` and `PreTrainedModel` implementations, safe weights, and `auto_map` registration. A normal Transformers installation can therefore load, train, resave, and publish it.

The first portable contract is `tb_parallel`, version 1. It uses bias-free grouped-query causal attention, non-interleaved RoPE, RMSNorm, a gated SiLU MLP, and one parallel residual block:

```text
normalized = rms_norm(hidden)
output = hidden + attention(normalized) + mlp(normalized)
```

Attention and the MLP consume the same normalized state. This saves the second per-block normalization and gives a future compiled execution plan two independent branches. It is a distinct architecture rather than a Llama checkpoint with a changed label.

## Create and train in Lisp

```lisp
(load "scripts/load.lisp")

(tb:with-resource
    (model (tb:make-portable-causal-lm
            :vocab-size 32000
            :hidden-size 512
            :intermediate-size 1536
            :layers 8
            :attention-heads 8
            :kv-heads 4
            :context-length 2048
            :tie-word-embeddings t
            :seed 104729
            :device :gpu))
  (tb:attach-tokenizer model "./my-tokenizer/")
  (tb:with-resource (optimizer (tb:make-adamw :learning-rate 3e-4))
    (tb:train-step model optimizer input-ids :labels labels)
    (tb:save-training-checkpoint model optimizer ".build/checkpoint/"))
  (tb:save-pretrained model ".build/python-ready/"))
```

Initialization is repeatable for a fixed seed in a given Lisp/runtime build: a versioned Park-Miller stream and Box-Muller transform produce all matrix and embedding values, while RMSNorm scales start at one. `:SEED` must be between 1 and 2147483646. Floating-point transcendental results are not promised to be bit-identical across different Common Lisp implementations. The constructor owns its MLX backend and tensors; dispose the model normally.

`TB:MAKE-PORTABLE-CONFIG` exposes the exact configuration object separately for inspection and tooling. Both constructors validate dimensions before allocating an MLX backend. Version 1 requires `hidden-size = attention-heads * head-dim`, an even head dimension, a query-head count divisible by the KV-head count, zero dropout, and FP32 execution.

Attach a Hugging Face tokenizer directory with `TB:ATTACH-TOKENIZER`. It requires `tokenizer.json`, validates the complete Rust Tokenizers vocabulary including added tokens against the model's embedding-table bounds, and snapshots its assets. A tokenizer may use fewer IDs than the embedding table contains. Matching bounds does not prove that token meanings agree with pretrained weights: the caller must choose the intended token-to-ID mapping and matching model BOS/EOS/padding IDs.

After attachment, `ENCODE-TEXT`, `DECODE-TOKENS`, and text prompts to `GENERATE` work directly. Model exports, full training checkpoints, and Hub staging include the byte-exact snapshot even if the original tokenizer directory changes. Invalid or out-of-range tokenizers leave the current tokenizer intact. The snapshot includes standard tokenizer JSON/configuration, added/special tokens, vocabulary/merges, SentencePiece assets, and a chat template when present. Model generation configuration remains model-owned. Attachment does not implement native chat-template rendering or arbitrary Python tokenizer wrappers.

The API works for native models, including imported architectures. Replacing a tokenizer intentionally changes text preparation; it does not resize or retrain embeddings. Rebuild the Rust tokenizer library when upgrading to this release. A new model without an attachment accepts token IDs and reports a clear condition when asked to tokenize text.

[`ENCODE-BATCH`](tokenization.md) also uses the attached tokenizer, returning padded IDs and masks for batched inference/training. A decoder uses the first two return values; the third contains tokenizer segment IDs, which the current portable architecture does not consume.

## Load in Python

The export contains `configuration_tb_parallel.py`, `modeling_tb_parallel.py`, `config.json`, and one or more standard safetensors files. Load the model through the ordinary AutoClass API:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    ".build/python-ready",
    trust_remote_code=True,
    local_files_only=True,
)
```

Transformers requires `trust_remote_code=True` because a custom architecture necessarily executes its supplied Python implementation. Review those two small source files and pin a Hub revision before trusting a remote repository. Native Lisp loading validates the exact `auto_map` class names and replaces supplied Python files with the package's versioned implementation on export; it does not execute them.

The same export can return through the resident worker:

```lisp
(tb:with-resource
    (model (tb:from-pretrained ".build/python-ready/"
                               :execution :python
                               :auto-class "AutoModelForCausalLM"
                               :trust-remote-code t
                               :local-files-only t))
  (tb:with-resource (logits (tb:forward model input-ids))
    (print (tb:tensor-shape logits))))
```

Standard Llama, GPT-2, Qwen2, and BERT exports continue to use built-in Transformers classes and do not need remote-code trust.

## Publication and extension

`TB:PUSH-TO-HUB` includes the emitted configuration and model sources in native publication staging. Use `:DRY-RUN-DIRECTORY` to inspect the exact model repository before contacting the Hub.

`TB:PYTHON-ARCHITECTURE-FILES` is a CLOS extension point. Its default method emits nothing for built-in Transformers architectures. A custom native model class can specialize it with an association list from safe root filenames to versioned source pathnames; the common transactional exporter validates and copies them. The model's exported configuration must contain matching AutoClass metadata, and an independent fresh Transformers process must validate configuration, complete weight coverage, logits, loss, and gradients.

Version 1 Python execution performs ordinary full-prefix attention and does not export a Python KV-cache contract. Native execution supports the block-growing cache and greedy generation. Quantization, mixed precision, dropout training, left padding, Python cached generation, and additional portable component choices remain later contracts; they will use new explicit configuration versions instead of changing version 1 semantics.
