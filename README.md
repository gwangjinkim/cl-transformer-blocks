# cl-transformer-blocks

Common Lisp model composition with native MLX computation and standard Hugging Face checkpoints. The native adapters cover explicitly bounded Llama, Qwen2, GPT-2, and BERT masked-language-model configurations. Each adapter exposes only the inference, training, caching, tokenization, and export behavior verified for that family.

The current MLX-based design and API replace the earlier MGL experiment.

## Start here

New users should begin with the **[hands-on guide](docs/hands-on-guide.md)**. It walks through installation, a first local Hugging Face model, Python round-trip verification, native LoRA training, the complete query-teacher application, unsupported and multimodal models through the Python worker, Lisp-defined architectures, publishing, resource ownership, and troubleshooting.

The central distinction is explicit:

| Path | Best for | Execution |
|---|---|---|
| **Native** | Qualified Llama, Qwen2, GPT-2, BERT masked-LM, and Lisp-defined architectures | Common Lisp model graph, MLX tensor kernels, Rust Hugging Face tokenizer; no Python inference or training |
| **Python worker** | Other installed Transformers AutoModel classes, processors, media, advanced generation, and broader PEFT | Common Lisp controls a resident Python/PyTorch model process |

Both paths read and write normal Hugging Face artifacts. Native compatibility is deliberately architecture-specific; the worker is the broader compatibility route and is never described as native execution.

Bootstrap and run the small CPU acceptance suite:

```sh
git clone https://github.com/gwangjinkim/cl-transformer-blocks.git
cd cl-transformer-blocks
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
uv run --no-sync python scripts/run-tests.py --device cpu
```

Then follow the [first useful model workflow](docs/hands-on-guide.md#first-useful-model-local-text-generation-and-python-export), or run the measured [Lisp query-teacher experiment](docs/query-teacher.md).

Try the [Lisp query-teacher experiment](docs/query-teacher.md): generate labeled application queries in Lisp, fine-tune SmolLM2 natively, and reload the adapter in Python. The first measured run improved exact test commands from 0/36 to 21/36; a keyword baseline still scored 35/36. The demo publishes both successes and failures and distinguishes a working training workflow from application readiness.

## What works

- Build a decoder from public [Lisp Transformer components](docs/components.md): rotary attention, SwiGLU, and sequential/parallel residual blocks with per-layer head counts and widths. Train the full model or a LoRA/rsLoRA adapter natively, and export the versioned `tb_composed` architecture with matching Python code and standard PEFT adapters.
- Start those components with [pretrained Llama weights](docs/pretrained-components.md), including the qualified SmolLM2-135M checkpoint. Convert to an equivalent stack, retain its tokenizer, and train/export through the same native APIs.
- [Select, reorder or repeat pretrained blocks](docs/layer-reuse.md) to build a different composed stack with copied weights. Train its independent layers in Lisp or Python; evaluate quality after changing the architecture.
- [Distill a pretrained teacher into an edited student](docs/distillation.md) using native temperature-scaled KL, optional supervised loss, full-model or LoRA updates, and ordinary Python/PEFT exchange. [Reusable teacher targets](docs/distillation-targets.md) let Lisp train after disposing the teacher and accept targets from otherwise unsupported Python causal-model architectures. [Lazy target datasets](docs/distillation-datasets.md) add atomic multi-batch publication, bounded training memory, deterministic epoch shuffling, and exact iterator continuation. The [streaming native producer](docs/distillation-streaming.md) also bounds target memory while Lisp creates a dataset from a resident teacher and an extensible application source. [FP16 target storage](docs/distillation-storage.md) reduces precision, [top-k targets](docs/distillation-top-k.md) replace the full vocabulary with explicit classes plus an exact aggregate tail, and [content-addressed datasets](docs/distillation-integrity.md) bind exact Lisp/Python target bytes to resumable iterator state.
- CPU and Apple Metal execution through MLX C; model mathematics are written in Lisp. No Python interpreter is used for native inference, tokenization, differentiation, or export.
- Llama with RMSNorm, SiLU-gated feed-forward layers, default unscaled RoPE, MHA/GQA, and tied or separate output embeddings. The tested checkpoints are the generated tiny fixture and **SmolLM2-135M** at a pinned Hugging Face revision.
- Qwen2 with learned Q/K/V biases, full causal attention, GQA, default RoPE, native training, and standard Qwen2 BPE tokenization. The pinned real Qwen2.5-0.5B checkpoint is verified on CPU and Metal; see [qwen2.md](docs/qwen2.md).
- GPT-2/DistilGPT-2 with learned positions, packed QKV, Conv1D biases, LayerNorm, and GELU variants; see [gpt2.md](docs/gpt2.md).
- BERT masked-language models with bidirectional attention, explicit sentence-pair token-type IDs, word/position/type embeddings, post-LayerNorm encoder blocks, WordPiece tokenization, same-position masked-token training, standard PEFT LoRA/rsLoRA, and standard export. The pinned real `bert-base-uncased` checkpoint is verified on CPU and Metal; see [bert.md](docs/bert.md).
- Standard local or Hugging Face Hub `config.json`, single/sharded safetensors loading and export, canonical parameter names and shape validation. Native Hub sources are resolved through the locked helper and can be pinned to an immutable revision.
- Machine-readable compatibility inspection reports native support or its exact rejection reason before device allocation, while loaded native and Python models report their actual backend, device/API, dtype, and operations; see [capabilities.md](docs/capabilities.md).
- Rust Hugging Face Tokenizers handles `tokenizer.json`. Exact encode/decode behavior is verified for the documented Llama, Qwen2, GPT-2, and BERT acceptance fixtures.
- Native [batched text and sentence-pair tokenization](docs/tokenization.md) returns ready-to-use IDs, attention masks, and segment IDs, with right padding and explicit longest-first truncation.
- Batch inference, absolute positions, right-padding masks, block-growing token/chunk KV caching, and single-sequence greedy generation.
- Standard LoRA/rsLoRA PEFT import/export and merging, masked training, momentum SGD/AdamW, gradient clipping, and resumable native full-model/adapter optimizer checkpoints are implemented; see [training.md](docs/training.md).
- MLX differentiates the Lisp-defined model. Every tiny-fixture parameter gradient and the exported SGD update are checked against independent Transformers results.
- Reproducible native benchmarks cover C-versus-Lisp MLX matrix multiplication and full-model prefill, cached decode, and training against PyTorch/Transformers. CPU/Metal reports include raw samples, numerical fingerprints, memory evidence, environment versions, and source/checkpoint hashes; see [performance.md](docs/performance.md).
- An explicit resident [Python compatibility worker](docs/python-worker.md) selects any installed Transformers `AutoModel*` class from a local directory or the Hub, applies saved tokenizer chat templates, runs its AutoProcessor, accepts numeric or local image/audio/video inputs, exposes Transformers beam/sampling generation, and performs resumable scheduled or accumulated SGD/AdamW training on CPU, Apple MPS, or CUDA. It can load or create standard PEFT LoRA adapters, train them from Lisp, publish adapter repositories, and merge them into ordinary Transformers models. Named inference tensors can use a validated binary transport instead of per-element JSON.

**Native compatibility is architecture-specific.** Scaled RoPE variants, quantized checkpoints, stochastic training dropout, custom tokenizer wrappers/chat-template execution, left padding, sampling/beam search, advanced PEFT variants, mixed precision, and distributed training are future native work. Unknown native model/configuration features raise conditions and may be loaded through the explicit Python worker. Generation assets are preserved on export; the current generic implements its documented greedy arguments rather than the full Transformers generation API.

MLX supports Linux CPU and CUDA as well as Apple Metal. The public Linux CPU suite runs in GitHub Actions, and the repository includes a manually triggered NVIDIA qualification workflow. **CUDA has not yet been verified by this project on real NVIDIA hardware.** M-series support follows actual MLX/macOS/device compatibility and test results; untested chip generations are not certified by name.

## Reproduce the build and tests

Build tools: Python 3.12 (managed by `uv`), `uv` 0.6.4, SBCL, CMake 3.25+, a C/C++20 compiler, and Rust/rustup. On the Mac, install Xcode and select a working developer directory. Tests were run with SBCL 2.6.7 and Apple Clang 17. Rust 1.92.0 is pinned by `rust-toolchain.toml`.

```sh
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
uv run --no-sync python scripts/run-tests.py --device cpu
uv run --no-sync python scripts/run-tests.py --device gpu
uv run --no-sync python scripts/run-training-tests.py --device cpu
uv run --no-sync python scripts/run-training-tests.py --device gpu
uv run --no-sync python scripts/run-component-tests.py --device cpu
uv run --no-sync python scripts/run-component-tests.py --device gpu
uv run --no-sync python scripts/run-gpt2-tests.py --device cpu
uv run --no-sync python scripts/run-gpt2-tests.py --device gpu
uv run --no-sync python scripts/run-qwen2-tests.py --device cpu
uv run --no-sync python scripts/run-qwen2-tests.py --device gpu
uv run --no-sync python scripts/run-bert-tests.py --device cpu
uv run --no-sync python scripts/run-bert-tests.py --device gpu
uv run --no-sync python scripts/run-python-worker-tests.py --device cpu
uv run --no-sync python scripts/run-python-worker-tests.py --device gpu
uv run --no-sync python scripts/run-portable-tests.py --device cpu
uv run --no-sync python scripts/run-portable-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --teacher-device mps
uv run --no-sync python scripts/model-benchmark.py --device cpu
uv run --no-sync python scripts/model-benchmark.py --device gpu
```

On Linux use `uv sync --frozen --extra cpu`. On a suitable NVIDIA machine choose **one** of `--extra cuda12` or `--extra cuda13`, then run the GPU suite. The CUDA wheel and installed NVIDIA driver must be compatible. A missing requested GPU fails the test; it does not silently fall back to CPU.

To verify the real checkpoint (downloads about 270 MB, with additional FP32 export/reference memory and disk usage):

```sh
uv run --no-sync python scripts/run-tests.py --device cpu --real
uv run --no-sync python scripts/run-tests.py --device gpu --real
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --real --local-files-only --teacher-device mps
```

The build verifies SHA-256-pinned Lisp and MLX C source archives, uses `Cargo.lock`, and locates the native MLX SDK in the locked wheel. Tests start clean SBCL processes with a project-only ASDF source registry. A global Quicklisp setup is unnecessary. Python is build/download/reference-test tooling; the Lisp runtime loads the resulting native shared libraries directly.

Use `TB_MLX_LIBRARY` and `TB_TOKENIZER_LIBRARY` for separately installed native libraries. Their transitive MLX libraries must also be loadable. Moving the development environment may require rebuilding to update native library search paths.

## Lisp usage

Load a supported native checkpoint directly from the Hub, preferably at an immutable revision:

```lisp
(load "scripts/load.lisp")

(tb:with-resource (model (tb:from-pretrained "HuggingFaceTB/SmolLM2-135M"
                                           :revision "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
                                           :device :gpu))
  (let ((ids (tb:encode-text model "Common Lisp is" :add-special-tokens nil)))
    (format t "~A~%"
            (tb:decode-tokens model
                              (tb:generate model ids :max-new-tokens 20)
                              :skip-special-tokens t)))
  (tb:save-pretrained model ".build/my-export/"))
```

Pass `:max-shard-size` as a positive byte count to write canonical `model-00001-of-000NN.safetensors` files and `model.safetensors.index.json`. Tensors are kept whole, so a tensor larger than the limit occupies its own shard. The same keyword is forwarded by the arbitrary AutoClass Python worker.

Native and Python-worker `save-pretrained`, adapter-save, and training-checkpoint calls build all applicable weights, configuration, tokenizer/processor assets, and optimizer sidecars in a hidden sibling directory. Publication then uses an atomic rename for a new destination or an atomic directory exchange when replacing an existing directory on macOS and Linux. A reader opening the published path therefore sees the complete previous export or the complete replacement. Publication cleans up the old directory after the exchange. This visibility guarantee does not include power-loss durability; applications requiring that must also arrange filesystem-specific `fsync` handling.

`tb:download-from-hub` exposes snapshot resolution separately when an application wants to inspect or reuse the local directory. `tb:push-to-hub` accepts native models as well as worker models; use `:dry-run-directory` to inspect the exact standard artifacts without network access.

Python can then load the export without this package:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(".build/my-export", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(".build/my-export", local_files_only=True)
```

For raw logits, pass a rank-two Lisp array of IDs to `tb:forward`; the returned owned tensor has shape `(batch time vocabulary)`. Use `tb:tensor-array` only when a host copy is needed, and dispose the tensor with `tb:with-resource`. `tb:make-cache` creates independent mutable generation state and reserves K/V positions in blocks of 256 by default; `:growth-step` selects another positive block size. `tb:cache-length` reports committed tokens and `tb:cache-capacity` reports reserved positions. A model, backend, or cache is not safe for concurrent mutation; give concurrent sessions separate caches and serialize model updates. Dispose caches and output tensors before their model/backend.

`tb:sgd-step` returns the pre-update loss and updates the active native parameters. Decoder models minimize shifted next-token cross entropy; BERT masked LM selects explicitly labeled positions without shifting. For reusable optimizer state, masked labels, gradient clipping and adapter training, use `tb:train-step`; see [the training guide](docs/training.md).

The Python compatibility path can process high-level multimodal inputs, train full models or PEFT adapters, save resumable training state, and publish a standard model or adapter repository directly from Lisp. See [contributing-models.md](docs/contributing-models.md) for the complete workflow and credential boundary.

Common Lisp can also originate a new `tb_parallel` causal-LM architecture. `TB:MAKE-PORTABLE-CAUSAL-LM` constructs deterministic native MLX parameters, and `SAVE-PRETRAINED` emits the matching Python configuration/model sources plus AutoClass metadata. A fresh ordinary Transformers process loads the result with `trust_remote_code=True`; see [portable-models.md](docs/portable-models.md).

Use `(tb:attach-tokenizer model "./tokenizer-directory/")` to give a native model a complete text interface. Token IDs are checked against its embedding table, and tokenizer assets are snapshotted for model exports, full training checkpoints, and Hub publication.

## Architecture and validation

[docs/architecture.md](docs/architecture.md) explains ownership and extension points, while [docs/validation.md](docs/validation.md) records independent numerical evidence and tolerances. Dependency locks and GitHub workflows make the public verification commands reproducible.

MIT licensed. Downloaded dependencies and models retain their own licenses; source provenance is recorded in `dependencies/sources.lock.json`.
