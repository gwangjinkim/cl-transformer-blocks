# Explicit Python compatibility worker

The Python worker gives Common Lisp an explicit path to Hugging Face architectures that do not yet have a native adapter. It keeps the model, AutoProcessor, and optimizer state resident in one child process and exposes model-level inference, training, checkpoint, export, and publication operations. This broadens practical Transformers compatibility while preserving an honest distinction between native MLX execution and Python execution.

The shared `FORWARD` and `TRAIN-STEP` APIs accept `:TOKEN-TYPE-IDS` alongside `:ATTENTION-MASK`. Segment arrays must match the input-ID shape; the loaded Transformers model defines their meaning and accepted range. Native and worker BERT segment inference and training are compared in the [BERT gate](bert.md).

```lisp
(load "scripts/load.lisp")

(tb:with-resource
    (model (tb:from-pretrained "Qwen/Qwen2.5-0.5B-Instruct"
                               :execution :python
                               :revision "an-immutable-commit-hash"
                               :device :gpu
                               :auto-class "AutoModelForCausalLM"))
  (format t "Device: ~A~%"
          (gethash "actual_device" (tb:python-worker-info model)))
  (let ((ids (tb:encode-text model "Common Lisp" :add-special-tokens nil)))
    (format t "~A~%" (tb:generate model ids :max-new-tokens 20)))
  (tb:with-resource
      (optimizer (tb:make-adamw :learning-rate 1e-5))
    (tb:python-train-step
     model optimizer
     `(("input_ids" . ,#2A((1 2 3)))
       ("attention_mask" . ,#2A((1 1 1)))
       ("labels" . ,#2A((1 2 3)))))))
  (tb:python-make-lora model :rank 8 :alpha 16
                             :target-modules '("q_proj" "v_proj"))
  (tb:python-save-adapter model ".build/python-trained-adapter/"))
```

`FROM-PRETRAINED` accepts a local directory or Hub repository identifier in this mode. Network access follows Hugging Face settings. Use `:local-files-only t` for an offline/reproducible load and pin `:revision` to an immutable commit for a reproducible Hub load. `HF_TOKEN` and cache environment variables remain in the worker environment; credentials are not serialized by Lisp.

## Supported worker contract

The convenience tasks are `:causal-lm`, `:masked-lm`, `:sequence-classification`, `:seq2seq-lm`, and `:base`. The default is `:causal-lm`. For other installed Transformers tasks, pass the exact class name with `:auto-class`, such as `"AutoModelForImageClassification"`, `"AutoModelForSpeechSeq2Seq"`, or `"AutoModelForTimeSeriesPrediction"`. Only names beginning with `AutoModel` are accepted, and the resolved object must implement `from_pretrained`.

- `TB:FORWARD` accepts a nonempty rank-two integer ID array and an optional same-shaped attention mask. It returns an owned `PYTHON-TENSOR`; use `TB:TENSOR-ARRAY` and dispose it normally.
- `TB:PYTHON-FORWARD` accepts an association list of named model arguments. Lisp arrays of any rank become `int64` or FP32 Torch tensors; simple scalar arguments pass through. Select one or more tensor fields from the model output with `:outputs`. Returned tensors are owned and must be disposed. Pass `:TRANSPORT :BINARY` for file-backed little-endian tensor payloads in both directions; JSON remains the default.
- `TB:PYTHON-PROCESS` runs the checkpoint's AutoProcessor and returns selected prepared tensors with their integer, boolean, or FP32 dtype preserved. `TB:PYTHON-PROCESSOR-FORWARD` and `TB:PYTHON-PROCESSOR-TRAIN-STEP` keep prepared tensors inside the worker and directly invoke the model.
- `TB:ENCODE-TEXT` and `TB:DECODE-TOKENS` use the checkpoint's Transformers tokenizer, including wrapper behavior beyond the native Rust tokenizer contract.
- `TB:MAKE-CHAT-MESSAGE` creates a JSON-compatible role/content message with optional extra fields. `TB:PYTHON-APPLY-CHAT-TEMPLATE` applies the tokenizer's saved template and returns either rendered text or a flat token-ID vector. It supports generation-prompt and continue-final-message modes plus additional JSON-compatible tokenizer options.
- `TB:GENERATE` delegates deterministic greedy generation to the loaded Python model.
- `TB:PYTHON-GENERATE` passes named tensors plus JSON-compatible Transformers generation options and returns selected tensor fields such as `sequences` and `sequences_scores`. `TB:PYTHON-PROCESSOR-GENERATE` prepares high-level inputs first. `:seed` makes sampling repeatable on the same backend without permanently changing its RNG state; different device kernels need not produce the same sampled sequence.
- `TB:PYTHON-TRAIN-STEP` invokes the model with named inputs, requires its standard scalar `loss`, differentiates all parameters whose `requires_grad` is true, optionally clips the global gradient norm, and performs a stateful SGD or AdamW update. The ordinary `TB:TRAIN-STEP` is also available as a causal-LM convenience wrapper.
- `TB:PYTHON-TRAIN-MICROBATCHES` averages scalar losses and accumulated gradients across a nonempty list, then clips and updates once. `TB:CONFIGURE-PYTHON-SCHEDULER` installs any scheduler accepted by the pinned Transformers `get_scheduler` before the first update; inspect its step, name, and current rate with `TB:PYTHON-OPTIMIZER-INFO`.
- `TB:PYTHON-MAKE-LORA` installs a PEFT LoRA adapter with explicit rank, alpha, target modules, dropout, bias, rsLoRA, DoRA, task type, and modules-to-save options. `TB:PYTHON-LOAD-ADAPTER` accepts a local adapter or Hub identifier. `TB:PYTHON-ADAPTER-INFO` reports active adapters, standard PEFT configurations, and trainable/total parameter counts.
- `TB:PYTHON-SAVE-ADAPTER` atomically publishes canonical `adapter_config.json` and `adapter_model.safetensors`. `TB:PYTHON-MERGE-ADAPTER` checks for stale optimizer ownership, safely merges selected adapters, and replaces the worker object with the plain Transformers model so `SAVE-PRETRAINED` emits a standalone model.
- `TB:SAVE-PRETRAINED` asks Transformers and its tokenizer/processor to write ordinary safe-serialization artifacts, including updates made from Lisp. `:MAX-SHARD-SIZE` forwards a positive byte limit for standard sharded output. The complete directory is staged beside its destination and published with one atomic rename or directory exchange. Fresh independent Python processes verify unchanged, sharded, and trained acceptance exports.
- `TB:SAVE-TRAINING-CHECKPOINT` additionally saves one optimizer's moments, completed-step count, and Torch CPU/device RNG state, then publishes the standard artifacts and state files together in one directory exchange. For a full model, load the checkpoint as the model source. For PEFT, load the original base and then load the checkpoint directory with `PYTHON-LOAD-ADAPTER :TRAINABLE T`. Construct a matching fresh Lisp optimizer and call `TB:RESTORE-TRAINING-CHECKPOINT` before the next step.
- `TB:PUSH-TO-HUB` stages model, tokenizer, processor, and model-card artifacts and uploads them with `HfApi`. `:dry-run-directory` performs the complete staging operation without network access. See [contributing-models.md](contributing-models.md).
- `TB:PYTHON-WORKER-INFO` reports the protocol version, PID, task, Python model/tokenizer classes, requested device, actual `cpu`/`mps`/`cuda` device, and full model configuration.

The model and its optimizer states remain resident, so calls do not reload parameters or lose Adam/SGD moments. Disposing a Lisp optimizer drops its corresponding Python optimizer state. Dispose the model to request a graceful shutdown and release Torch resources. A worker object is not safe for concurrent requests; serialize access or create one worker per concurrent session.

The worker launches the project `.venv/bin/python` when present, then falls back to `python3`. Set `TB_PYTHON` or pass `:python-executable` to select a compatible locked environment. Standard output is reserved for versioned JSON-lines frames. Python/model output is redirected to standard error so it cannot corrupt the protocol.

## Devices, dtype, and trust

`:device :gpu` selects CUDA when available and otherwise Apple MPS. It fails if neither is available; it never falls back to CPU. `:dtype` controls loaded model weights and accepts `"auto"`, `"float32"`, `"float16"`, or `"bfloat16"`. `:training-dtype` controls Torch autocast and accepts `"float32"`, `"float16"`, or `"bfloat16"`; CUDA FP16 also uses a GradScaler. Scheduler and scaler state are included in training checkpoints. `:python-threads` optionally fixes Torch's CPU thread count for reproducible comparisons. Support still depends on Torch, the model, and the installed hardware.

Remote custom model code is disabled by default. `:trust-remote-code t` has the same security meaning as in Transformers: repository-supplied Python executes inside the worker with the user's permissions and inherited environment. Use it only for a reviewed, pinned repository revision.

Named inputs and selected forward results use JSON by default and are intended for inspection, validation, and modest tensors. `PYTHON-FORWARD :TRANSPORT :BINARY` writes raw int64/FP32 inputs and int64/FP32/boolean outputs under a private worker-owned temporary directory; protocol frames contain only validated path, dtype, and shape metadata. Both sides validate exact byte length and delete each transfer after consumption. `:max-output-elements` still bounds binary results and defaults to 2,000,000. Processor-forward, generation, and training tensor arguments do not yet expose this binary option; those operations keep prepared inputs, parameters, gradients, and optimizer state inside Python where possible.

`MAKE-PYTHON-FILE-INPUT` identifies local image, audio, or video files for processor calls. Image decoding is locked through Pillow. Audio and video decoding use the installed Transformers backends and may require model-specific packages such as librosa, torchcodec, or PyAV. Numeric Lisp arrays can be supplied directly when decoding is managed elsewhere.

Direct parameter enumeration/transfer, non-LoRA PEFT methods, shared-memory or socket tensor transport, binary transport for operations other than named forward, nested non-tensor outputs such as per-step score tuples, token streaming, cancellation, and concurrent requests are not implemented. Chat templates still depend on the checkpoint's installed tokenizer and template; missing or invalid templates raise a compatibility condition. Model-specific Python packages and custom-code requirements still apply.

The worker checks finite loss and gradients before updating and finite trainable parameters afterward. Unlike the native transactional optimizer, it does not keep a complete copy of large Python weights for rollback if an optimizer creates a nonfinite parameter. Save checkpoints at appropriate intervals. Training-state files are loaded with `torch.load(weights_only=True)` and require an exact optimizer configuration match. Optimizer, scheduler, GradScaler, and Torch device RNG state are restored; arbitrary Python/NumPy application RNG state is outside this checkpoint version.

## Acceptance test

The suite creates a seeded tiny OPT checkpoint, an architecture intentionally absent from the native registry, plus a deterministic adapter produced by independent PEFT code. It checks CPU/MPS worker logits, exact saved chat-template rendering and tokenization, AutoProcessor text preparation, direct processor inference/training, tokenizer wrappers, greedy and beam generation, seeded sampling, output-size enforcement, process persistence, and graceful disposal. It accumulates microbatches under a linear schedule using BF16 CPU or FP16 MPS autocast, then resumes optimizer/scheduler/scaler state in a new worker and verifies the next update against uninterrupted training. It also loads the Python-created adapter, creates and trains rsLoRA from Lisp, resumes adapter optimizer state, stages an adapter Hub repository, and verifies adapter and merged exports in fresh PEFT/Transformers processes. A no-network unit test verifies local image decoding and the complete mocked HfApi publication boundary.

```sh
uv run --no-sync python scripts/run-python-worker-tests.py --device cpu
uv run --no-sync python scripts/run-python-worker-tests.py --device gpu
```

The GPU run used MPS locally. The CUDA workflow runs the same required test on an actual NVIDIA runner when one is provided; that workflow has not yet run, so CUDA remains unqualified.
