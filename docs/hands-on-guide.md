# Hands-on guide: use Hugging Face models from Common Lisp

`cl-transformer-blocks` lets a Common Lisp program load, run, train, and export
Transformer checkpoints that use Hugging Face conventions. It is useful when the
model should be part of the Lisp application rather than a separate HTTP service,
and when the result must remain usable by ordinary Python tooling.

There are two execution paths:

1. **Native MLX execution** loads supported `config.json`, safetensors, and
   tokenizer assets. Common Lisp defines the model graph and owns the model;
   MLX performs tensor operations and differentiation, and Hugging Face's Rust
   Tokenizers library executes `tokenizer.json`. No Python interpreter runs
   inference or training.
2. **The explicit Python worker** loads an installed Transformers `AutoModel*`
   class in a resident child process. Lisp controls inference, processing,
   generation, training, PEFT, checkpointing, and export, while PyTorch performs
   the model work. This path covers far more architectures, including many
   vision and audio models, but it is a Python integration path rather than
   native Lisp execution.

Both paths read and write standard Hugging Face artifacts. A model trained on
the native path can be loaded by Python, and a compatible model or PEFT adapter
trained in Python can be loaded by the native path.

This package is ready for controlled experiments and applications that fit its
documented contracts. It is not yet a universal, drop-in Common Lisp replacement
for all of Transformers.

Use the guide in this order:

1. [Choose the execution path](#choose-the-right-path).
2. [Install and verify the project](#install-and-verify-the-project).
3. [Run and export the first useful model](#first-useful-model-local-text-generation-and-python-export).
4. Pick a use case: [application fine-tuning](#use-case-teach-a-lisp-application-a-small-command-language), [Lisp/Python LoRA exchange](#use-case-fine-tune-in-lisp-and-continue-in-python), [multimodal models](#use-case-work-with-an-unsupported-or-multimodal-model), or [architecture research](#use-case-start-architecture-research-with-pretrained-weights).
5. Read the [current strengths and weaknesses](#current-strengths-and-weaknesses) before choosing a production workload.

## Choose the right path

Start by deciding whether native execution or broad compatibility matters more
for this model.

| Need | Recommended path | What to expect |
|---|---|---|
| Run or fine-tune a qualified Llama, Qwen2, or zero-dropout compatible model from a supported family | Native | Lisp model graph, MLX kernels, native differentiation, standard export |
| Run DistilGPT-2 or BERT masked LM locally | Native | Qualified inference and export; native training depends on the checkpoint's zero-dropout configuration |
| Use an architecture without a native adapter | Python worker | Select an installed `AutoModel*` class explicitly; PyTorch remains resident behind the Lisp API |
| Use image, audio, video, chat-template, beam-search, or sampling features | Python worker | AutoProcessor and the full documented generation bridge are available |
| Design a Transformer from Lisp components | Native `tb_composed` or `tb_parallel` | Train in Lisp and export matching Python source plus safetensors |
| Modify the layer stack of a pretrained Llama | Native composed-model tools | Preserve or copy pretrained weights, then evaluate and retrain the changed architecture |
| Publish a Lisp-trained model for Python users | Either | Export standard files locally or stage and push a Hugging Face repository |

For a local checkpoint, ask the package what the native loader supports before
allocating a device:

```lisp
(load "scripts/load.lisp")

(let ((report (tb:inspect-pretrained "./my-model/")))
  (format t "Native: ~:[no~;yes~]~%" (getf report :supported-p))
  (format t "Model type: ~S~%" (getf report :model-type))
  (format t "Operations: ~S~%" (getf report :operations))
  (unless (getf report :supported-p)
    (format t "Reason: ~A~%" (getf report :reason))))
```

Inspection of a local directory reads its configuration and checks tokenizer
assets. Inspection of a Hub identifier resolves the safe checkpoint snapshot and
may therefore download it. Use `:revision` for reproducibility and
`:local-files-only t` when network access must be forbidden.

## Install and verify the project

The reproducible development setup uses the repository's pinned Python, Rust,
and native dependency versions. Python is needed for bootstrap, Hub downloads,
reference tests, and the optional worker. It is not part of native model
execution after the libraries and checkpoint are present.

You need:

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- SBCL
- CMake 3.25 or newer
- a C/C++20 compiler
- Rust through `rustup`
- Xcode command-line tools on macOS

Clone and bootstrap from the repository root:

```sh
git clone https://github.com/gwangjinkim/cl-transformer-blocks.git
cd cl-transformer-blocks
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
```

`bootstrap.py` verifies pinned source archives, builds the native MLX C bridge
and Rust tokenizer library, and prepares the project-local Lisp dependencies.
It does not require a global Quicklisp setup.

Run the small CPU acceptance suite before downloading a real model:

```sh
uv run --no-sync python scripts/run-tests.py --device cpu
```

On an Apple-silicon Mac, verify actual Metal execution separately:

```sh
uv run --no-sync python scripts/run-tests.py --device gpu
```

`:device :gpu` and `--device gpu` require a real accelerator. They fail instead
of silently falling back to CPU. On Apple-silicon macOS, native `:gpu` means
Metal; in the Python worker it means MPS. Linux CPU runs in GitHub Actions. The
CUDA workflow has not yet been qualified by this project on real NVIDIA hardware.

On Linux, select the MLX wheel explicitly:

```sh
# CPU
uv sync --frozen --extra cpu

# NVIDIA: choose one wheel family, according to the host driver/toolkit.
uv sync --frozen --extra cuda12
# or: uv sync --frozen --extra cuda13
```

## First useful model: local text generation and Python export

This example downloads the exact SmolLM2-135M revision used by the acceptance
suite, generates text locally, prints the actual backend, and writes a standard
Transformers checkpoint.

Create `hello-model.lisp` in the repository root:

```lisp
(load "scripts/load.lisp")

(defparameter *model-id* "HuggingFaceTB/SmolLM2-135M")
(defparameter *revision*
  "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")

(tb:with-resource
    (model (tb:from-pretrained *model-id*
                               :revision *revision*
                               :device :cpu))
  (let ((capabilities (tb:model-capabilities model)))
    (format t "Backend: ~S, device API: ~S, dtype: ~S~%"
            (getf capabilities :backend)
            (getf capabilities :device-api)
            (getf capabilities :execution-dtype)))

  (let* ((prompt "Common Lisp is")
         (prompt-ids (tb:encode-text model prompt
                                     :add-special-tokens nil))
         (all-ids (tb:generate model prompt-ids
                               :max-new-tokens 24)))
    (format t "~A~%"
            (tb:decode-tokens model all-ids
                              :skip-special-tokens t)))

  (tb:save-pretrained model ".build/hello-export/"))
```

Run it in a clean Lisp process:

```sh
sbcl --noinform --no-sysinit --no-userinit --script hello-model.lisp
```

Change `:device :cpu` to `:device :gpu` after the Metal test passes. The first
run downloads roughly 270 MB of model and tokenizer assets. The checkpoint's
BF16 values are promoted to FP32 for current native execution and export, so
the resident model and exported files require more memory and disk than the
source weights.

Now verify that an ordinary Python process can load the Lisp export. Create
`verify-export.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = ".build/hello-export"
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

inputs = tokenizer("Common Lisp is", return_tensors="pt")
tokens = model.generate(**inputs, max_new_tokens=24, do_sample=False)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

Run it in the pinned Python environment:

```sh
uv run --no-sync python verify-export.py
```

Python does not need `cl-transformer-blocks` to load this export. It sees the
standard Llama configuration, safetensors, tokenizer, and generation assets.

SmolLM2-135M is a small base language model rather than a polished assistant.
Its completion may be awkward, inaccurate, or repetitive. The point of this
first workflow is local execution and exact interchange, not chat quality.

## Qualified real checkpoints

These real checkpoints have dedicated acceptance gates. A family name alone is
not a guarantee: the checkpoint configuration must satisfy that adapter's exact
contract.

| Checkpoint and pinned revision | Native use | Approximate download |
|---|---|---:|
| `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | Llama inference, cached greedy generation, full/LoRA training, export | 270 MB |
| `distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa` | GPT-2 inference, cache, greedy generation, export; its configured dropout prevents current native training | 350 MB |
| `Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987` | Qwen2 inference, cached greedy generation, full/LoRA training, export | 988 MB |
| `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` | BERT masked-LM inference, WordPiece and sentence pairs, export; current native training requires zero configured dropout | 440 MB |

Run the corresponding real gate when a result needs independent Python parity
evidence:

```sh
uv run --no-sync python scripts/run-tests.py --device cpu --real
uv run --no-sync python scripts/run-gpt2-tests.py --device cpu --real
uv run --no-sync python scripts/run-qwen2-tests.py --device cpu --real
uv run --no-sync python scripts/run-bert-tests.py --device cpu --real
```

Replace `cpu` with `gpu` to collect Metal evidence. These commands download
models and also produce reference and export artifacts, so budget more disk than
the model size alone.

See the individual contracts for accepted positions, masks, biases, heads,
training behavior, and rejection cases:

- [Llama validation](validation.md#acceptance-suite)
- [GPT-2 and DistilGPT-2](gpt2.md)
- [Qwen2](qwen2.md)
- [BERT masked LM](bert.md)

## Use case: teach a Lisp application a small command language

The most complete application example is the
[query-teacher experiment](query-teacher.md). A Lisp application defines a safe,
small command language for querying issue records:

```text
(issues STATUS COMPONENT PERIOD)
```

Common Lisp constructs supervised examples, tokenizes them, trains a native LoRA
adapter on SmolLM2-135M, validates generated commands without `READ` or `EVAL`,
executes valid commands, and exports both a standard PEFT adapter and a merged
Transformers model.

Run the complete Metal experiment:

```sh
uv run --no-sync python scripts/run-query-teacher.py \
  --device gpu --download --output .build/my-query-teacher
```

For a fast mechanics check on CPU:

```sh
uv run --no-sync python scripts/run-query-teacher.py \
  --device cpu --download --smoke --output .build/my-query-smoke
```

After a full run, ask the trained model from a clean Lisp process:

```sh
sbcl --noinform --no-sysinit --no-userinit \
  --script examples/query-teacher/ask.lisp \
  .build/my-query-teacher/merged \
  'Show open tokenizer issues from the last week.' gpu
```

A measured local run improved exact test commands from 0/36 before training to
21/36 after training. A simple keyword baseline scored 35/36. That result proves
the native training and Python interchange workflow, while also showing that a
small fine-tune is not automatically better than ordinary application logic.
The guide publishes every failed prediction and explains the evaluation design.

This pattern is valuable when an application already owns strong Lisp domain
logic:

- define a bounded output language or data structure;
- create and validate training examples with Lisp functions and CLOS objects;
- fine-tune only the language interface;
- reject invalid model output before it reaches application code;
- keep deterministic application semantics outside the model;
- export the learned adapter for Python evaluation or distribution.

Possible applications include a natural-language interface for a Lisp knowledge
base, routing requests to generic functions, normalizing laboratory commands, or
extracting a small validated record from free text. A model is most compelling
when language variation makes simple rules inadequate. Always compare it with a
rule-based baseline.

## Use case: fine-tune in Lisp and continue in Python

Native LoRA and rsLoRA are available for the documented projections of Llama,
Qwen2, BERT masked LM, and `tb_composed`. The base weights stay frozen while MLX
differentiates and updates the adapter tensors.

The following is a mechanics example with illustrative token IDs:

```lisp
(load "scripts/load.lisp")

(tb:with-resource
    (model (tb:from-pretrained "./supported-base/" :device :gpu))
  (tb:make-lora model
                :rank 8
                :alpha 16
                :targets '("q_proj" "v_proj")
                :seed 17)

  (tb:with-resource
      (optimizer (tb:make-adamw :learning-rate 0.001
                                :weight-decay 0.01))
    (dotimes (step 10)
      (format t "step ~D, loss ~F~%"
              step
              (tb:train-step model optimizer
                             #2A((3 4 5 6))
                             :labels #2A((-100 -100 5 6))
                             :max-grad-norm 1.0)))
    (tb:save-training-checkpoint
     model optimizer ".build/native-checkpoint/"))

  (tb:save-adapter model ".build/lisp-adapter/")
  (tb:merge-adapter model)
  (tb:save-pretrained model ".build/lisp-merged/"))
```

For decoder models, labels use next-token prediction and `-100` excludes a
position from the loss. A real data pipeline must tokenize prompt and answer
carefully, mask prompt and padding labels, batch equal shapes, retain EOS
supervision, and hold out independent evaluation data. Use
[`examples/query-teacher/training.lisp`](../examples/query-teacher/training.lisp)
as the complete working template.

Load the adapter in ordinary Python PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("./supported-base")
adapted = PeftModel.from_pretrained(
    base, ".build/lisp-adapter", is_trainable=True
)

# Continue with a normal PyTorch/Transformers training loop.
adapted.save_pretrained(".build/python-continued-adapter")
```

Bring the Python result back to Lisp by reloading the unchanged base and calling
`tb:load-adapter`. The base identity, adapter targets, ranks, shapes, and complete
tensor coverage are validated. The exact supported PEFT subset and checkpoint
resume rules are documented in [native training](training.md).

## Use case: work with an unsupported or multimodal model

When native inspection rejects a model, select the Python worker explicitly.
The worker keeps the model loaded, so repeated calls do not reload all weights.

```lisp
(load "scripts/load.lisp")

(tb:with-resource
    (model (tb:from-pretrained "organization/model-name"
                               :execution :python
                               :auto-class "AutoModelForCausalLM"
                               :revision "immutable-commit"
                               :device :cpu))
  (format t "Worker: ~S~%" (tb:python-worker-info model))
  (let* ((ids (tb:encode-text model "Common Lisp can call"))
         (generated (tb:generate model ids :max-new-tokens 20)))
    (format t "~A~%" (tb:decode-tokens model generated))))
```

This code uses Transformers and PyTorch. The explicit `:execution :python`
argument prevents the broader path from being mistaken for native execution.
Pass `:local-files-only t` for an offline load. Remote custom model code stays
disabled unless you deliberately pass `:trust-remote-code t` for a reviewed,
pinned revision.

For a photograph classifier, AutoProcessor can decode the local image and feed
the resulting tensors directly to the model inside the worker:

```lisp
(let ((image (tb:make-python-file-input "example.png" :image)))
  (tb:with-resource
      (model (tb:from-pretrained "organization/vision-model"
                                 :execution :python
                                 :auto-class
                                 "AutoModelForImageClassification"
                                 :revision "immutable-commit"
                                 :device :cpu))
    (let ((outputs
            (tb:python-processor-forward
             model `(("images" . ,image))
             :outputs '("logits"))))
      (unwind-protect
           (format t "Class logits: ~S~%"
                   (tb:tensor-array
                    (cdr (assoc "logits" outputs :test #'equal))))
        (mapc (lambda (entry) (tb:dispose (cdr entry))) outputs)))))
```

Replace the repository, revision, and AutoClass with a checkpoint whose model
card specifies image classification. Interpret output indices with that
checkpoint's `id2label` configuration. Speech, audio, and video models use the
same processor pattern with `:audio` or `:video` file inputs and the appropriate
AutoClass. Some processors need additional packages such as `librosa`,
`torchcodec`, or PyAV; those are model-specific and are not installed by the
base project.

The Python worker also exposes saved chat templates, beam search, sampling,
processor-driven generation, full-model updates, PEFT adapters, gradient
accumulation, learning-rate schedules, mixed-precision autocast, resumable Torch
state, and Hub publication. See the [worker contract](python-worker.md) and the
[model contribution workflow](contributing-models.md).

Calling the worker through Lisp can still be useful: the application, data model,
validation, orchestration, and extension points remain in Common Lisp, while an
unported model supplies one capability. It does not satisfy a requirement for a
Python-free runtime.

## Use case: build and exchange a Transformer defined in Lisp

There are two native custom-model APIs:

- `TB:MAKE-PORTABLE-CAUSAL-LM` creates the versioned `tb_parallel` architecture.
- `TB:MAKE-TRANSFORMER` composes rotary-attention, SwiGLU, and sequential or
  parallel residual block descriptors into the versioned `tb_composed`
  architecture.

Run the small component example:

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/composed-transformer.lisp
```

It creates and trains a tiny random model to demonstrate model construction,
native differentiation, and export. It is too small and insufficiently trained
to be a useful language model.

A custom export contains safetensors, `config.json`, `auto_map`, and reviewed
matching Python model/configuration source. Python loads it as follows:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    ".build/composed-example",
    trust_remote_code=True,
    local_files_only=True,
)
```

The Python files implement the documented component vocabulary; this is not a
general Lisp-to-Python compiler. Review custom model code before enabling
`trust_remote_code`, as with any Transformers repository that ships Python code.

See [portable models](portable-models.md), [component construction](components.md),
and the [portable LoRA example](../examples/composed-lora.lisp).

## Use case: start architecture research with pretrained weights

Random initialization is rarely the useful starting point. A supported Llama
checkpoint can instead be converted into an equivalent `tb_composed` stack while
retaining its current weights and tokenizer:

```lisp
(load "scripts/load.lisp")

(tb:with-resource
    (source (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
  (tb:save-composed-pretrained
   source ".build/pretrained-components/"))

(tb:with-resource
    (model (tb:from-pretrained ".build/pretrained-components/" :device :gpu))
  (format t "~D Lisp-visible blocks~%"
          (length (tb:transformer-blocks model))))
```

From there you can:

- inspect the block descriptors in Lisp;
- train the complete model or a LoRA adapter;
- select, reorder, remove, or repeat blocks into a new checkpoint;
- distill a teacher into the edited student;
- train in Python and load the result back into native Lisp.

Changing the layer stack changes the model. Removing half the layers does not
automatically produce a good compressed model, and repeating a block does not
automatically add useful capacity. The included recomposition example produced
degraded text before retraining. Treat these APIs as architecture research tools
and measure the resulting model on its intended workload.

Follow [pretrained component conversion](pretrained-components.md),
[layer reuse](layer-reuse.md), and [distillation](distillation.md) for the exact
weight, tokenizer, provenance, training, and export contracts.

## Batch text, masks, and sentence pairs

`TB:ENCODE-BATCH` performs one native tokenizer call for a batch and returns
ordinary Lisp arrays for IDs, attention masks, and token-type IDs:

```lisp
(tb:with-resource
    (model (tb:from-pretrained "./bert-masked-lm/" :device :gpu))
  (multiple-value-bind (ids mask segments)
      (tb:encode-batch model
                       '("Common Lisp" "Transformers")
                       :text-pairs '("is [MASK]." "use attention."))
    (tb:with-resource
        (logits (tb:forward model ids
                            :attention-mask mask
                            :token-type-ids segments))
      (format t "Logit shape: ~S~%" (tb:tensor-shape logits)))))
```

Native batching uses right padding. Truncation is opt-in, and oversized inputs
raise a condition by default. Decoder adapters accept IDs and attention masks;
the current BERT adapter additionally accepts token-type IDs. See
[native tokenization](tokenization.md).

## Save, resume, and publish safely

Model, adapter, and training-checkpoint saves are staged in a sibling directory
and atomically published. Readers see the previous complete directory or the new
complete directory rather than a half-written mixture. This is an atomic
visibility guarantee, not power-loss durability.

Use separate destinations for the source model, training checkpoints, adapters,
and merged model:

```text
.build/experiment/
  checkpoint/
  adapter/
  merged/
```

Before publishing to the Hub, stage exactly what would be uploaded:

```lisp
(tb:push-to-hub model "account/model-name"
                :dry-run-directory ".build/hub-preview/"
                :model-card "# Model name\n\nTrained from Common Lisp.")
```

Inspect the directory, then call `TB:PUSH-TO-HUB` without
`:dry-run-directory` when you intend to perform the network publication.
Authenticate outside Lisp with the normal Hugging Face CLI or `HF_TOKEN`.
Credentials are not written into the model request or checkpoint.

## Resource ownership in application code

Models, optimizers, caches, and returned tensors own native or worker resources.
Use `TB:WITH-RESOURCE` so they are disposed during both normal return and Lisp
condition unwinding:

```lisp
(tb:with-resource (model (tb:from-pretrained "./model/"))
  (tb:with-resource (output (tb:forward model #2A((1 2 3))))
    ;; TENSOR-ARRAY copies tensor data to a Lisp array when host access is needed.
    (print (tb:tensor-shape output))))
```

Avoid copying full logits with `TENSOR-ARRAY` unless the Lisp application needs
their values. Keep computation in native tensors where the API permits it.
Dispose caches and outputs before their model/backend. A model, cache, optimizer,
or Python worker is not safe for concurrent mutation; give concurrent sessions
separate state and serialize updates.

## Current strengths and weaknesses

### Strong today

- Exact, validated standard-artifact interchange for the documented native
  architecture contracts.
- CPU and Apple Metal execution with MLX kernels under a Common Lisp model API.
- Native tokenization, cached greedy decoder inference, full-model training, and
  a strict PEFT LoRA/rsLoRA subset where documented.
- Standard exports that fresh Python Transformers or PEFT processes can load.
- Lisp-defined custom architecture exports with matching reviewed Python code.
- A broad, explicit worker path for installed Transformers AutoModel classes,
  AutoProcessor media, advanced generation, training, and publication.
- Tests that compare weights, complete logits, gradients, updates, cache behavior,
  tokenizer behavior, exports, and fresh-process Python reloads.

### Limited today

- Native support is architecture- and configuration-specific. “Llama-like” or
  “Qwen-like” does not mean automatically compatible.
- Native execution is currently FP32. Quantized checkpoints, mixed-precision
  native training, and low-memory loading are not implemented.
- Native generation is single-sequence greedy generation. Sampling, beam search,
  and full chat-template behavior require the Python worker.
- Native batching uses right padding; advanced tokenizer wrappers, left padding,
  overflow windows, and native chat templates are absent.
- Native training requires zero configured dropout and does not yet provide
  distributed training, gradient accumulation, or scheduling.
- Native PEFT covers a strict LoRA/rsLoRA subset. DoRA and other advanced variants
  require the worker.
- CUDA has a reproducible workflow but still needs real NVIDIA qualification.
- Custom Lisp architectures require `trust_remote_code=True` in Python because
  Transformers must load their emitted Python implementation.
- The worker is broad but still depends on Python, PyTorch, compatible installed
  model classes, and any model-specific media packages.
- Model compatibility does not imply model quality. Fine-tuning requires a real
  dataset, independent evaluation, baselines, and application-level validation.

## Troubleshooting

### Native loading rejects the model

Run `TB:INSPECT-PRETRAINED` and read `:REASON`. Do not edit the model type or
silently approximate unsupported mathematics. Choose `:execution :python` or add
a new native adapter with independent parity tests.

### `:device :gpu` fails

The package does not fall back to CPU. First run the GPU acceptance command and
inspect `TB:MODEL-CAPABILITIES` or `TB:PYTHON-WORKER-INFO` for the actual device
API. Use `:device :cpu` until the accelerator path passes.

### A library cannot be loaded after moving the checkout

Re-run:

```sh
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
```

The built libraries may contain search paths tied to the project environment.
Advanced installations can set `TB_MLX_LIBRARY` and `TB_TOKENIZER_LIBRARY` to
separately installed libraries whose transitive dependencies are also loadable.

### A Hub model cannot be downloaded

Check the repository ID, pinned revision, network access, and Hugging Face access
permissions. Authenticate outside Lisp for gated repositories. Once cached, use
`:local-files-only t` to make accidental network access a failure.

### The model output is poor

Confirm whether the checkpoint is a base model or an instruction/chat model,
whether the correct tokenizer and chat template are used, and whether the task
fits its size. Greedy generation deliberately provides fewer controls than
Transformers generation. For an instruct checkpoint that depends on a saved
chat template, use the Python worker.

### Native training uses too much memory

Current native weights, gradients, and optimizer state are FP32. Prefer LoRA,
shorter sequences, smaller batches, and a small checkpoint. Use the worker when
mixed precision, accumulation, or a broader PEFT method is necessary.

## Where to go next

| Goal | Documentation |
|---|---|
| Understand the model/backend boundary | [Architecture](architecture.md) |
| Check exact support before loading | [Capabilities](capabilities.md) |
| Tokenize batches and sentence pairs | [Tokenization](tokenization.md) |
| Train or exchange PEFT adapters | [Native training](training.md) |
| Use arbitrary AutoModel classes or media | [Python worker](python-worker.md) |
| Publish a Lisp-controlled model | [Contributing models](contributing-models.md) |
| Build a portable Lisp architecture | [Portable models](portable-models.md) |
| Compose heterogeneous blocks | [Components](components.md) |
| Start components with pretrained weights | [Pretrained components](pretrained-components.md) |
| Change and retrain a layer stack | [Layer reuse](layer-reuse.md) |
| Distill a model | [Distillation](distillation.md) |
| Reproduce numerical claims | [Validation](validation.md) |
| Measure performance correctly | [Performance](performance.md) |
