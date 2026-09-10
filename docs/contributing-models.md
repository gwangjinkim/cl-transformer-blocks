# Contributing Hugging Face models from Common Lisp

The Python execution path can load a Transformers AutoModel, prepare its inputs, train it under Lisp control, resume the training state, export ordinary artifacts, and publish them to a Hugging Face model repository. The model remains a normal Transformers checkpoint; Python consumers do not need this Common Lisp package.

Supported native models can also be loaded and published directly:

```lisp
(tb:with-resource
    (model (tb:from-pretrained "HuggingFaceTB/SmolLM2-135M"
                               :revision "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
                               :device :gpu))
  (tb:push-to-hub model "account/lisp-trained-model"
                  :commit-message "Publish native Lisp checkpoint"))
```

The native runtime invokes the locked `huggingface_hub` helper only for snapshot resolution and publication. Model execution and safe serialization remain native. Set `:LOCAL-FILES-ONLY T` to forbid downloads and `:CACHE-DIRECTORY` to select the snapshot cache. The helper downloads configuration, safetensors, and tokenizer/generation assets; it does not download or execute pickle checkpoints or repository Python code.

## Originate an architecture in Common Lisp

The versioned `tb_parallel` model can be initialized and trained without a pre-existing Python class. Its export emits the corresponding Python `PretrainedConfig` and `PreTrainedModel`, exact `auto_map` entries, and standard safetensors. Python consumers load it with `trust_remote_code=True`; no cl-transformer-blocks installation is needed on their side.

```lisp
(tb:with-resource
    (model (tb:make-portable-causal-lm
            :vocab-size 32000 :hidden-size 512 :intermediate-size 1536
            :layers 8 :attention-heads 8 :kv-heads 4
            :context-length 2048 :seed 104729 :device :gpu))
  (tb:attach-tokenizer model "./my-tokenizer/")
  ;; Native TRAIN-STEP or adapter-free optimizer training can update the model here.
  (tb:save-pretrained model ".build/my-lisp-model/"))
```

See [portable-models.md](portable-models.md) for the architecture, trust boundary, tokenizer responsibility, extension generic, and current precision/cache limits.

## Processor-driven inference and training

`PYTHON-PROCESS` returns selected AutoProcessor tensors for inspection. Use `PYTHON-PROCESSOR-FORWARD` or `PYTHON-PROCESSOR-TRAIN-STEP` for real workloads so prepared tensors do not make a JSON round trip.

```lisp
(let ((image (tb:make-python-file-input "example.png" :image)))
  (tb:with-resource
      (model (tb:from-pretrained "organization/vision-model"
                                 :execution :python
                                 :auto-class "AutoModelForImageClassification"
                                 :revision "immutable-commit"))
    (let ((outputs (tb:python-processor-forward
                    model `(("images" . ,image)) :outputs '("logits"))))
      (unwind-protect
           (print (tb:tensor-array (cdr (assoc "logits" outputs :test #'equal))))
        (mapc (lambda (entry) (tb:dispose (cdr entry))) outputs)))))
```

Processor input values may be strings, lists/vectors, numeric Lisp arrays, JSON-compatible scalars, or `MAKE-PYTHON-FILE-INPUT` objects. File media types are `:IMAGE`, `:AUDIO`, and `:VIDEO`. Pass processor switches such as padding or a sampling rate through `:options`; pass labels and other arguments that bypass the processor through `:model-inputs`.

For chat checkpoints, apply the exact saved Transformers template before generation:

```lisp
(let ((messages (list (tb:make-chat-message "system" "Answer concisely.")
                      (tb:make-chat-message "user" "Explain CLOS generics."))))
  (tb:python-apply-chat-template model messages :add-generation-prompt t))
```

Pass `:TOKENIZE T` to receive the tokenizer's flat ID vector. Extra message fields support tool-call or multimodal structures when the checkpoint template understands them; optional template keyword arguments go in `:OPTIONS`.

## Resumable training

```lisp
(tb:with-resource (optimizer (tb:make-adamw :learning-rate 1e-5))
  (tb:python-processor-train-step
   model optimizer '(("text" . "Common Lisp trains Transformers"))
   :model-inputs `(("labels" . ,labels)))
  (tb:save-training-checkpoint model optimizer ".build/checkpoint/"))
```

Configure a learning-rate schedule before the first update with `CONFIGURE-PYTHON-SCHEDULER`; `PYTHON-TRAIN-MICROBATCHES` performs one update from several smaller batches. Pass `:training-dtype "bfloat16"` or `"float16"` to `FROM-PRETRAINED` for autocast when the model and device support it.

Resume by loading `.build/checkpoint/` with `FROM-PRETRAINED`, constructing an optimizer with exactly the same options, and calling `RESTORE-TRAINING-CHECKPOINT`. Version 1 restores Torch optimizer moments, scheduler state, GradScaler state, the completed-step count, CPU RNG state, and the active CUDA or MPS RNG state. It does not contain arbitrary Python/NumPy application RNG state. `tb_training_state.pt` is loaded with PyTorch's restricted `weights_only=True` loader.

For parameter-efficient training, attach or import standard PEFT LoRA without transferring parameters through Lisp:

```lisp
(tb:python-make-lora model :rank 16 :alpha 32
                           :target-modules '("q_proj" "k_proj" "v_proj" "o_proj")
                           :rslora t)
(tb:with-resource (optimizer (tb:make-adamw :learning-rate 2e-4))
  (tb:python-train-step model optimizer batch)
  (tb:save-training-checkpoint model optimizer ".build/adapter-checkpoint/"))
(tb:python-save-adapter model ".build/adapter/")
```

To resume an adapter checkpoint, load the original base model, call `PYTHON-LOAD-ADAPTER` on the checkpoint with `:TRAINABLE T`, then restore a matching optimizer. `PUSH-TO-HUB` stages an adapter repository while PEFT is active. Call `PYTHON-MERGE-ADAPTER` after disposing its optimizer to make a standalone model; subsequent `SAVE-PRETRAINED` or `PUSH-TO-HUB` writes full Transformers weights.

## Hub publication

```lisp
(tb:push-to-hub model "account/model-name"
                :private t
                :commit-message "Publish Lisp-trained checkpoint"
                :model-card "# Model name\n\nTrained from Common Lisp.")
```

Authenticate outside Lisp with the ordinary Hugging Face CLI or `HF_TOKEN`. The worker inherits that environment; the token is never serialized into a request or checkpoint. Publication creates the model repository when needed and uploads a temporary directory containing only the model, tokenizer, processor, and model card. Use `:revision` for a branch, `:create-pr t` for a Hub pull request, and `:dry-run-directory` to inspect the exact artifacts without contacting the Hub.

The same credential rule applies to native publication. Native `PUSH-TO-HUB` first serializes a fresh standard checkpoint into an isolated staging directory, uploads that folder through `HfApi`, and removes temporary staging after success or failure.

Remote custom code remains disabled unless `:trust-remote-code t` is explicitly supplied while loading a reviewed, immutable model revision. Publishing is an external operation and is only performed when `PUSH-TO-HUB` is called without `:dry-run-directory`.
