# Train an edited model from its pretrained teacher

Version 0.32 adds native knowledge distillation. A frozen teacher supplies a probability distribution over the vocabulary at each selected next-token position. The student learns to approximate that distribution using full-model updates or LoRA, with Common Lisp controlling both architectures and training. MLX performs the tensor computation and differentiation; Python is used by the independent acceptance tests and for exchanging trained artifacts. Version 0.33 adds [reusable teacher targets](distillation-targets.md), allowing the teacher to be disposed before repeated student updates and accepting targets created by an arbitrary compatible Python causal teacher. Version 0.34 adds [lazy target datasets](distillation-datasets.md) with deterministic shuffling and exact iterator continuation. Version 0.35 adds [streaming native production](distillation-streaming.md), so Lisp writes and releases each target before creating the next one.

This supplies a training path after [removing, reordering or repeating pretrained layers](layer-reuse.md). It does not guarantee recovery of the original model's quality.

After updating the checkout, run `uv sync --frozen` and `uv run --no-sync python scripts/bootstrap.py` to rebuild the native bridge with the new loss operation before using the API.

## Train and exchange a student

Prepare an edited composed checkpoint as described in the layer-reuse guide, then load it alongside its original teacher:

```lisp
(load "scripts/load.lisp")

(tb:with-resource (teacher (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
  (tb:with-resource (student (tb:from-pretrained ".build/recomposed-example/" :device :gpu))
    (tb:make-lora student :rank 4 :alpha 8 :seed 37)
    (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.0005 :weight-decay 0.0))
      ;; Token IDs must mean the same thing to both models. Use real task text
      ;; for useful training; these IDs only illustrate the API.
      (tb:distill-step student teacher optimizer #2A((3 4 5 6))
                       :temperature 2.0 :hard-weight 0.3
                       :labels #2A((-100 -100 5 6)) :max-grad-norm 1.0)
      (tb:save-training-checkpoint student optimizer ".build/distilled-checkpoint/"))
    (tb:save-adapter student ".build/distilled-adapter/")))
```

Omit `make-lora` to train all unique student base parameters. `distill-step` uses the ordinary native SGD/AdamW ownership, gradient clipping and atomic update rules. It returns the pre-update scalar loss, advances only the student model/optimizer version, and invalidates stale student caches. The teacher stays unchanged. Use ordinary full-model or adapter checkpoint restoration to resume, supplying the teacher, batches, temperature and hard weight again. Those objective/data choices and the teacher weights are not stored in the student optimizer checkpoint.

`distillation-loss-and-gradients` takes the same student, teacher, inputs and objective options and returns the scalar loss plus owned named gradient tensors. Dispose those gradients after use, as with `loss-and-gradients`. This computes gradients even when you only want the loss. `model-capabilities` and `inspect-pretrained` advertise `:distillation` for the qualified native families.

Export a full student with `save-pretrained`, an adapter with `save-adapter`, or explicitly merge its adapter first. Ordinary Transformers/PEFT loads the resulting standard artifacts. A composed student still needs its emitted source files and `trust_remote_code=True`. Python can continue training and resave for fresh native Lisp reload; distillation adds no special inference format.

## Objective and limits

For selected positions, let `p = softmax(teacher_logits / T)` and `q = softmax(student_logits / T)`. The objective is:

```text
(1 - hard_weight) * T² * mean_positions(sum_vocabulary(p * (log(p) - log(q))))
  + hard_weight * mean_positions(cross_entropy(student_logits, next_token_label))
```

Temperature defaults to `1.0` and must be positive and finite. Hard weight defaults to `0.0` (pure distillation) and lies in `[0, 1]`; `1.0` is ordinary supervised cross entropy. Both terms use the same positions: causal next-token shifting, `-100` ignored labels, and target padding excluded. Omitted labels use input IDs as next-token targets; explicit labels can restrict distillation to completion tokens. Ignoring a loss position does not remove its input token from attention. Empty supervision is rejected even for pure distillation.

The first contract supports the exact native Llama and `tb_composed` classes, FP32 execution, zero dropout, and right padding. Teacher and student must be separate live models on the same CPU/GPU device with equal vocabulary sizes. **The caller must ensure identical token-ID meanings**; vocabulary size cannot establish this. The source-and-edited-student workflow retains the same tokenizer. Unknown subclasses, Python-worker models and other native families are rejected by this API. Each model retains its own configuration/context restrictions.

Teacher inference occurs before student differentiation, with no student parameter overrides in scope. Selected teacher logits stay native and are explicitly detached in the KL operation. Stable log-sum-exp avoids taking a logarithm of underflowed probabilities. Losses and gradients must be finite before any optimizer update. Online calls compute the complete vocabulary distribution and rerun the teacher per call, so both models and their activations must fit on the chosen device. Use the version 0.33 target API to compute once and remove the teacher during training, the version 0.34 dataset API to train lazily over many targets, or the version 0.38 [top-k format](distillation-top-k.md) to store a coarsened distribution. Top-k is a different offline objective; hidden-state loss and vocabulary chunking remain outside this online contract.

## Reproducible experiment

With the pinned SmolLM2 checkpoint cached:

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/distill-pretrained.lisp
uv run --no-sync python scripts/verify-distillation-example.py
```

The [example](../examples/distill-pretrained.lisp) rebuilds the 15-layer alternating-layer student, creates rank-4 query/value LoRA factors with seed 37, streams separate lazy train/held target datasets one batch at a time, and disposes the teacher. It performs 20 deterministically shuffled AdamW updates over four fixed training texts and evaluates two held-out texts, with at most 24 tokens per text, temperature 2, pure KL, learning rate 0.0005, zero weight decay and gradient clipping at 1. Outputs in `.build/distillation-example/` include both datasets, paired optimizer/iterator state, the adapter, report and native verification logits.

| Mean per-example temperature-scaled KL on the local Metal run | Before | After |
|---|---:|---:|
| Four training texts | 10.634911 | 8.204338 |
| Two held-out texts | 10.862764 | 10.454003 |

The independent Python verifier checks both dataset manifests, the iterator boundary, all four measurements and the exported adapter's held-text logits. The student's demonstration generation remains poor after this tiny experiment. Six short texts do not establish generalization or recovered language quality; this demonstrates a working learning/interchange loop, with its limitations visible. A useful compressed model needs substantially better training data and a separate task evaluation.

## Acceptance tests

```sh
uv run --no-sync python scripts/run-distillation-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-tests.py --device gpu --real --local-files-only
```

Each runner includes layer-reuse qualification; the real path also includes the full pretrained conversion gate. Tiny tied/untied stacks cover pure KL at temperature 0.7, mixed KL/CE at temperature 2, hard-only CE, and nonzero imported LoRA factors. Independent PyTorch computes every gradient and three clipped momentum-SGD updates. Native checkpoint continuation is exact; teachers and adapter bases remain frozen; caches reject stale versions; resources are released. Fresh Python loads all exports, continues distillation and resaves, and native Lisp checks the returned logits.

Real SmolLM2 tests use the original 30-layer Llama teacher and edited 15-layer composed student, with all 60 query/value LoRA gradient tensors and the same continuation checks. Python must load the original BF16 checkpoint with `dtype=torch.float32` to match native precision. CPU/Metal measurements are in [validation.md](validation.md), with reports at `.build/distillation-DEVICE/validation.json` and `real-validation.json`. CUDA remains pending external hardware qualification.
