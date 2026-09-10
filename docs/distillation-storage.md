# Store distillation logits in FP16

Version 0.36 adds an optional FP16 storage format for reusable [distillation batches](distillation-targets.md) and [lazy datasets](distillation-datasets.md). It halves the tensor payload on disk while keeping the native objective, gradients and optimizer arithmetic in FP32.

Dataset manifest version and batch manifest version are independent. Since version 0.37 every newly produced dataset uses [content-addressed dataset format v2](distillation-integrity.md), whether its nested batches use FP32 batch format v1 or FP16 batch format v2.

FP32 remains the default. Choose FP16 when target disk or transfer size matters and the measured quantization error is acceptable for the training run.

## Common Lisp producers

Pass `:storage-dtype :float16` when publishing one target, a collection of existing targets, or a dataset streamed from a native teacher:

```lisp
(tb:save-distillation-batch target ".build/target-fp16/"
                            :storage-dtype :float16)

(tb:save-distillation-dataset targets ".build/dataset-fp16/"
                              :dataset-id "corpus-v3"
                              :storage-dtype :float16)

(tb:save-distillation-dataset-from-teacher
 teacher examples ".build/streamed-fp16/"
 :dataset-id "corpus-v3" :storage-dtype :float16)
```

`save-distillation-batch` and `save-distillation-dataset` default to `:preserve`: a freshly created batch is FP32, while a loaded FP16 batch is saved as FP16 again. An explicit `:float32` or `:float16` converts storage for the new artifact without changing the live batch. `save-distillation-dataset-from-teacher` defaults to FP32 because it starts from examples rather than stored batches.

`distillation-batch-storage-dtype` reports `:float32` or `:float16` provenance. The batch's live `teacher_logits` tensor is always FP32. Loading checks the raw safetensor dtype before conversion, so metadata cannot disguise another representation.

FP16 conversion is evaluated and checked for finite values before publication. A value outside the finite FP16 range aborts the staged publication and preserves the previous destination.

## Python producers

Teacher execution precision and artifact storage precision are separate options:

```sh
uv run --no-sync python scripts/create-distillation-batch.py \
  --teacher organization/teacher-model \
  --revision COMMIT_SHA \
  --batch-json batch.json \
  --output target-fp16 \
  --dtype float32 \
  --storage-dtype float16

uv run --no-sync python scripts/create-distillation-dataset.py \
  --teacher organization/teacher-model \
  --revision COMMIT_SHA \
  --dataset-json batches.json \
  --dataset-id corpus-v3 \
  --output dataset-fp16 \
  --dtype float32 \
  --storage-dtype float16
```

`--dtype` controls `AutoModelForCausalLM` execution on CPU, MPS or CUDA. `--storage-dtype` controls the CPU safetensor after selected logits have been converted to FP32. Both default to `float32`. The manifest retains the teacher execution dtype as provenance.

## Versioned artifact contract

The accepted batch pairs are:

| Format version | Manifest dtype | Safetensor dtype | Native in-memory dtype |
|---:|---|---|---|
| 1 | `float32` | FP32 | FP32 |
| 2 | `float16` | FP16 | FP32 |
| 3, top-k | `float32` | FP32 probabilities, int32 indices/sentinel | FP32 probabilities, int32 indices |
| 4, top-k | `float16` | FP16 probabilities, int32 indices/sentinel | FP32 probabilities, int32 indices |

Other versions, crossed pairs such as version 1 plus `float16`, and a tensor whose raw dtype disagrees with the valid pair are rejected. Dense `teacher.safetensors` contains exactly one rank-two tensor named `teacher_logits`; top-k tensor names and validation are documented separately. IDs, labels, mask, selected positions, vocabulary and tokenizer-meaning preconditions are unchanged. A dataset may mix supported batch versions because each batch manifest is validated independently.

FP16 storage costs:

```text
2 bytes × selected positions × vocabulary size
```

The tiny five-position, 32-token target falls from 640 to 320 payload bytes. Five SmolLM2 positions fall from 983,040 to 491,520 bytes. Safetensors headers and JSON metadata add a small fixed overhead.

FP16 is lossy. Across the qualified real SmolLM2 CPU/Metal runs, the maximum stored-logit difference from a fresh FP32 teacher was `0.00793`; native/PyTorch gradient disagreement based on the same stored target was at most `1.25e-4`; three clipped momentum-SGD updates differed by at most `3.73e-9`. These measurements qualify the test workload, not every teacher or objective. Keep FP32 for exact preservation or measure the effect on the intended corpus.

Dense versions still store every vocabulary logit. FP16 reduces the constant factor but not growth with vocabulary or selected positions. Version 0.38 supplies a separately versioned [top-k distribution](distillation-top-k.md) in FP32 or FP16; quantized distributions and hidden-state targets remain separate contracts.

## Reproduce the qualification

```sh
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --teacher-device mps
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --real --local-files-only --teacher-device mps
```

The dataset runner includes the single-target gate. It exercises native and Python FP32/FP16 producers, materialized and streaming Lisp datasets, lazy restore-to-FP32, exact FP16 resave, malformed artifacts, shuffled training, optimizer and iterator continuation, and fresh Transformers/PEFT reload. CUDA runs through the same existing manual workflow but still needs external hardware evidence.
