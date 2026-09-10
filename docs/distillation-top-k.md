# Compress reusable teacher targets with top-k distributions

Version 0.38 adds a sparse target representation for [reusable distillation batches](distillation-targets.md) and [lazy datasets](distillation-datasets.md). Instead of storing every teacher logit, it stores the teacher's `k` most probable token classes and one exact aggregate event for all remaining classes. Common Lisp and Python can both produce and consume the same safetensors/JSON contract.

## Objective

At target temperature `T`, the producer computes the full teacher log probabilities. For every selected next-token position it stores:

- `teacher_topk_log_probs`: `[positions, k]` FP32 or FP16 log probabilities;
- `teacher_topk_indices`: `[positions, k]` int32 vocabulary indices;
- `teacher_tail_log_prob`: `[positions, 1]`, the stable `logsumexp` over every class outside the top-k set; and
- `teacher_vocab_size`: one int32 value that makes the sparse tensor payload independently checkable against its manifest.

Training computes the student's full stable log probabilities at the same `T`, gathers its probability for each recorded class, and aggregates all other student classes into one tail event. It minimizes token-mean `T² * KL(q || p)` over these `k + 1` events. FP16 targets are renormalized after load/conversion. The optional supervised mixture remains `(1-w)*soft + w*CE`.

This coarsened KL is a principled lower bound on the full-vocabulary KL by the data-processing inequality. It preserves exact teacher and student tail mass, while discarding how the teacher distributes that mass among individual tail tokens. It therefore changes the objective and should be evaluated for the intended task. The target temperature is part of the artifact; training with another temperature is rejected.

## Common Lisp

```lisp
(let ((target nil))
  (tb:with-resource (teacher (tb:from-pretrained "teacher/" :device :gpu))
    (setf target
          (tb:make-top-k-distillation-batch
           teacher ids :labels labels :attention-mask mask
           :top-k 64 :temperature 2.0)))
  (unwind-protect
       (progn
         (tb:save-distillation-batch target "target/" :storage-dtype :float16)
         (tb:distill-batch-step student target optimizer
                                :temperature 2.0 :hard-weight 0.3))
    (tb:dispose target)))
```

`distillation-batch-representation`, `distillation-batch-top-k`, and `distillation-batch-temperature` expose the contract. To create a content-addressed dataset while retaining only one target at a time:

```lisp
(tb:save-distillation-dataset-from-teacher
 teacher examples "targets/" :dataset-id "corpus-v1"
 :top-k 64 :temperature 2.0 :storage-dtype :float16)
```

`save-distillation-dataset` also accepts already materialized dense and top-k batches. A dataset may contain either representation because every nested batch is independently versioned and hashed.

## Python producer

```sh
uv run --no-sync python scripts/create-distillation-batch.py \
  --teacher organization/model --batch-json batch.json --output target \
  --top-k 64 --temperature 2.0 --storage-dtype float16

uv run --no-sync python scripts/create-distillation-dataset.py \
  --teacher organization/model --dataset-json batches.json \
  --dataset-id corpus-v1 --output targets \
  --top-k 64 --temperature 2.0 --storage-dtype float16
```

The Python route accepts a dense-logit `AutoModelForCausalLM` even when its architecture has no native Lisp adapter. Native Lisp student execution remains limited to the documented Llama/composed models and requires matching token-ID meanings.

## Formats and size

Dense formats remain version 1/FP32 and version 2/FP16. Top-k formats are version 3/FP32 and version 4/FP16 and require `representation: "top_k"`, `top_k`, and `temperature` in the manifest. Load rejects crossed versions and dtypes, unexpected tensors, non-finite or positive log probabilities, duplicate/out-of-range indices, a mismatched vocabulary sentinel, and incompatible shapes.

For `n` selected positions and vocabulary `V`, tensor payload sizes are:

```text
dense FP32: 4nV
dense FP16: 2nV
top-k FP32: n(8k + 4) + 4
top-k FP16: n(6k + 2) + 4
```

The top-k formulas include int32 indices, one tail probability per position, and the vocabulary sentinel. For five positions from SmolLM2's 49,152-token vocabulary at `k=64`, FP32 payload falls from 983,040 bytes to 2,584 bytes, about 380 times smaller; FP16 top-k uses 1,934 bytes, about 508 times smaller than dense FP32. Headers and JSON add fixed overhead.

## Reproduce the checks

```sh
uv run --no-sync python scripts/run-distillation-target-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-target-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --teacher-device mps
```

The gates independently compare each stored top probability and exact tail against PyTorch teacher logits, then compare all native gradients and three optimizer updates with a fresh PyTorch implementation of the coarsened objective. They cover Lisp/Python batch producers, materialized and streaming datasets, FP32/FP16, exact resave, content hashes, malformed tensors/metadata, native resource cleanup, and standard model or PEFT export. CUDA uses the same dataset workflow on the external NVIDIA runner and remains unqualified until that evidence exists.
