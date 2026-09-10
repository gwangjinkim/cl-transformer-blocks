# Train from lazy, shuffled teacher-target datasets

Version 0.34 groups reusable [teacher-target batches](distillation-targets.md) into a validated dataset. Lisp can now build or consume many offline teacher batches without keeping every logit tensor, a teacher, or Python resident during student training. One target is loaded onto the selected MLX device for one access and released afterward.

Version 0.35 adds the recommended [streaming native producer](distillation-streaming.md), which also avoids holding every target while Lisp creates the dataset.

Version 0.36 adds optional [FP16 target storage](distillation-storage.md) to every producer. Loading remains lazy and restores one target to FP32 only when accessed.

Version 0.37 adds [content-addressed integrity](distillation-integrity.md). Every exact batch manifest and safetensor is SHA-256 verified at dataset open and again before lazy access; iterator state binds to the resulting ordered dataset identity.

Version 0.38 lets every producer write [top-k distributions with exact aggregate tails](distillation-top-k.md). Lazy loading, hashing, shuffling and continuation work unchanged for dense or sparse nested batches.

This layer supplies four missing training controls:

- atomic publication of an ordered target collection;
- deterministic local shuffling with an explicit epoch and seed;
- a cursor that advances only after successful native optimizer updates; and
- portable iterator state for exact continuation beside the model/optimizer checkpoint; and
- cross-language content identity and corruption detection.

## Build a dataset in Common Lisp

`save-distillation-dataset` accepts a nonempty list or vector of live `distillation-batch` objects. All batches must use the same vocabulary and validated FP32 training targets; they may use dense logits or top-k probabilities. `:storage-dtype :float16` writes the representation's FP16 format; the default preserves each batch's source storage dtype. `dataset-id` is a required application-level name; the generated content digest prevents a saved cursor from attaching to different bytes under the same name.

Use `save-distillation-dataset-from-teacher` when starting from a native teacher and input examples. It writes and disposes each target before requesting the next example; `save-distillation-dataset` remains useful when targets already exist.

```lisp
(let ((targets nil))
  (unwind-protect
       (progn
         (tb:with-resource (teacher
                            (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
           (setf targets
                 (mapcar (lambda (ids)
                           (tb:make-distillation-batch teacher ids))
                         training-batches)))
         ;; The whole dataset replaces DESTINATION in one directory publication.
         (tb:save-distillation-dataset
          targets ".build/train-targets/" :dataset-id "my-corpus-v1"
          :storage-dtype :float16))
    (mapc #'tb:dispose targets)))
```

Publication writes this layout:

```text
train-targets/
  distillation-dataset.json
  batches/
    000000/
      distillation.json
      teacher.safetensors
    000001/
      distillation.json
      teacher.safetensors
    ...
```

The dataset manifest fixes the ordered canonical paths, batch count, vocabulary, selected-position counts, format version, caller-supplied dataset ID, hashes of each exact nested file, and an ordered content digest. Loading verifies every batch and reads the target files for hashing, but it does not create an MLX backend or allocate target tensors. `load-distillation-dataset-batch` verifies the selected files again, then loads one physical zero-based index as an owned `distillation-batch`; dispose it after use. The precise portable framing and legacy behavior are documented in [content-addressed datasets](distillation-integrity.md).

## Create the same dataset with one Python teacher load

The Python producer accepts a JSON object containing a nonempty `batches` array. Each member has the same `input_ids`, optional `labels`, and optional `attention_mask` fields as the single-batch utility.

```json
{
  "batches": [
    {
      "input_ids": [[1, 234, 567, 890]],
      "labels": [[-100, -100, 567, 890]],
      "attention_mask": [[1, 1, 1, 1]]
    },
    {
      "input_ids": [[1, 345, 678]],
      "labels": [[-100, 345, 678]],
      "attention_mask": [[1, 1, 1]]
    }
  ]
}
```

```sh
uv run --no-sync python scripts/create-distillation-dataset.py \
  --teacher organization/teacher-model \
  --revision COMMIT_SHA \
  --dataset-json batches.json \
  --dataset-id my-corpus-v1 \
  --output train-targets \
  --dtype float32 \
  --storage-dtype float16
```

The teacher is loaded once, evaluated batch by batch, and released after the process exits. The complete output is staged beside a new destination and appears only after all targets and the dataset manifest have been written. Existing destinations are refused. `--local-files-only`, reviewed `--trust-remote-code`, and `--device cpu|mps|cuda` have the same meanings as the single-target utility. Teacher execution and FP32/FP16 storage are separate choices; execution dtype is recorded as provenance.

Python creation broadens the set of teachers. Native dataset training still requires a qualified native Llama or `tb_composed` student. All batches and the student must use identical token-ID meanings; a dataset ID and equal vocabulary size do not prove tokenizer equivalence.

## Deterministic epochs and training

`load-distillation-dataset` takes the native target device and optional shuffle settings. Opening is lazy and starts at epoch zero, position zero.

```lisp
(let ((dataset (tb:load-distillation-dataset
                ".build/train-targets/" :device :gpu :shuffle t :seed 37)))
  (tb:with-resource (student
                     (tb:from-pretrained ".build/recomposed-student/" :device :gpu))
    (tb:make-lora student :rank 8 :alpha 16 :seed 37)
    (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.0005))
      (dotimes (epoch 5)
        (tb:start-distillation-dataset-epoch dataset epoch)
        (loop while
          (tb:distill-dataset-step
           student dataset optimizer :temperature 2.0
           :hard-weight 0.3 :max-grad-norm 1.0)))
      (tb:save-training-checkpoint student optimizer ".build/checkpoint/")
      (tb:save-distillation-dataset-state dataset ".build/dataset-state/"))))
```

The shuffle is specified independently of the Common Lisp implementation's random state. Start with physical indices `0..count-1`, initialize unsigned 32-bit state to `(seed + epoch) mod 2^32`, then run Fisher–Yates from the last index down to one. Before each swap update state with:

```text
state = (1664525 × state + 1013904223) mod 2^32
swap(index, state mod (index + 1))
```

Both seed and epoch must be unsigned 32-bit integers. With three batches, seed 41 and epoch 3 produce order `[1, 0, 2]` in Lisp and Python.

`distill-dataset-step` returns the loss and physical batch index as two values. At the end of the epoch it returns `nil` without changing the model, optimizer, or cursor. It loads and disposes the target internally and increments the position only after `distill-batch-step` succeeds. A shape, compatibility, finite-value, or optimizer error therefore leaves that batch available for retry.

`next-distillation-dataset-batch` supports custom processing. It returns an owned batch plus its physical index and consumes the position after the batch loads successfully. Later application failure cannot roll that cursor back automatically, so use `distill-dataset-step` for transactional training.

## Exact continuation

`save-distillation-dataset-state` atomically writes `dataset-state.json` with the dataset ID, verified content digest, vocabulary, batch count, shuffle flag, seed, epoch, and next unconsumed position. Restore the model/optimizer and iterator from the same logical training point:

```lisp
(let ((dataset (tb:load-distillation-dataset
                ".build/train-targets/" :device :gpu :shuffle t :seed 37)))
  (tb:restore-distillation-dataset-state dataset ".build/dataset-state/")
  ;; Restore the corresponding model and optimizer, then continue DISTILL-DATASET-STEP.
  )
```

Restore rejects a different dataset ID, content digest, vocabulary, count, shuffle choice, seed, invalid epoch, or cursor. The dataset ID remains a caller-controlled semantic name; version 0.37 supplies the independently computed content identity. Dataset state and the model/optimizer checkpoint are two independently atomic directories; publish both at the same application checkpoint boundary. A process failure between the two publications can leave different generations, so retain or name checkpoint generations when stronger cross-directory recovery is required.

## Reproduce the acceptance gates

```sh
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --real --local-files-only
```

The runner first executes the reusable-target, online-distillation, layer-reuse, and conversion gates. It then builds the same three-batch FP32 and FP16 datasets independently in Lisp and Python, including native materialized and streaming production. It checks every target and student gradient, recomputes every digest in Python, corrupts metadata and safetensor bytes, verifies the portable shuffle, rejects malformed manifests and state, injects publication failures, and resumes FP16 training from paired model/optimizer and content-bound iterator checkpoints. Fresh PyTorch independently repeats the shuffled clipped momentum-SGD updates and compares the standard full-model or PEFT export.

CPU and actual Apple Metal pass both the tiny full-model and real SmolLM2/LoRA paths; measurements are in [validation.md](validation.md). The updated [real example](../examples/distill-pretrained.lisp) streams separate train and held datasets, releases its teacher before training, saves iterator state, and exports an ordinary PEFT adapter.

The current loader performs synchronous hashing and per-batch disk-to-device loading. The version 0.35 native producer streams target creation, version 0.36 adds FP16 storage, version 0.37 verifies immutable local snapshots, and version 0.38 adds top-k distributions. The loader does not prefetch, combine variable batches, pack sequences, shard work across processes, or stream remote datasets. Those features need measured memory/throughput goals and their own reproducible state and artifact contracts. CUDA remains pending the external NVIDIA workflow.
