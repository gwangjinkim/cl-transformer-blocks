# Reuse teacher targets without a resident teacher

Version 0.33 adds reusable distillation batches. Compute selected teacher logits once, dispose the teacher, and run many full-model or LoRA student updates from the saved targets. A versioned safetensors/JSON format works in both directions: native Common Lisp can create it, and an ordinary Python `AutoModelForCausalLM` can create targets that native Lisp consumes.

This matters when the teacher is larger than the student. Online [distillation](distillation.md) keeps both models and their activations available for every step. Reusable targets replace the live teacher during student training with one matrix of shape `(selected next-token positions, vocabulary size)` per batch. Version 0.36 can [store that matrix in FP16](distillation-storage.md). Version 0.38 can instead [store a top-k distribution plus exact aggregate tail](distillation-top-k.md), often reducing large-vocabulary payloads by hundreds of times.

## Native Common Lisp workflow

```lisp
(load "scripts/load.lisp")

(let ((target nil))
  ;; MAKE-DISTILLATION-BATCH evaluates and materializes the selected logits.
  ;; The returned object owns its own MLX backend.
  (tb:with-resource (teacher (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
    (setf target
          (tb:make-distillation-batch
           teacher #2A((1 234 567 890))
           :labels #2A((-100 -100 567 890))
           :attention-mask #2A((1 1 1 1)))))

  ;; TEACHER is disposed here. TARGET remains usable and can be persisted.
  (unwind-protect
       (progn
         (tb:save-distillation-batch target ".build/teacher-target/"
                                     :storage-dtype :float16)
         (tb:with-resource (student (tb:from-pretrained ".build/recomposed-example/" :device :gpu))
           (tb:make-lora student :rank 4 :alpha 8)
           (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.0005))
             (dotimes (epoch 10)
               (tb:distill-batch-step student target optimizer
                                      :temperature 2.0 :hard-weight 0.3
                                      :max-grad-norm 1.0)))
           (tb:save-adapter student ".build/student-adapter/")))
    (tb:dispose target)))
```

`make-distillation-batch` snapshots the input IDs, effective labels and attention mask. Later caller mutations cannot change the target. Omitted labels use the input IDs, as in ordinary causal training. Padding and `-100` labels determine the selected next-token positions when the target is created.

`distillation-batch-loss-and-gradients` returns the loss and owned student gradients. `distill-batch-step` uses the same native objective, optimizer ownership, clipping, cache invalidation and checkpoint behavior as online `distill-step`. Temperature and hard weight are chosen at training time, because the artifact stores raw logits. You can reuse one target with different valid objective settings. A student checkpoint does not contain its targets; reload the target artifacts and resupply them after restoring the optimizer.

`save-distillation-batch` publishes `teacher.safetensors` and `distillation.json` atomically. `load-distillation-batch` creates an owned CPU or GPU backend:

```lisp
(tb:with-resource (target (tb:load-distillation-batch ".build/teacher-target/" :device :gpu))
  (tb:with-resource (student (tb:from-pretrained ".build/recomposed-example/" :device :gpu))
    (multiple-value-bind (loss gradients)
        (tb:distillation-batch-loss-and-gradients student target :temperature 2.0)
      (unwind-protect
           (format t "Loss: ~F~%" loss)
        (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)))))
```

The loaded target and student must use the same device and vocabulary size. The caller must still guarantee identical token-ID meanings. The exact IDs, labels and mask in the artifact prevent accidentally applying its logits to another batch, but equal vocabulary size and equal integer IDs do not prove equal tokenizer semantics.

## Create targets from any Python causal teacher

The utility loads any dense-logit causal model accepted by `AutoModelForCausalLM`, including an architecture that has no native Lisp adapter. The resulting teacher is no longer needed during Lisp training:

```json
{
  "input_ids": [[1, 234, 567, 890]],
  "attention_mask": [[1, 1, 1, 1]],
  "labels": [[-100, -100, 567, 890]]
}
```

```sh
uv run --no-sync python scripts/create-distillation-batch.py \
  --teacher organization/teacher-model \
  --revision COMMIT_SHA \
  --batch-json batch.json \
  --output teacher-target \
  --dtype float32 \
  --storage-dtype float16
```

Use `--local-files-only` for a local/cached source, `--trust-remote-code` only for reviewed custom model code, and `--device mps` or `--device cuda` when desired. `--dtype` controls teacher execution and is recorded as provenance; `--storage-dtype` independently selects `float32` or `float16`. Both default to FP32. The utility stages files beside a new destination and refuses to replace an existing path. Native publication supports atomic replacement of an existing target directory.

This Python route broadens which models can act as teachers, not which architectures execute natively in Lisp. The student remains one of the qualified native Llama or `tb_composed` models and must use the same token IDs. Models without standard dense causal logits, encoder-only models, vision/audio models and task-specific heads require a separate target contract.

## Artifact contract and cost

Dense `teacher.safetensors` contains exactly one rank-two tensor named `teacher_logits`. Version one pairs FP32 metadata and storage; version two pairs FP16 metadata and storage. Sparse versions three and four use the four tensors documented in the [top-k contract](distillation-top-k.md). `distillation.json` also records vocabulary size, teacher source/revision provenance, input IDs, effective labels, and an optional attention mask. Python-created metadata may record the teacher model type and execution dtype. Load verifies representation, raw tensor dtypes, shapes and values, restores FP16 floating targets to owned FP32 training tensors, and rejects crossed versions/dtypes, empty supervision, out-of-vocabulary values, non-binary masks, and unexpected tensors. See the [storage contract](distillation-storage.md).

The raw-logit format keeps temperature adjustable at training time. FP32 preserves the stored values; FP16 applies the measured approximation described in the [storage guide](distillation-storage.md). Payload storage is:

```text
4 bytes × selected positions × vocabulary size (FP32)
2 bytes × selected positions × vocabulary size (FP16)
```

Five positions for SmolLM2's 49,152-token vocabulary occupy 983,040 FP32 or 491,520 FP16 tensor bytes. Large corpora can become much larger than the model checkpoint. Split them into workload-sized batch artifacts and account for disk/device space. Target logits can also reveal information about the teacher and its input; treat them according to the same data-access policy as the source text and model outputs.

The dense batch contract stores full logits and permits a different temperature on every training call. The [top-k contract](distillation-top-k.md) fixes temperature when materialized and optimizes a coarsened distribution. Hidden-state/attention matching and target prefetching remain separate work. Version 0.34 adds [lazy dataset indexing, deterministic shuffling, and iterator continuation](distillation-datasets.md) over either batch representation.

## Reproduce the checks

```sh
uv run --no-sync python scripts/run-distillation-target-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-target-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-target-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-target-tests.py --device gpu --real --local-files-only
```

The runner includes layer reuse and online distillation first. Tests compare live-teacher and cached losses/gradients, dispose the teacher before reuse, mutate the caller arrays, save/reload FP32 and FP16 targets on an independent backend, run three clipped momentum-SGD steps, and check resource cleanup and failed atomic replacement. Fresh Python verifies native-created logits. Targets generated by the Python utility drive native gradients and updates, which Python independently reproduces. The gate also requires exact FP16 resave and strict rejection of crossed metadata/tensor dtypes. One tiny teacher is `tb_parallel`, which this native online distillation API deliberately rejects; its Python-created target still trains the composed Lisp student.

The real gate uses the original SmolLM2 teacher, its 15-layer composed student and an existing trainable LoRA adapter. CPU/Metal evidence and exact counts are in [validation.md](validation.md). The [real example](../examples/distill-pretrained.lisp) now groups its six targets into lazy train/held datasets and disposes the teacher before its 20 student updates. CUDA remains pending the external hardware workflow.
