# Produce distillation datasets with bounded native memory

Version 0.35 adds a streaming Common Lisp producer for the [distillation dataset](distillation-datasets.md) format. The earlier native producer accepts live `distillation-batch` objects, so every full-vocabulary teacher-logit tensor must exist before publication. The streaming producer keeps the native teacher resident but creates, writes and disposes one target per source callback. Version 0.36 lets that callback write [FP16 target storage](distillation-storage.md), and version 0.37 publishes the same [content-addressed dataset](distillation-integrity.md) as the materialized and Python producers.

This bounds additional native target memory by the largest individual batch. It also lets an application expose data from a file, database or generated corpus through a CLOS generic without first building one in-memory list.

## Create examples and publish a dataset

`make-distillation-example` snapshots rank-two input IDs, effective labels and an optional attention mask. It checks matching shapes, int32-compatible values, binary masks and nonempty next-token supervision. The teacher-specific vocabulary bound is checked during production.

```lisp
(let ((examples
        (list
         (tb:make-distillation-example
          #2A((1 234 567 890))
          :labels #2A((-100 -100 567 890))
          :attention-mask #2A((1 1 1 1)))
         (tb:make-distillation-example
          #2A((1 345 678))
          :labels #2A((-100 345 678))))))
  (tb:with-resource
      (teacher (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
    (tb:save-distillation-dataset-from-teacher
     teacher examples ".build/train-targets/" :dataset-id "my-corpus-v2"
     :storage-dtype :float16)))
```

Lists and vectors implement `map-distillation-examples` directly. The producer invokes the source in order. For each example it runs the native teacher, materializes the selected FP32 logits under a temporary owned backend, synchronously saves that batch as FP32 or FP16, and disposes the target before asking the source for another example. It does not mutate the teacher or advance its model version. FP16 conversion is also temporary and is released before the callback returns.

The resulting directory is the same portable dataset format produced by `save-distillation-dataset` and `scripts/create-distillation-dataset.py`. Training, deterministic shuffling, iterator state and Python verification require no new loader.

## Stream from an application source

Specialize `map-distillation-examples` for an application class. The method calls its function once per `distillation-example`; it does not need to know the final count in advance.

```lisp
(defclass line-corpus ()
  ((path :initarg :path :reader corpus-path)
   (model :initarg :model :reader corpus-tokenizer)))

(defmethod tb:map-distillation-examples (function (source line-corpus))
  (with-open-file (stream (corpus-path source))
    (loop for text = (read-line stream nil)
          while text
          for tokens = (tb:encode-text
                        (corpus-tokenizer source) text :add-special-tokens t)
          do (funcall function
                      (tb:make-distillation-example
                       (make-array (list 1 (length tokens))
                                   :initial-contents (list tokens))))))
  source)
```

Custom methods must yield sequentially. They may release or reuse their own source storage after the callback returns because production completes that example synchronously. Concurrent callbacks against one teacher are outside this contract.

An empty source, a value other than `distillation-example`, invalid tokens for the teacher, nonfinite logits, a source error, an I/O error, or a publication failure aborts the operation. The destination remains the previous complete dataset. Partial staging data is removed. Retrying starts source enumeration from the beginning; version 0.35 does not resume a partially produced dataset.

## Choosing the producer

- Use `save-distillation-dataset-from-teacher` for native teachers and large target collections. It is the recommended native creation path.
- Use `save-distillation-dataset` when the application already owns reusable `distillation-batch` objects or wants to combine previously computed targets.
- Use `scripts/create-distillation-dataset.py` for any compatible Python `AutoModelForCausalLM`, especially architectures without a qualified native adapter. It also keeps one teacher resident and writes one target at a time.

All three paths default to full FP32 logits. `:storage-dtype :float16` or Python `--storage-dtype float16` reduces stored precision. Version 0.38 also lets the streaming Lisp producer accept `:top-k` and `:temperature`, and the Python producer accepts matching command-line options; see [top-k distributions](distillation-top-k.md). Streaming reduces peak native memory but does not reduce teacher computation. Quantized distributions remain a separate contract.

## Reproduce the gates

```sh
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --real --local-files-only
```

The suite supplies three variable-shape examples through a custom source. It checks that live native handles return to the teacher-only baseline after every callback, caller-array mutations cannot alter snapshots, source errors and invalid/empty streams preserve an existing dataset, and the teacher version is unchanged. It compares FP32 and FP16 streamed targets exactly with their materialized native artifacts, while fresh PyTorch independently compares both precisions with the teacher. Tiny and real SmolLM2 paths pass on CPU and actual Apple Metal; see [validation.md](validation.md).

The updated [real example](../examples/distill-pretrained.lisp) streams separate four-batch training and two-batch held datasets, disposes the teacher, then performs lazy shuffled LoRA distillation with exact optimizer and iterator continuation.
