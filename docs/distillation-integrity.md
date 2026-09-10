# Content-addressed distillation datasets

Version 0.37 makes newly produced distillation datasets content-addressed. Lisp and Python compute SHA-256 over the exact bytes of every nested `distillation.json` and `teacher.safetensors`, then derive one ordered dataset identity. A saved iterator is bound to that identity, so a caller cannot resume against replaced targets merely because the replacement reused the same dataset ID, vocabulary and batch count.

All three producers emit dataset manifest version two:

- `save-distillation-dataset` for already materialized Lisp targets;
- `save-distillation-dataset-from-teacher` for memory-bounded native production; and
- `scripts/create-distillation-dataset.py` for any compatible Python `AutoModelForCausalLM` teacher.

The nested batch format is unchanged. Its version still describes target storage: batch version one is FP32 and batch version two is FP16. Dataset version two describes collection integrity and may contain either target dtype.

## Manifest contract

Each ordered entry adds two lowercase hexadecimal digests:

```json
{
  "path": "batches/000000",
  "selected_positions": 5,
  "manifest_sha256": "...64 lowercase hexadecimal characters...",
  "weights_sha256": "...64 lowercase hexadecimal characters..."
}
```

`manifest_sha256` hashes the exact bytes of that batch's `distillation.json`. `weights_sha256` hashes the exact bytes of `teacher.safetensors`. The outer `content_sha256` hashes this ASCII framing, where `<LF>` is byte `0a` and `<TAB>` is byte `09`:

```text
cl-transformer-blocks-distillation-dataset-v2<LF>
vocab_size=<decimal><LF>
batch_count=<decimal><LF>
000000<TAB><selected_positions><TAB><manifest_sha256><TAB><weights_sha256><LF>
000001<TAB><selected_positions><TAB><manifest_sha256><TAB><weights_sha256><LF>
...
```

The six-digit index is the entry's physical zero-based position. Digests are lowercase. Decimal values have no sign, grouping or leading zeros, apart from the fixed-width index. This framing binds vocabulary, count, order, selected-position counts, batch metadata and weight bytes without depending on JSON object-key order or whitespace in the outer manifest.

The caller's `dataset_id` remains a semantic name and is deliberately outside the content digest. Two byte-identical datasets can have different application names and the same name can identify a later revision, but iterator restore checks both the semantic ID and content digest.

## Loading and continuation

`load-distillation-dataset` verifies the outer digest, both exact files for every entry, and the existing batch metadata contract. It does not create an MLX backend or allocate target tensors, although hashing necessarily reads every target file from disk. `distillation-dataset-content-sha256` returns the verified digest.

`load-distillation-dataset-batch`, `next-distillation-dataset-batch`, and `distill-dataset-step` rehash the selected entry immediately before loading it. A file replaced after the dataset was opened is therefore rejected before it can train the student, and a failed training load leaves the iterator position unchanged.

Iterator state for a verified dataset uses state format version two and stores `content_sha256`. Restore requires the exact content digest in addition to dataset ID, vocabulary, count, shuffle setting and seed. Model/optimizer and iterator state are still separate atomic directories; save both at one application checkpoint boundary.

Legacy dataset manifest version one remains readable. Its `distillation-dataset-content-sha256` value is `nil`, its iterator state remains version one, and no file-content verification is claimed. Republish legacy targets with a current producer to obtain a verified dataset.

SHA-256 detects accidental corruption and replacement relative to the manifest or a saved iterator. It does not authenticate an untrusted dataset when an attacker can replace the files and their manifest together. Loading from an actively modified directory also has a small check-to-open race; use immutable local snapshots for training.

## Reproduce the checks

```sh
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --teacher-device mps
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-distillation-dataset-tests.py --device gpu --real --local-files-only --teacher-device mps
```

The suite uses standard SHA-256 vectors, recomputes every Lisp/Python artifact with Python `hashlib`, corrupts both nested metadata and safetensor bytes, checks rejection at initial open and later lazy access, verifies that a cursor cannot attach to different content with the same semantic ID, and exercises explicit version-one compatibility.
