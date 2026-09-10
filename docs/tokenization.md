# Native batched text and sentence pairs

`TB:ENCODE-BATCH` runs the model's `tokenizer.json` with native Hugging Face Tokenizers. It returns three ordinary rank-two Lisp arrays as multiple values: input IDs, attention masks, and token-type IDs. Rows follow input order. The arrays require no `DISPOSE` call and can feed native inference or training directly.

```lisp
(tb:with-resource (model (tb:from-pretrained "./bert-masked-lm/" :device :gpu))
  (multiple-value-bind (ids mask segments)
      (tb:encode-batch model '("Common Lisp" "Fast")
                       :text-pairs '("is [MASK] and fun" "Lisp"))
    (tb:with-resource (logits (tb:forward model ids :attention-mask mask
                                                 :token-type-ids segments))
      (format t "pair logits: ~S~%" (tb:tensor-shape logits)))))
```

For masked-token training, construct a label array with the same shape as `ids`, put target IDs at supervised positions and `-100` elsewhere, then pass `ids`, `mask`, and `segments` to `TRAIN-STEP`. Native BERT training requires zero configured dropout. Native decoder models accept IDs and masks; omit the third value when calling their inference/training APIs, since their current contracts do not accept segment IDs.

| Argument | Contract |
|---|---|
| `texts` | Nonempty list or vector of strings. A bare string is not a batch. |
| `:text-pairs` | `NIL` for single texts, or an equally sized list/vector of second strings. An empty string still denotes a second sequence. |
| `:add-special-tokens` | Defaults to `T`; the checkpoint's postprocessor supplies special tokens and segment rules. |
| `:padding` | `:LONGEST` (default) pads to the longest encoded row; `:MAX-LENGTH` pads to the explicit maximum. Both pad on the right. Padding segment IDs are zero. |
| `:max-length` | Optional positive ceiling including special tokens, no greater than native model context. Required for fixed padding or truncation. Without it, model context is the ceiling. |
| `:truncation` | `NIL` (default) rejects oversized rows. `T` requests rightward, longest-first truncation, reserving space for special tokens. |
| `:pad-token-id` | Defaults to the model configuration's `pad_token_id`. An override must fit the model vocabulary. A padding ID is required only when a row actually needs padding. |

The result always has a rectangular, nonempty shape. An empty text can produce valid special tokens; a row that encodes to zero tokens is rejected. Invalid input shapes, embedded NUL characters, oversized outputs, insufficient special-token space, and emitted IDs outside the model vocabulary signal `COMPATIBILITY-ERROR`.

The maximum is a strict model-input ceiling even without truncation. This is more restrictive than Transformers calls that ignore `max_length` under some padding options. No implicit truncation, left padding, overflow windows, offsets, Python tokenizer wrappers, or chat-template execution are provided here. The [Python processor API](python-worker.md) remains the broader path for those features and for Python-worker models; `ENCODE-BATCH` currently has a native-model implementation.

Padding uses the selected numeric ID and the engine's attention mask. A literal padding-token string in the original text remains attended; only added padding is masked out. Tokenizer and model vocabularies, special IDs, and token meanings must agree. An `ATTACH-TOKENIZER` asset snapshot also serves batch calls on Lisp-created models.

Each batch crosses the FFI once. Untruncated calls borrow the resident tokenizer; truncation uses a private native clone so successful and failed calls cannot change later tokenization or exported assets. No measured tokenizer speedup is claimed. Rebuild the Rust tokenizer library when upgrading:

```sh
uv run --no-sync python scripts/bootstrap.py
CARGO_TARGET_DIR=.build/tokenizer cargo test --locked --offline --manifest-path native/tokenizer/Cargo.toml
uv run --no-sync python scripts/run-bert-tests.py --device cpu
uv run --no-sync python scripts/run-bert-tests.py --device gpu
```

The BERT gate compares all three arrays exactly with independent Transformers references for single texts and pairs, special tokens on/off, fixed/longest padding, short truncation, empty pair members, Unicode, and literal padding tokens. It verifies native logits and masked-token loss, fresh Python pair-tokenizer reloads, rejection cases, and unchanged resident settings. Real-checkpoint variants and the portable-model gate exercise the same API; see [validation.md](validation.md).
