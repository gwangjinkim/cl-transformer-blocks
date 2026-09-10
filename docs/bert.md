# BERT masked-language-model contract

The native BERT adapter loads standard tied-head Hugging Face `BertForMaskedLM` safetensors and computes masked-language-model logits in MLX under Common Lisp. It covers the bidirectional encoder path, native WordPiece tokenization, and ordinary Transformers export on CPU and Apple Metal.

```lisp
(load "scripts/load.lisp")
(tb:with-resource (model (tb:from-pretrained "./bert-masked-lm/" :device :gpu))
  (let* ((ids (tb:encode-text model "Common Lisp is [MASK]."))
         (batch (make-array (list 1 (length ids)) :initial-contents (list ids))))
    (tb:with-resource (logits (tb:forward model batch))
      (format t "logit shape: ~S~%" (tb:tensor-shape logits))))
  (tb:save-pretrained model ".build/my-bert-export/"))
```

The implementation follows the [Transformers BERT model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) and [BERT configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/configuration_bert.py). The test fixture and reference logits are created by the locked Transformers version.

## Supported behavior

| Feature | Contract |
|---|---|
| Architecture | `BertForMaskedLM` with a tied word-embedding/decoder matrix. |
| Embeddings | Word, learned absolute position, and caller-supplied token-type embeddings followed by LayerNorm; omitted token-type IDs default to zero. |
| Encoder | Bidirectional multihead attention, learned projection biases, post-attention and post-MLP LayerNorm, exact GELU. |
| Masks | Rank-two right-padding masks; valid tokens cannot attend to padded keys. |
| Head | Dense, exact GELU, LayerNorm, tied decoder projection, and learned vocabulary bias. |
| Tokenization | Native execution of the checkpoint's `tokenizer.json`; the fixture verifies BERT WordPiece special tokens and lowercasing. |
| Training | Explicit same-position labels with `-100` ignore markers; full-model or PEFT LoRA gradients and native optimizers when both configured dropouts are zero. |
| Adapters | Standard bias-free, zero-dropout PEFT LoRA/rsLoRA on the `query` and `value` projection suffixes; import, native creation/training, export, and merge. |
| Checkpoints | Exact canonical parameter validation and standard single/sharded FP32 export. |

Pass `:TOKEN-TYPE-IDS` to `FORWARD`, `LOSS-AND-GRADIENTS`, `TRAIN-STEP`, or `SGD-STEP` for sentence-pair segments. Supply a rank-two integer array matching the input IDs, with each value in `[0, type_vocab_size)`, including padded positions. Segment IDs are batch inputs and must be supplied again when resuming training. They do not alter the attention mask or the same-position masked-LM labeling rule.

```lisp
(tb:with-resource (model (tb:from-pretrained "./bert-masked-lm/" :device :gpu))
  ;; Illustrative IDs: use the values produced by your checkpoint's tokenizer.
  (let ((ids #2A((2 5 3 4 3)))
        (segments #2A((0 0 0 1 1)))
        (labels #2A((-100 -100 -100 7 -100))))
    (tb:with-resource (logits (tb:forward model ids :token-type-ids segments))
      (format t "pair logit shape: ~S~%" (tb:tensor-shape logits)))
    (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
      (tb:train-step model optimizer ids :labels labels :token-type-ids segments))))
```

The generic Python-worker `FORWARD` and `TRAIN-STEP` also pass `:TOKEN-TYPE-IDS` to Transformers. Native decoder families reject supplied token-type IDs. Use native [`ENCODE-BATCH`](tokenization.md) to construct padded input IDs, attention masks, and segment IDs directly from text pairs; `ENCODE-TEXT` remains the single-text ID-vector API.

This adapter does not accept `BertModel`, classification/QA heads, decoder or cross-attention BERT, relative positions, untied MLM heads, or KV caches. Load those models with `:execution :python` and the appropriate AutoClass. Dropout configuration is accepted for evaluation because Transformers disables it in `eval()`; native training requires both dropout probabilities to be zero. Native BERT LoRA accepts PEFT's built-in `BertForMaskedLM` automatic mapping and rejects arbitrary/custom mappings.

## Reproduce

```sh
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py
uv run --no-sync python scripts/run-bert-tests.py --device cpu
uv run --no-sync python scripts/run-bert-tests.py --device gpu
uv run --no-sync python scripts/run-bert-tests.py --device cpu --real
uv run --no-sync python scripts/run-bert-tests.py --device gpu --real
```

The real gate pins `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` and downloads its 440 MB safetensors checkpoint plus tokenizer/configuration assets. This historical artifact stores LayerNorm parameters as `gamma`/`beta` and includes the complete pooler/next-sentence group; native loading validates and canonicalizes those known forms, then export writes the modern masked-LM parameter set.

The suite compares complete batched logits, tokenizer output, every tiny full-model and adapter gradient, full-model and adapter SGD updates, native reload, checkpoint values, LoRA merge, and fresh Python `AutoModelForMaskedLM`/PEFT reloads. The real gate additionally checks all 202 masked-LM parameters, tokenizer behavior, complete logits, canonical export, and fresh Transformers reload. See [validation.md](validation.md) for observed errors and remaining hardware limits.
