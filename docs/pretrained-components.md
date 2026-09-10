# Start Lisp components from pretrained Llama weights

Version 0.30 adds `save-composed-pretrained`: convert a supported native Llama model into an equivalent `tb_composed` checkpoint. Reload that checkpoint to inspect its public Lisp block descriptions, run it, train the full model or LoRA adapters, and exchange the result with Python. The weights come from the pretrained model, so this path does not start from random initialization.

## Convert and use a model

After bootstrap, download and qualify the pinned SmolLM2 checkpoint:

```sh
uv run --no-sync python scripts/run-composition-import-tests.py --device gpu --real
```

This uses `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Add `--local-files-only` for subsequent runs using that cached snapshot. CPU works with `--device cpu`.

```lisp
(load "scripts/load.lisp")

(tb:with-resource (source (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
  (tb:save-composed-pretrained source ".build/my-pretrained-blocks/"
                              :max-shard-size 16777216))

(tb:with-resource (model (tb:from-pretrained ".build/my-pretrained-blocks/" :device :gpu))
  ;; These 30 block descriptions correspond to the loaded, pretrained weights.
  (format t "~D blocks~%" (length (tb:transformer-blocks model)))
  (let* ((ids (tb:encode-text model "Common Lisp is a programming language"
                              :add-special-tokens nil))
         (tokens (tb:generate model ids :max-new-tokens 16)))
    (format t "~A~%" (tb:decode-tokens model tokens :skip-special-tokens t))))
```

The [runnable example](../examples/pretrained-components.lisp) does this conversion and text generation. Set `TB_MODEL` to another local supported Llama checkpoint; `TB_DEVICE` defaults to CPU.

```sh
TB_DEVICE=gpu sbcl --noinform --no-sysinit --no-userinit \
  --script examples/pretrained-components.lisp
```

## Train in Lisp, use in Python

Use the converted model itself to retain its pretrained weights. `transformer-blocks` returns descriptions; passing those descriptions to `make-transformer` constructs a separately initialized model, not a weight copy.

```lisp
(tb:with-resource (model (tb:from-pretrained ".build/my-pretrained-blocks/" :device :gpu))
  (tb:make-lora model :rank 8 :alpha 16 :targets '("q_proj" "v_proj"))
  ;; Prepare token IDs, attention masks and supervised labels for your own dataset.
  ;; Train with TRAIN-STEP and a native optimizer; see training.md.
  (tb:save-adapter model ".build/my-pretrained-adapter/"))
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    ".build/my-pretrained-blocks", trust_remote_code=True, local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    ".build/my-pretrained-blocks", trust_remote_code=True, local_files_only=True,
)
adapted = PeftModel.from_pretrained(base, ".build/my-pretrained-adapter", is_trainable=True)
```

The Lisp snippet creates and saves an identity adapter; actual learning requires training data and updates. The acceptance suite performs a real native AdamW adapter update on the converted SmolLM2 model and verifies the result in Python PEFT. The separate [training guide](training.md) covers supervision, clipping, optimizer checkpoints and merging. A full-model update can be exported with `save-pretrained`; Python can train and resave the converted architecture for native Lisp reload.

## Exact conversion boundary

The supported source is the exact native Llama class, with bias-free SiLU/SwiGLU projections, pre-RMSNorm sequential residuals, zero dropout and unscaled split-half RoPE. `hidden_size` must equal `num_attention_heads * head_dim`; the broader native Llama loader also supports head layouts that composed version one cannot represent. MHA, GQA, MQA, tied and untied vocabulary heads are covered. Unknown numerical variants and subclasses are rejected, even when their weight names happen to match.

The converter derives all sequential block descriptions from validated source settings, checks complete canonical weight names and shapes, and exports the current base weights. It also accepts a base explicitly updated in Lisp. Active adapters require an explicit `merge-adapter` or `remove-adapter` first. Merging includes their contribution in the base; removing excludes it.

This operation preserves the represented architecture. To select, reorder or repeat converted blocks, use the separate [layer-reuse API](layer-reuse.md) added in version 0.31; changing the stack changes the model function and requires quality evaluation. Replacing residual topologies, transplanting weights into arbitrary new blocks, and converting Qwen2/GPT-2/BERT into this component vocabulary remain outside this contract. Those models retain their existing native adapters. The output identifies itself as `tb_composed`, so Python needs the four emitted source files and explicit `trust_remote_code=True`. Returning to a standard Llama configuration after architectural experimentation is not an automatic public operation.

Native tensors are borrowed only during synchronous serialization through a private export view. No extra model initialization or Lisp-array copy of the weights occurs. Only the completed directory is returned; reloading creates an independently owned model. Keep the source alive and avoid concurrent mutation during conversion, as with ordinary export.

Publication stages all files and atomically replaces the destination; failures preserve the previous complete output. Export over the source directory is rejected. Tokenizer files and attached tokenizer snapshots are preserved. If `generation_config.json` exists, its other settings are retained and `use_cache` becomes false, because that separate file overrides the model config in Transformers. Native cached greedy generation remains available. FP16/BF16 input checkpoints follow the existing FP32 conversion/export policy; exact weight transport means exact native FP32 values, not identical checkpoint bytes.

## Reproduce the checks

```sh
uv run --no-sync python scripts/run-composition-import-tests.py --device cpu
uv run --no-sync python scripts/run-composition-import-tests.py --device gpu
uv run --no-sync python scripts/run-composition-import-tests.py --device cpu --real --local-files-only
uv run --no-sync python scripts/run-composition-import-tests.py --device gpu --real --local-files-only
```

Tiny fixtures originate in ordinary Transformers and use sharded safetensors. Tests cover exact weights, complete full-model gradients, native SGD updates before/after conversion, LoRA training, Python training/resave/native reload, tied heads, tokenizers, cached greedy tokens, generation settings, source immutability, unsupported configurations, atomic failure recovery and native handle cleanup. Real SmolLM2 checks cover every weight, logits, tokenization, cached greedy output and a native-trained adapter reloaded in Python. Reports are `.build/composition-import-DEVICE/validation.json` and `real-validation.json`; numerical evidence is in [validation.md](validation.md). CUDA requires the external hardware workflow.
