# Compatibility and capability inspection

`INSPECT-PRETRAINED` checks whether a checkpoint configuration is supported by the native runtime before loading weights or allocating an MLX device. It returns a property list instead of signaling for an unknown architecture or a recognized architecture with unsupported semantics.

```lisp
(let ((report (tb:inspect-pretrained ".build/models/qwen2.5-0.5b/")))
  (if (getf report :supported-p)
      (tb:from-pretrained ".build/models/qwen2.5-0.5b/" :device :gpu)
      (tb:from-pretrained ".build/models/qwen2.5-0.5b/"
                          :execution :python :device :gpu)))
```

Local inspection reads only `config.json` and checks tokenizer asset presence. A Hub source uses the same safe-artifact resolver as native loading, so the current helper resolves the complete safetensors/tokenizer snapshot. Pin `:REVISION`, and use `:LOCAL-FILES-ONLY T` when inspection must not access the network.

The report includes:

| Key | Meaning |
|---|---|
| `:SUPPORTED-P` | The configuration passes the registered native semantic contract. This does not replace complete weight-name and shape validation during loading. |
| `:REASON` | Native rejection text for an unsupported report. |
| `:EXECUTION`, `:BACKEND` | `:NATIVE`/`:MLX` for inspection and native models, or `:PYTHON`/`:PYTORCH` for a loaded worker model. |
| `:MODEL-TYPE`, `:ARCHITECTURE` | Preserved Hugging Face model type and the selected native family or loaded Transformers class. |
| `:OPERATIONS` | Supported forward, training, cache, generation, tokenization, PEFT, checkpoint, export, processor, publication, or custom-Python export operations. Native BERT also advertises `:TOKEN-TYPE-IDS` for explicit segment inputs. |
| `:SUPPORTED-DEVICES` | Device kinds implemented by the execution path. It is not a claim that the current machine has a GPU. |
| `:REQUESTED-DEVICE`, `:DEVICE`, `:DEVICE-API` | Populated for loaded models. The API distinguishes native Metal/CUDA and worker MPS/CUDA execution. |
| `:CHECKPOINT-DTYPE`, `:EXECUTION-DTYPE`, `:PRECISION-POLICY` | Stored precision, actual or fixed execution precision, and conversion/ownership policy. |
| `:TOKENIZER`, `:PROCESSOR` | Available tokenizer asset/worker class and processor class. Native tokenizer availability also adds `:BATCH-TOKENIZATION` to `:OPERATIONS`. |
| `:QUANTIZATION`, `:COMPILED-P` | Explicit current quantization and compiled-execution state. |
| `:MASK-POLICY`, `:RESTRICTIONS` | The accepted mask convention and concise semantic limits. |
| `:PYTHON-FALLBACK-P` | Whether the explicit Transformers worker is the available broader route. |

`MODEL-CAPABILITIES` reports a loaded model's actual state:

```lisp
(tb:with-resource
    (model (tb:from-pretrained ".build/models/bert-base-uncased/" :device :gpu))
  (let ((report (tb:model-capabilities model)))
    (format t "~S ~S ~S~%"
            (getf report :backend)
            (getf report :device-api)
            (getf report :operations))))
```

Architecture-specific capability methods are ordinary CLOS methods. A new native adapter extends `ARCHITECTURE-CAPABILITIES` alongside its configuration validator, parameter schema, and graph methods. The report never changes execution automatically; callers explicitly select `:EXECUTION :NATIVE` or `:EXECUTION :PYTHON`.

The Lisp-originated `tb_parallel` and `tb_composed` models report `:CUSTOM-PYTHON-EXPORT`. Their checkpoints include reviewed AutoConfig/AutoModel source files and exact `auto_map` metadata; this does not weaken the requirement for an explicit `:TRUST-REMOTE-CODE T` when selecting Python execution. The [composed model](components.md) supports full-model training and advertises `:LORA`/`:RSLORA` for its seven attention/feed-forward projections. Save and reload a newly constructed base before attaching an adapter so its PEFT metadata identifies reusable base weights.
