"""Versioned vocabulary for Lisp-composed rotary-attention/SwiGLU decoders."""

import copy
import math

from transformers import PretrainedConfig


def _positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {name}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {name}")
    if integer and (not isinstance(value, int) or value > 2147483647):
        raise ValueError(f"invalid integer {name}")


class TBComposedConfig(PretrainedConfig):
    model_type = "tb_composed"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        block_configs=None,
        num_hidden_layers=None,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        composition_version=1,
        use_cache=False,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_configs = (
            copy.deepcopy(block_configs)
            if block_configs is not None
            else [
                {
                    "attention": {
                        "type": "rope",
                        "heads": 8,
                        "kv_heads": 8,
                        "theta": 10000.0,
                    },
                    "feed_forward": {"type": "swiglu", "intermediate_size": 1536},
                    "rms_norm_eps": 1e-5,
                    "residual": "sequential",
                }
            ]
        )
        if not isinstance(self.block_configs, list):
            raise ValueError("block descriptions must be a list")
        self.num_hidden_layers = (
            len(self.block_configs) if num_hidden_layers is None else num_hidden_layers
        )
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.composition_version = composition_version
        self.use_cache = use_cache
        if (
            type(composition_version) is not int
            or composition_version != 1
            or use_cache is not False
            or not isinstance(tie_word_embeddings, bool)
        ):
            raise ValueError("unsupported composition version or execution settings")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "max_position_embeddings",
        ):
            _positive(getattr(self, name), name, integer=True)
        for name in ("rms_norm_eps", "initializer_range"):
            _positive(getattr(self, name), name)
        if (
            not isinstance(self.block_configs, list)
            or len(self.block_configs) != self.num_hidden_layers
        ):
            raise ValueError("block descriptions must match num_hidden_layers")
        for block in self.block_configs:
            if not isinstance(block, dict) or set(block) != {
                "attention",
                "feed_forward",
                "rms_norm_eps",
                "residual",
            }:
                raise ValueError("invalid block description")
            attention, ffn = block["attention"], block["feed_forward"]
            if not isinstance(attention, dict) or set(attention) != {
                "type",
                "heads",
                "kv_heads",
                "theta",
            }:
                raise ValueError("invalid attention description")
            if not isinstance(ffn, dict) or set(ffn) != {"type", "intermediate_size"}:
                raise ValueError("invalid feed-forward description")
            if (
                attention["type"] != "rope"
                or ffn["type"] != "swiglu"
                or block["residual"] not in ("sequential", "parallel")
            ):
                raise ValueError("unregistered component semantics")
            for name in ("heads", "kv_heads"):
                _positive(attention[name], name, integer=True)
            _positive(attention["theta"], "theta")
            _positive(ffn["intermediate_size"], "intermediate_size", integer=True)
            _positive(block["rms_norm_eps"], "block epsilon")
            if (
                attention["heads"] % attention["kv_heads"]
                or hidden_size % attention["heads"]
                or (hidden_size // attention["heads"]) % 2
            ):
                raise ValueError("invalid grouped-attention dimensions")
