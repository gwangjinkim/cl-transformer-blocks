"""Portable Python implementation of the explicit Lisp component vocabulary.

The emitted parallel-model support files supply the already verified tensor
primitives and Transformers protocol. This file defines the composed topology.
"""

from types import SimpleNamespace

from torch import nn

from .configuration_tb_composed import TBComposedConfig
from .modeling_tb_parallel import (
    TBParallelAttention,
    TBParallelBackbone,
    TBParallelForCausalLM,
    TBParallelMLP,
    TBParallelPreTrainedModel,
    TBParallelRMSNorm,
)


class TBComposedBlock(nn.Module):
    def __init__(self, hidden_size, block):
        super().__init__()
        attention = block["attention"]
        config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=attention["heads"],
            num_key_value_heads=attention["kv_heads"],
            head_dim=hidden_size // attention["heads"],
            rope_theta=attention["theta"],
            intermediate_size=block["feed_forward"]["intermediate_size"],
        )
        self.residual = block["residual"]
        self.input_layernorm = TBParallelRMSNorm(hidden_size, block["rms_norm_eps"])
        self.self_attn = TBParallelAttention(config)
        self.mlp = TBParallelMLP(config)
        if self.residual == "sequential":
            self.post_attention_layernorm = TBParallelRMSNorm(
                hidden_size, block["rms_norm_eps"]
            )

    def forward(self, hidden, attention_mask=None):
        normalized = self.input_layernorm(hidden)
        residual = hidden + self.self_attn(normalized, attention_mask=attention_mask)
        ffn_input = (
            normalized
            if self.residual == "parallel"
            else self.post_attention_layernorm(residual)
        )
        return residual + self.mlp(ffn_input)


class TBComposedBackbone(TBParallelBackbone):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TBComposedBlock(config.hidden_size, block)
                for block in config.block_configs
            ]
        )
        self.norm = TBParallelRMSNorm(config.hidden_size, config.rms_norm_eps)


class TBComposedForCausalLM(TBParallelForCausalLM):
    config_class = TBComposedConfig
    _no_split_modules = ["TBComposedBlock"]

    def __init__(self, config):
        TBParallelPreTrainedModel.__init__(self, config)
        self.model = TBComposedBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
