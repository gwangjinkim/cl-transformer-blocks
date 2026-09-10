"""PyTorch implementation emitted with the Lisp-defined parallel decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_tb_parallel import TBParallelConfig


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(value, positions, theta):
    dimension = value.shape[-1]
    inverse = 1.0 / (
        theta
        ** (torch.arange(0, dimension, 2, device=value.device).float() / dimension)
    )
    frequencies = torch.outer(positions.float(), inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos()[None, None].to(dtype=value.dtype)
    sine = embedding.sin()[None, None].to(dtype=value.dtype)
    return value * cosine + _rotate_half(value) * sine


class TBParallelRMSNorm(nn.Module):
    def __init__(self, dimension, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, hidden):
        variance = hidden.float().square().mean(-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + self.epsilon)
        return normalized.to(dtype=hidden.dtype) * self.weight


class TBParallelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)

    def forward(self, hidden, attention_mask=None, position_offset=0):
        batch, steps, _ = hidden.shape
        positions = torch.arange(
            position_offset,
            position_offset + steps,
            device=hidden.device,
        )

        def split_heads(projection, count):
            return projection.view(batch, steps, count, self.head_dim).transpose(1, 2)

        query = _apply_rope(split_heads(self.q_proj(hidden), self.heads), positions, self.theta)
        key = _apply_rope(split_heads(self.k_proj(hidden), self.kv_heads), positions, self.theta)
        value = split_heads(self.v_proj(hidden), self.kv_heads)
        groups = self.heads // self.kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        allowed = torch.ones(
            steps, steps, dtype=torch.bool, device=hidden.device
        ).tril()
        allowed = allowed[None, None].expand(batch, 1, steps, steps)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attended = torch.matmul(probabilities, value)
        joined = attended.transpose(1, 2).reshape(batch, steps, -1)
        return self.o_proj(joined)


class TBParallelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TBParallelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = TBParallelRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = TBParallelAttention(config)
        self.mlp = TBParallelMLP(config)

    def forward(self, hidden, attention_mask=None):
        normalized = self.input_layernorm(hidden)
        return (
            hidden
            + self.self_attn(normalized, attention_mask=attention_mask)
            + self.mlp(normalized)
        )


class TBParallelBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TBParallelBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = TBParallelRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty time)")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask values must be zero or one")
            if torch.any(attention_mask[:, 0] == 0):
                raise ValueError("a batch row cannot be fully masked or left padded")
            if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
                raise ValueError("only right padding is supported")
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
        return self.norm(hidden)


class TBParallelPreTrainedModel(PreTrainedModel):
    config_class = TBParallelConfig
    base_model_prefix = "model"
    _no_split_modules = ["TBParallelBlock"]
    _supports_flash_attn = False
    _supports_sdpa = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, TBParallelRMSNorm):
            module.weight.data.fill_(1.0)


class TBParallelForCausalLM(TBParallelPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TBParallelBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
        past_key_values=None,
        **kwargs,
    ):
        del kwargs
        if input_ids is None:
            raise ValueError("input_ids are required")
        if output_attentions or output_hidden_states:
            raise NotImplementedError("attention and hidden-state outputs are not implemented")
        if use_cache or past_key_values is not None:
            raise NotImplementedError("Python KV caching is not implemented for portable v1")
        hidden = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if not return_dict:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=loss, logits=logits)
