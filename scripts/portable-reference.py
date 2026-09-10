"""Independent Torch reference and fresh Transformers custom-code verification."""

import argparse
import json
import math
from pathlib import Path
import shutil

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F


torch.set_num_threads(1)
IDS = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
LABELS = torch.tensor([[-100, 4, 5, -100], [-100, -100, 4, 3]])


def rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(value, positions, theta):
    dimension = value.shape[-1]
    inverse = 1.0 / (theta ** (torch.arange(0, dimension, 2).float() / dimension))
    frequencies = torch.outer(positions.float(), inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return value * embedding.cos()[None, None] + rotate_half(value) * embedding.sin()[None, None]


class RMSNorm(nn.Module):
    def __init__(self, dimension, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, hidden):
        variance = hidden.float().square().mean(-1, keepdim=True)
        return (hidden.float() * torch.rsqrt(variance + self.epsilon)).type_as(hidden) * self.weight


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        hidden = config["hidden_size"]
        self.theta = config["rope_theta"]
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)

    def forward(self, hidden, attention_mask=None):
        batch, steps, _ = hidden.shape
        positions = torch.arange(steps, device=hidden.device)

        def heads(projection, count):
            return projection.view(batch, steps, count, self.head_dim).transpose(1, 2)

        query = apply_rope(heads(self.q_proj(hidden), self.heads), positions, self.theta)
        key = apply_rope(heads(self.k_proj(hidden), self.kv_heads), positions, self.theta)
        value = heads(self.v_proj(hidden), self.kv_heads)
        groups = self.heads // self.kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        allowed = torch.ones(steps, steps, dtype=torch.bool, device=hidden.device).tril()
        allowed = allowed[None, None].expand(batch, 1, steps, steps)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).type_as(query)
        attended = torch.matmul(probabilities, value)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, steps, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class ParallelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, hidden, attention_mask=None):
        normalized = self.input_layernorm(hidden)
        return hidden + self.self_attn(normalized, attention_mask) + self.mlp(normalized)


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            [ParallelBlock(config) for _ in range(config["num_hidden_layers"])]
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.norm(hidden)


class ReferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, attention_mask=None):
        return self.lm_head(self.model(input_ids, attention_mask))


def configuration():
    return {
        "architectures": ["TBParallelForCausalLM"],
        "model_type": "tb_parallel",
        "portable_architecture_version": 1,
        "parallel_residual": True,
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 40,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "dtype": "float32",
    }


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = configuration()
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "Common": 3, "Lisp": 4}
    vocabulary.update({f"token{i}": i for i in range(5, 32)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                                  bos_token="<bos>", eos_token="<eos>", unk_token="<pad>")
    fast.save_pretrained(root / "text-assets")
    fast.add_tokens(["overflow"])
    fast.save_pretrained(root / "invalid-text-assets")
    torch.manual_seed(73)
    model = ReferenceModel(config).float().eval()
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    weights = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    output = model(IDS, MASK)
    shifted_logits = output[:, :-1].contiguous()
    shifted_labels = LABELS[:, 1:].contiguous()
    loss = F.cross_entropy(
        shifted_logits.view(-1, config["vocab_size"]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().contiguous()
        for name, parameter in model.named_parameters()
    }
    save_file(gradients, str(root / "gradients.safetensors"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.sub_(0.01 * parameter.grad)
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(root / "updated.safetensors"),
        metadata={"format": "pt"},
    )
    (root / "reference.json").write_text(
        json.dumps(
            {
                "input_ids": IDS.tolist(),
                "attention_mask": MASK.tolist(),
                "labels": LABELS.tolist(),
                "logits": output.detach().tolist(),
                "loss": loss.item(),
            }
        )
        + "\n"
    )


def verify(source, exported, updated=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    source = Path(source)
    exported = Path(exported)
    config = AutoConfig.from_pretrained(
        exported, trust_remote_code=True, local_files_only=True
    )
    assert type(config).__name__ == "TBParallelConfig"
    try:
        type(config)(parallel_residual=False)
    except ValueError as error:
        assert "residual topology" in str(error)
    else:
        raise AssertionError("portable Python config accepted another topology as v1")
    model = AutoModelForCausalLM.from_pretrained(
        exported,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    expected = load_file(str(source / ("updated.safetensors" if updated else "model.safetensors")))
    assert model.state_dict().keys() == expected.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=2e-6 if updated else 0, rtol=2e-4 if updated else 0)
    reference = ReferenceModel(configuration()).eval()
    reference.load_state_dict(expected)
    with torch.no_grad():
        torch.testing.assert_close(
            model(IDS, attention_mask=MASK).logits,
            reference(IDS, MASK),
            atol=3e-5,
            rtol=3e-4,
        )
        generated = model.generate(
            IDS[:1, :2], max_new_tokens=2, do_sample=False, use_cache=False,
            eos_token_id=None,
        )
        assert generated.shape == (1, 4)
    model.train()
    result = model(IDS, attention_mask=MASK, labels=LABELS)
    assert result.loss.isfinite()
    result.loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    try:
        model(IDS, attention_mask=torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))
    except ValueError as error:
        assert "left padded" in str(error)
    else:
        raise AssertionError("portable Python model accepted unsupported left padding")
    print("Fresh Transformers custom-code reload matches:", exported)


def smoke(exported):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    exported = Path(exported)
    tokenizer = AutoTokenizer.from_pretrained(exported, local_files_only=True,
                                              trust_remote_code=False)
    assert tokenizer.encode("Common Lisp", add_special_tokens=False) == [3, 4]
    assert tokenizer.decode([3, 4]) == "Common Lisp"
    config = AutoConfig.from_pretrained(
        exported, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        exported,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    )
    assert type(config).__name__ == "TBParallelConfig"
    assert model.lm_head.weight is model.model.embed_tokens.weight
    result = model(IDS, attention_mask=MASK, labels=LABELS)
    assert result.logits.shape == (2, 4, 32)
    assert result.logits.isfinite().all() and result.loss.isfinite()
    result.loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(IDS, attention_mask=MASK).logits
    resaved = exported.with_name(exported.name + "-python-resaved")
    shutil.rmtree(resaved, ignore_errors=True)
    model.save_pretrained(resaved, safe_serialization=True)
    tokenizer.save_pretrained(resaved)
    assert (resaved / "configuration_tb_parallel.py").is_file()
    assert (resaved / "modeling_tb_parallel.py").is_file()
    reloaded = AutoModelForCausalLM.from_pretrained(
        resaved,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(IDS, attention_mask=MASK).logits,
            expected,
            atol=0,
            rtol=0,
        )
    (resaved / "python-reference.json").write_text(
        json.dumps(
            {
                "input_ids": IDS.tolist(),
                "attention_mask": MASK.tolist(),
                "logits": expected.tolist(),
            }
        )
        + "\n"
    )
    print("Transformers trained and resaved Lisp-created model:", resaved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fixture", "verify", "smoke"])
    parser.add_argument("path")
    parser.add_argument("exported", nargs="?")
    parser.add_argument("--updated", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "fixture":
        fixture(arguments.path)
    elif arguments.action == "verify":
        verify(arguments.path, arguments.exported, arguments.updated)
    else:
        smoke(arguments.path)
