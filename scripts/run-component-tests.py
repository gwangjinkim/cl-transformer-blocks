"""Independent functional Torch oracle and fresh AutoClass checks for Lisp blocks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def verify(directory):
    import torch
    from safetensors.torch import load_file
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)
    ids = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.tensor([[-100, 4, 5, -100], [-100, -100, 4, 3]])
    config = json.loads((directory / "config.json").read_text())
    weights = {
        k: v.requires_grad_()
        for k, v in load_file(str(directory / "model.safetensors")).items()
    }

    def oracle(weights, config, ids, mask):
        def norm(x, name, eps):
            return (
                x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weights[name]
            )

        def rotate(x, theta):
            # Complex multiplication implements split-half RoPE independently
            # of the emitted Python model's sine/cosine rotate-half code.
            d = x.shape[-1]
            frequencies = theta ** (-torch.arange(0, d, 2).float() / d)
            angles = torch.arange(x.shape[-2]).float()[:, None] * frequencies[None, :]
            rotation = torch.polar(torch.ones_like(angles), angles)
            rotated = torch.complex(x[..., : d // 2], x[..., d // 2 :]) * rotation
            return torch.cat((rotated.real, rotated.imag), dim=-1)

        hidden = F.embedding(ids, weights["model.embed_tokens.weight"])
        for index, block in enumerate(config["block_configs"]):
            prefix = f"model.layers.{index}."
            normalized = norm(
                hidden, prefix + "input_layernorm.weight", block["rms_norm_eps"]
            )
            a = block["attention"]
            d = config["hidden_size"] // a["heads"]

            def projection(name, heads):
                value = F.linear(
                    normalized, weights[prefix + "self_attn." + name + ".weight"]
                )
                return value.reshape(ids.shape[0], ids.shape[1], heads, d).transpose(
                    1, 2
                )

            q = rotate(projection("q_proj", a["heads"]), a["theta"])
            k = rotate(projection("k_proj", a["kv_heads"]), a["theta"])
            v = projection("v_proj", a["kv_heads"])
            k = k.repeat_interleave(a["heads"] // a["kv_heads"], dim=1)
            v = v.repeat_interleave(a["heads"] // a["kv_heads"], dim=1)
            allowed = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool).tril()[
                None, None
            ]
            allowed = allowed & mask[:, None, None, :].bool()
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            joined = attended.transpose(1, 2).reshape(*ids.shape, config["hidden_size"])
            residual = hidden + F.linear(
                joined, weights[prefix + "self_attn.o_proj.weight"]
            )
            ff = (
                normalized
                if block["residual"] == "parallel"
                else norm(
                    residual,
                    prefix + "post_attention_layernorm.weight",
                    block["rms_norm_eps"],
                )
            )
            gate = F.silu(F.linear(ff, weights[prefix + "mlp.gate_proj.weight"]))
            up = F.linear(ff, weights[prefix + "mlp.up_proj.weight"])
            hidden = residual + F.linear(
                gate * up, weights[prefix + "mlp.down_proj.weight"]
            )
        hidden = norm(hidden, "model.norm.weight", config["rms_norm_eps"])
        head = (
            "model.embed_tokens.weight"
            if config["tie_word_embeddings"]
            else "lm_head.weight"
        )
        return F.linear(hidden, weights[head])

    expected = oracle(weights, config, ids, mask)
    native = load_file(str(directory / "native-logits.safetensors"))["logits"]
    torch.testing.assert_close(native, expected, atol=2e-6, rtol=3e-4)
    loss = F.cross_entropy(expected[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
    assert (
        abs(
            loss.item()
            - json.loads((directory / "native-loss.json").read_text())["loss"]
        )
        < 2e-6
    )
    loss.backward()
    gradients = load_file(str(directory / "native-gradients.safetensors"))
    updated = load_file(str(directory / "updated/model.safetensors"))
    assert set(gradients) == set(weights) == set(updated)
    for name, weight in weights.items():
        torch.testing.assert_close(gradients[name], weight.grad, atol=2e-6, rtol=3e-4)
        torch.testing.assert_close(
            updated[name], weight - 0.01 * weight.grad, atol=2e-6, rtol=3e-4
        )

    # A fresh Transformers loader reads only exported files, not source imports.
    for path in (directory, directory / "sharded"):
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        ).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(ids, attention_mask=mask).logits, native, atol=2e-6, rtol=3e-4
            )
    model = AutoModelForCausalLM.from_pretrained(
        directory, trust_remote_code=True, local_files_only=True
    ).eval()
    model(ids, attention_mask=mask, labels=labels).loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, gradients[name], atol=2e-6, rtol=3e-4
        )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    optimizer.step()
    output = directory / "python-updated"
    model.save_pretrained(output)
    with torch.no_grad():
        logits = model(ids, attention_mask=mask).logits
        generated = model.generate(
            torch.tensor([[3, 4]]),
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            eos_token_id=None,
        )
    (output / "reference.json").write_text(
        json.dumps({"logits": logits.tolist(), "generated": generated[0].tolist()})
    )

    tied_dir = directory / "tied"
    tied_weights = {
        k: v.requires_grad_()
        for k, v in load_file(str(tied_dir / "model.safetensors")).items()
    }
    tied_config = json.loads((tied_dir / "config.json").read_text())
    tied_logits = oracle(tied_weights, tied_config, ids[:1], torch.ones_like(ids[:1]))
    F.cross_entropy(
        tied_logits[:, :-1].reshape(-1, 32), ids[:1, 1:].reshape(-1)
    ).backward()
    tied_gradients = load_file(str(tied_dir / "native-gradients.safetensors"))
    assert set(tied_gradients) == set(tied_weights)
    for name, weight in tied_weights.items():
        torch.testing.assert_close(
            tied_gradients[name], weight.grad, atol=2e-6, rtol=3e-4
        )
    tied = AutoModelForCausalLM.from_pretrained(
        tied_dir, trust_remote_code=True, local_files_only=True
    ).eval()
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    with torch.no_grad():
        torch.testing.assert_close(
            tied(ids[:1]).logits, tied_logits, atol=2e-6, rtol=3e-4
        )
    for bad in (
        {"composition_version": True},
        {"block_configs": []},
        {"block_configs": "invalid"},
        {"hidden_size": 15},
        {"use_cache": True},
    ):
        try:
            model.config.__class__(**{**config, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid configuration accepted: {bad}")
    report = {
        "logits_max_abs_error": (native - expected.detach()).abs().max().item(),
        "gradient_max_abs_error": max(
            (gradients[n] - w.grad).abs().max().item() for n, w in weights.items()
        ),
        "sgd_max_abs_error": max(
            (updated[n] - (w - 0.01 * w.grad)).abs().max().item()
            for n, w in weights.items()
        ),
        "tied_gradient_max_abs_error": max(
            (tied_gradients[n] - w.grad).abs().max().item()
            for n, w in tied_weights.items()
        ),
        "canonical_parameters": len(weights),
        "canonical_tied_parameters": len(tied_weights),
    }
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        f"Independent functional oracle, all gradients/SGD, AutoClass export, and tied weights passed: {directory}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.verify:
        verify(args.verify.resolve())
        return
    env = {**os.environ, "TB_DEVICE": args.device}
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-components.lisp",
        env=env,
    )
    run(
        sys.executable,
        "scripts/run-component-tests.py",
        "--verify",
        f".build/components-created-{args.device}",
    )
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-components-roundtrip.lisp",
        env=env,
    )
    print(f"All composed Transformer gates passed: {args.device}")


if __name__ == "__main__":
    main()
