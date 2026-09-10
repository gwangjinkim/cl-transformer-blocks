"""Verify selected/repeated Lisp blocks against independently rearranged PyTorch modules."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = {"identity": [0, 1], "drop": [1], "reverse": [1, 0], "repeat": [1, 0, 1]}


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def cases(device, real):
    output = ROOT / f".build/recomposition-{device}"
    if real:
        return [
            (
                ROOT / f".build/composition-import-{device}/real",
                output / "real",
                list(range(0, 30, 2)),
            )
        ]
    source = ROOT / f".build/components-created-{device}"
    return [
        (source if kind == "untied" else source / "tied", output / kind / name, indices)
        for kind in ("untied", "tied")
        for name, indices in SELECTIONS.items()
    ]


def python_gate(device, real, phase):
    import torch
    from torch import nn
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    ids = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.tensor([[-100, 4, 5, -100], [-100, -100, 4, 3]])
    report = {}

    def load(path):
        return AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        ).eval()

    def max_error(actual, expected):
        torch.testing.assert_close(
            actual, expected, atol=1e-3 if real else 2e-6, rtol=3e-4
        )
        return (actual - expected).abs().max().item()

    for source, target, indices in cases(device, real):
        original = load(source)
        # This reference uses only source modules and the explicit test selection.
        # It never reads the Lisp-produced target configuration or target weights.
        expected = copy.deepcopy(original)
        expected.model.layers = nn.ModuleList(
            [copy.deepcopy(original.model.layers[i]) for i in indices]
        )
        expected.config.block_configs = [
            copy.deepcopy(original.config.block_configs[i]) for i in indices
        ]
        expected.config.num_hidden_layers = len(indices)
        if phase == "fixture":
            target.parent.mkdir(parents=True, exist_ok=True)
            with torch.set_grad_enabled(not real):
                out = expected(ids, attention_mask=mask, labels=labels)
            reference = {
                "input_ids": ids.tolist(),
                "attention_mask": mask.tolist(),
                "labels": labels.tolist(),
                "logits": out.logits.detach().tolist(),
                "loss": out.loss.item(),
                "indices": indices,
            }
            (target.parent / f"{target.name}-reference.json").write_text(
                json.dumps(reference) + "\n"
            )
            if not real:
                out.loss.backward()
                save_file(
                    {n: p.grad.contiguous() for n, p in expected.named_parameters()},
                    str(target.parent / f"{target.name}-gradients.safetensors"),
                )
            continue
        actual = load(target)
        a, b = dict(expected.named_parameters()), dict(actual.named_parameters())
        assert a.keys() == b.keys()
        assert actual.config.block_configs == expected.config.block_configs
        assert actual.config.num_hidden_layers == len(indices)
        for name in a:
            torch.testing.assert_close(b[name], a[name], atol=0, rtol=0)
        if actual.config.tie_word_embeddings:
            assert actual.lm_head.weight is actual.model.embed_tokens.weight
        provenance = json.loads((target / "recomposition.json").read_text())
        assert provenance == {
            "format_version": 1,
            "source_id": str(source.resolve()) + "/",
            "source_revision": None,
            "source_layer_count": original.config.num_hidden_layers,
            "source_modified": False,
            "layer_indices": indices,
        }
        with torch.no_grad():
            logits = expected(ids, attention_mask=mask).logits
            native = load_file(str(target / "native-logits.safetensors"))["logits"]
            metrics = {
                "canonical_parameters": len(a),
                "layers": len(indices),
                "native_logits_max_abs_error": max_error(native, logits),
                "python_logits_max_abs_error": max_error(
                    actual(ids, attention_mask=mask).logits, logits
                ),
            }
            options = dict(max_new_tokens=4, do_sample=False, eos_token_id=None)
            generated = actual.generate(torch.tensor([[3, 4]]), **options)[0].tolist()
            assert (
                generated
                == expected.generate(torch.tensor([[3, 4]]), **options)[0].tolist()
            )
            assert generated == json.loads(
                (target / "native-generation.json").read_text()
            )
        if real:
            assert (target / "resaved/recomposition.json").read_bytes() == (
                target / "recomposition.json"
            ).read_bytes()
            first = AutoTokenizer.from_pretrained(
                source, trust_remote_code=True, local_files_only=True
            )
            second = AutoTokenizer.from_pretrained(
                target, trust_remote_code=True, local_files_only=True
            )
            assert first.encode("Common Lisp is fast and fun") == second.encode(
                "Common Lisp is fast and fun"
            )
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
                "chat_template.jinja",
            ):
                if (source / name).is_file():
                    assert (source / name).read_bytes() == (target / name).read_bytes()
        adapted = PeftModel.from_pretrained(load(target), target / "adapter").eval()
        with torch.no_grad():
            metrics["trained_adapter_logits_max_abs_error"] = max_error(
                adapted(ids, attention_mask=mask).logits,
                load_file(str(target / "adapter-logits.safetensors"))["logits"],
            )
        if not real:
            expected(ids, attention_mask=mask, labels=labels).loss.backward()
            actual(ids, attention_mask=mask, labels=labels).loss.backward()
            gradients = load_file(str(target / "native-gradients.safetensors"))
            assert gradients.keys() == a.keys()
            metrics["gradient_max_abs_error"] = max(
                max_error(gradients[n], a[n].grad) for n in a
            )
            for name in a:
                max_error(b[name].grad, a[name].grad)
            updated = load(target / "updated")
            for name, value in updated.named_parameters():
                max_error(value, a[name] - 0.01 * a[name].grad)
            if target.name == "repeat":
                x = "model.layers.0.mlp.down_proj.weight"
                y = "model.layers.2.mlp.down_proj.weight"
                assert b[x] is not b[y] and torch.equal(b[x], b[y])
                changed = dict(updated.named_parameters())
                assert not torch.equal(changed[x], changed[y])
            assert (target / "updated/recomposition.json").read_bytes() == (
                target / "recomposition.json"
            ).read_bytes()
            # Ordinary Python training and resave must come back to native Lisp.
            torch.optim.SGD(actual.parameters(), lr=0.02).step()
            continued = target / "python-continued"
            actual.save_pretrained(continued)
            with torch.no_grad():
                out = actual(ids, attention_mask=mask, labels=labels)
            (continued / "reference.json").write_text(
                json.dumps({"logits": out.logits.tolist(), "loss": out.loss.item()})
                + "\n"
            )
        report[str(target.relative_to(ROOT / f".build/recomposition-{device}"))] = (
            metrics
        )
    if phase == "verify":
        destination = (
            ROOT
            / f".build/recomposition-{device}"
            / ("real-validation.json" if real else "validation.json")
        )
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--phase", choices=["fixture", "verify"])
    args = parser.parse_args()
    if args.phase:
        python_gate(args.device, args.real, args.phase)
        return
    # Include existing independent base-model qualification before selecting layers.
    if args.real:
        run(
            sys.executable,
            "scripts/run-composition-import-tests.py",
            "--device",
            args.device,
            "--real",
            *(["--local-files-only"] if args.local_files_only else []),
        )
    else:
        run(sys.executable, "scripts/run-component-tests.py", "--device", args.device)
    flags = ["--device", args.device] + (["--real"] if args.real else [])
    env = {
        **os.environ,
        "TB_DEVICE": args.device,
        "TB_RECOMPOSITION_REAL": "1" if args.real else "0",
        "HF_HUB_OFFLINE": "1",
    }
    run(sys.executable, __file__, *flags, "--phase", "fixture", env=env)
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-recomposition.lisp",
        env=env,
    )
    run(sys.executable, __file__, *flags, "--phase", "verify", env=env)
    if not args.real:
        run(
            "sbcl",
            "--noinform",
            "--no-sysinit",
            "--no-userinit",
            "--script",
            "scripts/test-recomposition.lisp",
            env={**env, "TB_RECOMPOSITION_ROUNDTRIP": "1"},
        )
    print(f"Recomposed layer reuse passed: {args.device}, real={args.real}")


if __name__ == "__main__":
    main()
