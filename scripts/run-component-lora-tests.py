"""Qualify composed Lisp models against ordinary Python PEFT, on the chosen device."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def python_gate(directory, phase):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)
    ids = torch.tensor([[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    labels = torch.tensor([[-100, -100, 5, 6, -100], [-100, 5, -100, 3, 8]])
    report = {}

    def base(path):
        # Load only exported architecture files in this fresh process.
        return AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        ).eval()

    def parameters(model):
        return {
            name.replace(".default.", "."): p
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def compare(actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-4)
        return (actual - expected).abs().max().item()

    for kind in ("untied", "tied"):
        base_path = directory if kind == "untied" else directory / "tied"
        for variant in ("standard", "rslora"):
            folder = directory / "lora" / kind / variant
            if phase == "fixture":
                torch.manual_seed(1709)
                model = get_peft_model(
                    base(base_path),
                    LoraConfig(
                        r=3,
                        lora_alpha=6,
                        target_modules=TARGETS,
                        lora_dropout=0.0,
                        bias="none",
                        task_type="CAUSAL_LM",
                        use_rslora=variant == "rslora",
                    ),
                ).eval()
                model.peft_config["default"].base_model_name_or_path = str(base_path)
                with torch.no_grad():
                    # Nonzero factors exercise both A and B gradients immediately.
                    for p in parameters(model).values():
                        p.uniform_(-0.08, 0.08)
                model.save_pretrained(folder)
                out = model(ids, attention_mask=mask, labels=labels)
                out.loss.backward()
                assert len(parameters(model)) == 28
                save_file(
                    {k: p.grad.contiguous() for k, p in parameters(model).items()},
                    str(folder / "gradients.safetensors"),
                )
                reference = {
                    "input_ids": ids.tolist(),
                    "attention_mask": mask.tolist(),
                    "labels": labels.tolist(),
                    "logits": out.logits.detach().tolist(),
                    "loss": out.loss.item(),
                }
                for algorithm in ("sgd", "adamw"):
                    updated = PeftModel.from_pretrained(
                        base(base_path), folder, is_trainable=True
                    ).eval()
                    params = list(parameters(updated).values())
                    optimizer = (
                        torch.optim.SGD(
                            params, lr=0.01, momentum=0.9, weight_decay=0.02
                        )
                        if algorithm == "sgd"
                        else torch.optim.AdamW(
                            params,
                            lr=0.001,
                            betas=(0.8, 0.95),
                            weight_decay=0.02,
                            foreach=False,
                        )
                    )
                    losses = []
                    for _ in range(3):
                        optimizer.zero_grad()
                        loss = updated(ids, attention_mask=mask, labels=labels).loss
                        losses.append(loss.item())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 0.05, foreach=False)
                        optimizer.step()
                    updated.save_pretrained(folder / algorithm)
                    reference[algorithm] = losses
                (folder / "reference.json").write_text(json.dumps(reference) + "\n")
                continue

            expected = PeftModel.from_pretrained(base(base_path), folder).eval()
            with torch.no_grad():
                logits = expected(ids, attention_mask=mask).logits
            native = load_file(str(folder / "native-logits.safetensors"))["logits"]
            metrics = {"logits_max_abs_error": compare(native, logits)}
            a = load_file(str(folder / "gradients.safetensors"))
            b = load_file(str(folder / "native-gradients.safetensors"))
            assert a.keys() == b.keys() and len(a) == 28
            metrics["gradient_max_abs_error"] = max(compare(b[k], a[k]) for k in a)
            for suffix in ("exported", "sgd", "adamw"):
                reference = folder if suffix == "exported" else folder / suffix
                exported = folder / ("native-" + suffix)
                a = load_file(str(reference / "adapter_model.safetensors"))
                b = load_file(str(exported / "adapter_model.safetensors"))
                assert a.keys() == b.keys()
                metrics[suffix + "_parameter_max_abs_error"] = max(
                    compare(b[k], a[k]) for k in a
                )
                if suffix == "exported":
                    assert all(torch.equal(a[k], b[k]) for k in a)
                actual = PeftModel.from_pretrained(base(base_path), exported).eval()
                ref = PeftModel.from_pretrained(base(base_path), reference).eval()
                with torch.no_grad():
                    compare(
                        actual(ids, attention_mask=mask).logits,
                        ref(ids, attention_mask=mask).logits,
                    )
            merged = base(folder / "merged")
            expected_merged = expected.merge_and_unload()
            assert merged.state_dict().keys() == expected_merged.state_dict().keys()
            metrics["merged_parameter_max_abs_error"] = max(
                compare(value, expected_merged.state_dict()[name])
                for name, value in merged.state_dict().items()
            )
            with torch.no_grad():
                metrics["merged_logits_max_abs_error"] = compare(
                    merged(ids, attention_mask=mask).logits, logits
                )
            if kind == "tied":
                assert merged.lm_head.weight is merged.model.embed_tokens.weight
            for algorithm in ("sgd", "adamw"):
                # Native optimizer sidecars leave an ordinary PEFT checkpoint.
                PeftModel.from_pretrained(
                    base(base_path), folder / (algorithm + "-resume")
                )
            created = PeftModel.from_pretrained(
                base(base_path), folder / "created", is_trainable=True
            ).eval()
            native_created = load_file(str(folder / "created-logits.safetensors"))[
                "logits"
            ]
            with torch.no_grad():
                metrics["created_logits_max_abs_error"] = compare(
                    created(ids, attention_mask=mask).logits, native_created
                )
                generated = created.generate(
                    torch.tensor([[3, 4]]),
                    max_new_tokens=4,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=None,
                )[0].tolist()
            assert generated == json.loads(
                (folder / "created-generation.json").read_text()
            )
            # Continue training a Lisp-created adapter in Python, then return
            # the saved result to a fresh native Lisp process.
            params = list(parameters(created).values())
            assert len(params) == 28
            optimizer = torch.optim.SGD(params, lr=0.01)
            created(ids, attention_mask=mask, labels=labels).loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in params
            )
            optimizer.step()
            continued = folder / "python-continued"
            created.save_pretrained(continued)
            with torch.no_grad():
                out = created(ids, attention_mask=mask, labels=labels)
            (continued / "reference.json").write_text(
                json.dumps({"logits": out.logits.tolist(), "loss": out.loss.item()})
                + "\n"
            )
            report[f"{kind}-{variant}"] = metrics
    if phase == "verify":
        (directory / "lora-validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        print(
            "Fresh PEFT import, all gradients/updates, native creation, merge and generation passed"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--phase", choices=["fixture", "verify"])
    args = parser.parse_args()
    directory = ROOT / f".build/components-created-{args.device}"
    if args.phase:
        python_gate(directory, args.phase)
        return
    env = {**os.environ, "TB_DEVICE": args.device, "HF_HUB_OFFLINE": "1"}
    # Keep the independent functional base-model oracle and its report alongside
    # the new PEFT evidence, rather than qualifying adapters over unchecked bases.
    run(
        sys.executable,
        "scripts/run-component-tests.py",
        "--device",
        args.device,
        env=env,
    )
    run(
        sys.executable, __file__, "--device", args.device, "--phase", "fixture", env=env
    )
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-component-lora.lisp",
        env=env,
    )
    run(sys.executable, __file__, "--device", args.device, "--phase", "verify", env=env)
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-component-lora.lisp",
        env={**env, "TB_COMPONENT_LORA_PHASE": "roundtrip"},
    )
    print(f"All composed LoRA/rsLoRA gates passed: {args.device}")


if __name__ == "__main__":
    main()
