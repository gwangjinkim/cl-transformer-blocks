"""Native teacher/student loss, gradients, updates and Python interchange."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def cases(device, real):
    output = ROOT / f".build/distillation-{device}"
    if real:
        return [
            (
                ROOT / ".build/models/smollm2",
                ROOT / f".build/recomposition-{device}/real",
                output / "real",
                True,
                2.0,
                0.3,
            )
        ]
    result = []
    for kind in ("untied", "tied"):
        teacher = ROOT / f".build/components-created-{device}"
        if kind == "tied":
            teacher /= "tied"
        student = ROOT / f".build/recomposition-{device}/{kind}/drop"
        for name, adapter, temperature, hard in [
            ("soft", False, 0.7, 0.0),
            ("mixed", False, 2.0, 0.3),
            ("hard", False, 1.0, 1.0),
            ("adapter", True, 2.0, 0.3),
        ]:
            result.append(
                (teacher, student, output / kind / name, adapter, temperature, hard)
            )
    return result


def python_gate(device, real, phase):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)
    ids = torch.tensor([[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    labels = torch.tensor([[-100, -100, 5, 6, -100], [-100, 5, -100, 3, 8]])
    selected = (labels[:, 1:] != -100) & mask[:, 1:].bool()
    manifest, report = [], {}

    def load(path):
        return AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()

    def parameters(model):
        return {
            n.replace(".default.", "."): p
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def compare(a, b):
        torch.testing.assert_close(a, b, atol=1e-3 if real else 2e-6, rtol=3e-4)
        return (a - b).abs().max().item()

    for teacher_path, student_path, folder, adapter, temperature, hard in cases(
        device, real
    ):
        teacher = load(teacher_path)
        teacher.requires_grad_(False)
        with torch.no_grad():
            teacher_logits = teacher(ids, attention_mask=mask).logits[:, :-1][selected]
            teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)

        def objective(model):
            logits = model(ids, attention_mask=mask).logits[:, :-1][selected]
            soft = (
                F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    teacher_logp,
                    log_target=True,
                    reduction="batchmean",
                )
                * temperature**2
            )
            supervised = F.cross_entropy(logits, labels[:, 1:][selected])
            return (1 - hard) * soft + hard * supervised

        if phase == "fixture":
            folder.mkdir(parents=True, exist_ok=True)
            model = load(student_path)
            if adapter:
                torch.manual_seed(1409)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=2,
                        lora_alpha=4,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0,
                        bias="none",
                        task_type="CAUSAL_LM",
                    ),
                ).eval()
                model.peft_config["default"].base_model_name_or_path = str(student_path)
                with torch.no_grad():
                    for p in parameters(model).values():
                        p.uniform_(-0.03, 0.03)
                model.save_pretrained(folder / "initial-adapter")
            optimizer = torch.optim.SGD(
                parameters(model).values(), lr=0.01, momentum=0.9, weight_decay=0.02
            )
            losses = []
            for step in range(3):
                optimizer.zero_grad()
                loss = objective(model)
                losses.append(loss.item())
                loss.backward()
                if step == 0:
                    save_file(
                        {n: p.grad.contiguous() for n, p in parameters(model).items()},
                        str(folder / "gradients.safetensors"),
                    )
                torch.nn.utils.clip_grad_norm_(parameters(model).values(), 0.05)
                optimizer.step()
            save_file(
                {n: p.detach().contiguous() for n, p in parameters(model).items()},
                str(folder / "updated.safetensors"),
            )
            with torch.no_grad():
                save_file(
                    {"logits": model(ids, attention_mask=mask).logits},
                    str(folder / "expected-logits.safetensors"),
                )
            entry = dict(
                teacher=str(teacher_path) + "/",
                student=str(student_path) + "/",
                folder=str(folder) + "/",
                adapter=adapter,
                temperature=temperature,
                hard_weight=hard,
                input_ids=ids.tolist(),
                attention_mask=mask.tolist(),
                labels=labels.tolist(),
                losses=losses,
            )
            manifest.append(entry)
        else:
            gradients = load_file(str(folder / "native-gradients.safetensors"))
            expected_gradients = load_file(str(folder / "gradients.safetensors"))
            assert gradients.keys() == expected_gradients.keys()
            exported = (
                PeftModel.from_pretrained(
                    load(student_path), folder / "exported", is_trainable=True
                ).eval()
                if adapter
                else load(folder / "exported")
            )
            weights = parameters(exported)
            expected_weights = load_file(str(folder / "updated.safetensors"))
            assert weights.keys() == expected_weights.keys()
            metrics = {
                "trainable_tensors": len(weights),
                "gradient_max_abs_error": max(
                    compare(gradients[n], expected_gradients[n]) for n in gradients
                ),
                "updated_parameter_max_abs_error": max(
                    compare(weights[n], expected_weights[n]) for n in weights
                ),
            }
            with torch.no_grad():
                logits = exported(ids, attention_mask=mask).logits
                metrics["native_export_logits_max_abs_error"] = compare(
                    logits,
                    load_file(str(folder / "native-logits.safetensors"))["logits"],
                )
                metrics["reference_update_logits_max_abs_error"] = compare(
                    logits,
                    load_file(str(folder / "expected-logits.safetensors"))["logits"],
                )
            # A Python contribution uses the same distillation objective and remains
            # an ordinary full-model or PEFT artifact for a fresh native reload.
            objective(exported).backward()
            torch.optim.SGD(parameters(exported).values(), lr=0.005).step()
            exported.save_pretrained(folder / "python-continued")
            with torch.no_grad():
                save_file(
                    {"logits": exported(ids, attention_mask=mask).logits},
                    str(folder / "python-logits.safetensors"),
                )
            report[str(folder.relative_to(ROOT / f".build/distillation-{device}"))] = (
                metrics
            )
        assert all(p.grad is None for p in teacher.parameters())
    output = ROOT / f".build/distillation-{device}"
    prefix = "real-" if real else ""
    if phase == "fixture":
        (output / f"{prefix}cases.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    else:
        (output / f"{prefix}validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
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
    flags = ["--device", args.device] + (["--real"] if args.real else [])
    run(
        sys.executable,
        "scripts/run-recomposition-tests.py",
        *flags,
        *(["--local-files-only"] if args.local_files_only else []),
    )
    env = {
        **os.environ,
        "TB_DEVICE": args.device,
        "HF_HUB_OFFLINE": "1",
        "TB_DISTILLATION_REAL": "1" if args.real else "0",
    }
    run(sys.executable, __file__, *flags, "--phase", "fixture", env=env)
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-distillation.lisp",
        env=env,
    )
    run(sys.executable, __file__, *flags, "--phase", "verify", env=env)
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-distillation.lisp",
        env={**env, "TB_DISTILLATION_ROUNDTRIP": "1"},
    )
    print(f"Native distillation passed: {args.device}, real={args.real}")


if __name__ == "__main__":
    main()
