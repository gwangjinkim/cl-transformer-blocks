"""Verify native- and Python-created reusable teacher targets on CPU or Metal."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCH = {
    "input_ids": [[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]],
    "attention_mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
    "labels": [[-100, -100, 5, 6, -100], [-100, 5, -100, 3, 8]],
}


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def make_wrong_dtype_fixture(source, destination):
    from safetensors.torch import load_file, save_file

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    path = destination / "teacher.safetensors"
    tensors = load_file(str(path))
    save_file({name: value.half() for name, value in tensors.items()}, path)


def make_wrong_v2_dtype_fixture(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    path = destination / "distillation.json"
    manifest = json.loads(path.read_text())
    manifest["format_version"] = 2
    manifest["dtype"] = "float16"
    path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")


def make_bad_top_k_indices_fixture(source, destination, *, duplicate):
    from safetensors.torch import load_file, save_file

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    path = destination / "teacher.safetensors"
    tensors = load_file(str(path))
    indices = tensors["teacher_topk_indices"].clone()
    if duplicate:
        indices[0, 1] = indices[0, 0]
    else:
        manifest = json.loads((destination / "distillation.json").read_text())
        indices[0, 0] = manifest["vocab_size"]
    tensors["teacher_topk_indices"] = indices
    save_file(tensors, path)


def verify(device, real):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    teacher_path = (
        ROOT / ".build/models/smollm2"
        if real
        else ROOT / f".build/components-created-{device}"
    )
    student_path = (
        ROOT / f".build/recomposition-{device}/real"
        if real
        else ROOT / f".build/recomposition-{device}/untied/drop"
    )
    target_root = ROOT / f".build/distillation-targets-{device}"
    if real:
        target_root /= "real"
    ids = torch.tensor(BATCH["input_ids"])
    mask = torch.tensor(BATCH["attention_mask"])
    labels = torch.tensor(BATCH["labels"])
    selected = (labels[:, 1:] != -100) & mask[:, 1:].bool()
    tolerance = 1e-3 if real else 2e-6
    half_tolerance = 1e-2 if real else 1e-3

    def teacher_logits(path, trust_remote_code):
        teacher = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()
        with torch.no_grad():
            return teacher(ids[:, :-1], attention_mask=mask[:, :-1]).logits[selected]

    reference = teacher_logits(teacher_path, not real)
    top_k = 64 if real else 4
    target_errors = {}
    stored_targets = {}
    target_references = {
        "native": ("dense", reference),
        "native-fp16": ("dense", reference),
        "native-fp16-resaved": ("dense", reference),
        "python": ("dense", reference),
        "python-fp16": ("dense", reference),
        "native-topk": ("top_k", reference),
        "native-topk-fp16": ("top_k", reference),
        "native-topk-fp16-resaved": ("top_k", reference),
        "python-topk": ("top_k", reference),
        "python-topk-fp16": ("top_k", reference),
    }
    if not real:
        target_references["python-foreign"] = (
            "dense",
            teacher_logits(ROOT / f".build/portable-lisp-created-{device}", True),
        )
    for kind, (representation, expected) in target_references.items():
        directory = target_root / kind
        manifest = json.loads((directory / "distillation.json").read_text())
        fp16 = "fp16" in kind
        assert manifest["format_version"] == (
            (4 if fp16 else 3) if representation == "top_k" else (2 if fp16 else 1)
        )
        assert manifest["dtype"] == ("float16" if fp16 else "float32")
        values = load_file(str(directory / "teacher.safetensors"))
        float_dtype = torch.float16 if fp16 else torch.float32
        if representation == "dense":
            assert "representation" not in manifest
            assert values.keys() == {"teacher_logits"}
            stored = values["teacher_logits"]
            assert stored.dtype == float_dtype
            torch.testing.assert_close(
                stored.float(),
                expected,
                atol=half_tolerance if fp16 else tolerance,
                rtol=3e-4,
            )
            stored_targets[kind] = {"representation": "dense", "logits": stored.float()}
            target_errors[kind] = float((stored.float() - expected).abs().max())
        else:
            assert manifest["representation"] == "top_k"
            assert manifest["top_k"] == top_k and manifest["temperature"] == 2.0
            assert values.keys() == {
                "teacher_topk_log_probs",
                "teacher_topk_indices",
                "teacher_tail_log_prob",
                "teacher_vocab_size",
            }
            top = values["teacher_topk_log_probs"]
            indices = values["teacher_topk_indices"]
            tail = values["teacher_tail_log_prob"]
            assert top.dtype == float_dtype and tail.dtype == float_dtype
            assert indices.dtype == torch.int32
            assert values["teacher_vocab_size"].dtype == torch.int32
            assert values["teacher_vocab_size"].tolist() == [reference.shape[-1]]
            assert top.shape == indices.shape == (int(selected.sum()), top_k)
            assert tail.shape == (int(selected.sum()), 1)
            assert torch.all((indices >= 0) & (indices < reference.shape[-1]))
            assert torch.all(torch.sort(indices.long(), dim=-1).values.diff(dim=-1) != 0)
            expected_log_probs = F.log_softmax(expected / 2.0, dim=-1)
            expected_top = expected_log_probs.gather(-1, indices.long())
            expected_tail = torch.logsumexp(
                expected_log_probs.scatter(
                    -1, indices.long(), torch.full_like(top.float(), -torch.inf)
                ),
                dim=-1,
                keepdim=True,
            )
            top_error = float((top.float() - expected_top).abs().max())
            tail_error = float((tail.float() - expected_tail).abs().max())
            torch.testing.assert_close(
                top.float(), expected_top,
                atol=half_tolerance if fp16 else tolerance, rtol=3e-4,
            )
            torch.testing.assert_close(
                tail.float(), expected_tail,
                atol=half_tolerance if fp16 else tolerance, rtol=3e-4,
            )
            stored_targets[kind] = {
                "representation": "top_k",
                "top": top.float(),
                "indices": indices.long(),
                "tail": tail.float(),
            }
            target_errors[kind] = max(top_error, tail_error)
    for stem in ("native", "native-topk"):
        original = load_file(str(target_root / f"{stem}-fp16/teacher.safetensors"))
        resaved = load_file(str(target_root / f"{stem}-fp16-resaved/teacher.safetensors"))
        assert original.keys() == resaved.keys()
        for name in original:
            assert torch.equal(original[name], resaved[name])
    gradient_errors = {}
    update_errors = {}
    interop_kinds = [
        "native-fp16", "python", "python-fp16",
        "native-topk-fp16", "python-topk", "python-topk-fp16",
    ]
    if not real:
        interop_kinds.append("python-foreign")
    for kind in interop_kinds:
        base = AutoModelForCausalLM.from_pretrained(
            student_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()
        student = (
            PeftModel.from_pretrained(
                base, student_path / "adapter", is_trainable=True
            ).eval()
            if real
            else base
        )
        parameters = {
            name.replace(".default.", "."): value
            for name, value in student.named_parameters()
            if value.requires_grad
        }
        target = stored_targets[kind]

        def objective():
            logits = student(ids[:, :-1], attention_mask=mask[:, :-1]).logits[selected]
            if target["representation"] == "dense":
                soft = F.kl_div(
                    F.log_softmax(logits / 2.0, dim=-1),
                    F.log_softmax(target["logits"] / 2.0, dim=-1),
                    log_target=True, reduction="batchmean",
                ) * 4
            else:
                teacher_all = torch.cat((target["top"], target["tail"]), dim=-1)
                teacher_logp = F.log_softmax(teacher_all, dim=-1)
                student_logp = F.log_softmax(logits / 2.0, dim=-1)
                student_top = student_logp.gather(-1, target["indices"])
                student_tail = torch.logsumexp(
                    student_logp.scatter(
                        -1,
                        target["indices"],
                        torch.full_like(target["top"], -torch.inf),
                    ),
                    dim=-1,
                    keepdim=True,
                )
                student_all = torch.cat((student_top, student_tail), dim=-1)
                soft = (
                    torch.exp(teacher_logp) * (teacher_logp - student_all)
                ).sum(dim=-1).mean() * 4
            return 0.7 * soft + 0.3 * F.cross_entropy(
                logits, labels[:, 1:][selected]
            )

        objective().backward()
        native_gradients = load_file(
            str(target_root / kind / "native-gradients.safetensors")
        )
        assert native_gradients.keys() == parameters.keys()
        gradient_error = 0.0
        for name, parameter in parameters.items():
            torch.testing.assert_close(
                native_gradients[name], parameter.grad, atol=tolerance, rtol=3e-4
            )
            gradient_error = max(
                gradient_error,
                float((native_gradients[name] - parameter.grad).abs().max()),
            )
        gradient_errors[kind] = gradient_error
        optimizer = torch.optim.SGD(
            parameters.values(), lr=0.01, momentum=0.9, weight_decay=0.02
        )
        for _ in range(3):
            optimizer.zero_grad()
            objective().backward()
            torch.nn.utils.clip_grad_norm_(parameters.values(), 0.05)
            optimizer.step()
        actual_base = AutoModelForCausalLM.from_pretrained(
            student_path if real else target_root / kind / "native-updated",
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()
        actual = (
            PeftModel.from_pretrained(
                actual_base, target_root / kind / "native-updated", is_trainable=True
            ).eval()
            if real
            else actual_base
        )
        actual_parameters = {
            name.replace(".default.", "."): value
            for name, value in actual.named_parameters()
            if value.requires_grad
        }
        assert parameters.keys() == actual_parameters.keys()
        update_error = 0.0
        for name, expected in parameters.items():
            torch.testing.assert_close(
                actual_parameters[name], expected, atol=tolerance, rtol=3e-4
            )
            update_error = max(
                update_error,
                float((actual_parameters[name] - expected).detach().abs().max()),
            )
        update_errors[kind] = update_error
    elements = reference.numel()
    return {
        "selected_positions": int(selected.sum()),
        "vocab_size": reference.shape[1],
        "native_teacher_logits_max_abs_error": target_errors["native"],
        "native_fp16_teacher_logits_max_abs_error": target_errors["native-fp16"],
        "native_fp16_resave_exact": True,
        "python_teacher_logits_max_abs_error": target_errors["python"],
        "python_fp16_teacher_logits_max_abs_error": target_errors["python-fp16"],
        "native_fp16_target_gradient_max_abs_error": gradient_errors["native-fp16"],
        "python_target_gradient_max_abs_error": gradient_errors["python"],
        "python_fp16_target_gradient_max_abs_error": gradient_errors["python-fp16"],
        "native_fp16_updated_parameter_max_abs_error": update_errors["native-fp16"],
        "python_fp16_updated_parameter_max_abs_error": update_errors["python-fp16"],
        "native_topk_teacher_probability_max_abs_error": target_errors["native-topk"],
        "python_topk_teacher_probability_max_abs_error": target_errors["python-topk"],
        "native_topk_fp16_gradient_max_abs_error": gradient_errors["native-topk-fp16"],
        "python_topk_gradient_max_abs_error": gradient_errors["python-topk"],
        "python_topk_fp16_gradient_max_abs_error": gradient_errors["python-topk-fp16"],
        "native_topk_fp16_updated_parameter_max_abs_error": update_errors["native-topk-fp16"],
        "python_topk_fp16_updated_parameter_max_abs_error": update_errors["python-topk-fp16"],
        "foreign_teacher_logits_max_abs_error": target_errors.get("python-foreign"),
        "foreign_target_gradient_max_abs_error": gradient_errors.get("python-foreign"),
        "fp32_teacher_payload_bytes": elements * 4,
        "fp16_teacher_payload_bytes": elements * 2,
        "top_k": top_k,
        "topk_fp32_payload_bytes": int(selected.sum()) * (top_k * 8 + 4) + 4,
        "topk_fp16_payload_bytes": int(selected.sum()) * (top_k * 6 + 2) + 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    flags = ["--device", args.device] + (["--real"] if args.real else [])
    run(
        sys.executable,
        "scripts/run-distillation-tests.py",
        *flags,
        *(["--local-files-only"] if args.local_files_only else []),
    )
    if not args.real:
        run(
            sys.executable,
            "scripts/run-portable-tests.py",
            "--device",
            args.device,
        )
    target_root = ROOT / f".build/distillation-targets-{args.device}"
    if args.real:
        target_root /= "real"
    target_root.mkdir(parents=True, exist_ok=True)
    input_path = target_root / "batch.json"
    input_path.write_text(json.dumps(BATCH) + "\n")
    python_target = target_root / "python"
    if python_target.exists():
        shutil.rmtree(python_target)
    run(
        sys.executable,
        "scripts/create-distillation-batch.py",
        "--teacher",
        ROOT / ".build/models/smollm2"
        if args.real
        else ROOT / f".build/components-created-{args.device}",
        "--batch-json",
        input_path,
        "--output",
        python_target,
        *([] if args.real else ["--trust-remote-code"]),
        "--local-files-only",
    )
    top_k = "64" if args.real else "4"
    for name, storage_dtype in (
        ("python-topk", "float32"),
        ("python-topk-fp16", "float16"),
    ):
        destination = target_root / name
        if destination.exists():
            shutil.rmtree(destination)
        run(
            sys.executable,
            "scripts/create-distillation-batch.py",
            "--teacher",
            ROOT / ".build/models/smollm2"
            if args.real
            else ROOT / f".build/components-created-{args.device}",
            "--batch-json",
            input_path,
            "--output",
            destination,
            "--storage-dtype",
            storage_dtype,
            "--top-k",
            top_k,
            "--temperature",
            "2.0",
            *([] if args.real else ["--trust-remote-code"]),
            "--local-files-only",
        )
    python_half_target = target_root / "python-fp16"
    if python_half_target.exists():
        shutil.rmtree(python_half_target)
    run(
        sys.executable,
        "scripts/create-distillation-batch.py",
        "--teacher",
        ROOT / ".build/models/smollm2"
        if args.real
        else ROOT / f".build/components-created-{args.device}",
        "--batch-json",
        input_path,
        "--output",
        python_half_target,
        "--storage-dtype",
        "float16",
        *([] if args.real else ["--trust-remote-code"]),
        "--local-files-only",
    )
    if not args.real:
        foreign_target = target_root / "python-foreign"
        if foreign_target.exists():
            shutil.rmtree(foreign_target)
        run(
            sys.executable,
            "scripts/create-distillation-batch.py",
            "--teacher",
            ROOT / f".build/portable-lisp-created-{args.device}",
            "--batch-json",
            input_path,
            "--output",
            foreign_target,
            "--trust-remote-code",
            "--local-files-only",
        )
    make_wrong_dtype_fixture(python_target, target_root / "malformed-fp16")
    make_wrong_v2_dtype_fixture(python_target, target_root / "malformed-fp32-v2")
    make_bad_top_k_indices_fixture(
        target_root / "python-topk", target_root / "malformed-topk-duplicate", duplicate=True
    )
    make_bad_top_k_indices_fixture(
        target_root / "python-topk", target_root / "malformed-topk-range", duplicate=False
    )
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-distillation-targets.lisp",
        env={
            **os.environ,
            "TB_DEVICE": args.device,
            "HF_HUB_OFFLINE": "1",
            "TB_DISTILLATION_TARGETS_REAL": "1" if args.real else "0",
        },
    )
    report = verify(args.device, args.real)
    (target_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        f"Reusable native/Python distillation targets passed: "
        f"{args.device}, real={args.real}"
    )


if __name__ == "__main__":
    main()
