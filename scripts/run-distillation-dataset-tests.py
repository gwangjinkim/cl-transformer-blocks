"""Verify lazy native/Python distillation datasets and exact iterator resumption."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        "input_ids": [[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]],
        "labels": [[-100, -100, 5, 6, -100], [-100, 5, -100, 3, 8]],
        "attention_mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
    },
    {
        "input_ids": [[4, 3, 7, 5]],
        "labels": [[-100, 3, -100, 5]],
        "attention_mask": [[1, 1, 1, 1]],
    },
    {
        "input_ids": [[8, 6, 4, 0, 0]],
        "labels": [[-100, -100, 4, -100, -100]],
        "attention_mask": [[1, 1, 1, 0, 0]],
    },
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_sha256(vocab_size, entries):
    framing = (
        "cl-transformer-blocks-distillation-dataset-v2\n"
        f"vocab_size={vocab_size}\n"
        f"batch_count={len(entries)}\n"
        + "".join(
            f"{index:06d}\t{entry['selected_positions']}\t"
            f"{entry['manifest_sha256']}\t{entry['weights_sha256']}\n"
            for index, entry in enumerate(entries)
        )
    )
    return hashlib.sha256(framing.encode("ascii")).hexdigest()


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def epoch_order(count, seed, epoch):
    order = list(range(count))
    state = (seed + epoch) % (2**32)
    for index in range(count - 1, 0, -1):
        state = (1664525 * state + 1013904223) % (2**32)
        target = state % (index + 1)
        order[index], order[target] = order[target], order[index]
    return order


def verify(device, real, output_root):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    native = output_root / "native"
    native_fp16 = output_root / "native-fp16"
    python = output_root / "python"
    python_fp16 = output_root / "python-fp16"
    streamed = output_root / "native-streamed"
    streamed_fp16 = output_root / "native-streamed-fp16"
    native_topk_fp16 = output_root / "native-topk-fp16"
    python_topk_fp16 = output_root / "python-topk-fp16"
    streamed_topk_fp16 = output_root / "native-streamed-topk-fp16"
    directories = {
        "native": native,
        "native_fp16": native_fp16,
        "python": python,
        "python_fp16": python_fp16,
        "streamed": streamed,
        "streamed_fp16": streamed_fp16,
        "native_topk_fp16": native_topk_fp16,
        "python_topk_fp16": python_topk_fp16,
        "streamed_topk_fp16": streamed_topk_fp16,
    }
    manifests = {}
    for kind, directory in directories.items():
        manifest = json.loads((directory / "distillation-dataset.json").read_text())
        manifests[kind] = manifest
        assert manifest["format"] == "cl-transformer-blocks-distillation-dataset"
        assert manifest["format_version"] == 2
        assert manifest["dataset_id"] == "tiny-three-v1"
        assert manifest["batch_count"] == len(CASES)
        assert [entry["path"] for entry in manifest["batches"]] == [
            f"batches/{index:06d}" for index in range(len(CASES))
        ]
        assert manifest["content_sha256"] == dataset_content_sha256(
            manifest["vocab_size"], manifest["batches"]
        )
        for entry in manifest["batches"]:
            batch_directory = directory / entry["path"]
            assert entry["manifest_sha256"] == sha256_file(
                batch_directory / "distillation.json"
            )
            assert entry["weights_sha256"] == sha256_file(
                batch_directory / "teacher.safetensors"
            )
    assert manifests["native"]["content_sha256"] == manifests["streamed"][
        "content_sha256"
    ]
    assert manifests["native_fp16"]["content_sha256"] == manifests[
        "streamed_fp16"
    ]["content_sha256"]
    assert manifests["native_topk_fp16"]["content_sha256"] == manifests[
        "streamed_topk_fp16"
    ]["content_sha256"]
    teacher_path = (
        ROOT / ".build/models/smollm2"
        if real
        else ROOT / f".build/components-created-{device}"
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        trust_remote_code=not real,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    tolerance = 1e-3 if real else 2e-6
    target_error = {kind: 0.0 for kind in directories}
    for index, case in enumerate(CASES):
        ids = torch.tensor(case["input_ids"])
        labels = torch.tensor(case["labels"])
        mask = torch.tensor(case["attention_mask"])
        selected = (labels[:, 1:] != -100) & mask[:, 1:].bool()
        with torch.no_grad():
            expected = teacher(ids[:, :-1], attention_mask=mask[:, :-1]).logits[
                selected
            ]
        for kind, directory in directories.items():
            values = load_file(
                str(directory / f"batches/{index:06d}/teacher.safetensors")
            )
            fp16 = kind.endswith("fp16")
            topk = "topk" in kind
            batch_manifest = json.loads(
                (directory / f"batches/{index:06d}/distillation.json").read_text()
            )
            assert batch_manifest["format_version"] == (
                4 if topk and fp16 else 3 if topk else 2 if fp16 else 1
            )
            assert batch_manifest["dtype"] == ("float16" if fp16 else "float32")
            artifact_tolerance = 1e-2 if real and fp16 else 1e-3 if fp16 else tolerance
            if not topk:
                actual = values["teacher_logits"]
                assert actual.dtype == (torch.float16 if fp16 else torch.float32)
                torch.testing.assert_close(
                    actual.float(), expected, atol=artifact_tolerance, rtol=3e-4
                )
                error = float((actual.float() - expected).abs().max())
            else:
                top = values["teacher_topk_log_probs"]
                indices = values["teacher_topk_indices"].long()
                tail = values["teacher_tail_log_prob"]
                assert batch_manifest["representation"] == "top_k"
                assert batch_manifest["top_k"] == (64 if real else 4)
                assert batch_manifest["temperature"] == 2.0
                assert values["teacher_vocab_size"].tolist() == [expected.shape[-1]]
                expected_logp = F.log_softmax(expected / 2.0, dim=-1)
                expected_top = expected_logp.gather(-1, indices)
                expected_tail = torch.logsumexp(
                    expected_logp.scatter(
                        -1, indices, torch.full_like(top.float(), -torch.inf)
                    ), dim=-1, keepdim=True,
                )
                torch.testing.assert_close(
                    top.float(), expected_top, atol=artifact_tolerance, rtol=3e-4
                )
                torch.testing.assert_close(
                    tail.float(), expected_tail, atol=artifact_tolerance, rtol=3e-4
                )
                error = max(
                    float((top.float() - expected_top).abs().max()),
                    float((tail.float() - expected_tail).abs().max()),
                )
            target_error[kind] = max(target_error[kind], error)

    student_path = (
        ROOT / f".build/recomposition-{device}/real"
        if real
        else ROOT / f".build/recomposition-{device}/untied/drop"
    )
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
    optimizer = torch.optim.SGD(
        parameters.values(), lr=0.01, momentum=0.9, weight_decay=0.02
    )
    order = epoch_order(len(CASES), 41, 3)
    for index in order:
        case = CASES[index]
        ids = torch.tensor(case["input_ids"])
        labels = torch.tensor(case["labels"])
        mask = torch.tensor(case["attention_mask"])
        selected = (labels[:, 1:] != -100) & mask[:, 1:].bool()
        targets = load_file(
            str(native_fp16 / f"batches/{index:06d}/teacher.safetensors")
        )["teacher_logits"].float()
        optimizer.zero_grad()
        logits = student(ids[:, :-1], attention_mask=mask[:, :-1]).logits[selected]
        loss = 0.7 * F.kl_div(
            F.log_softmax(logits / 2.0, dim=-1),
            F.log_softmax(targets / 2.0, dim=-1),
            log_target=True,
            reduction="batchmean",
        ) * 4 + 0.3 * F.cross_entropy(logits, labels[:, 1:][selected])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), 0.05)
        optimizer.step()
    actual_base = AutoModelForCausalLM.from_pretrained(
        student_path if real else output_root / "native-updated",
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    actual = (
        PeftModel.from_pretrained(
            actual_base, output_root / "native-updated", is_trainable=True
        ).eval()
        if real
        else actual_base
    )
    actual_parameters = {
        name.replace(".default.", "."): value
        for name, value in actual.named_parameters()
        if value.requires_grad
    }
    assert actual_parameters.keys() == parameters.keys()
    update_error = 0.0
    for name, expected in parameters.items():
        torch.testing.assert_close(
            actual_parameters[name], expected, atol=tolerance, rtol=3e-4
        )
        update_error = max(
            update_error,
            float((actual_parameters[name] - expected).detach().abs().max()),
        )
    state = json.loads((output_root / "state/dataset-state.json").read_text())
    assert state["format_version"] == 2
    assert state["dataset_id"] == "tiny-three-v1"
    assert state["content_sha256"] == manifests["native_fp16"]["content_sha256"]
    assert state["epoch"] == 3 and state["position"] == 1
    return {
        "order": order,
        "content_sha256": {
            kind: manifest["content_sha256"]
            for kind, manifest in manifests.items()
        },
        "native_target_max_abs_error": target_error["native"],
        "native_fp16_target_max_abs_error": target_error["native_fp16"],
        "python_target_max_abs_error": target_error["python"],
        "python_fp16_target_max_abs_error": target_error["python_fp16"],
        "streamed_target_max_abs_error": target_error["streamed"],
        "streamed_fp16_target_max_abs_error": target_error["streamed_fp16"],
        "native_topk_fp16_probability_max_abs_error": target_error["native_topk_fp16"],
        "python_topk_fp16_probability_max_abs_error": target_error["python_topk_fp16"],
        "streamed_topk_fp16_probability_max_abs_error": target_error["streamed_topk_fp16"],
        "updated_parameter_max_abs_error": update_error,
        "fp32_teacher_payload_bytes": sum(
            load_file(str(native / f"batches/{index:06d}/teacher.safetensors"))[
                "teacher_logits"
            ].numel()
            * 4
            for index in range(len(CASES))
        ),
        "fp16_teacher_payload_bytes": sum(
            load_file(
                str(native_fp16 / f"batches/{index:06d}/teacher.safetensors")
            )["teacher_logits"].numel()
            * 2
            for index in range(len(CASES))
        ),
        "topk_fp16_payload_bytes": sum(
            sum(value.numel() * value.element_size() for value in load_file(
                str(native_topk_fp16 / f"batches/{index:06d}/teacher.safetensors")
            ).values())
            for index in range(len(CASES))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--teacher-device", choices=["cpu", "mps", "cuda"], default="cpu"
    )
    parser.add_argument("--prepared", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    flags = ["--device", args.device] + (["--real"] if args.real else [])
    if not args.prepared:
        run(
            sys.executable,
            "scripts/run-distillation-target-tests.py",
            *flags,
            *(["--local-files-only"] if args.local_files_only else []),
        )
    output_root = ROOT / f".build/distillation-datasets-{args.device}"
    if args.real:
        output_root /= "real"
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "dataset-input.json"
    input_path.write_text(json.dumps({"batches": CASES}) + "\n")
    python_output = output_root / "python"
    if python_output.exists():
        shutil.rmtree(python_output)
    teacher_path = (
        ROOT / ".build/models/smollm2"
        if args.real
        else ROOT / f".build/components-created-{args.device}"
    )
    run(
        sys.executable,
        "scripts/create-distillation-dataset.py",
        "--teacher",
        teacher_path,
        "--dataset-json",
        input_path,
        "--dataset-id",
        "tiny-three-v1",
        "--output",
        python_output,
        "--device",
        args.teacher_device,
        *([] if args.real else ["--trust-remote-code"]),
        "--local-files-only",
    )
    python_topk_half_output = output_root / "python-topk-fp16"
    if python_topk_half_output.exists():
        shutil.rmtree(python_topk_half_output)
    run(
        sys.executable,
        "scripts/create-distillation-dataset.py",
        "--teacher",
        teacher_path,
        "--dataset-json",
        input_path,
        "--dataset-id",
        "tiny-three-v1",
        "--output",
        python_topk_half_output,
        "--device",
        args.teacher_device,
        "--storage-dtype",
        "float16",
        "--top-k",
        "64" if args.real else "4",
        "--temperature",
        "2.0",
        *([] if args.real else ["--trust-remote-code"]),
        "--local-files-only",
    )
    python_half_output = output_root / "python-fp16"
    if python_half_output.exists():
        shutil.rmtree(python_half_output)
    run(
        sys.executable,
        "scripts/create-distillation-dataset.py",
        "--teacher",
        teacher_path,
        "--dataset-json",
        input_path,
        "--dataset-id",
        "tiny-three-v1",
        "--output",
        python_half_output,
        "--device",
        args.teacher_device,
        "--storage-dtype",
        "float16",
        *([] if args.real else ["--trust-remote-code"]),
        "--local-files-only",
    )
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-distillation-datasets.lisp",
        env={
            **os.environ,
            "TB_DEVICE": args.device,
            "HF_HUB_OFFLINE": "1",
            "TB_DISTILLATION_DATASET_REAL": "1" if args.real else "0",
            "TB_DISTILLATION_DATASET_PYTHON": "1",
        },
    )
    report = verify(args.device, args.real, output_root)
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Distillation datasets passed: {args.device}, real={args.real}")


if __name__ == "__main__":
    main()
