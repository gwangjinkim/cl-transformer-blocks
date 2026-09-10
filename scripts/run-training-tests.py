"""Run the reproducible PEFT and masked optimizer milestone, requiring the chosen device."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    run(sys.executable, "scripts/reference.py", "fixture", ".build/fixtures/tiny")
    run(
        sys.executable,
        "scripts/peft-reference.py",
        "fixture",
        ".build/fixtures/tiny",
        ".build/fixtures/peft",
    )
    run(
        "sbcl",
        "--noinform",
        "--no-userinit",
        "--no-sysinit",
        "--script",
        "scripts/test-training.lisp",
        env={**os.environ, "TB_DEVICE": args.device},
    )
    for variant in ["standard", "rslora"]:
        source = f".build/fixtures/peft/{variant}"
        for suffix in ["", "-sgd", "-adamw"]:
            run(
                sys.executable,
                "scripts/peft-reference.py",
                "verify",
                ".build/fixtures/tiny",
                source + ("/" + suffix[1:] if suffix else ""),
                f".build/peft-{variant}{suffix}-{args.device}",
            )
        run(
            sys.executable,
            "scripts/peft-reference.py",
            "verify",
            ".build/fixtures/tiny",
            source,
            f".build/peft-merged-{variant}-{args.device}",
            "--merged",
        )
    # Full-model optimizer results are compared directly to independent Torch state_dict.
    import torch
    from transformers import AutoModelForCausalLM
    expected = AutoModelForCausalLM.from_pretrained(
        ROOT / ".build/fixtures/peft/full/adamw", local_files_only=True
    )
    actual = AutoModelForCausalLM.from_pretrained(
        ROOT / f".build/full-adamw-{args.device}", local_files_only=True
    )
    for name, parameter in expected.state_dict().items():
        torch.testing.assert_close(
            parameter, actual.state_dict()[name], atol=3e-6, rtol=3e-4
        )
    from peft import PeftModel
    from safetensors.torch import load_file

    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            ROOT / ".build/fixtures/tiny", local_files_only=True
        ),
        ROOT / f".build/peft-native-{args.device}",
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        torch.testing.assert_close(
            model(ids, attention_mask=mask).logits,
            load_file(
                str(ROOT / f".build/native-adapter-logits-{args.device}.safetensors")
            )["logits"],
            atol=3e-5,
            rtol=3e-4,
        )
    # Native training sidecars must leave ordinary Transformers/PEFT artifacts usable.
    AutoModelForCausalLM.from_pretrained(
        ROOT / f".build/native-full-checkpoint-{args.device}", local_files_only=True
    )
    PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            ROOT / ".build/fixtures/tiny", local_files_only=True
        ),
        ROOT / f".build/native-adapter-checkpoint-{args.device}",
    )
    print(f"All PEFT/training gates passed: {args.device}")


if __name__ == "__main__":
    main()
