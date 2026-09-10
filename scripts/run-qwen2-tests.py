"""Reproduce native Qwen2 acceptance against pinned Transformers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    arguments = parser.parse_args()
    if arguments.real:
        from huggingface_hub import snapshot_download

        destination = ROOT / ".build/models/qwen2.5-0.5b"
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=destination,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        (destination / "revision.txt").write_text(MODEL_REVISION + "\n")
        run(sys.executable, "scripts/qwen2-reference.py", "reference", destination)
    else:
        run(sys.executable, "scripts/reference.py", "fixture", ".build/fixtures/tiny")
        run(sys.executable, "scripts/qwen2-reference.py", "fixture", ".build/fixtures/qwen2")
    environment = {**os.environ, "TB_DEVICE": arguments.device}
    if arguments.real:
        environment["TB_QWEN2_REAL"] = "1"
    else:
        environment.pop("TB_QWEN2_REAL", None)
    run(
        "sbcl",
        "--noinform",
        "--no-userinit",
        "--no-sysinit",
        "--script",
        "scripts/test-qwen2.lisp",
        env=environment,
    )
    if arguments.real:
        run(
            sys.executable,
            "scripts/qwen2-reference.py",
            "verify",
            ".build/models/qwen2.5-0.5b",
            f".build/qwen2-export-real-{arguments.device}",
        )
        print(f"All Qwen2 acceptance gates passed: {arguments.device}, real=True")
        return
    run(
        sys.executable,
        "scripts/qwen2-reference.py",
        "verify",
        ".build/fixtures/qwen2",
        f".build/qwen2-export-{arguments.device}",
    )
    run(
        sys.executable,
        "scripts/qwen2-reference.py",
        "verify",
        ".build/fixtures/qwen2",
        f".build/qwen2-updated-{arguments.device}",
        "--updated",
    )
    run(
        sys.executable,
        "scripts/qwen2-reference.py",
        "verify-adapter",
        ".build/fixtures/qwen2",
        f".build/qwen2-adapter-{arguments.device}",
    )
    run(
        sys.executable,
        "scripts/qwen2-reference.py",
        "verify-adapter",
        ".build/fixtures/qwen2",
        f".build/qwen2-merged-{arguments.device}",
        "--merged",
    )
    print(f"All Qwen2 acceptance gates passed: {arguments.device}, real=False")


if __name__ == "__main__":
    main()
