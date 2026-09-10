"""Reproduce native BERT masked-LM acceptance against pinned Transformers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    arguments = parser.parse_args()
    if arguments.real:
        from huggingface_hub import snapshot_download

        destination = ROOT / ".build/models/bert-base-uncased"
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=destination,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        (destination / "revision.txt").write_text(MODEL_REVISION + "\n")
        run(sys.executable, "scripts/bert-reference.py", "reference", destination)
    else:
        run(sys.executable, "scripts/bert-reference.py", "fixture", ".build/fixtures/bert")
        run(
            sys.executable, "scripts/bert-lora-reference.py", "fixture",
            ".build/fixtures/bert", ".build/fixtures/bert-lora",
        )
        run(
            sys.executable, "scripts/bert-lora-reference.py", "fixture",
            ".build/fixtures/bert", ".build/fixtures/bert-rslora", "--rslora",
        )
    environment = {**os.environ, "TB_DEVICE": arguments.device}
    if arguments.real:
        environment["TB_BERT_REAL"] = "1"
    else:
        environment.pop("TB_BERT_REAL", None)
    run(
        "sbcl", "--noinform", "--no-userinit", "--no-sysinit", "--script", "scripts/test-bert.lisp",
        env=environment,
    )
    if arguments.real:
        run(
            sys.executable, "scripts/bert-reference.py", "verify",
            ".build/models/bert-base-uncased", f".build/bert-export-real-{arguments.device}",
        )
        print(f"All BERT masked-LM acceptance gates passed: {arguments.device}, real=True")
        return
    run(
        sys.executable, "scripts/bert-reference.py", "verify", ".build/fixtures/bert",
        f".build/bert-export-{arguments.device}",
    )
    run(
        sys.executable, "scripts/bert-reference.py", "verify", ".build/fixtures/bert",
        f".build/bert-updated-{arguments.device}", "--updated",
    )
    run(
        sys.executable, "scripts/bert-reference.py", "verify", ".build/fixtures/bert",
        f".build/bert-token-type-updated-{arguments.device}", "--updated", "--token-types",
    )
    for variant in ("lora", "rslora"):
        for suffix, options in (
            ("", ()),
            ("-sgd", ("--updated",)),
            ("-merged", ("--merged",)),
        ):
            run(
                sys.executable, "scripts/bert-lora-reference.py", "verify",
                ".build/fixtures/bert", f".build/fixtures/bert-{variant}",
                f".build/bert-{variant}{suffix}-{arguments.device}", *options,
            )
    for variant in ("lora", "rslora"):
        run(
            sys.executable, "scripts/bert-lora-reference.py", "verify",
            ".build/fixtures/bert", f".build/fixtures/bert-{variant}",
            f".build/bert-{variant}-native-{arguments.device}", "--native",
        )
    print(f"All BERT masked-LM acceptance gates passed: {arguments.device}, real=False")


if __name__ == "__main__":
    main()
