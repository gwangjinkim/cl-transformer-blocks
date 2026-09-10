"""Reproduce Lisp-defined portable architecture and Transformers export gates."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    os.chdir(ROOT)
    run(sys.executable, "scripts/portable-reference.py", "fixture",
        ".build/fixtures/portable")
    run("sbcl", "--noinform", "--no-userinit", "--no-sysinit", "--script",
        "scripts/test-portable.lisp", env={**os.environ, "TB_DEVICE": args.device})
    for suffix in ("export", "sharded"):
        run(sys.executable, "scripts/portable-reference.py", "verify",
            ".build/fixtures/portable", f".build/portable-{suffix}-{args.device}")
    run(sys.executable, "scripts/portable-reference.py", "verify",
        ".build/fixtures/portable", f".build/portable-updated-{args.device}",
        "--updated")
    run(sys.executable, "scripts/portable-reference.py", "smoke",
        f".build/portable-lisp-created-{args.device}")
    run("sbcl", "--noinform", "--no-userinit", "--no-sysinit", "--script",
        "scripts/test-portable-resave.lisp",
        env={**os.environ, "TB_DEVICE": args.device})
    print(f"All portable architecture gates passed: {args.device}")


if __name__ == "__main__":
    main()
