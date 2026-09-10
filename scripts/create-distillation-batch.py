"""Create one portable teacher-logit batch for native Common Lisp distillation."""

import argparse
import json
from pathlib import Path

from distillation_artifacts import create_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--batch-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16", "auto"], default="float32"
    )
    parser.add_argument(
        "--storage-dtype", choices=["float32", "float16"], default="float32"
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    batch = json.loads(Path(args.batch_json).read_text())
    result = create_batch(
        args.teacher,
        batch,
        args.output,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        storage_dtype=args.storage_dtype,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    print(result)


if __name__ == "__main__":
    main()
