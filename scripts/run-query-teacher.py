"""Lisp teaches SmolLM2 a bounded query language; fresh Python verifies exports."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
COMMAND = re.compile(r"\(issues :(open|closed|any) :(tokenizer|training|storage|any) :(week|month|any)\)")


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def query_ids(command):
    match = COMMAND.fullmatch(command.strip())
    if match is None:
        return None
    wanted_status, wanted_component, period = match.groups()
    issues = [(1, "open", "tokenizer", 2), (2, "closed", "tokenizer", 8),
              (3, "open", "training", 6), (4, "closed", "storage", 29),
              (5, "open", "storage", 30), (6, "closed", "training", 40)]
    return [i for i, status, component, age in issues
            if wanted_status in ("any", status) and wanted_component in ("any", component)
            and (period == "any" or age < {"week": 7, "month": 30}[period])]


def keyword_prediction(text):
    """Transparent hand-written baseline, not a learned or production parser."""
    text = text.lower()
    opened = "open" in text or "unresolved" in text
    closed = "closed" in text
    status = "open" if opened and not closed else "closed" if closed and not opened else "any"
    component = next((x for x in ("tokenizer", "training", "storage") if x in text), "any")
    if "tokenization" in text:
        component = "tokenizer"
    period = "week" if "week" in text or "seven" in text else "month" if "month" in text or "thirty" in text else "any"
    return f"(issues :{status} :{component} :{period})"


def metrics(rows):
    result = {}
    groups = {
        "validation": [r for r in rows if r["split"] == "validation"],
        "test": [r for r in rows if r["split"] == "test"],
        "test_seen_combinations": [r for r in rows if r["split"] == "test" and not r["held_combination"]],
        "test_held_combinations": [r for r in rows if r["split"] == "test" and r["held_combination"]],
    }
    for name, group in groups.items():
        if group:
            result[name] = {
                "count": len(group),
                "exact": sum(r["prediction"].strip() == r["target"] for r in group),
                "valid": sum(COMMAND.fullmatch(r["prediction"].strip()) is not None for r in group),
                "execution_match": sum(query_ids(r["prediction"]) == query_ids(r["target"]) for r in group),
            }
    return result


def verify(source, output):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    dataset = read(output / "dataset.json")
    baseline = read(output / "baseline.json")
    adapted = read(output / "adapted.json")
    training = read(output / "training.json")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    by_request = {r["request"]: r for r in dataset}
    assert len(by_request) == len(dataset) == 246
    train_targets = {r["target"] for r in dataset if r["split"] == "train"}
    assert len(train_targets) == 30
    for row in dataset:
        assert query_ids(row["target"]) == row["expected_issue_ids"]
        if row["split"] == "test":
            assert (row["target"] not in train_targets) == row["held_combination"]
    expected = [r["request"] for r in dataset if r["split"] != "train"]
    if training["smoke"]:
        expected = expected[:2]
    assert [r["request"] for r in baseline] == expected
    assert [r["request"] for r in adapted] == expected
    for row in baseline + adapted:
        reference = by_request[row["request"]]
        assert row["target"] == reference["target"]
        assert row["split"] == reference["split"]
        assert row["held_combination"] == reference["held_combination"]
        assert row["prompt_ids"] == tokenizer.encode(reference["prompt"], add_special_tokens=False)
        assert tokenizer.decode(row["generated_ids"], skip_special_tokens=True) == row["prediction"]
        assert row["correct"] == (row["prediction"].strip() == row["target"])
        assert row["valid"] == (query_ids(row["prediction"]) is not None)
        if row["valid"]:
            assert row["issue_ids"] == query_ids(row["prediction"])

    def load_model(path):
        return AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        ).eval()

    base = load_model(source)
    with torch.no_grad():
        for row in baseline:
            ids = torch.tensor([row["prompt_ids"]])
            tokens = base.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=24, eos_token_id=0, pad_token_id=0)
            assert tokens[0, ids.shape[1]:].tolist() == row["generated_ids"], row["request"]
    print(f"Python baseline generation matches all {len(baseline)} requests", flush=True)
    model = PeftModel.from_pretrained(base, output / "adapter", local_files_only=True).eval()
    max_error = 0.0
    with torch.no_grad():
        for i in range(2):
            ids = torch.tensor([read(output / f"probe-{i}.json")])
            actual = model(ids).logits
            native = load_file(str(output / f"probe-{i}.safetensors"))["logits"]
            torch.testing.assert_close(actual, native, atol=5e-4, rtol=3e-4)
            max_error = max(max_error, (actual - native).abs().max().item())
        for index, row in enumerate(adapted):
            ids = torch.tensor([row["prompt_ids"]])
            tokens = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                    max_new_tokens=24, eos_token_id=0, pad_token_id=0)
            assert tokens[0, ids.shape[1]:].tolist() == row["generated_ids"], row["request"]
            print(f"Python adapter generation {index + 1}/{len(adapted)} matches", flush=True)
    del model, base
    merged = load_model(output / "merged")
    with torch.no_grad():
        for i in range(2):
            ids = torch.tensor([read(output / f"probe-{i}.json")])
            actual = merged(ids).logits
            native = load_file(str(output / f"probe-{i}.safetensors"))["logits"]
            torch.testing.assert_close(actual, native, atol=5e-4, rtol=3e-4)
            max_error = max(max_error, (actual - native).abs().max().item())
    result = {
        "smoke": training["smoke"], "baseline": metrics(baseline), "adapted": metrics(adapted),
        "keyword_baseline": metrics([{**r, "prediction": keyword_prediction(r["request"])} for r in adapted]),
        "python_baseline_generation_matches": len(baseline),
        "python_adapter_generation_matches": len(adapted), "logits_max_abs_error": max_error,
        "training_seconds": training["training_seconds"], "training_steps": len(training["losses"]),
        "first_loss": training["losses"][0], "last_loss": training["losses"][-1],
    }
    write(output / "results.json", result)
    write(output / "report.json", {
        "results": result, "training": training, "environment": read(output / "environment.json"),
        "verification_source_sha256": digest(Path(__file__)),
        "artifact_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                            if p.is_file() and p.suffix in (".json", ".safetensors")
                            and p.name != "report.json"},
        "predictions": [{"request": after["request"], "split": after["split"],
                         "held_combination": after["held_combination"], "target": after["target"],
                         "before": before["prediction"], "after": after["prediction"],
                         "correct": after["correct"]}
                        for before, after in zip(baseline, adapted, strict=True)],
    })
    print(json.dumps(result, indent=2))
    if not training["smoke"]:
        assert result["adapted"]["test"]["exact"] > result["baseline"]["test"]["exact"], "No held-out improvement"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", type=Path, default=ROOT / ".build/models/smollm2")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory; artifacts can exceed 1 GB")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Download the pinned public checkpoint")
    parser.add_argument("--smoke", action="store_true", help="One update and two evaluations; no quality claim")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    source, output = args.model.resolve(), args.output.resolve()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.verify_only:
        verify(source, output)
        return
    if output.exists():
        parser.error("output already exists; use a new directory or --verify-only")
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=source,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
        (source / "revision.txt").write_text(REVISION + "\n")
    if not (source / "revision.txt").exists() or (source / "revision.txt").read_text().strip() != REVISION:
        parser.error("missing pinned checkpoint; supply --download")
    run("sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script", "tests/query-teacher.lisp")
    run("sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script", "scripts/test-query-training.lisp")
    output.mkdir(parents=True)
    sources = [*sorted((ROOT / "examples/query-teacher").glob("*.lisp")),
               Path(__file__), ROOT / "uv.lock", *sorted((ROOT / "src").glob("*.lisp"))]
    write(output / "environment.json", {
        "model_id": MODEL, "revision": REVISION, "device": args.device,
        "platform": platform.platform(), "python": sys.version,
        "sbcl": subprocess.check_output(["sbcl", "--version"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "packages": {p: importlib.metadata.version(p) for p in ["mlx", "torch", "transformers", "peft", "tokenizers"]},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "checkpoint_sha256": {p.name: digest(p) for p in sorted(source.iterdir())
                              if p.suffix in (".json", ".safetensors", ".txt")},
    })
    run("sbcl", "--noinform", "--no-sysinit", "--no-userinit", "--script", "examples/query-teacher/run.lisp",
        env={**os.environ, "TB_QUERY_MODEL": str(source), "TB_QUERY_OUTPUT": str(output),
             "TB_DEVICE": args.device, "TB_QUERY_EPOCHS": str(1 if args.smoke else args.epochs),
             "TB_QUERY_BATCH": str(args.batch_size), "TB_QUERY_SMOKE": "1" if args.smoke else "0"})
    verify(source, output)


if __name__ == "__main__":
    main()
