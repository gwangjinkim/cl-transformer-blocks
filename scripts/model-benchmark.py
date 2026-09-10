"""Benchmark full-model Lisp/MLX and PyTorch prefill, cached decode, and training."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("prefill", "decode", "training")


def validate_settings(config, *, batch, prefill_tokens, decode_tokens,
                      training_tokens, repeats, warmups):
    values = (batch, prefill_tokens, decode_tokens, training_tokens, repeats)
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("batch, token counts, and repeats must be positive integers")
    if not isinstance(warmups, int) or warmups < 1:
        raise ValueError("warmups must be a positive integer")
    context = int(config.get("max_position_embeddings", 0))
    if not context or prefill_tokens + decode_tokens > context or training_tokens > context:
        raise ValueError(f"benchmark token counts exceed model context length {context}")
    if int(config.get("vocab_size", 0)) <= 3:
        raise ValueError("benchmark requires a vocabulary larger than three tokens")


def summarize(workload, samples, *, batch, tokens):
    median = statistics.median(samples)
    return {
        "workload": workload,
        "samples_seconds": samples,
        "median_seconds": median,
        "batch": batch,
        "tokens_per_iteration": batch * tokens,
        "tokens_per_second": batch * tokens / median,
    }


def validate_pair(native, python):
    if native.get("workload") != python.get("workload"):
        raise ValueError("frontend workload mismatch")
    if not math.isclose(float(native["fingerprint"]), float(python["fingerprint"]),
                        rel_tol=2e-3, abs_tol=2e-3):
        raise ValueError(
            f"frontend fingerprint mismatch: {native['fingerprint']} vs {python['fingerprint']}"
        )


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def actual_torch_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("requested PyTorch GPU is unavailable; CPU fallback is forbidden")


def token_ids(batch, tokens, vocabulary, device, offset=0):
    values = (torch.arange(batch * tokens, device=device) + offset) % (vocabulary - 3) + 3
    return values.reshape(batch, tokens)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def torch_memory_start(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def torch_memory_result(device):
    result = {"process_peak_rss_bytes": peak_rss_bytes()}
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        result["allocated_bytes_after"] = torch.mps.current_allocated_memory()
        result["driver_bytes_after"] = torch.mps.driver_allocated_memory()
    return result


def python_worker(args):
    device = actual_torch_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).to(device)
    vocabulary = model.config.vocab_size
    prefill = token_ids(args.batch, args.prefill_tokens, vocabulary, device)
    training = token_ids(args.batch, args.training_tokens, vocabulary, device)
    next_ids = token_ids(args.batch, 1, vocabulary, device, args.prefill_tokens)

    def prefill_once():
        model.eval()
        with torch.inference_mode():
            output = model(prefill, use_cache=False).logits
        sync(device)
        return output

    def training_once():
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(training[:, :-1], use_cache=False).logits
        loss = functional.cross_entropy(
            logits.reshape(-1, vocabulary), training[:, 1:].reshape(-1)
        )
        loss.backward()
        sync(device)
        return float(loss.detach().float().cpu())

    def decode_once(fingerprint=False):
        model.eval()
        with torch.inference_mode():
            cache = model(prefill, use_cache=True).past_key_values
            sync(device)
            start = time.perf_counter()
            output = None
            for _ in range(args.decode_tokens):
                output = model(next_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                sync(device)
            seconds = (time.perf_counter() - start) / args.decode_tokens
        return seconds, (float(output.logits.float().sum().cpu()) if fingerprint else None)

    operation = {"prefill": prefill_once, "training": training_once}.get(args.workload)
    fingerprint = None
    for _ in range(args.warmups):
        if args.workload == "decode":
            _, fingerprint = decode_once(fingerprint=True)
        elif args.workload == "prefill":
            fingerprint = float(operation().float().sum().cpu())
        else:
            fingerprint = operation()
    torch_memory_start(device)
    samples = []
    for _ in range(args.repeats):
        if args.workload == "decode":
            seconds, _ = decode_once()
        else:
            start = time.perf_counter()
            output = operation()
            seconds = time.perf_counter() - start
            del output
        samples.append(seconds)
    result = {
        "frontend": "python-pytorch",
        "workload": args.workload,
        "actual_device": device.type,
        "dtype": "float32",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "samples_seconds": samples,
        "fingerprint": fingerprint,
        "allocator_memory": torch_memory_result(device),
    }
    print(json.dumps(result))


def measured(command, env=None):
    output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
    try:
        return json.loads(next(line for line in reversed(output.splitlines()) if line.startswith("{")))
    except (StopIteration, json.JSONDecodeError) as error:
        raise RuntimeError(f"benchmark worker returned no JSON:\n{output}") from error


def source_hashes(source):
    paths = [source / "config.json"]
    index = source / "model.safetensors.index.json"
    if index.exists():
        paths.append(index)
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        paths.extend(source / name for name in sorted(set(weight_map.values())))
    else:
        paths.append(source / "model.safetensors")
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.exists()
    }


def source_revision(source):
    path = source / "revision.txt"
    return path.read_text().strip() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--source", type=Path, default=Path(".build/fixtures/benchmark-llama"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--training-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["python"], help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = (ROOT / args.source).resolve() if not args.source.is_absolute() else args.source
    if args.worker:
        python_worker(args)
        return
    if not (args.source / "config.json").exists():
        if args.source == (ROOT / ".build/fixtures/benchmark-llama").resolve():
            subprocess.run(
                [sys.executable, "scripts/model-benchmark-fixture.py", args.source],
                cwd=ROOT, check=True,
            )
        else:
            parser.error(f"checkpoint does not exist: {args.source}")
    config = json.loads((args.source / "config.json").read_text())
    validate_settings(config, batch=args.batch, prefill_tokens=args.prefill_tokens,
                      decode_tokens=args.decode_tokens, training_tokens=args.training_tokens,
                      repeats=args.repeats, warmups=args.warmups)
    settings = {
        "batch": args.batch, "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens, "training_tokens": args.training_tokens,
        "warmups": args.warmups, "repeats": args.repeats,
    }
    environment = {
        **os.environ,
        "TB_DEVICE": args.device,
        "TB_MODEL_BENCH_SOURCE": str(args.source) + os.sep,
        **{f"TB_MODEL_BENCH_{key.split('_')[0].upper()}": str(value)
           for key, value in settings.items()},
    }
    records = []
    for workload in WORKLOADS:
        environment["TB_MODEL_BENCH_WORKLOAD"] = workload
        common = [
            "--device", args.device, "--source", str(args.source),
            "--batch", str(args.batch), "--prefill-tokens", str(args.prefill_tokens),
            "--decode-tokens", str(args.decode_tokens),
            "--training-tokens", str(args.training_tokens),
            "--warmups", str(args.warmups), "--repeats", str(args.repeats),
            "--workload", workload,
        ]
        frontends = {}
        order = ("native", "python") if workload != "decode" else ("python", "native")
        for frontend in order:
            if frontend == "native":
                command = ["sbcl", "--noinform", "--no-userinit", "--no-sysinit",
                           "--script", "scripts/model-benchmark.lisp"]
                frontends[frontend] = measured(command, environment)
            else:
                command = [sys.executable, "scripts/model-benchmark.py", "--worker", "python", *common]
                frontends[frontend] = measured(command)
        validate_pair(frontends["native"], frontends["python"])
        tokens = {"prefill": args.prefill_tokens,
                  "decode": 1, "training": args.training_tokens - 1}[workload]
        for frontend, raw in frontends.items():
            raw.update(summarize(workload, raw["samples_seconds"],
                                 batch=args.batch, tokens=tokens))
        records.append({"workload": workload, "frontends": frontends,
                        "native_over_python": frontends["native"]["median_seconds"] /
                                              frontends["python"]["median_seconds"]})
        print(f"{workload:8} native={frontends['native']['median_seconds']:.6f}s "
              f"python={frontends['python']['median_seconds']:.6f}s "
              f"ratio={records[-1]['native_over_python']:.3f}")
    parameter_count = records[0]["frontends"]["python"]["parameter_count"]
    try:
        display_source = str(args.source.relative_to(ROOT))
    except ValueError:
        display_source = str(args.source)
    fixture_manifest = args.source / "benchmark-fixture.json"
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": display_source, "source_revision": source_revision(args.source),
        "fixture_manifest": (json.loads(fixture_manifest.read_text())
                             if fixture_manifest.exists() else None),
        "artifact_sha256": source_hashes(args.source),
        "benchmark_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("scripts/model-benchmark.py", "scripts/model-benchmark.lisp",
                         "scripts/model-benchmark-fixture.py", "native/mlx_bridge.c",
                         "src/mlx.lisp", "src/llama.lisp", "src/gpt2.lisp")
        },
        "model_type": config.get("model_type"), "parameter_count": parameter_count,
        "requested_device": args.device, "dtype": "float32", "settings": settings,
        "records": records,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": (subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip() if platform.system() == "Darwin" else platform.processor()),
            "python": sys.version, "sbcl": subprocess.check_output(["sbcl", "--version"], text=True).strip(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("mlx", "torch", "transformers", "safetensors")},
        },
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
    }
    output = args.output or ROOT / f".build/model-benchmark-{args.device}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
