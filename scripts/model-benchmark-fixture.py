"""Create the deterministic standard Hugging Face checkpoint used by model benchmarks."""

import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def create(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1701)
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).float().eval()
    model.save_pretrained(root)
    manifest = {
        "format_version": 1,
        "seed": 1701,
        "purpose": "reproducible full-model performance comparison",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (root / "benchmark-fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Benchmark fixture ready: {root} ({manifest['parameter_count']} parameters)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default=".build/fixtures/benchmark-llama")
    create(parser.parse_args().destination)
