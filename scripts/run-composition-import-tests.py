"""Qualify faithful Llama checkpoint conversion to public Lisp Transformer components."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)


def paths(device, real):
    output = ROOT / f".build/composition-import-{device}"
    if real:
        return [(ROOT / ".build/models/smollm2", output / "real")]
    return [
        (ROOT / ".build/fixtures/composition-import" / name, output / name)
        for name in ("tied", "untied", "mha", "mqa")
    ]


def python_gate(device, real, phase):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.set_num_threads(1)
    ids = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.tensor([[-100, 4, 5, -100], [-100, -100, 4, 3]])
    report = {}
    for index, (source, destination) in enumerate(paths(device, real)):
        if phase == "fixture" and not real:
            torch.manual_seed(1901 + index)
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=28,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads={"mha": 4, "mqa": 1}.get(source.name, 2),
                max_position_embeddings=64,
                rms_norm_eps=3e-5,
                tie_word_embeddings=source.name != "untied",
                attention_dropout=0.0,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                rope_parameters={"rope_type": "default", "rope_theta": 100000.0},
            )
            model = LlamaForCausalLM(config).float().eval()
            model.generation_config.use_cache = True
            model.save_pretrained(source, max_shard_size="1KB")
            words = [
                "<unk>",
                "<bos>",
                "<eos>",
                "Common",
                "Lisp",
                "is",
                "fast",
                "and",
                "fun",
            ]
            words += [f"token{i}" for i in range(len(words), 32)]
            tokenizer = Tokenizer(
                models.WordLevel(dict(zip(words, range(32))), unk_token="<unk>")
            )
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
                pad_token="<unk>",
            ).save_pretrained(source)
        original = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        if phase == "fixture":
            destination.mkdir(parents=True, exist_ok=True)
            out = original(ids, attention_mask=mask, labels=labels)
            text = "Common Lisp is fast and fun"
            reference = {
                "input_ids": ids.tolist(),
                "attention_mask": mask.tolist(),
                "labels": labels.tolist(),
                "logits": out.logits.detach().tolist(),
                "loss": out.loss.item(),
                "text": text,
                "tokens": tokenizer.encode(text, add_special_tokens=False),
            }
            # Keep references outside the atomically replaced conversion directory.
            (destination.parent / f"{destination.name}-reference.json").write_text(
                json.dumps(reference) + "\n"
            )
            if not real:
                out.loss.backward()
                save_file(
                    {n: p.grad.contiguous() for n, p in original.named_parameters()},
                    str(
                        destination.parent / f"{destination.name}-gradients.safetensors"
                    ),
                )
            continue

        converted = AutoModelForCausalLM.from_pretrained(
            destination, trust_remote_code=True, local_files_only=True
        ).eval()
        assert converted.config.model_type == "tb_composed"
        a, b = dict(original.named_parameters()), dict(converted.named_parameters())
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        assert (
            original.config.tie_word_embeddings == converted.config.tie_word_embeddings
        )
        if converted.config.tie_word_embeddings:
            assert converted.lm_head.weight is converted.model.embed_tokens.weight
        atol, rtol = (1e-3, 3e-4) if real else (2e-6, 3e-4)
        with torch.no_grad():
            expected = original(ids, attention_mask=mask).logits
            actual = converted(ids, attention_mask=mask).logits
        native = load_file(str(destination / "native-logits.safetensors"))["logits"]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(native, expected, atol=atol, rtol=rtol)
        metrics = {
            "canonical_parameters": len(a),
            "python_logits_max_abs_error": (actual - expected).abs().max().item(),
            "native_logits_max_abs_error": (native - expected).abs().max().item(),
        }
        exported_tokenizer = AutoTokenizer.from_pretrained(
            destination, trust_remote_code=True, local_files_only=True
        )
        text = "Common Lisp is fast and fun"
        assert tokenizer.encode(text) == exported_tokenizer.encode(text)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ):
            if (source / name).exists():
                assert (source / name).read_bytes() == (destination / name).read_bytes()
        generation = json.loads((destination / "generation_config.json").read_text())
        old_generation = json.loads((source / "generation_config.json").read_text())
        assert generation == {**old_generation, "use_cache": False}
        with torch.no_grad():
            # Deliberately omit use_cache: GenerationConfig must choose the valid path.
            generated = converted.generate(
                torch.tensor([[3, 4]]),
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
            )[0].tolist()
            expected_tokens = original.generate(
                torch.tensor([[3, 4]]),
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
            )[0].tolist()
        assert (
            generated
            == expected_tokens
            == json.loads((destination / "native-generation.json").read_text())
        )
        adapted = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                destination, trust_remote_code=True, local_files_only=True
            ),
            destination / "adapter",
        ).eval()
        with torch.no_grad():
            adapted_logits = adapted(ids, attention_mask=mask).logits
        native_adapted = load_file(str(destination / "adapter-logits.safetensors"))[
            "logits"
        ]
        torch.testing.assert_close(adapted_logits, native_adapted, atol=atol, rtol=rtol)
        metrics["trained_adapter_logits_max_abs_error"] = (
            (adapted_logits - native_adapted).abs().max().item()
        )
        if not real:
            original(ids, attention_mask=mask, labels=labels).loss.backward()
            converted(ids, attention_mask=mask, labels=labels).loss.backward()
            gradients = load_file(str(destination / "native-gradients.safetensors"))
            assert gradients.keys() == a.keys()
            for name in a:
                torch.testing.assert_close(
                    b[name].grad, a[name].grad, atol=2e-6, rtol=3e-4
                )
                torch.testing.assert_close(
                    gradients[name], a[name].grad, atol=2e-6, rtol=3e-4
                )
            metrics["gradient_max_abs_error"] = max(
                (gradients[n] - a[n].grad).abs().max().item() for n in a
            )
            for suffix in (
                ["updated", "updated-source"] if source.name == "tied" else ["updated"]
            ):
                updated = AutoModelForCausalLM.from_pretrained(
                    destination / suffix, trust_remote_code=True, local_files_only=True
                )
                for name, p in updated.named_parameters():
                    torch.testing.assert_close(
                        p, a[name] - 0.01 * a[name].grad, atol=2e-6, rtol=3e-4
                    )
            # Train the converted architecture in Python, resave, and reload in Lisp.
            optimizer = torch.optim.SGD(converted.parameters(), lr=0.02)
            optimizer.step()
            continued = destination / "python-continued"
            converted.save_pretrained(continued)
            exported_tokenizer.save_pretrained(continued)
            with torch.no_grad():
                out = converted(ids, attention_mask=mask, labels=labels)
            (continued / "reference.json").write_text(
                json.dumps({"logits": out.logits.tolist(), "loss": out.loss.item()})
                + "\n"
            )
        report[destination.name] = metrics
    if phase == "verify":
        output = (
            ROOT
            / f".build/composition-import-{device}"
            / ("real-validation.json" if real else "validation.json")
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--phase", choices=["fixture", "verify"])
    args = parser.parse_args()
    if args.phase:
        python_gate(args.device, args.real, args.phase)
        return
    if args.real:
        source = ROOT / ".build/models/smollm2"
        pin = source / "revision.txt"
        if args.local_files_only:
            if not pin.is_file() or pin.read_text().strip() != REVISION:
                raise RuntimeError(
                    "Pinned SmolLM2 snapshot is not present; run without --local-files-only to fetch it"
                )
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_dir=source,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                ],
            )
            pin.write_text(REVISION + "\n")
    flags = ["--device", args.device] + (["--real"] if args.real else [])
    env = {
        **os.environ,
        "TB_DEVICE": args.device,
        "TB_COMPOSITION_REAL": "1" if args.real else "0",
        "HF_HUB_OFFLINE": "1",
    }
    run(sys.executable, __file__, *flags, "--phase", "fixture", env=env)
    run(
        "sbcl",
        "--noinform",
        "--no-sysinit",
        "--no-userinit",
        "--script",
        "scripts/test-composition-import.lisp",
        env=env,
    )
    run(sys.executable, __file__, *flags, "--phase", "verify", env=env)
    if not args.real:
        run(
            "sbcl",
            "--noinform",
            "--no-sysinit",
            "--no-userinit",
            "--script",
            "scripts/test-composition-import.lisp",
            env={**env, "TB_COMPOSITION_ROUNDTRIP": "1"},
        )
    print(f"Pretrained component conversion passed: {args.device}, real={args.real}")


if __name__ == "__main__":
    main()
