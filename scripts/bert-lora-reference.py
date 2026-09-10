"""Independent PEFT LoRA fixture and reload checks for BERT masked LM."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForMaskedLM


torch.set_num_threads(1)
IDS = torch.tensor([[2, 5, 4, 3, 0], [2, 6, 7, 4, 3]])
MASK = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
LABELS = torch.tensor([[-100, -100, 7, -100, -100], [-100, 5, -100, -100, -100]])
TOKEN_TYPES = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])


def base(path):
    return AutoModelForMaskedLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()


def trainable(model):
    return {
        name.replace(".default.", "."): parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def fixture(base_path, destination, rslora=False):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(47)
    config = LoraConfig(
        r=3 if rslora else 2,
        lora_alpha=4,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=rslora,
    )
    model = get_peft_model(base(base_path), config).eval()
    model.peft_config["default"].base_model_name_or_path = str(Path(base_path).resolve())
    with torch.no_grad():
        for parameter in trainable(model).values():
            parameter.uniform_(-0.08, 0.08)
    model.save_pretrained(root)
    output = model(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES, labels=LABELS)
    output.loss.backward()
    (root / "reference.json").write_text(
        json.dumps(
            {
                "input_ids": IDS.tolist(),
                "attention_mask": MASK.tolist(),
                "token_type_ids": TOKEN_TYPES.tolist(),
                "labels": LABELS.tolist(),
                "logits": output.logits.detach().tolist(),
                "loss": output.loss.item(),
            }
        )
    )
    save_file(
        {
            name: parameter.grad.contiguous()
            for name, parameter in trainable(model).items()
        },
        str(root / "gradients.safetensors"),
    )
    with torch.no_grad():
        for parameter in trainable(model).values():
            parameter.sub_(0.01 * parameter.grad)
    model.save_pretrained(root / "sgd")


def verify(base_path, source, exported, updated=False, merged=False, native=False):
    source = Path(source)
    adapter = source / "sgd" if updated else source
    expected = (
        base(base_path)
        if native
        else PeftModel.from_pretrained(base(base_path), adapter).eval()
    )
    if merged:
        expected = expected.merge_and_unload()
        actual = base(exported)
    else:
        actual = PeftModel.from_pretrained(base(base_path), exported).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            actual(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES).logits,
            expected(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES).logits,
            atol=3e-5,
            rtol=3e-4,
        )
    if not merged and not native:
        expected_weights = load_file(str(adapter / "adapter_model.safetensors"))
        actual_weights = load_file(str(Path(exported) / "adapter_model.safetensors"))
        assert expected_weights.keys() == actual_weights.keys()
        for name, value in expected_weights.items():
            torch.testing.assert_close(
                value,
                actual_weights[name],
                atol=2e-6 if updated else 0,
                rtol=2e-4 if updated else 0,
            )
    print("Fresh BERT PEFT reload matches:", exported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fixture", "verify"])
    parser.add_argument("base")
    parser.add_argument("source")
    parser.add_argument("exported", nargs="?")
    parser.add_argument("--updated", action="store_true")
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--rslora", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "fixture":
        fixture(arguments.base, arguments.source, arguments.rslora)
    else:
        verify(
            arguments.base,
            arguments.source,
            arguments.exported,
            arguments.updated,
            arguments.merged,
            arguments.native,
        )
