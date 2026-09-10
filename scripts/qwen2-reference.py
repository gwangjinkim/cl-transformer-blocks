"""Independent Qwen2 fixtures and native-export verification."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
)


torch.set_num_threads(1)
IDS = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
LABELS = torch.tensor([[-100, 4, 5, -100], [-100, -100, 4, 3]])


def load(root):
    return AutoModelForCausalLM.from_pretrained(
        root,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()


def write_reference(root, training=True):
    root = Path(root)
    model = load(root)
    output = model(IDS, attention_mask=MASK, labels=LABELS if training else None)
    with torch.no_grad():
        generated = torch.tensor([[3, 4]])
        for _ in range(4):
            next_token = model(generated).logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    (root / "reference.json").write_text(
        json.dumps(
            {
                "input_ids": IDS.tolist(),
                "attention_mask": MASK.tolist(),
                "logits": output.logits.detach().tolist(),
                "generated": generated[0].tolist(),
                **(
                    {"labels": LABELS.tolist(), "loss": output.loss.item()}
                    if training
                    else {}
                ),
            }
        )
    )
    if training:
        output.loss.backward()
        save_file(
            {name: parameter.grad.contiguous() for name, parameter in model.named_parameters()},
            str(root / "gradients.safetensors"),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.sub_(0.01 * parameter.grad)
        model.save_pretrained(root / "updated")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    text = "Common Lisp is fast and fun"
    ids = tokenizer.encode(text, add_special_tokens=False)
    (root / "tokenizer-reference.json").write_text(
        json.dumps({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    )


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(53)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config).float().eval()
    model.set_attn_implementation("eager")
    # Nonzero projection biases make omission immediately visible in logits/gradients.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")):
                parameter.uniform_(-0.12, 0.12)
    model.save_pretrained(root)
    characters = sorted(set("CommonLispisfastandfun"))
    vocabulary = {token: index for index, token in enumerate(["<unk>", "<bos>", "<eos>", "Ġ"] + characters)}
    tokenizer = Qwen2Tokenizer(
        vocab=vocabulary,
        merges=[],
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<unk>",
    )
    tokenizer.save_pretrained(root)
    write_reference(root)
    torch.manual_seed(59)
    adapter = get_peft_model(
        load(root),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).eval()
    adapter.peft_config["default"].base_model_name_or_path = str(root.resolve())
    with torch.no_grad():
        for _, parameter in adapter.named_parameters():
            if parameter.requires_grad:
                parameter.uniform_(-0.08, 0.08)
    adapter_folder = root / "adapter"
    adapter.save_pretrained(adapter_folder)
    with torch.no_grad():
        logits = adapter(IDS, attention_mask=MASK).logits
    (adapter_folder / "reference.json").write_text(json.dumps({"logits": logits.tolist()}))


def verify(source, destination, updated=False):
    source, destination = Path(source), Path(destination)
    expected = load(source / "updated" if updated else source)
    actual = load(destination)
    assert expected.config.model_type == actual.config.model_type == "qwen2"
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(
            value,
            actual.state_dict()[name],
            atol=2e-6 if updated else 0,
            rtol=2e-4 if updated else 0,
        )
    with torch.no_grad():
        torch.testing.assert_close(
            expected(IDS, attention_mask=MASK).logits,
            actual(IDS, attention_mask=MASK).logits,
            atol=3e-5,
            rtol=3e-4,
        )
    source_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    exported_tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    assert source_tokenizer.encode("Common Lisp is fast and fun") == exported_tokenizer.encode(
        "Common Lisp is fast and fun"
    )
    print("Fresh Qwen2 reload matches:", destination)


def verify_adapter(base, destination, merged=False):
    base, destination = Path(base), Path(destination)
    expected = PeftModel.from_pretrained(load(base), base / "adapter").eval()
    actual = load(destination) if merged else PeftModel.from_pretrained(load(base), destination).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            expected(IDS, attention_mask=MASK).logits,
            actual(IDS, attention_mask=MASK).logits,
            atol=3e-5,
            rtol=3e-4,
        )
    print("Fresh Qwen2 PEFT reload matches:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fixture", "reference", "verify", "verify-adapter"])
    parser.add_argument("path")
    parser.add_argument("exported", nargs="?")
    parser.add_argument("--updated", action="store_true")
    parser.add_argument("--merged", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "fixture":
        fixture(arguments.path)
    elif arguments.action == "reference":
        write_reference(arguments.path, training=False)
    elif arguments.action == "verify":
        verify(arguments.path, arguments.exported, arguments.updated)
    else:
        verify_adapter(arguments.path, arguments.exported, arguments.merged)
