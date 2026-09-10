"""Independent BERT masked-LM fixtures and native-export verification."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
)


torch.set_num_threads(1)
IDS = torch.tensor([[2, 5, 4, 3, 0], [2, 6, 7, 4, 3]])
MASK = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
LABELS = torch.tensor([[-100, -100, 7, -100, -100], [-100, 5, -100, -100, -100]])
TOKEN_TYPES = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])


def load(path):
    return AutoModelForMaskedLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()


def reference(root, training=True):
    root = Path(root)
    model = load(root)
    output = model(IDS, attention_mask=MASK, labels=LABELS if training else None)
    payload = {
        "input_ids": IDS.tolist(),
        "attention_mask": MASK.tolist(),
        "logits": output.logits.detach().tolist(),
    }
    if training:
        payload.update({"labels": LABELS.tolist(), "loss": output.loss.item()})
    (root / "reference.json").write_text(json.dumps(payload))
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
    encoded = tokenizer.encode(text, add_special_tokens=True)
    (root / "tokenizer-reference.json").write_text(
        json.dumps({"text": text, "ids": encoded, "decoded": tokenizer.decode(encoded)})
    )
    model = load(root)
    output = model(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES,
                   labels=LABELS if training else None)
    payload = {"input_ids": IDS.tolist(), "attention_mask": MASK.tolist(),
               "token_type_ids": TOKEN_TYPES.tolist(), "logits": output.logits.detach().tolist()}
    if training:
        payload.update({"labels": LABELS.tolist(), "loss": output.loss.item()})
        output.loss.backward()
        save_file({name: parameter.grad.contiguous() for name, parameter in model.named_parameters()},
                  str(root / "token-type-gradients.safetensors"))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.sub_(0.01 * parameter.grad)
        model.save_pretrained(root / "token-type-updated")
    (root / "token-type-reference.json").write_text(json.dumps(payload))
    model = load(root)
    tokenizer_cases = []
    for texts, pairs, special, padding, maximum, truncate in (
        (["Common Lisp", "Fast"], None, True, "longest", None, False),
        (["Common Lisp", "Fast"], ["is [MASK] and fun", "Lisp"], True, "longest", None, False),
        (["Common Lisp", "Fast"], ["is fast and fun", "Lisp"], False, "max_length", 10, False),
        (["Common Lisp is fast and fun", "Lisp"], ["is fast and fun", "Common"], True, "max_length", 7, True),
        (["Common Lisp is fast and fun", "Lisp"], ["is fast and fun", "Common"], False, "longest", 5, True),
        (["Common Lisp", ""], ["", "Lisp"], True, "longest", None, False),
        (["Lisp café", "Lisp"], None, True, "longest", None, False),
        (["[PAD] Lisp", "Lisp"], None, True, "longest", None, False),
        (["Common Lisp", "Fast"], None, False, "max_length", 1, True),
    ):
        encoded = tokenizer(texts, text_pair=pairs, add_special_tokens=special,
                            padding=padding, max_length=maximum, truncation=truncate,
                            return_token_type_ids=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits.tolist()
        tokenizer_cases.append({
            "texts": texts, "text_pairs": pairs, "special": special, "padding": padding,
            "max_length": maximum, "truncation": truncate,
            **{name: value.tolist() for name, value in encoded.items()}, "logits": logits,
        })
        if training and tokenizer.mask_token_id in encoded["input_ids"]:
            labels = torch.full_like(encoded["input_ids"], -100)
            labels[encoded["input_ids"] == tokenizer.mask_token_id] = tokenizer.convert_tokens_to_ids("fast")
            with torch.no_grad():
                loss = model(**encoded, labels=labels).loss.item()
            tokenizer_cases[-1].update({"labels": labels.tolist(), "loss": loss})
    (root / "batch-tokenizer-reference.json").write_text(json.dumps(tokenizer_cases))


def fixture(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(67)
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )
    model = BertForMaskedLM(config).float().eval()
    model.set_attn_implementation("eager")
    model.save_pretrained(root)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "common", "lisp", "is", "fast", "and", "fun"]
    tokens.extend(f"token{i}" for i in range(len(tokens), 32))
    tokenizer = BertTokenizer(
        vocab={token: index for index, token in enumerate(tokens)},
        do_lower_case=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(root)
    (root / "vocab.txt").write_text("\n".join(tokens) + "\n")
    reference(root)
    weights = load_file(str(root / "model.safetensors"))
    invalid = root.parent / "bert-invalid-pretraining"
    invalid.mkdir(exist_ok=True)
    shutil.copyfile(root / "config.json", invalid / "config.json")
    save_file(
        {**weights, "bert.pooler.dense.bias": torch.zeros(config.hidden_size)},
        str(invalid / "model.safetensors"),
    )
    duplicate = root.parent / "bert-duplicate-layernorm"
    duplicate.mkdir(exist_ok=True)
    shutil.copyfile(root / "config.json", duplicate / "config.json")
    save_file(
        {
            **weights,
            "bert.embeddings.LayerNorm.gamma": weights[
                "bert.embeddings.LayerNorm.weight"
            ].clone(),
        },
        str(duplicate / "model.safetensors"),
    )


def verify(source, destination, updated=False, token_types=False):
    source = Path(source)
    update_folder = "token-type-updated" if token_types else "updated"
    expected, actual = load(source / update_folder if updated else source), load(destination)
    assert expected.config.model_type == actual.config.model_type == "bert"
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
        torch.testing.assert_close(
            expected(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES).logits,
            actual(IDS, attention_mask=MASK, token_type_ids=TOKEN_TYPES).logits,
            atol=3e-5, rtol=3e-4,
        )
    a = AutoTokenizer.from_pretrained(source, local_files_only=True)
    b = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    assert a.encode("Common Lisp is fast and fun") == b.encode("Common Lisp is fast and fun")
    pair_options = {"text_pair": ["is fast and fun", "Lisp"], "padding": "max_length",
                    "max_length": 7, "truncation": True, "return_token_type_ids": True}
    assert a(["Common Lisp", "Fast"], **pair_options) == b(["Common Lisp", "Fast"], **pair_options)
    print("Fresh BERT masked-LM reload matches:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fixture", "reference", "verify"])
    parser.add_argument("path")
    parser.add_argument("exported", nargs="?")
    parser.add_argument("--updated", action="store_true")
    parser.add_argument("--token-types", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "fixture":
        fixture(arguments.path)
    elif arguments.action == "reference":
        reference(arguments.path, training=False)
    else:
        verify(arguments.path, arguments.exported, arguments.updated, arguments.token_types)
