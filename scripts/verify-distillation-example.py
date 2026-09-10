"""Independently verify the native distillation example's exported adapter and metrics."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / ".build/distillation-example"
report = json.loads((output / "report.json").read_text())
assert report["shuffle"] is True
assert report["train_dataset_id"] == "smollm2-demo-train-v1"
assert report["held_dataset_id"] == "smollm2-demo-held-v1"
for name, count, dataset_id in (
    ("train-targets", 4, report["train_dataset_id"]),
    ("held-targets", 2, report["held_dataset_id"]),
):
    manifest = json.loads((output / name / "distillation-dataset.json").read_text())
    assert manifest["dataset_id"] == dataset_id
    assert manifest["batch_count"] == count
state = json.loads((output / "dataset-state/dataset-state.json").read_text())
assert state["dataset_id"] == report["train_dataset_id"]
assert state["epoch"] == 4 and state["position"] == 4
torch.set_num_threads(1)
teacher = AutoModelForCausalLM.from_pretrained(
    report["teacher"], local_files_only=True, dtype=torch.float32
).eval()
base = AutoModelForCausalLM.from_pretrained(
    report["student"], trust_remote_code=True, local_files_only=True
).eval()
tokenizer = AutoTokenizer.from_pretrained(report["teacher"], local_files_only=True)


def ids(text):
    return torch.tensor([tokenizer.encode(text, add_special_tokens=False)[:24]])


@torch.no_grad()
def disagreement(model, texts):
    losses = []
    temperature = report["temperature"]
    for text in texts:
        tokens = ids(text)
        a = F.log_softmax(model(tokens).logits[0, :-1] / temperature, dim=-1)
        b = F.log_softmax(teacher(tokens).logits[0, :-1] / temperature, dim=-1)
        losses.append(
            (
                F.kl_div(a, b, log_target=True, reduction="batchmean") * temperature**2
            ).item()
        )
    return sum(losses) / len(losses)


for split in ("train", "held"):
    before = disagreement(base, report[f"{split}_texts"])
    torch.testing.assert_close(
        torch.tensor(before),
        torch.tensor(report[f"{split}_before"]),
        atol=1e-3,
        rtol=3e-4,
    )
adapted = PeftModel.from_pretrained(base, output / "adapter").eval()
for split in ("train", "held"):
    after = disagreement(adapted, report[f"{split}_texts"])
    torch.testing.assert_close(
        torch.tensor(after),
        torch.tensor(report[f"{split}_after"]),
        atol=1e-3,
        rtol=3e-4,
    )
with torch.no_grad():
    logits = adapted(ids(report["held_texts"][0])).logits
    torch.testing.assert_close(
        logits,
        load_file(str(output / "native-logits.safetensors"))["logits"],
        atol=1e-3,
        rtol=3e-4,
    )
print(
    "Python PEFT confirms the exported adapter, held-text logits and all four disagreement metrics."
)
