"""Shared portable target and target-dataset creation for validation CLIs."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_sha256(vocab_size, entries):
    framing = (
        "cl-transformer-blocks-distillation-dataset-v2\n"
        f"vocab_size={vocab_size}\n"
        f"batch_count={len(entries)}\n"
        + "".join(
            f"{index:06d}\t{entry['selected_positions']}\t"
            f"{entry['manifest_sha256']}\t{entry['weights_sha256']}\n"
            for index, entry in enumerate(entries)
        )
    )
    return hashlib.sha256(framing.encode("ascii")).hexdigest()


def integer_matrix(value, name):
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(row, list) and row for row in value)
    ):
        raise ValueError(f"{name} must be a nonempty matrix")
    if any(len(row) != len(value[0]) for row in value):
        raise ValueError(f"{name} must be rectangular")
    if any(
        not isinstance(item, int) or isinstance(item, bool)
        for row in value
        for item in row
    ):
        raise ValueError(f"{name} must contain integers")
    return torch.tensor(value, dtype=torch.long)


def load_teacher(
    teacher,
    *,
    revision=None,
    local_files_only=False,
    trust_remote_code=False,
    device="cpu",
    dtype="float32",
):
    try:
        dtype_value = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }[dtype]
    except KeyError as error:
        raise ValueError("dtype must be float32, float16, bfloat16, or auto") from error
    return (
        AutoModelForCausalLM.from_pretrained(
            teacher,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            dtype=dtype_value,
        )
        .eval()
        .to(device)
    )


def prepare_target(model, batch, device):
    if not isinstance(batch, dict):
        raise ValueError("distillation batch must be a JSON object")
    ids = integer_matrix(batch.get("input_ids"), "input_ids")
    if ids.shape[1] < 2:
        raise ValueError("distillation requires at least two tokens")
    labels = integer_matrix(batch.get("labels", batch["input_ids"]), "labels")
    mask = (
        integer_matrix(batch["attention_mask"], "attention_mask")
        if "attention_mask" in batch
        else None
    )
    if labels.shape != ids.shape or (mask is not None and mask.shape != ids.shape):
        raise ValueError("input_ids, labels, and attention_mask must have equal shapes")
    if mask is not None and not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask values must be zero or one")
    selected = labels[:, 1:] != -100
    if mask is not None:
        selected &= mask[:, 1:].bool()
    if not torch.any(selected):
        raise ValueError("distillation batch has no selected next-token positions")
    vocab_size = int(model.config.vocab_size)
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError("input ID outside teacher vocabulary")
    valid_labels = labels[labels != -100]
    if torch.any(valid_labels < 0) or torch.any(valid_labels >= vocab_size):
        raise ValueError("label outside teacher vocabulary")
    ids_device = ids.to(device)
    mask_device = mask.to(device) if mask is not None else None
    with torch.no_grad():
        logits = model(
            ids_device[:, :-1],
            attention_mask=mask_device[:, :-1] if mask_device is not None else None,
        ).logits
        logits = logits[selected.to(device)].float().cpu().contiguous()
    if (
        logits.shape != (int(selected.sum()), vocab_size)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("teacher returned incompatible or non-finite logits")
    return ids, labels, mask, logits


def validate_top_k(vocab_size, top_k, temperature):
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 0 < top_k < vocab_size
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "top-k distillation requires 0 < top_k < vocabulary and positive finite temperature"
        )


def write_target(
    directory,
    model,
    teacher,
    revision,
    batch,
    device,
    storage_dtype="float32",
    top_k=None,
    temperature=1.0,
):
    ids, labels, mask, logits = prepare_target(model, batch, device)
    try:
        storage_torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
        }[storage_dtype]
    except KeyError as error:
        raise ValueError("storage_dtype must be float32 or float16") from error
    if top_k is None:
        if temperature != 1.0:
            raise ValueError("temperature applies only when top_k is supplied")
        tensors = {"teacher_logits": logits.to(storage_torch_dtype)}
        format_version = 1 if storage_dtype == "float32" else 2
    else:
        validate_top_k(int(model.config.vocab_size), top_k, temperature)
        teacher_log_probs = F.log_softmax(logits / float(temperature), dim=-1)
        top_log_probs, top_indices = torch.topk(
            teacher_log_probs, top_k, dim=-1, sorted=True
        )
        masked = teacher_log_probs.scatter(
            -1, top_indices, torch.full_like(top_log_probs, -torch.inf)
        )
        tail_log_prob = torch.logsumexp(masked, dim=-1, keepdim=True)
        tensors = {
            "teacher_topk_log_probs": top_log_probs.to(storage_torch_dtype),
            "teacher_topk_indices": top_indices.to(torch.int32),
            "teacher_tail_log_prob": tail_log_prob.to(storage_torch_dtype),
            "teacher_vocab_size": torch.tensor(
                [int(model.config.vocab_size)], dtype=torch.int32
            ),
        }
        format_version = 3 if storage_dtype == "float32" else 4
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("teacher targets overflowed the requested storage dtype")
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "teacher.safetensors")
    manifest = {
        "format": "cl-transformer-blocks-distillation-batch",
        "format_version": format_version,
        "dtype": storage_dtype,
        "vocab_size": int(model.config.vocab_size),
        "teacher_source": str(teacher),
        "teacher_revision": revision,
        "teacher_model_type": model.config.model_type,
        "teacher_execution_dtype": str(model.dtype).removeprefix("torch."),
        "input_ids": ids.tolist(),
        "labels": labels.tolist(),
    }
    if top_k is not None:
        manifest.update(
            {"representation": "top_k", "top_k": top_k, "temperature": temperature}
        )
    if mask is not None:
        manifest["attention_mask"] = mask.tolist()
    (directory / "distillation.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n"
    )
    return int(logits.shape[0])


def publication_stage(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    return output, Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))


def create_batch(
    teacher,
    batch,
    output,
    *,
    revision=None,
    local_files_only=False,
    trust_remote_code=False,
    device="cpu",
    dtype="float32",
    storage_dtype="float32",
    top_k=None,
    temperature=1.0,
):
    model = load_teacher(
        teacher,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
        dtype=dtype,
    )
    output, stage = publication_stage(output)
    try:
        write_target(
            stage,
            model,
            teacher,
            revision,
            batch,
            device,
            storage_dtype,
            top_k,
            temperature,
        )
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def create_dataset(
    teacher,
    dataset,
    dataset_id,
    output,
    *,
    revision=None,
    local_files_only=False,
    trust_remote_code=False,
    device="cpu",
    dtype="float32",
    storage_dtype="float32",
    top_k=None,
    temperature=1.0,
):
    if (
        not isinstance(dataset, dict)
        or not isinstance(dataset.get("batches"), list)
        or not dataset["batches"]
    ):
        raise ValueError("dataset must contain a nonempty batches array")
    if (
        not isinstance(dataset_id, str)
        or not 1 <= len(dataset_id) <= 256
        or not dataset_id.isprintable()
    ):
        raise ValueError(
            "dataset ID must be a nonempty printable string of at most 256 characters"
        )
    model = load_teacher(
        teacher,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
        dtype=dtype,
    )
    output, stage = publication_stage(output)
    try:
        entries = []
        for index, batch in enumerate(dataset["batches"]):
            relative = f"batches/{index:06d}"
            selected = write_target(
                stage / relative,
                model,
                teacher,
                revision,
                batch,
                device,
                storage_dtype,
                top_k,
                temperature,
            )
            batch_directory = stage / relative
            entries.append(
                {
                    "path": relative,
                    "selected_positions": selected,
                    "manifest_sha256": sha256_file(
                        batch_directory / "distillation.json"
                    ),
                    "weights_sha256": sha256_file(
                        batch_directory / "teacher.safetensors"
                    ),
                }
            )
        manifest = {
            "format": "cl-transformer-blocks-distillation-dataset",
            "format_version": 2,
            "dataset_id": dataset_id,
            "vocab_size": int(model.config.vocab_size),
            "batch_count": len(entries),
            "content_sha256": dataset_content_sha256(
                int(model.config.vocab_size), entries
            ),
            "batches": entries,
        }
        (stage / "distillation-dataset.json").write_text(
            json.dumps(manifest, separators=(",", ":")) + "\n"
        )
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output
