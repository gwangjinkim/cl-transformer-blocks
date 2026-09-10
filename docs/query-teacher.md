# A Lisp application teaches a small model its command language

This runnable experiment loads **SmolLM2-135M**, builds labeled application queries in Common Lisp, trains a native LoRA adapter, and exports both a PEFT adapter and a merged Transformers model. The learned model can be reloaded into Lisp or ordinary Python. Model computation, training, tokenization, and export use the native path; Python orchestrates reproduction, optionally downloads the checkpoint, and independently verifies the result.

The first measured result is useful but limited: **21/36 exact commands after training, compared with 0/36 before training**. A simple keyword baseline scores **35/36**. This demonstrates working training and interchange, not a reason to replace a simple rule-based application with a model. The failures remain in the published [run report](results/query-teacher-metal.json).

## Run it

From the repository root, after installing the build tools described in the README:

```sh
uv sync --frozen
uv run --no-sync python scripts/bootstrap.py

# First download: pinned public weights, about 270 MB.
# A full run produces over 600 MB of local model, checkpoint, and evidence files.
uv run --no-sync python scripts/run-query-teacher.py \
  --device gpu --download --output .build/my-query-teacher

# Once the checkpoint is cached, omit --download. CPU is also available.
# Smoke mode runs one update and two requests; it makes no quality claim.
uv run --no-sync python scripts/run-query-teacher.py \
  --device cpu --smoke --output .build/my-query-smoke
```

Use a new output directory for each run; the runner refuses to overwrite an existing experiment. To repeat only Python verification:

```sh
uv run --no-sync python scripts/run-query-teacher.py \
  --verify-only --output .build/my-query-teacher
```

Try the merged model directly from Lisp:

```sh
sbcl --noinform --no-sysinit --no-userinit \
  --script examples/query-teacher/ask.lisp \
  .build/my-query-teacher/merged \
  'Show open tokenizer issues from the last week.' gpu
```

The checked local run printed:

```text
Model:  (issues :open :tokenizer :week)
Matching example issue IDs: (1)
```

This command executes against the six example issue records in `domain.lisp`. It does not connect to an issue tracker. Replace those records with application data deliberately. Output validation establishes that a command is permitted, not that it expresses the user's intended query correctly.

The manually dispatched [query-teacher workflow](../.github/workflows/query-teacher.yml) offers hosted Linux CPU or an existing trusted Metal runner. It defaults to smoke mode. No workflow has been run remotely for this milestone, and Linux/CUDA qualification remains separate.

## What Lisp contributes

| File | Responsibility |
|---|---|
| [domain.lisp](../examples/query-teacher/domain.lisp) | Query structures, canonical rendering, strict command validation, application execution, deterministic training curriculum, and separately authored evaluation requests. Runs without MLX or Python. |
| [training.lisp](../examples/query-teacher/training.lisp) | Native tokenization, prompt-masked batches, deterministic shuffle, LoRA/AdamW training, greedy evaluation, and export. |
| [run.lisp](../examples/query-teacher/run.lisp) | Clean SBCL entry point driven by the runner's environment variables. |
| [ask.lisp](../examples/query-teacher/ask.lisp) | Reload the merged model, translate a request, validate the command, and execute it on sample records. |
| [run-query-teacher.py](../scripts/run-query-teacher.py) | Reproduction metadata, optional pinned download, independent application semantics, Transformers/PEFT generation and numerical verification, and reports. No Python training. |

The teacher starts with valid query structures and renders both their commands and templated natural-language requests. Correct answers are known by construction; Lisp also executes the commands to record their expected issue IDs. This is supervised fine-tuning, not reinforcement learning or a model autonomously inventing its training data.

The application language is deliberately bounded:

```text
(issues STATUS COMPONENT PERIOD)
STATUS    = :open | :closed | :any
COMPONENT = :tokenizer | :training | :storage | :any
PERIOD    = :week | :month | :any
```

There are 36 valid commands. The validator matches canonical strings against those commands and never invokes `READ` or `EVAL` on model output. A week means integer age < 7 days and a month means age < 30 days; the dataset does not depend on wall-clock time. This is not an arbitrary Lisp program generator or a general natural-language query engine.

## Experiment design

- **Checkpoint:** `HuggingFaceTB/SmolLM2-135M` at `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, using the pretrained base model, not an instruction-tuned model. BF16 source values are executed and exported as FP32.
- **Training:** 30 command combinations × 6 templates = 180 examples. Four epochs, batch size 4, 180 optimizer updates. Prompt tokens and right padding are excluded from the loss; answer tokens and EOS are supervised. Prompt and answer are tokenized separately with the answer's leading space preserved.
- **Adapter:** rank 8, alpha 16, initialization seed 17; q/v/o and gate/up/down projections; zero dropout. Native AdamW uses learning rate 0.001, weight decay 0.01, and global gradient clipping at 1.0. The frozen base remains unchanged until an explicit merge after evaluation.
- **Validation:** a seventh template applied to the 30 seen combinations. No checkpoint selection or hyperparameter adjustment was performed using these results.
- **Test:** 36 separately authored requests, one per command. Six combinations (canonical indices divisible by seven) are absent from training and validation. All individual field values still occur in training.
- **Decoding:** identical instruction prompt before/after, greedy generation, at most 24 new tokens, EOS token 0. No few-shot examples, constrained decoding, or output repair. The zero-shot base baseline should not be described as the best result prompting could achieve.
- **Scoring:** exact canonical command, syntactic validity, and execution agreement on the sample database are separate metrics. Different commands can happen to return the same records; execution agreement alone overstates semantic accuracy.

The data is small and synthetic, and the test sentences were authored within this project. This is not an external benchmark. Any subsequent data revisions informed by these failures must treat these test sentences as development data and introduce a new untouched evaluation set before claiming improved generalization.

## Observed result

One local Apple Metal run on 2026-09-10, with pinned dependencies and native FP32 computation:

| Evaluation | Pretrained model | Lisp-trained adapter | Keyword baseline |
|---|---:|---:|---:|
| Validation: unseen template, seen combinations | 0/30 | 30/30 | 30/30 |
| Test: all separately authored requests | 0/36 | **21/36 (58.3%)** | **35/36 (97.2%)** |
| Test: seen combinations | 0/30 | 17/30 | 29/30 |
| Test: held-out combinations | 0/6 | **4/6** | 6/6 |

The trained model produced 33/36 syntactically valid test commands and 26/36 matching database results. The exact-command score, 21/36, is the primary metric.

Training plus its final training-checkpoint save took **96.56 seconds**; the complete native phase took about **116.9 seconds**, excluding Python verification. First and last minibatch losses were 2.74936 and 0.00021334, respectively; these are different minibatches, not a held-out loss comparison. The full run's MLX counters reported roughly **1.69 GB peak active allocation**, with 0.55 GB active and 5.12 GB cached at measurement time. Cached allocations are separate from the active peak. These counters are not whole-process RSS or a universal memory requirement.

The main failure pattern is instructive. Validation preserves the training template's field order and wording, while the test varies them. Some requests containing “unresolved” produce the invalid `:unresolved` status. Many requests for both statuses produce the valid but wrong `:open`. Perfect template validation and low training loss did not establish reliable paraphrase understanding.

Independent CPU Python verification reproduced **all 66 pretrained and all 66 adapted generation sequences exactly**, including the failures. It also compared complete teacher-forced logits on two probes for both the PEFT adapter and merged standard model, using `atol=5e-4, rtol=3e-4`. Maximum observed absolute error was 0.00052834; the test uses the combined absolute-plus-relative tolerance. CPU and Metal one-update smoke runs passed, and the merged checkpoint reloaded into native Lisp for the CLI example above. The full curriculum was run on Metal; full CPU training and CUDA were not measured.

## Artifacts and next milestones

Each output directory contains the full dataset, before/after predictions and token IDs, losses, timing, environment and source/checkpoint hashes, numerical probes, a resumable native checkpoint, standard adapter files, a merged model, and a compact `report.json`. The committed report retains all 66 before/after prediction strings and artifact hashes; large weights remain untracked. Seeds and source hashes aid reproduction but do not promise bit-identical training across devices or library versions.

The local model's `revision.txt` declares its pinned source revision. The runner also records hashes of the actual files; the marker alone is not a cryptographic proof that a locally edited checkpoint equals the Hub snapshot. Use a fresh download directory when qualifying another environment.

Python can use the resulting artifacts without this Lisp package:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(".build/my-query-teacher/merged")
tokenizer = AutoTokenizer.from_pretrained(".build/my-query-teacher/merged")

# Alternatively, distribute the small adapter alongside a pinned base reference.
base = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
)
adapted = PeftModel.from_pretrained(base, ".build/my-query-teacher/adapter")
```

The next useful milestones are richer supervised wording with a new independent test set, then a real application dataset where rules are insufficient. A stronger prompt-only baseline should accompany that experiment. A public component-construction API remains a separate architectural priority. None of these requires changing the Rust tokenizer or moving model computation into Python.

An honest article opening is: **“My Lisp application taught a small Transformer its command language—and Python reproduced what it learned, including its mistakes.”**
