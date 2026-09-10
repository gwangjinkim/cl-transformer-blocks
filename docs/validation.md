# Validation evidence and reproducibility

## Version 0.38: top-k distillation targets

The [top-k distillation suite](distillation-top-k.md) passes on CPU and actual Apple Metal for the tiny full-model fixture and the pinned real SmolLM2/LoRA path. Lisp/MLX and Python independently create batch format v3/FP32 and v4/FP16 targets containing explicit log probabilities, int32 vocabulary indices, an exact aggregate tail log probability and a vocabulary sentinel. Native load validates the representation and uses the same full-model/LoRA optimizer, clipping, checkpoint and export paths as dense distillation. Materialized, streaming and Python-created content-addressed datasets also accept the representation without loading targets at dataset open.

Fresh PyTorch checks each stored top probability by gathering the artifact's indices from the full teacher distribution, recomputes the excluded-class tail with stable `logsumexp`, independently implements the coarsened `k + 1` KL, and compares every trainable gradient and three clipped momentum-SGD updates. It does not assume a common ordering between MLX `argpartition` and PyTorch `topk`. Tests also cover exact FP16 resave, fixed-temperature rejection, crossed format/dtype pairs, duplicate or out-of-range indices, false manifest vocabulary, non-finite/positive probabilities, atomic replacement and native handle cleanup.

The tiny target gates pass 32,282 native checks per device and the expanded dataset gates pass 36,472. Native top-k probabilities differ from the PyTorch teacher by at most `2.38e-7`; all sparse-objective gradients differ by at most `5.22e-8`, and exported updates by at most `3.73e-9`. The tiny five-position top-4 payload is 184 bytes in FP32 or 134 bytes in FP16, compared with 640 bytes for dense FP32.

The real target gates pass 519,591 native checks per device and the expanded dataset gates pass 1,364,332. For five selected positions and SmolLM2's 49,152-token vocabulary, dense FP32 uses 983,040 bytes; top-64 uses 2,584 FP32 or 1,934 FP16 bytes, about 380 or 508 times smaller. Native FP32 top/tail differences are at most `2.72e-5` on CPU and `8.92e-5` on Metal. Across Lisp/Python FP32/FP16 sparse targets, gradient differences are at most `3.92e-5` on CPU and `1.39e-4` on Metal; exported updates differ by at most `1.82e-8` and `6.03e-8`. The three-batch real dataset falls from 1,572,864 dense FP32 tensor bytes to 3,100 top-64 FP16 bytes.

These measurements qualify artifact mathematics and interchange. Coarsened KL is intentionally a lower bound on full-vocabulary KL and no model-quality equivalence is claimed. Choosing `k`, temperature and precision requires an application evaluation. CUDA executes the same runner in the manual NVIDIA workflow and still awaits external hardware evidence.

## Version 0.37: content-addressed distillation datasets

The [dataset-integrity suite](distillation-integrity.md) passes on CPU and actual Apple Metal for the tiny full-model fixture and pinned real SmolLM2/LoRA path. Lisp materialized/streaming producers and the Python one-resident-teacher producer emit dataset manifest v2. Python `hashlib` independently reproduces every exact nested-file digest and the documented ASCII ordered dataset identity. Identical native materialized and streaming contents have identical digests for both FP32 and FP16 storage.

Native opening reads and verifies all files while allocating no target tensors. Each later indexed or training access rechecks the chosen files before loading. Tests flip a safetensor byte and append valid JSON whitespace after a dataset object was opened; both initial open and lazy access reject the change, and the cursor stays unconsumed. Malformed file and outer digests are rejected. Iterator state v2 restores only against its exact dataset content, including when another dataset has the same ID, vocabulary, count and iteration settings. Explicit legacy dataset/state v1 loading remains functional and reports no content identity.

The tiny CPU/Metal native gates pass 25,653/25,654 checks, and real CPU/Metal gates each pass 1,190,596 checks. The existing numerical results remain unchanged: tiny shuffled updates differ from independent PyTorch by at most `1.86e-9`; real updates differ by at most `3.73e-9`. Real native FP32 target errors are `2.07e-4` on CPU and `5.68e-4` on Metal; real FP16 maxima are `0.007835` and `0.007950`. The new checks also retain standard model/PEFT reload, exact model/optimizer/iterator continuation, lazy handle cleanup, and 1,572,864→786,432 byte FP16 payload reduction.

Standard empty, `abc`, and multi-block SHA-256 vectors pass in the dependency-free Common Lisp implementation. The first cross-language run correctly rejected a canonical-framing mismatch caused by a Common Lisp string containing `\n` rather than LF; explicit character output fixed it. CUDA executes the same runner in the manual NVIDIA workflow and still awaits external hardware evidence.

## Version 0.36: FP16 distillation target storage

The [FP16 target-storage suite](distillation-storage.md) passes on CPU and actual Apple Metal for the tiny full-model fixture and real SmolLM2/LoRA path. Lisp batch, materialized-dataset and streaming producers and the Python batch/dataset producers emit strict version-two FP16 `teacher_logits` safetensors. Native load verifies the raw dtype and restores FP32 tensors before loss and optimization. Version-one FP32 remains the default; a loaded FP16 batch reports `:float16` and an ordinary resave reproduces its stored tensor exactly.

The tiny five-position target has 160 logits. Its logical payload falls from 640 to 320 bytes. The maximum FP16 target error is `5.743e-5` on both CPU and Metal; gradients computed independently in PyTorch differ by at most `5.96e-8`; three clipped momentum-SGD updates differ by at most `1.86e-9`. The three-batch dataset falls from 1,024 to 512 payload bytes and preserves shuffled order `[1, 0, 2]`.

The real five-position target has 245,760 logits. Its logical payload falls from 983,040 to 491,520 bytes. Across CPU and Metal, the maximum FP16 stored-logit error is `0.007927`, the maximum native/PyTorch gradient error using the same stored target is `1.245e-4`, and three exported updates differ by at most `3.73e-9`. The three-batch dataset falls from 1,572,864 to 786,432 payload bytes; its largest FP16 target error is `0.007969`. Tiny target/dataset native gates pass 17,924/25,636 checks per device. Real target/dataset gates pass 288,650/1,190,578 checks per device, including standard model or PEFT reload and exact optimizer/iterator continuation.

The tests reject invalid storage keywords, FP16 tensors labeled version-one/FP32, FP32 tensors labeled version-two/FP16, crossed version/dtype metadata, shape/selection errors and publication failures. FP16 conversions are checked for finite values before publication. Native handles return to baseline after temporary casts and lazy loads. The existing Linux CUDA workflow automatically runs the same FP16 paths; hardware evidence remains pending externally.

## Version 0.35: memory-bounded native dataset production

The [streaming-producer suite](distillation-streaming.md) passes on CPU and actual Apple Metal. A custom CLOS source yields three variable-shape `distillation-example` snapshots to one resident native teacher. After each synchronous callback, the live native handle count is exactly the teacher-only baseline: the full FP32 target has been saved and disposed before the source can yield another example. The teacher model version remains unchanged.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny streamed target logits vs PyTorch teacher | 2.98e-8 | 4.47e-8 |
| Tiny streamed target logits vs materialized native dataset | 0 | 0 |
| Real streamed target logits vs PyTorch SmolLM2 | 2.07e-4 | 5.68e-4 |
| Real streamed target logits vs materialized native dataset | 0 | 0 |

Caller mutation of every source ID, label and mask after example construction cannot change the artifacts. A source failure after two completed callbacks, an empty source and an invalid yielded value all leave the previous complete destination unchanged. The stream is not resumed internally; an application retry enumerates the source again. Fresh PyTorch checks the third streamed dataset beside the materialized Lisp and Python datasets. Tiny native gates pass 14,624 scalar/behavior checks per device; real SmolLM2/LoRA gates pass 624,168, including exact comparison of every streamed full-vocabulary logit. Existing shuffled training, optimizer/iterator continuation and exported full-model/PEFT checks retain their version 0.34 numerical results.

The updated real Metal example uses the list-source method to stream four training and two held batches, then releases its teacher and lazily trains exactly as before. Fresh Python verification preserves train KL 10.634911→8.204338, held KL 10.862764→10.454003, adapter parameters, held-text logits and the iterator boundary. Full logits still occupy about 14 MB for these six short batches; the improvement is bounded production memory, not artifact compression.

## Version 0.34: lazy distillation datasets and exact data continuation

The [dataset suite](distillation-datasets.md) passes on CPU and actual Apple Metal. Native and Python producers independently publish three variable-shape target batches under the same strict dataset contract. The Python producer loads its teacher once. Native opening validates every manifest and selected-position count while retaining zero target tensors; physical access and each training step load and dispose one owned target backend.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny native-dataset target logits vs PyTorch teacher | 2.98e-8 | 4.47e-8 |
| Tiny Python-dataset target logits vs PyTorch teacher | 0 | 0 |
| Tiny shuffled updated parameters vs independent PyTorch | 1.86e-9 | 9.31e-10 |
| Real native-dataset target logits vs PyTorch SmolLM2 | 2.07e-4 | 5.68e-4 |
| Real Python-dataset target logits vs PyTorch SmolLM2 | 0 | 0 |
| Real shuffled LoRA parameters vs independent PyTorch | 3.73e-9 | 3.73e-9 |

Lisp and Python produce physical order `[1, 0, 2]` for three batches at seed 41, epoch 3. The native suite saves a full-model or LoRA model/optimizer checkpoint and dataset cursor after the first update, finishes the epoch, then reloads both artifacts and reproduces every remaining loss and final trainable parameter exactly. Failed updates leave the cursor unchanged. Dataset and state publication failures preserve the previous complete directories. Invalid versions, IDs, counts, paths, selected-position metadata, epochs, cursors and mismatched shuffle settings are rejected without changing iterator state. Final handle counts equal their starting values.

Tiny native gates pass 14,340 scalar/behavior checks per device; real SmolLM2/LoRA gates pass 230,924. Fresh Python validates both producers, repeats the shuffled clipped momentum-SGD sequence, and reloads the standard model or PEFT adapter. Reports are `.build/distillation-datasets-DEVICE/validation.json` and `real/validation.json`. Existing reusable-target and parent conversion/distillation gates run first in the reproducible workflow.

The updated real Metal example creates separate four-batch train and two-batch held datasets, releases the teacher and all in-memory targets, performs 20 lazy shuffled AdamW updates, and saves paired optimizer and iterator state. Mean teacher disagreement changes from 10.634911 to 8.204338 on the training texts and from 10.862764 to 10.454003 on held-out texts. Fresh Python PEFT verifies both datasets, the state boundary, adapter, logits and all four metrics. Generation remains poor; this is an interchange and resumability demonstration rather than evidence of useful compression. CUDA remains pending the external hardware workflow.

## Version 0.33: reusable Python/native distillation targets

The [reusable-target suite](distillation-targets.md) passes on CPU and Apple Metal. Native-created batches produce exactly the same loss and every student gradient as the live native teacher, then remain usable after the teacher is disposed and caller-owned input/label/mask arrays are mutated. Atomic native save and fresh independent-backend reload preserve the result. A Python `AutoModelForCausalLM` utility creates the same target format; native Lisp consumes it for three clipped momentum-SGD updates, and fresh Python independently checks all gradients and exported parameters.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny native-created target logits vs Python teacher | 2.98e-8 | 4.47e-8 |
| Tiny Python-created target logits vs Python teacher | 0 | 0 |
| Tiny cached native gradients vs PyTorch | 3.73e-8 | 4.47e-8 |
| Tiny unsupported-online `tb_parallel` target gradients vs PyTorch | 5.96e-8 | 5.96e-8 |
| Real native-created target logits vs Python SmolLM2 | 1.59e-4 | 5.68e-4 |
| Real Python-created target logits vs Python SmolLM2 | 0 | 0 |
| Real cached native LoRA gradients vs PyTorch | 2.72e-5 | 4.29e-5 |

The tiny artifact has five selected positions and vocabulary 32, requiring 640 tensor bytes. The real artifact has five positions and SmolLM2 vocabulary 49,152, requiring 983,040 bytes. Tiny gates pass 10,746 native scalar/behavior checks per device after final cross-architecture, strict-rejection and capability assertions; real gates pass 173,182. Malformed versions/dtypes, matrices, IDs, labels, masks, empty selection, vocabulary/logit shapes and unexpected target objects are rejected. This includes an actual FP16 safetensor paired with a false FP32 manifest. Injected publication failure preserves the old complete directory, and all targets, independent backends, students, adapters, optimizers and gradients return to the initial native handle count.

The tiny cross-architecture case uses a Python `tb_parallel` teacher that online native distillation deliberately rejects; its artifact nevertheless trains the native composed student and all updates agree with Python. The real case uses the pinned 30-layer SmolLM2 teacher, 15-layer composed student and all 60 trainable query/value LoRA tensors. The Python creator executes the teacher on CPU and writes FP32 logits; that artifact loads and trains on CPU or Metal. Native-created target differences use the established real tolerance `atol=1e-3, rtol=3e-4`; tiny tolerance remains `atol=2e-6, rtol=3e-4`. Reports are `.build/distillation-targets-DEVICE/validation.json` and `real/validation.json`.

At version 0.33 the real example saved six individual target batches (about 14 MB total), disposed the teacher, and obtained the same 20-step Metal train/held-out measurements and standard PEFT adapter as version 0.32. Version 0.34 supersedes its manual target loop with the dataset workflow documented above. Full logits reduce repeated teacher compute and live model memory at the cost of storage proportional to selected positions times vocabulary size. CUDA remains pending the external hardware workflow.

## Version 0.32: native teacher/student distillation

The [distillation suite](distillation.md) passes on CPU and Apple Metal. Eight tiny tied/untied cases cover pure KL at temperature 0.7, a KL/CE mixture at temperature 2, hard-only CE, and nonzero imported query/value LoRA factors. Independent PyTorch selects the same next-token positions using ignored labels and padding, computes `T² * KL(teacher || student)` plus the weighted supervised term, and verifies every student gradient and three clipped momentum-SGD updates. Hard weight one agrees exactly with the ordinary native supervised loss and gradients.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny student gradients vs PyTorch | 1.04e-7 | 1.04e-7 |
| Tiny student parameters after three updates | 3.73e-9 | 3.73e-9 |
| Tiny exported student logits, native vs Python | 5.96e-8 | 8.94e-8 |
| Real student adapter gradients vs PyTorch | 4.08e-5 | 4.65e-5 |
| Real adapter parameters after three updates | 1.86e-9 | 3.73e-9 |
| Real exported student logits, native vs Python PEFT | 9.01e-5 | 2.29e-4 |

The real gate uses the pinned original 30-layer SmolLM2-135M Llama teacher and the 15-layer composed student from layer reuse. All 60 query/value adapter gradient tensors are checked. Tiny tolerances remain `atol=2e-6, rtol=3e-4`; real tolerances remain `atol=1e-3, rtol=3e-4`. Python loads the original checkpoint with explicit `dtype=torch.float32`, matching the native FP32 policy rather than Transformers' checkpoint-selected BF16 default.

Native optimizer continuation is exact. Teacher versions, parameter bindings and logits remain unchanged; adapter training retains the student base bindings. Invalid options and empty supervision fail, stale student caches are rejected, and teacher/student/gradient handles return to baseline. Ordinary Python loads each exported full model or adapter, performs another distillation update, saves it, and fresh native Lisp reproduces the returned logits. Native tiny gates pass 43,246 scalar/behavior checks per device plus 2,569 after Python continuation. Real gates pass 549,662 checks per device plus 491,522 after Python continuation. Reports are `.build/distillation-DEVICE/validation.json` and `real-validation.json`.

At version 0.32 the real Metal example performed 20 sequential native AdamW LoRA updates. Its mean per-example temperature-scaled KL decreased from 10.634911 to 8.176231 on four training texts and from 10.862764 to 10.382759 on two held-out texts. Version 0.34 uses the lazy shuffled dataset experiment documented above. The example's generated continuation remains poor; these small-data results do not establish useful compression or recovered language quality. CUDA remains pending external hardware qualification.

The shared supervised-loss callback retains passing native training/PEFT, composed LoRA/rsLoRA and 7,204-check core suites on CPU and Metal. GPT-2, Qwen2 and BERT masked-LM CPU interchange also passes. Native C compilation, forced clean Lisp compilation without native initialization, frozen offline dependency audit, Python lint/syntax and workflow/documentation checks pass at version 0.32.0.

## Version 0.31: selected and repeated pretrained blocks

The [layer-reuse suite](layer-reuse.md) passes on CPU and Apple Metal. Eight tiny cases cover identity, removal, reversal and repetition of heterogeneous sequential/parallel blocks, with tied and untied vocabulary heads. Independent Python references copy and rearrange original source modules without reading the native-produced target configuration or weights. Complete selected weights are exact, and repeated layers become unequal after native SGD, verifying independent training. Every full-model gradient, masked loss, SGD update, native AdamW checkpoint continuation, cached greedy output and native-trained LoRA export passes. Python trains and resaves each edited model, then fresh Lisp reload reproduces its logits/loss and differentiates all parameters.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny native logits vs independently rearranged Python | 5.96e-8 | 5.96e-8 |
| Tiny native full-model gradients | 2.98e-7 | 2.98e-7 |
| Tiny native-trained adapter logits vs Python PEFT | 5.96e-8 | 5.96e-8 |
| Real selected-stack native logits vs rearranged Python | 1.19e-4 | 3.71e-4 |
| Real native-trained adapter logits vs Python PEFT | 6.48e-5 | 9.27e-5 |

The real case selects layers `0, 2, …, 28` from the pinned converted SmolLM2-135M model: 15 layers and all 137 canonical parameter tensors transported exactly. Tokenizer assets, native/Python greedy tokens, and one masked/clipped native AdamW query/value LoRA update pass, including ordinary PEFT reload. Exported Python logits equal the independently rearranged reference exactly in all tiny and real cases. Tolerances remain `atol=2e-6, rtol=3e-4` for tiny fixtures and `atol=1e-3, rtol=3e-4` for the real model.

Native selection gates pass 48,867 tiny and 786,439 real scalar/behavior checks per device, plus 2,072 tiny Python-training/native-reload checks. Invalid/circular selections, mutated configurations, missing unselected weights, active adapters, unknown subclasses and source overwrites are rejected. Source logits/configuration remain unchanged; injected publication failure preserves the previous output; native handle counts return to baseline. Provenance survives ordinary native resave, including when tokenizer assets are attached in memory.

The real runner also reruns the full pretrained conversion gate. Existing component/LoRA/optimizer and tiny conversion suites remain green on CPU/Metal, along with the 7,204-check core suite per device and fresh ordinary Transformers reloads. Clean forced Lisp compilation, pinned Python lint, frozen offline dependency audit and workflow/document checks pass. Reports are `.build/recomposition-DEVICE/validation.json` and `real-validation.json`.

The runnable [real layer-removal example](../examples/recompose-pretrained.lisp) executes on Metal. Its reduced model repeats punctuation on the demonstrated prompt; numerical interchange is not evidence of retained language quality. No compression-quality or speed improvement is claimed. The workflows include the new gates; Linux/CUDA execution remains unqualified locally.

## Version 0.30: pretrained component conversion

The [conversion suite](pretrained-components.md) passes on CPU and Apple Metal for sharded Transformers Llama fixtures with tied/untied heads and MHA/GQA/MQA attention. Converted exports preserve every canonical native FP32 weight exactly. All gradients, native SGD before/after conversion, Python SGD/resave/native reload, tokenizer assets, cached greedy tokens and native-trained LoRA exports match independent Python references. Native conversion tests pass 3,176 scalar/behavior checks per device, with 1,032 further checks after Python training. Source configuration/logits/generation assets stay unchanged; invalid classes/layouts/weights and active adapters are rejected. Injected publication failure preserves the preceding output, and all native tensor handles are released.

| Maximum absolute difference | CPU | Metal |
|---|---:|---:|
| Tiny converted native logits vs original Python Llama | 5.96e-8 | 5.96e-8 |
| Tiny converted native gradients vs original Python Llama | 2.38e-7 | 2.38e-7 |
| Tiny trained adapter logits vs Python PEFT | 5.96e-8 | 5.96e-8 |
| Real converted native logits vs original Python SmolLM2 | 1.67e-4 | 3.18e-4 |
| Real trained adapter logits vs Python PEFT | 3.35e-4 | 5.25e-4 |

The real gate uses `HuggingFaceTB/SmolLM2-135M` revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`: 30 blocks, 272 canonical tensors, tied embeddings, exact FP32 weight transport, tokenization and cached greedy generation, followed by one native AdamW update of query/value LoRA factors and fresh PEFT verification. Native tests pass 1,179,661 scalar/behavior checks per device. Python's converted model and original Llama have identical real-fixture logits; tiny Python differences peak at 1.49e-8. Tiny tolerances are `atol=2e-6, rtol=3e-4`; the established real native tolerance is `atol=1e-3, rtol=3e-4`. This qualifies the tested FP32 path, not quantized execution or unrestricted generation options. No model-quality improvement is claimed from the single training update.

The [real-model example](../examples/pretrained-components.lisp) runs native generation through the converted blocks. Reports are `.build/composition-import-DEVICE/validation.json` and `real-validation.json`. New ordinary CI covers tiny conversion; the Metal and manual external CUDA workflows include the real checkpoint. CUDA/Linux execution remains unqualified.

## Version 0.29: composed-model LoRA/rsLoRA

The [composed adapter runner](../scripts/run-component-lora-tests.py) executes the independent component oracle first, followed by ordinary PEFT 0.20.0 fixtures and fresh Lisp/Python processes. CPU and Apple Metal pass standard LoRA and rsLoRA over all seven projection types in a two-layer mixed sequential/parallel model, for both tied and untied vocabulary heads. Each variant checks all 28 A/B gradients against nonzero Python factors, three clipped/decayed momentum-SGD and AdamW updates, bit-exact native checkpoint continuation, frozen base weights, exact unchanged adapter transport, every merged weight, and merged logits. Native-created adapters pass identity initialization, stale-cache rejection, trained heterogeneous cache chunks, fresh Python logits and exact greedy tokens. Python continues training them, and fresh native Lisp reload reproduces the saved logits/loss and computes all adapter gradients.

| Maximum absolute difference across the four variants | CPU | Metal |
|---|---:|---:|
| Imported adapter logits | 5.96e-8 | 7.45e-8 |
| Adapter gradients | 1.68e-8 | 2.33e-8 |
| Three-step SGD parameters | 7.45e-9 | 7.45e-9 |
| Three-step AdamW parameters | 5.37e-7 | 1.67e-6 |
| Merged weights | 7.45e-9 | 7.45e-9 |
| Merged logits | 5.96e-8 | 5.96e-8 |

Tolerance is `atol=2e-6, rtol=3e-4`; native resume, frozen base parameters, initial adapter identity, and unchanged adapter transport are exact. The main native adapter gate has 66,859 scalar/behavior checks per device and returns to its initial live tensor-handle count. The additional Python-continuation reload gate has 1,292 checks per device. Reports are written to `.build/components-created-DEVICE/lora-validation.json`, alongside the base component report. The runnable [Lisp example](../examples/composed-lora.lisp) constructs/trains a tiny base, trains rsLoRA, and exports adapter, native optimizer checkpoint, and merged model. This is bounded FP32, zero-dropout qualification; CUDA/Linux and arbitrary numerical components remain unqualified.

## Version 0.28: public component construction

The [component suite](components.md) passes on local CPU and Apple Metal. It checks a Lisp-created mixed sequential/parallel decoder with per-layer GQA dimensions against a separate functional Torch implementation, including all logits, all full/tied gradients, SGD updates, canonical weight coverage, heterogeneous cache chunks, causal isolation, sharded export, fresh Python AutoClass loading, Python training/resave, native reload, and exact AdamW continuation. CPU/Metal maxima are 5.96e-8/4.47e-8 for logits and 1.19e-7/1.19e-7 for full gradients, under `atol=2e-6, rtol=3e-4`. The custom numerical vocabulary is versioned; arbitrary Lisp functions and composed-model LoRA are outside this qualification.

Existing core tests retain 7,204 passing checks per device. Original portable-model/tokenizer gates pass CPU/Metal, and the cached real SmolLM2 CPU gate passes 2,064,803 checks with fresh Python export verification. Linux/CUDA runs remain unverified by this project.

The first deliverable was developed with failing API, native ABI, model-parity, gradient, tensor-layout, and configuration-contract tests before their implementations and fixes.

## Acceptance suite

`scripts/run-tests.py` generates its tiny fixture independently in Python, runs native ABI tests and a clean SBCL suite, and launches fresh Transformers processes to load native exports. Python is never called by the Lisp runtime.

- Matrix arithmetic, noncontiguous host copies, retained handles and idempotent disposal.
- Full batched logits against eager Transformers.
- Prefix causality, single-token and chunked cached logits.
- Right-padding isolation, left-padding/fully-masked rejection.
- Exact tokenizer IDs and decoding; greedy token sequence and EOS termination.
- Invalid IDs/ranks, unsupported architecture/configuration, stale caches after training.
- Sharded loading, missing/extra/misshaped checkpoint rejection, native export reload, and separate output embeddings.
- Every canonical tiny-fixture gradient against PyTorch autograd; mean shifted-token loss.
- Exact unchanged exported parameter values and shared embedding identity in Python; numerical equality after the native SGD update.
- Real SmolLM2 checkpoint loading, tokenization, cache/generation and Python export reload.

The model revision is `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The model's BF16 weights are promoted to FP32 for execution/export. Tests compare both runtimes at that precision.

## Numerical criteria

| Comparison | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Tiny model logits and cached logits | 2e-5 | 2e-4 |
| All tiny parameter gradients | 2e-6 | 2e-4 |
| Mean loss | 2e-5 | — |
| Python parameters after SGD | 2e-6 | 2e-5 |
| Unchanged exported weights after FP32 conversion | 0 | 0 |
| SmolLM2 native logits and cache | 1e-3 | 2e-4 |

Observed tiny-logit maximum absolute error was **5.96e-8** on both CPU and Metal. Observed full SmolLM2 maximum error against Torch FP32 was **1.67e-4 on CPU** and **3.18e-4 on Metal** for the fixed batched input.

The full-model tolerance was calibrated rather than applied to hide a failed mapping. Against independent eager Torch FP64, maximum deviations were 3.87e-4 for Torch FP32, 4.21e-4 for MLX CPU and 5.35e-4 for MLX Metal. All fixed-input greedy choices matched. These differences are consistent with accumulated FP32 reduction error across 30 layers. Reproduce this diagnostic with `scripts/precision-audit.py` after both real-model acceptance runs. Tiny-model criteria remain strict.

## Environment and limits

Local validation: arm64 macOS 15.5, SBCL 2.6.7, Apple Clang 17, MLX 0.32.2, MLX C commit `c74db5307cc8ce122f48d97ef951b30578674e7f`, Transformers 5.16.1, Torch 2.14.0, Safetensors 0.8.0, Tokenizers 0.23.2. Python and Rust dependency graphs, source checksums, and toolchain choices are committed. Successful harness runs also write `.build/validation-*.json` with the actual environment.

A locked source/dependency graph supports reproducible functional tests; different host compilers, SDKs, devices and floating-point kernels can still produce different binaries/numerical reductions. The project does not claim bit-identical cross-platform builds.

All four acceptance workflows (tiny CPU, tiny Metal, real CPU, real Metal) passed locally, including Python reload. Tiny suites each made 6,865 scalar/behavior checks; real suites each made 1,671,204. These counts include elementwise numeric comparisons, not distinct test cases. CPU and Metal were executed locally. Linux CPU CI and CUDA workflows are provided but have not been executed here. Neither a Mac container nor a mock validates CUDA kernels. Use the manual NVIDIA workflow on an existing real runner; it invokes `nvidia-smi` and requires successful GPU inference and differentiation. Hardware rental/provisioning is outside these workflows.

Milestone 1 did not certify training beyond its full-parameter SGD foundation; subsequent training evidence is recorded below. End-to-end C/Rust speed equivalence, production throughput, long-context memory bounds and mixed-precision accuracy still require separate benchmarks and qualification.

## Milestone 2: PEFT and training

CPU and Metal passed the independent PEFT 0.20.0 acceptance suite for standard LoRA and rsLoRA. Fixtures use nonzero A/B matrices; every adapter gradient matches within 2e-6 absolute / 2e-4 relative tolerance. Three native momentum-SGD and AdamW updates (with decay and global clipping) match Python exports within 2e-6 / 2e-4. Full-model masked AdamW matches within 3e-6 / 3e-4. Imported, merged, and native-created/trained adapter exports reload in fresh Python and reproduce logits.

Repeated-step tests warmed up four updates and ran twenty more: active MLX allocation stayed at 23,504 bytes on CPU and Metal, live handle counts remained constant, and disposal returned them to baseline. Base weights remained bit-identical during adapter-only training. Invalid targets, stale optimizers, unsupported adapter options, and base/revision mismatches are rejection gates. Run scripts/run-training-tests.py as described in training.md.

## Milestone 4: GPT-2

Five tiny fixtures (standard, untied, alternate scaling, exact GELU, legacy buffers) passed CPU/Metal logits, token-wise cache, tokenizer, greedy generation, every parameter gradient, SGD-update export, and fresh Transformers reload. The alternate-scaling fixture is sharded; a deliberately swapped shard index is rejected. A modified legacy causal mask is also rejected.

The pinned real DistilGPT-2 checkpoint passed CPU and Metal inference/cache/tokenizer/generation and exact learned-weight export/reload. Maximum fixed-input logit errors were 6.48e-5 on CPU and 7.63e-5 on Metal. Tiny errors stayed below 4.5e-8. Tiny logits/gradients use 3e-5/3e-6 absolute and 3e-4 relative tolerances; real logits use 1e-3 absolute / 3e-4 relative. Full-model comparisons are FP32. Real GPT-2 training with its nonzero dropout config is intentionally rejected.

Run scripts/run-gpt2-tests.py for each device, with and without --real. Details and the model revision are in gpt2.md. Linux/CUDA qualification remains pending actual execution on those platforms.

## Milestone 5a: native benchmark

The compiled C and clean SBCL benchmark programs passed analytical matrix readback on CPU and Metal at dimensions 16, 128, 512 and 1024. Each recorded case includes ten warmup operations and nine timed groups. The native ABI suite passed after introducing the shared benchmark declarations, and the C benchmark passed strict compiler warning checks. [performance.md](performance.md) records the method, measurements, variability, source hashes and limitations. Llama and PEFT CPU/Metal regression suites also passed after the GPT-2 milestone.

## Milestone 4b: explicit Python compatibility worker

A seeded tiny OPT model supplies the independent acceptance fixture because OPT is deliberately absent from the native architecture registry. Native loading rejects it. The explicit Python worker passed full masked logits, Transformers tokenizer encode/decode, deterministic generation, bounded-response rejection, process persistence, graceful shutdown, standard checkpoint export, exact parameter comparison, and fresh Python reload on CPU and Apple MPS. Maximum logit error against the independent CPU fixture was 0 on CPU and 2.98e-8 on MPS. The worker reported `cpu` and `mps` respectively as its actual execution devices; there was no GPU fallback. The fixture uses a different process for reference/export verification. CUDA runs the same gate in the manual NVIDIA workflow but remains unexecuted.

The expanded worker gate dynamically selects `AutoModelForCausalLM`, forwards rank-three FP32 `inputs_embeds`, and performs three stateful clipped AdamW steps under Lisp control. Trained logits differed from the independent three-step CPU Python reference by at most 8.94e-8 on CPU and 2.43e-6 on MPS. New CPU/MPS workers reloaded their contributed checkpoints and reproduced the corresponding pre-export logits exactly; standalone Python processes checked the trained parameters and logits against the independent reference. The same expanded gate is required by the pending CUDA workflow.

## Milestone 4d: processors, resumable checkpoints, and Hub publication

The worker acceptance gate now runs AutoProcessor text preparation, preserves int64 processor outputs, directly forwards and trains from processed batches, saves a two-step AdamW/Torch RNG checkpoint, and resumes the third update in a new worker. A mismatched optimizer is rejected before state installation. The no-network Python boundary test loads a local image through the same file descriptor protocol and mocks `HfApi` to verify repository creation, publication options, and the exact staged model/tokenizer/processor/model-card files. The complete gate passed on CPU and real Apple MPS; the existing trained-reference errors remained 8.94e-8 and 2.43e-6 respectively, and resumed updates matched uninterrupted execution within the same strict device tolerances.

The MLX bridge now reports native tensor dtype through the C ABI rather than assuming FP32. FP32 and int32 regressions passed as part of the 6,867-check tiny Llama suites on CPU and Metal. The full native PEFT/optimizer and five-variant GPT-2 suites also passed on both devices after the protocol extensions.

## Milestone 4e: generation and training controls

The worker gate matches independent CPU Transformers beam sequences and sequence scores, and seeded top-k sampling matches the CPU reference. Repeating the same seeded request produces the same tokens on CPU and MPS; cross-device sampled tokens are intentionally not required because Torch RNG kernels differ. Generation accepts arbitrary named tensors or AutoProcessor inputs and JSON-compatible `generate` options.

Training now accumulates an average gradient over named microbatches, advances a Transformers learning-rate scheduler once per optimizer update, and exposes the current step/rate. The checkpoint test uses BF16 autocast on CPU and FP16 autocast on MPS, saves after two scheduled AdamW updates, and confirms that a fresh worker restoring optimizer, scheduler, scaler, and RNG state reproduces the next uninterrupted loss and logits within strict device tolerances. The complete worker gate passed on CPU and real Apple MPS.

## Milestone 4f: worker PEFT contribution

The independent OPT fixture now includes a deterministic adapter written by PEFT 0.20.0. The worker loads it and matches its reference logits, reports the frozen/trainable parameter split, exports canonical safe adapter files, stages those files for Hub publication without network access, merges them, and exports a standalone model. A second path creates rsLoRA from Lisp, performs clipped AdamW updates, saves and exactly resumes adapter optimizer state, exports the adapter, and merges it. The complete gate passed on CPU and actual Apple MPS. Fresh standalone Python processes load each exported adapter with `PeftModel` and each merged model with `AutoModelForCausalLM`; their logits agree within 2e-6 absolute / 2e-5 relative tolerance.

## Milestone 4g: native Hub artifacts

The native `FROM-PRETRAINED` path now resolves a nonlocal repository ID through pinned `huggingface_hub`, forwards revision/offline/cache policy, records the resolved snapshot revision, and downloads only safe model and tokenizer/configuration patterns. Native `PUSH-TO-HUB` stages an ordinary model, tokenizer assets, and model card, then uses inherited credentials in a one-shot helper. Mocked tests prove the exact download/upload boundary without contacting the network or passing a token. A real subprocess check proves JSON reaches the helper as stdin and translates an expected offline cache miss. The tiny native CPU and actual Metal suites each passed 6,875 checks plus independent Python export reload.

## Milestone 4h: chat-template compatibility

The independent OPT tokenizer fixture now saves a Jinja chat template and Transformers generates reference rendered text and token IDs. Lisp constructs structured role/content messages, passes optional message fields and template options, and receives stable text or flat IDs even when Transformers returns a `BatchEncoding`. The complete worker gate passed on CPU and actual Apple MPS with exact rendering/tokenization equality.

## Milestone 5b: binary worker tensors

`PYTHON-FORWARD :TRANSPORT :BINARY` now carries arbitrary-rank FP32/int64 named inputs and FP32/int64/boolean selected outputs as raw little-endian files under a worker-owned temporary directory. Path containment, dtype, shape, and exact byte count are checked before reading; input and output files are removed after each request. The acceptance fixture transfers two tensors and one logits result, confirms the worker counters, and requires exact equality with the JSON route. The complete worker gate and malformed binary descriptor tests passed on CPU and actual Apple MPS.

The reproducible `scripts/benchmark-python-transport.lisp` benchmark used 8x64x16 FP32 embeddings, an 8x64 int64 mask, and 16,384 output logits after two warmups and nine timed calls. Median end-to-end CPU time fell from 27.472 ms for JSON to 2.511 ms for binary (0.091 ratio); actual Apple MPS fell from 29.441 ms to 4.155 ms (0.141 ratio). These local tiny-OPT transfer measurements include model execution and file I/O; they do not measure shared memory or large-model generation throughput.

## Milestone 4i: native Qwen2

A seeded Transformers 5.16.1 Qwen2 fixture with deliberately nonzero Q/K/V biases passed full masked logits, token-wise cached logits, greedy generation, Qwen2 BPE tokenization, every parameter gradient, one full-model SGD update, native save/reload, and fresh standalone Transformers reload on CPU and actual Apple Metal. A PEFT 0.20.0 Qwen2 LoRA with nonzero factors also passed native import, exact forward, export, merge, and fresh PEFT/Transformers reload. Maximum fixed-input base-model logit errors were 2.98e-8 on CPU and 3.73e-8 on Metal. Logits use 3e-5 absolute / 3e-4 relative tolerances; gradients use 4e-6 / 4e-4.

The configuration gate also covers the inactive `sliding_window`, direct `rope_theta`, and `use_mrope=false` fields found in the official Qwen2.5-0.5B configuration. Sliding attention, MRoPE, scaled RoPE, quantized native weights, and nonzero attention dropout remain explicit rejections.

The pinned 988 MB `Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987` checkpoint passed all 290 canonical native parameters, complete batched and token-cached logits, greedy generation, tokenizer behavior, native save/reload, FP32 canonical export, and fresh `AutoModelForCausalLM` reload. Maximum FP32 logit errors were 2.54e-4 on CPU and 7.96e-4 on actual Apple Metal, under 1e-3 absolute / 3e-4 relative criteria. The independently generated tiny fixture retains the stricter full-gradient, SGD, and PEFT gates. CUDA execution remains unqualified.

## Milestone 4j: native BERT masked LM

A seeded Transformers 5.16.1 `BertForMaskedLM` fixture passed right-padded bidirectional logits, BERT WordPiece tokenization, native save/reload, exact checkpoint transport, and fresh standalone `AutoModelForMaskedLM` reload on CPU and actual Apple Metal. Maximum logit errors were 5.22e-8 on CPU and 6.71e-8 on Metal, under 3e-5 absolute / 3e-4 relative criteria.

The initial native BERT contract supplied zero token-type IDs; milestone 4o below adds explicit segment inputs. BERT requires absolute positions, a tied decoder, and the masked-LM architecture. Masked-token loss uses explicit same-position labels, passes all 42 tiny-fixture parameter gradients against PyTorch, and exports a full-model SGD update that reloads in Python on CPU and actual Apple Metal. Standard PEFT LoRA and rsLoRA on query/value projections pass imported logits, all eight adapter gradients per variant, native SGD updates, native adapter creation, unchanged exports, and merged-model exports through fresh Python PEFT/Transformers reloads on both devices. Missing labels, no supervised positions, nonzero-dropout training, incompatible adapter tasks/targets, and custom automatic mappings are rejection tests. Decoder/cross-attention modes, relative positions, and KV caching remain unsupported. The explicit Python worker supports those broader AutoModel tasks.

The pinned 440 MB `google-bert/bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` safetensors checkpoint passed all 202 masked-LM parameter mappings, WordPiece tokenization, complete fixed-input logits, native save/reload, canonical parameter export, and fresh `AutoModelForMaskedLM` reload. Maximum FP32 logit errors were 5.53e-5 on CPU and 1.72e-4 on actual Apple Metal, under 5e-4 absolute / 3e-4 relative criteria. Its inactive historical `gradient_checkpointing` field is accepted; active checkpointing is rejected. The loader converts only `.LayerNorm.gamma`/`.beta` aliases and discards only a complete, correctly shaped four-tensor pooler/next-sentence group. Partial groups and simultaneous legacy/canonical LayerNorm names are rejection tests. CUDA execution has not been qualified.

## Milestone 5d: full-model performance baseline

A deterministic standard Hugging Face Llama checkpoint with 8,392,960 parameters passed matching Lisp/MLX and PyTorch fingerprints for FP32 prefill, cached one-token decode, and shifted-token forward/backward training on CPU and actual Apple Metal/MPS. The benchmark uses fresh processes, synchronized work, two warmups, five measured samples, fixed batches and context lengths, source/checkpoint hashes, and memory evidence. No timing threshold is a correctness gate.

On the M3 Max, native/PyTorch median latency ratios were 0.817 prefill, 0.923 decode, and 1.421 training on CPU; Metal/MPS ratios were 0.885, 0.318, and 2.145. These measurements show that training is the clearest current optimization target and that the small-model decode result needs confirmation on a representative real checkpoint. Raw measurements and the exact method are in [performance.md](performance.md). CUDA remains unmeasured.

## Milestone 2f: native training checkpoints

The native suite saves after two updates and compares the third uninterrupted update with a fresh model/optimizer restoration. Full-model AdamW restores two moment tensors for every trainable parameter; PEFT adapter momentum SGD restores one velocity tensor for every adapter parameter. CPU and actual Apple Metal continuation matched loss within 2e-6 and every resulting parameter within 2e-7 absolute / 2e-6 relative tolerance. Mismatched optimizer settings are rejected before state ownership changes.

The full-model checkpoint remains loadable by a fresh `AutoModelForCausalLM`, and the adapter checkpoint remains loadable by a fresh PEFT `PeftModel`. The JSON manifest and safe optimizer tensors are native training state rather than a proposed cross-framework optimizer format. CUDA remains pending actual hardware.

## Milestone 5e: block-growing KV cache

Tests request a three-token growth block and verify the observable `0 -> 3 -> 3 -> 6` capacity sequence while committed length advances `0 -> 2 -> 3 -> 5`. Every Llama layer retains the reserved shape, but attention receives only the committed prefix. Token-wise and chunked cached logits remain equal to full forward results, invalid growth steps are rejected, stale caches remain rejected, and all cache tensors are disposed with their owner.

The complete 6,912-check native Llama suite passed on CPU and actual Apple Metal. All five GPT-2 configuration fixtures and Qwen2's biased GQA fixture passed cached parity on both devices, including their independent Transformers export and PEFT checks. The reproducible full-model benchmark passed matching fingerprints on CPU/Metal and records the latency evidence in [performance.md](performance.md). CUDA remains unqualified.

## Milestone 1h: sharded Hugging Face export

Native `save-pretrained` accepts a positive `max-shard-size` byte count, groups sorted canonical parameters deterministically without splitting tensors, writes standard numbered safetensors shards, and emits `model.safetensors.index.json` with complete `weight_map` coverage and exact total storage bytes. A limit smaller than one tensor places that whole tensor in its own shard. Old native model weight artifacts are removed before a replacement export so single and sharded layouts cannot be mixed.

The tiny native checkpoint is forced into multiple 2 KiB shards, reloaded natively for logit parity, and loaded by a fresh ordinary `AutoModelForCausalLM` process with exact weights, tied embeddings, tokenizer behavior, and logits. A stale numbered shard is removed before replacement. The explicit AutoClass worker forwards the same option to Transformers and its 17-shard OPT export passes a separate fresh-process reload. The 7,195-check suite passes on CPU and Metal; CUDA remains unqualified.

## Lisp-defined portable architecture export

Milestone 5g extends this gate with a Python-produced WordLevel tokenizer. Tests cover native encoding/decoding and text generation, added-token overflow rejection, missing/malformed tokenizer rejection without losing the previous attachment, byte-exact export after source-file mutation, native/Python tokenizer reload, and exact next-loss continuation from a Lisp-created model's AdamW checkpoint. Tokenizer assets also appear in dry-run Hub staging. These checks are part of `scripts/run-portable-tests.py` in all existing device workflows.

Milestone 5f adds the versioned `tb_parallel` causal LM. The fixture is independently constructed in Torch with grouped-query attention, non-interleaved RoPE, one RMSNorm per block, parallel attention/MLP residual branches, and 19 canonical untied parameters. Native CPU logits differ by at most `2.99e-8`; native Metal logits differ by at most `4.48e-8`. Shifted loss, all gradients, cached decoding, greedy generation, ordinary and 2 KiB-sharded safetensors exports are checked against the independent reference.

Both exports include a custom `PretrainedConfig`, `PreTrainedModel`, and exact `auto_map`. Fresh ordinary `AutoConfig` and `AutoModelForCausalLM` processes load them with explicit `trust_remote_code=True`, reproduce weights/logits, reject left padding, run uncached generation, and backpropagate a finite loss. The resident Lisp Python worker rejects the same directory without custom-code trust and matches native logits and generation when trust is enabled.

A separate acceptance route starts from `TB:MAKE-PORTABLE-CAUSAL-LM`, checks repeatable seed initialization and tied parameter ownership, publishes a dry-run Hub directory containing both Python sources, and loads it in fresh Transformers. Python then performs an SGD update, resaves the standard model and its source files, and native Lisp loads that Python contribution back. The final native error is at most `5.97e-8` on CPU and `2.99e-8` on Metal. No network publication occurs in this gate.

## Milestone 1i: transactional model publication

Native and explicit Python-worker `save-pretrained` implementations now serialize every weight shard, index, configuration, tokenizer, and processor asset into a unique sibling staging directory. A completed stage is published through a single native filesystem call: ordinary rename when the target is absent, `renamex_np(..., RENAME_SWAP)` on macOS, or `renameat2(..., RENAME_EXCHANGE)` on Linux when replacing an existing directory. The exchanged old directory is then removed.

Acceptance tests seed the published path with a sentinel checkpoint, inject an error after the complete replacement has been staged, and verify that the sentinel remains readable while none of the staged configuration is exposed. A subsequent successful native or worker save replaces the sentinel, exposes matching standard weights and configuration, and leaves no staging directory. Independent Transformers reload remains part of both complete suites. The guarantee concerns atomic path visibility; no power-loss durability claim is made. Adapter and resumable training-checkpoint directories remain follow-up work.

## Milestone 2g: transactional adapters and training checkpoints

Native `save-adapter` and worker `python-save-adapter` now use the same staged publication primitive as full models. Native full-model/adapter checkpoints write the standard Transformers/PEFT artifacts, optimizer safetensors, and JSON manifest into one private stage. The worker similarly writes model or adapter artifacts, optimizer/scheduler/scaler/RNG state, and its manifest before publishing. Internal directory writers avoid an early nested model publication.

Failure injection runs only when the completed state manifest exists in the stage. This distinguishes complete-checkpoint publication from an earlier nested model-only commit. Native full-model and adapter checkpoints and worker full-model checkpoints preserve seeded predecessor directories under that failure, then pass deterministic continuation and fresh ordinary Transformers/PEFT reload after successful replacement. Direct native and worker adapter exports pass the same predecessor-preservation gate.

## Milestone 4o: BERT sentence-pair token-type inputs

The BERT gate now supplies distinct segment layouts in a padded two-row batch. It compares explicit segment logits and all 42 full-model parameter gradients with independent Transformers fixtures. Native LoRA/rsLoRA fixtures likewise use nonzero segment IDs for all eight adapter gradients, SGD updates, and fresh PEFT/merged-model reloads. Native and Python-worker generic inference/training agree, and a native AdamW checkpoint resumes the exact next loss after an intervening invalid batch is rejected.

Omitted and explicitly zero segment IDs produce bit-identical logits. Single-row results match the batched reference. Wrong ranks/shapes, negative IDs, IDs equal to `type_vocab_size`, and noninteger IDs are rejected; native decoders reject supplied segment IDs. Tiny BERT maximum segment-logit errors were `4.85e-8` on CPU and `5.59e-8` on Metal. The pinned real BERT checkpoint passed offline CPU/Metal execution and fresh Transformers export reload, with maximum segment-logit errors of `5.63e-5` and `2.11e-4`, respectively, within the existing `atol=5e-4, rtol=3e-4` real-model gate.

The shared native regression suite passed 7,204 checks on each device. GPT-2, portable-model interchange, and the full Python worker regression suites passed on CPU. Clean ASDF compilation, Python byte-compilation, Ruff, and frozen dependency synchronization passed for version 0.26.0. CUDA remains unqualified.

## Milestone 4p: native batched and paired tokenization

Nine independent Transformers cases compare native input IDs, attention masks, and segment IDs exactly: single texts, pairs, optional special tokens, fixed/longest right padding, longest-first truncation, empty pair members, Unicode, literal padding tokens, and a one-token limit with special tokens disabled. The resulting arrays feed native BERT inference and masked-token training. Invalid options, missing padding IDs when needed, zero-length encodings, insufficient special-token space, and malformed text batches are rejected. Repeated calls verify that truncation leaves the resident tokenizer unchanged; fresh Python exports retain identical pair-tokenizer behavior.

Tiny and cached real BERT gates passed on CPU and actual Metal/MPS. Maximum text-batch logit errors were `7.64e-8` / `6.71e-8` for tiny CPU/Metal and `9.45e-5` / `3.44e-4` for real CPU/Metal, under unchanged existing tolerances. Lisp-created portable models also pass attached-tokenizer batching and ordinary Python round trips on both devices. The shared native regression suite passed 7,204 checks per device.

An offline debug Rust ABI test exercises short truncation and failed-request recovery with overflow checking enabled. This gate is now part of Linux CPU, Metal, and manual CUDA workflows. The release tokenizer library rebuilt from the frozen Cargo lock; clean ASDF compilation, Python byte-compilation, Ruff, frozen dependency synchronization, workflow YAML parsing, and whitespace checks passed for version 0.27.0. No CUDA hardware qualification was performed.
