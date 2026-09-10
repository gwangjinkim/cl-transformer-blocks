# Native performance baselines

Milestone 5a provides a repeatable comparison between compiled C and Common Lisp calling the same MLX matrix-multiplication implementation. On the tested M3 Max, median Lisp times were within about 10% of C for these cases. These measurements establish that this tensor operation can retain native speed through the binding; they do not establish full-model throughput or superiority over another inference engine.

## Recorded results

Apple M3 Max, macOS 15.5, SBCL 2.6.7, Apple Clang 17, Release CMake build, MLX 0.32.2; recorded 8 September 2026. Times are microseconds per synchronized square FP32 matrix multiplication, taking the median of nine timed groups. A ratio above 1 means Lisp took longer.

| Device | Matrix dimension | C (µs) | Lisp (µs) | Lisp/C |
|---|---:|---:|---:|---:|
| CPU | 16 | 13.27 | 13.51 | 1.018 |
| CPU | 128 | 15.66 | 16.27 | 1.039 |
| CPU | 512 | 134.47 | 136.60 | 1.016 |
| CPU | 1024 | 803.30 | 745.10 | 0.928 |
| Metal | 16 | 177.93 | 178.78 | 1.005 |
| Metal | 128 | 172.50 | 185.28 | 1.074 |
| Metal | 512 | 232.05 | 249.60 | 1.076 |
| Metal | 1024 | 440.10 | 481.30 | 1.094 |

Raw group means, environment details, and source SHA-256 hashes are committed in [CPU results](benchmarks/m3-max-cpu.json) and [Metal results](benchmarks/m3-max-metal.json). The revision field identifies the integration parent used during development; `tree_dirty: true` records that the benchmark changes were not committed when measured. Source hashes identify the measured implementation.

## Reproduce

After the locked bootstrap described in the README:

```sh
uv run --no-sync python scripts/benchmark.py --device cpu --repeats 9
uv run --no-sync python scripts/benchmark.py --device gpu --repeats 9
```

Reports go to `.build/benchmark-cpu.json` and `.build/benchmark-gpu.json`; `--output` selects another path. Run sequentially with other substantial workloads stopped. A requested unavailable GPU fails. The same commands can collect a future CUDA baseline on an actual qualified NVIDIA machine.

Both frontends allocate one input matrix containing `1/N`, warm up ten multiplies, then create, evaluate/synchronize, and release each result. Groups contain 400, 200, 40, and 20 iterations for dimensions 16, 128, 512, and 1024 respectively. The C process and clean SBCL process execute separately, with order alternating across sizes. Input construction, process startup, library loading, warmup, JSON encoding, and host readback are outside timing. Output allocation, dispatch, evaluation, and disposal are inside timing. Lisp timing uses SBCL's internal real-time clock; C uses `CLOCK_MONOTONIC`.

Both frontends check every final output entry against the analytical result `1/N` within 1e-6 absolute error. The driver also compares their readbacks. The existing native ABI and model parity suites supply broader numerical and ownership checks; this constant-input workload is not a replacement for them.

## Interpretation and next measurements

These are medians of group means, not per-operation latency percentiles or confidence intervals. Metal group means varied substantially: for dimension 1024, C ranged from 413 to 729 µs and Lisp from 408 to 905 µs. Scheduling, power/thermal state, allocation and synchronization can affect the result. A ratio below 1 is not evidence that the Lisp wrapper improves the native kernel. No timing threshold is used as a correctness gate.

Synchronization after each operation emphasizes dispatch and completion latency, especially on tiny GPU matrices. Real model graphs can schedule and fuse several operations before synchronization, so these numbers must not be extrapolated to token generation or a claim that CPU is preferable for a whole model.

The operation-level result motivated the full-model prefill/decode/training baseline below. Rust uses the same native ABI but has not been benchmarked as a separate frontend.

## Full-model baseline

Milestone 5d compares the public Common Lisp/MLX model path with PyTorch/Transformers on the same deterministic standard Llama checkpoint. The generated fixture has 8,392,960 parameters, eight layers, 256 hidden features, grouped-query attention, an 8,192-token vocabulary, and a 256-token context. Its seed, configuration, model artifacts, generator, benchmark programs, and hashes are recorded in each report.

The workload uses FP32, batch one, 128-token prefill, 32 cached decode tokens, and a 64-token shifted next-token training batch. Each frontend runs in a fresh process. Two warmups precede five measured samples. Model loading and fingerprint validation are outside timing. Every GPU operation is synchronized. Decode reports latency per generated token; prefill and training report latency per complete operation. PyTorch training computes the same 63 shifted targets as Lisp. The Lisp training path also performs its production finite-value checks on the loss and every gradient.

Apple M3 Max, macOS 15.5, SBCL 2.6.7, MLX 0.32.2, Torch 2.14.0, and Transformers 5.16.1; recorded 9 September 2026:

| Device | Workload | Lisp/MLX latency | PyTorch latency | Lisp/Python | Lisp throughput | PyTorch throughput |
|---|---|---:|---:|---:|---:|---:|
| CPU | Prefill | 8.508 ms | 10.410 ms | 0.817 | 15,045 tok/s | 12,296 tok/s |
| CPU | Cached decode | 2.878 ms/token | 3.119 ms/token | 0.923 | 347 tok/s | 321 tok/s |
| CPU | Training forward/backward | 30.911 ms | 21.759 ms | 1.421 | 2,038 tok/s | 2,895 tok/s |
| Metal/MPS | Prefill | 4.373 ms | 4.944 ms | 0.885 | 29,271 tok/s | 25,890 tok/s |
| Metal/MPS | Cached decode | 2.465 ms/token | 7.747 ms/token | 0.318 | 406 tok/s | 129 tok/s |
| Metal/MPS | Training forward/backward | 28.820 ms | 13.434 ms | 2.145 | 2,186 tok/s | 4,690 tok/s |

The matching workload fingerprints passed with a 0.2% relative/absolute guard. The native reports reset and record MLX allocator peaks after warmup: roughly 67 MB for CPU inference, 81 MB for CPU training, 80 MB for Metal inference, and 118 MB for Metal training. PyTorch records process peak RSS on both devices, CUDA allocator peak when available, and current/driver MPS allocation because Torch exposes no resettable MPS peak counter in the locked release. These memory types are evidence within each runtime and are not directly comparable.

Raw reports are [CPU](benchmarks/m3-max-model-cpu.json) and [Metal](benchmarks/m3-max-model-metal.json). Reproduce them with the two README commands. Override `--source` to benchmark another supported local checkpoint and adjust batch/token counts within its configured context. The default fixture is generated automatically under `.build/`; downloads and model loading remain outside timing.

These results identify the next optimization targets. Current inference is already near PyTorch for prefill and faster in this small cached-decode workload, while native training is materially slower. The decode result may reflect Python/MPS dispatch and cache overhead for a small model and does not establish a broad engine ranking. Profile the training graph and finite-gradient checks before adding compilation or fusion. Confirm cache behavior on a representative real checkpoint and long context. Rust still reaches the same C ABI but has no separate full-model frontend measurement. Mixed precision, quantization, compiled execution, packed projections, and fixed-size or paged KV storage remain planned.

## Block-growing KV cache

Milestone 5e replaces per-call concatenation of the complete K/V history with the block-growing policy used by [MLX-LM's `KVCache`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py). Each layer reserves FP32 K/V positions in blocks of 256 by default, writes new keys and values through MLX `slice_update`, and passes only the committed prefix to attention. `make-cache :growth-step` can tune the positive block size; growth is capped at the model context length. `cache-length` and `cache-capacity` expose committed and reserved positions separately.

The update is transactional at the Lisp ownership boundary: the previous cache tensors remain owned until the new model graph evaluates successfully, after which every layer entry, length, capacity, and batch binding commit together. A failure leaves the old cache usable. This is block reservation rather than mutable device memory; MLX arrays remain functional values.

The same five-sample 8.39M-parameter benchmark recorded CPU cached decode at 2.970 ms/token versus PyTorch's 3.203 ms/token (ratio 0.927), and Apple Metal at 2.479 versus MPS's 8.101 ms/token (ratio 0.306). These short 128-token prefill plus 32-token decode runs mainly verify that reservation did not cause a material default-path regression; they do not quantify long-context scaling. The reports include hashes of the native bridge and Lisp cache/model sources in addition to the benchmark drivers: [CPU](benchmarks/m3-max-model-block-cache-cpu.json) and [Metal](benchmarks/m3-max-model-block-cache-metal.json). Fixed-size, rotating, paged, and quantized caches remain future specializations.
