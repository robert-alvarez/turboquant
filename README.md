# TurboQuant

From-scratch implementation of **TurboQuant** for KV cache compression in large language models.

> **Paper:** [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, 2025

TurboQuant compresses the KV cache from 16 bits to 2-8 bits per coordinate using a random orthogonal rotation followed by per-coordinate Lloyd-Max scalar quantization. The rotation makes the quantization error **data-oblivious** — a single precomputed codebook works optimally for any input vector, with no calibration data needed.

## Key results on Qwen2.5-3B-Instruct

| Bits | Cosine Sim | Top-1 Accuracy | Compression |
|------|-----------|----------------|-------------|
| 2 | 0.941 | 76% | 7.1x |
| 3 | 0.983 | 90% | 4.9x |
| 4 | 0.995 | 92% | 3.8x |
| 5 | 0.999 | 100% | 3.0x |

With **outlier channel handling** (mixed-precision, from the paper's Section 4.3):

| Config | Eff. Bits | Cosine Sim | Top-1 | Compression |
|--------|-----------|-----------|-------|-------------|
| 2.5-bit (32ch@4b + 96ch@2b) | 2.5 | 0.963 | 82% | 5.3x |
| 3.5-bit (32ch@5b + 96ch@3b) | 3.5 | 0.990 | 92% | 4.0x |

MSE-only consistently outperforms MSE+QJL at every bit budget for KV cache compression (see the [learner's guide](learners-guide.md) for why).

## Visualizations

### Per-channel variance before and after rotation
![Coordinate distribution](output/a_coordinate_distribution.png)

The top row shows the key insight: before rotation, per-channel standard deviations are highly non-uniform (outlier channels visible as spikes). After rotation, all channels have equal variance at the theoretical 1/sqrt(d) level.

### Quality vs bit width (all methods)
![Cosine similarity vs bits](output/b_cosine_vs_bits.png)

### Quality-compression tradeoff
![Compression tradeoff](output/c_compression_ratio.png)

### Inner product bias vs variance (MSE vs QJL)
![Inner product scatter](output/d_inner_product_scatter.png)

Left: MSE quantization is biased but tight. Right: MSE+QJL is unbiased but high variance. Softmax neutralizes the bias but amplifies the variance, which is why MSE wins for KV cache compression.

## Quick start

```bash
# Setup
cd turboquant
source .venv/bin/activate

# Run with Qwen2.5-3B (default)
python run_eval.py

# With Top-k accuracy evaluation
python run_eval.py --eval-top1

# Larger model, longer context, residual window
python run_eval.py --model Qwen/Qwen2.5-7B-Instruct \
                   --min-tokens 1000 \
                   --eval-top1 \
                   --residual-window 128

# Vision Language Model
python run_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct \
                   --image path/to/image.png

# Synthetic data only (no GPU / no model download)
python run_eval.py --skip-model --device cpu
```

## CLI flags

| Flag | Description |
|------|-------------|
| `--skip-model` | Use synthetic data instead of loading a real model |
| `--device {cuda,cpu}` | Device to use (default: cuda) |
| `--model MODEL` | HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct) |
| `--eval-top1` | Enable Top-k token match evaluation |
| `--n-generate N` | Tokens to generate for Top-k eval (default: 50) |
| `--residual-window W` | Keep last W tokens in FP16 (default: 0 = disabled) |
| `--min-tokens N` | Extend prompt to at least N tokens |
| `--image PATH` | Image path/URL for VLM mode (requires Qwen2.5-VL model) |

## Project structure

```
turboquant/                  # Core library (torch, numpy, scipy only)
    __init__.py              # Public API
    core.py                  # TurboQuantMSE, TurboQuantProd, TurboQuantOutlier,
                             #   codebooks, rotation, outlier detection
    bitpack.py               # Bit packing/unpacking
    serialize.py             # .tqkv binary disk format

eval/                        # Evaluation harness (+ transformers, matplotlib)
    gpu_eval.py              # MSE, prod, outlier, residual window, top-k
    disk_eval.py             # Disk round-trip compression tests
    metrics.py               # Quality metrics + teacher-forced generation
    model.py                 # Model loading (text + VLM)
    visualize.py             # Four matplotlib plots
    flashblade_worker.py     # Per-GPU worker for concurrent storage benchmarks

run_eval.py                  # CLI entry point
bench_flashblade.py          # FlashBlade storage benchmark (multi-GPU, GDS)
flashblade-benchmark-results.md  # Benchmark results and analysis
learners-guide.md            # In-depth technical guide (math, metrics, design decisions)
```

The core library is importable independently:

```python
from turboquant import TurboQuantMSE, compute_all_codebooks, generate_rotation_matrix

codebooks = compute_all_codebooks(d=128, bit_widths=[3, 4])
rotation = generate_rotation_matrix(d=128)
quantizer = TurboQuantMSE(d=128, bits=3, codebook=codebooks[3], rotation=rotation)

x_hat = quantizer.quantize_dequantize(my_vectors)
```

## What it implements

1. **Lloyd-Max codebook computation** for the Beta((d-1)/2, (d-1)/2) distribution arising from random rotation on S^{d-1}
2. **TurboQuant_MSE** (Algorithm 1): Haar-random orthogonal rotation + per-coordinate scalar quantization
3. **TurboQuant_prod** (Algorithm 2): MSE at (b-1) bits + 1-bit QJL residual correction for unbiased inner products
4. **Outlier channel handling**: Mixed-precision quantization enabling fractional bit rates (2.5, 3.0, 3.5, 4.0)
5. **Top-k teacher-forced evaluation**: Stable per-position accuracy metric for k = 1, 2, 4, 8, 16
6. **Residual window**: Keep recent W tokens in FP16 (from KIVI) for improved generation quality
7. **Disk serialization** (.tqkv): Binary format for compressed KV cache with full round-trip verification
8. **VLM support**: Quantize KV caches from vision-language models (Qwen2.5-VL)

## Supported models

All models with head_dim=128 work with precomputed codebooks (auto-recomputed otherwise):

| Model | FP16 Size | Layers | KV Heads |
|-------|-----------|--------|----------|
| Qwen/Qwen2.5-3B-Instruct | ~6 GB | 36 | 2 |
| Qwen/Qwen2.5-7B-Instruct | ~15 GB | 28 | 4 |
| Qwen/Qwen2.5-14B-Instruct | ~29 GB | 48 | 8 |
| Qwen/Qwen2.5-VL-3B-Instruct | ~8 GB | 36 | 2 |
| Qwen/Qwen2.5-VL-7B-Instruct | ~17 GB | 28 | 4 |

Tested on NVIDIA GB10 (128 GB unified memory).

## FlashBlade storage benchmark

`bench_flashblade.py` benchmarks KV cache checkpoint/restore on Everpure FlashBlade storage with GPU Direct Storage (GDS). It measures the end-to-end cost of swapping LLM sessions to and from storage — the critical operation for serving many concurrent users on a fixed GPU fleet.

### What it measures

The benchmark runs five tests:

1. **Storage I/O** — Single-GPU compress/write/read/decompress timing at each context length (1K-30K tokens) and bit width. Reports per-phase latency, compression ratio, and cosine similarity.
2. **Concurrent multi-GPU checkpoint/restore** — All GPUs simultaneously checkpoint a session, measuring aggregate throughput and per-worker variance under contention.
3. **Session capacity** — Sessions per terabyte at each context length and compression level.
4. **Session migration** — Compress on GPU 0, write to disk, read back, restore on GPU 1, then verify generation quality with teacher-forced Top-1 accuracy. Proves the compressed session produces identical predictions to the original.
5. **TurboQuant ON vs OFF with GDS** — The key comparison. Tests four configurations (FP16, FP16+GDS, TQ 3-bit, TQ 3-bit+GDS) at multiple context lengths with concurrent workers. Shows how compression + GDS combine.

### Key results (7 GPUs concurrent, Qwen2.5-7B-Instruct)

**Restore latency** (user-facing — the delay when a session is swapped back in):

| Context | FP16 | TQ+GDS | Speedup |
|---------|------|--------|---------|
| 10K tokens (573 MB) | 1,105 ms | 131 ms | **8.4x** |
| 30K tokens (1.72 GB) | 2,294 ms | 344 ms | **6.7x** |
| 50K tokens (2.87 GB) | 3,976 ms | 619 ms | **6.4x** |

**Round-trip** (full checkpoint + restore cycle — determines GPU utilization):

| Context | FP16 | TQ+GDS | Speedup |
|---------|------|--------|---------|
| 10K tokens | 1,961 ms | 1,647 ms | 1.2x |
| 30K tokens | 7,939 ms | 4,266 ms | 1.9x |
| 50K tokens | 18,483 ms | 3,311 ms | **5.6x** |

The advantage grows with context length. At 50K tokens, TQ+GDS is 5.6x faster while using 4.9x less storage. See [flashblade-benchmark-results.md](flashblade-benchmark-results.md) for full results.

### Running

```bash
# Standard benchmark (no GDS, ~7 minutes)
python bench_flashblade.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 0,1,2,3,4,5,6,7 \
  --context-lengths 1000,5000,10000,30000 \
  --bits 3,4 \
  --n-generate 50

# With GPU Direct Storage (adds GDS variants to Benchmark 5, ~10 minutes)
python bench_flashblade.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 0,1,2,3,4,5,6,7 \
  --bits 3,4 \
  --n-generate 50 \
  --gds

# Point output to FlashBlade mount for accurate I/O timing
python bench_flashblade.py \
  --output-dir /mnt/flashblade/tq_bench \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 0,1,2,3,4,5,6,7 \
  --gds
```

| Flag | Description |
|------|-------------|
| `--model MODEL` | HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct) |
| `--gpus 0,1,2,...` | Comma-separated GPU IDs (default: auto-detect GPUs with >20 GB free) |
| `--context-lengths N,N,...` | Context lengths to benchmark (default: 1000,5000,10000,30000) |
| `--bits B,B,...` | Bit widths for uniform quantization (default: 3,4) |
| `--migration-bits B` | Bit width for the session migration test (default: 3) |
| `--n-generate N` | Tokens for teacher-forced migration verification (default: 50) |
| `--output-dir DIR` | Where to write temp files during benchmarks (default: ./output/flashblade_bench) |
| `--gds` | Enable GPU Direct Storage via kvikio (requires kvikio and GDS-capable storage) |

GDS requires `kvikio` and a storage target accessible via RDMA. Without `--gds`, Benchmark 5 still runs but only compares FP16 vs TQ using CPU-mediated I/O.

## Dependencies

```
torch, torchvision, numpy, scipy, matplotlib, transformers, accelerate, qwen-vl-utils, kvikio (for GDS benchmarks)
```

The `.env` file in this directory stores `HF_TOKEN` and is auto-loaded at startup.

## Further reading

- **[Learner's guide](learners-guide.md)** — In-depth explanation of the algorithms, metrics, evaluation design, and results. Covers: why rotation works, the Beta distribution and Lloyd-Max quantization, MSE vs QJL bias-variance tradeoff, teacher-forced evaluation, outlier channels, residual windows, RoPE compatibility, and QJL's role in vector search.
- **[FlashBlade benchmark results](flashblade-benchmark-results.md)** — Full checkpoint/restore benchmark on Everpure FlashBlade with GPU Direct Storage. Shows how TurboQuant + GDS delivers 6-8x faster restore and 4.9x more sessions per TB.
- **[TurboQuant paper](https://arxiv.org/abs/2504.19874)** — Zandieh et al., 2025
- **[QJL paper](https://arxiv.org/abs/2406.03482)** — The 1-bit sign quantization used in Algorithm 2
- **[PolarQuant](https://arxiv.org/abs/2502.02617)** — Alternative approach using polar coordinates (not random rotation)
- **[KIVI](https://arxiv.org/abs/2402.02750)** — Source of the residual window technique
