# TurboQuant KV Cache Compression

From-scratch implementation of the TurboQuant algorithm for KV cache compression.

**Paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" — Zandieh et al., Google Research, arXiv:2504.19874v1

## Project structure

```
turboquant/
    turboquant/              # Core library (importable, dependency-light)
        __init__.py          # Public API exports
        core.py              # Quantizers, codebooks, rotation, outlier detection
        bitpack.py           # Bit packing/unpacking utilities
        serialize.py         # .tqkv binary disk format
    eval/                    # Evaluation harness (pulls in transformers, matplotlib)
        metrics.py           # MSE, cosine, IP correlation, Top-k teacher-forced eval
        model.py             # Model loading (text + VLM), KV cache capture
        visualize.py         # Four matplotlib plots
        gpu_eval.py          # GPU evaluation (uniform, prod, outlier, window, top-k)
        disk_eval.py         # Disk round-trip compression tests
    run_eval.py              # CLI entry point
    bench_flashblade.py      # FlashBlade storage benchmark (multi-GPU, GDS)
    eval/flashblade_worker.py  # Per-GPU worker for concurrent benchmarks
    flashblade-benchmark-results.md  # Benchmark results and analysis
    learners-guide.md        # In-depth technical guide
    output/                  # Generated PNGs and .tqkv files
    .venv/                   # Python 3.12 virtual environment
    .env                     # HF_TOKEN (auto-loaded)
```

The `turboquant/` package depends only on torch, numpy, scipy. The `eval/` modules add transformers, matplotlib, PIL as needed.

## Setup

```bash
source .venv/bin/activate
```

Dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `transformers`, `accelerate`, `qwen-vl-utils`, `kvikio` (for GDS benchmarks)

The `.env` file in this directory contains `HF_TOKEN` and is auto-loaded at startup.

## Running

```bash
# Full run with real model (Qwen2.5-3B-Instruct on GPU)
python run_eval.py

# With Top-1 token accuracy evaluation (keeps model in memory)
python run_eval.py --eval-top1

# Control number of tokens generated for Top-1 evaluation (default: 50)
python run_eval.py --eval-top1 --n-generate 100

# Synthetic data only (no model download, no Top-1)
python run_eval.py --skip-model

# CPU-only
python run_eval.py --skip-model --device cpu

# Larger models
python run_eval.py --model Qwen/Qwen2.5-7B-Instruct --eval-top1
python run_eval.py --model Qwen/Qwen2.5-14B-Instruct --eval-top1

# Longer context (extend prompt to N tokens)
python run_eval.py --min-tokens 1000 --eval-top1 --residual-window 128
python run_eval.py --min-tokens 5000 --eval-top1

# Vision Language Model (VLM)
python run_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct --image path/to/image.png
python run_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --image path/to/image.png --eval-top1
```

## FlashBlade storage benchmark

`bench_flashblade.py` measures checkpoint/restore performance for KV caches on Everpure FlashBlade storage. It loads a real model, captures KV caches at multiple context lengths, then runs five benchmarks:

1. **Storage I/O** — Single-GPU compress/write/read/decompress at each context length and bit width. Measures per-phase timing, compression ratio, and cosine similarity. Includes both uniform and mixed-precision (outlier) configs.
2. **Concurrent multi-GPU checkpoint/restore** — Spawns one worker per GPU, each independently checkpointing and restoring a session. Measures aggregate throughput and per-worker variance.
3. **Session capacity** — Computes sessions per terabyte at each context length and compression level.
4. **Session migration** — Full GPU-to-disk-to-GPU migration: compress on GPU 0, write to disk, read back, decompress, restore on GPU 1, verify with teacher-forced generation. Reports Top-1 accuracy and per-phase latency. Uses layer exemption to keep outlier layers' keys in FP16.
5. **TurboQuant ON vs OFF with GPU Direct Storage** — Compares four configurations (FP16, FP16+GDS, TQ, TQ+GDS) with concurrent workers. This is the key benchmark: it shows how compression + GDS combine to speed up restore (6-8x faster than FP16) and how the advantage grows with context length.

Each worker runs as a separate process (`eval/flashblade_worker.py`) with `CUDA_VISIBLE_DEVICES` set, communicating results via JSON files in a shared temp directory.

```bash
# Standard benchmark (Benchmarks 1-5, no GDS, ~7 minutes)
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
  --context-lengths 1000,10000,30000 \
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

### FlashBlade benchmark CLI flags

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

### GDS requirements

GPU Direct Storage requires `kvikio` and a storage target accessible via RDMA. The FlashBlade is mounted at `/mnt/data` via NFS4 with RDMA enabled. GDS bypasses the CPU on the I/O path — reads go directly from storage to GPU memory, and the decompression (bit unpack + inverse rotation) runs on GPU instead of CPU.

Without `--gds`, Benchmark 5 still runs but only compares FP16 vs TQ using CPU-mediated I/O. The GDS variants are the ones that show the real FlashBlade advantage.

Results are written to `flashblade-benchmark-results.md`.

## What it implements

1. **Lloyd-Max codebook computation** for Beta((d-1)/2, (d-1)/2) distribution via scipy.integrate
2. **TurboQuant_MSE** (Algorithm 1): Haar-random orthogonal rotation (QR of Gaussian matrix) + per-coordinate scalar quantization
3. **TurboQuant_prod** (Algorithm 2): MSE at (b-1) bits + 1-bit QJL residual correction
4. **Outlier channel handling**: Mixed-precision quantization — high-variance channels at higher bits, rest at lower bits. Enables fractional rates like 2.5-bit and 3.5-bit (paper Section 4.3)
5. **Top-1 token accuracy** (`--eval-top1`): Teacher-forced generation comparing original vs quantized KV cache predictions
6. **Layer exemption**: Auto-detects layers with anomalously large key norms (>4x median) and keeps their keys in FP16, preventing catastrophic attention distortion
7. **Bit packing**: Vectorized pack/unpack for arbitrary bit widths using numpy
8. **Binary disk format** (`.tqkv`): Stores rotation matrix + codebook (once) + packed indices/norms per layer/head
9. **Full disk round-trip**: GPU compress -> serialize -> deserialize on CPU -> dequantize -> verify

## CLI flags

| Flag | Description |
|------|-------------|
| `--skip-model` | Use synthetic data instead of loading a real model |
| `--device {cuda,cpu}` | Device to use (default: cuda) |
| `--model MODEL` | HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct) |
| `--eval-top1` | Enable Top-1 token match evaluation (requires model, adds ~5min with outlier configs) |
| `--n-generate N` | Tokens to generate for Top-1 eval (default: 50) |
| `--residual-window W` | Keep last W tokens in FP16 (0 = disabled, typical: 128). Requires `--eval-top1` |
| `--min-tokens N` | Extend prompt to at least N tokens (0 = use default ~168 token prompt) |
| `--image PATH` | Image path/URL for VLM evaluation (requires a Qwen2.5-VL model) |

## Key constants

- `DEFAULT_DIM = 128` (head dimension)
- `DEFAULT_SEED = 42` (rotation matrix seed)
- `BIT_WIDTHS = [2, 3, 4, 5, 6, 8]`
- QJL dequant scale: `sqrt(pi/2) / d`

## Architecture notes

- The rotation matrix is generated ONCE from a fixed seed and reused for all vectors (data-oblivious).
- After rotation, each coordinate follows a Beta distribution. The Lloyd-Max codebook is precomputed from this known distribution — no calibration data needed.
- MSE-only mode is the recommended default. MSE+QJL (prod mode) gives unbiased inner products but higher variance, which hurts Top-1 token accuracy in practice (3-bit MSE: 90% vs 3-bit MSE+QJL: 8%).
- The `.tqkv` binary format uses a header with magic bytes "TQKV", followed by the shared rotation matrix and codebook, then per-block packed indices and norms.
- `transformers >= 5.x` uses `DynamicCache` with `.layers[i].keys/.values` (not tuple indexing).

## Top-1 evaluation design

The `--eval-top1` flag uses **teacher-forced** generation to avoid error cascading in autoregressive decoding:
1. Generate a ground truth token sequence using the original (FP32) KV cache
2. For each quantized variant: feed the ground truth tokens as context at each step and record predictions
3. Compare per-position predictions against ground truth

This gives a stable, per-position accuracy metric. Without teacher forcing, a single wrong token at position 1 cascades into a completely different sequence, making the metric noisy and non-monotonic across bit widths.

The model is kept in memory during the GPU evaluation phase and freed before the disk compression tests.

## Outlier channel handling

The `TurboQuantOutlier` class splits d=128 channels into two groups by variance, each with its own rotation matrix and codebook at the appropriate subspace dimension. Predefined configs in `OUTLIER_CONFIGS`:

| Config | Outlier channels | High bits | Low bits | Effective rate |
|--------|-----------------|-----------|----------|----------------|
| 2.5-bit | 32 @ 4-bit | 4 | 2 | 2.5 |
| 3.5-bit | 32 @ 5-bit | 5 | 3 | 3.5 |
| 3.0-bit (mixed) | 64 @ 4-bit | 4 | 2 | 3.0 |
| 4.0-bit (mixed) | 64 @ 5-bit | 5 | 3 | 4.0 |

Channel identification uses per-channel variance across the entire KV cache. Each subspace gets its own codebook (computed for d=32, d=64, or d=96 as appropriate).

## Evaluation pipeline

`run_gpu_evaluation` runs three quantizer families in order and returns all results:
1. **TurboQuant_MSE** — uniform bit widths [2, 3, 4, 5, 6, 8]
2. **TurboQuant_prod** — MSE+QJL at the same bit widths
3. **TurboQuant with outlier handling** — four mixed-precision configs (2.5, 3.0, 3.5, 4.0 effective bits)
4. **Residual window** (if `--residual-window W` is set) — compares Top-1 with and without keeping last W tokens in FP16

Each family prints its own results table. A comparison table follows each section. When `--eval-top1` is active, Top-1 is measured for every row across all three families.

## Layer exemption

`identify_outlier_layers()` detects layers where mean key norm exceeds 4x the median across layers. These layers' keys are kept in FP16 during quantization (values are still quantized — they are robust).

This is critical for larger models: Qwen2.5-7B has Layer 0 (key_norm=274, 14x median) and Layer 27 (key_norm=242, 13x median) which, at moderate bit widths (3-6 bit), create systematically wrong attention patterns that catastrophically break generation. The 3B model's Layer 0 (key_norm=172) is less extreme and tolerates quantization.

The cost is negligible: typically 1-2 of 28+ layers exempt, affecting only keys (~7% of key storage stays FP16). The Top-1+LE column in the evaluation output shows the layer-exempt results alongside the baseline.

## Residual window

The `--residual-window W` flag (from KIVI) keeps the last W tokens in full FP16 precision and only compresses tokens [0, n_tokens-W). Recent tokens disproportionately affect generation quality because attention scores decay with distance.

The compression cost is small at long contexts: at 10K tokens with W=128, effective 3-bit compression is ~4.7x vs 4.9x without the window. The quality benefit is most visible with longer sequences where the compressed portion dominates storage.

## Supported models

All models with head_dim=128 work with the precomputed codebooks. Tested:

| Model | Params | FP16 Size | Layers | KV Heads | head_dim |
|-------|--------|-----------|--------|----------|----------|
| Qwen/Qwen2.5-3B-Instruct | 3B | ~6 GB | 36 | 2 | 128 |
| Qwen/Qwen2.5-7B-Instruct | 7.6B | ~15 GB | 28 | 4 | 128 |
| Qwen/Qwen2.5-14B-Instruct | 14.7B | ~29 GB | 48 | 8 | 128 |
| Qwen/Qwen2.5-VL-3B-Instruct | 3B | ~8 GB | 36 | 2 | 128 |
| Qwen/Qwen2.5-VL-7B-Instruct | 7.6B | ~17 GB | 28 | 4 | 128 |

All fit on the 128GB GB10. If head_dim != 128, codebooks are automatically recomputed.

## VLM support

The `--image` flag enables Vision Language Model evaluation. Image tokens from the vision encoder are processed through the same transformer layers and stored in the KV cache alongside text tokens. TurboQuant quantizes them identically — it's data-oblivious.

VLM mode uses the Qwen2.5-VL processor (manually assembled to work around an upstream config issue where `preprocessor_config.json` references `Qwen2VLImageProcessor` instead of the newer class name). Requires `torchvision` and `qwen-vl-utils`.

## Top-k evaluation

When `--eval-top1` is active, a Top-k summary table is printed for k = 1, 2, 4, 8, 16. This shows whether the ground truth token appears in the quantized model's top-k predictions at each position, using the same teacher-forced evaluation as Top-1.

## Future test items

1. **VLM evaluation** — Run `Qwen2.5-VL-3B` and `VL-7B` with an image to test whether vision tokens are harder or easier to quantize than text tokens.
2. **More generation tokens** — Increase `--n-generate` to 200-500 for more statistical power. Current 50-token window may hide accuracy drops in configs showing 100%.
3. **Adversarial prompts** — Test with math/code/reasoning-heavy prompts where token predictions are more brittle, rather than the current benign KV cache explanation prompt.
