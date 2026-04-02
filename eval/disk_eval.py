"""
Disk-based evaluation: compress -> serialize -> deserialize -> decompress -> verify.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from turboquant import (
    TurboQuantMSE,
    generate_rotation_matrix,
    serialize_compressed_kv,
    deserialize_compressed_kv,
    dequantize_from_disk,
    DEFAULT_SEED,
    DEFAULT_DIM,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run_disk_evaluation(kv_cache: dict, codebooks: dict, device: str = "cuda"):
    """
    Demonstrate full round-trip: compress -> serialize -> deserialize -> decompress -> verify.
    """
    d = kv_cache["head_dim"]
    n_layers = kv_cache["n_layers"]
    n_heads = kv_cache["n_heads"]
    n_tokens = kv_cache["n_tokens"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)

    print("\n" + "=" * 80)
    print("Part 2: Disk-Based KV Cache Compression Round-Trip")
    print("=" * 80)
    print(f"KV cache shape: {n_layers} layers x {n_heads} heads x {n_tokens} tokens x {d} dim")

    # Save FP16 baseline
    fp16_path = OUTPUT_DIR / "kv_cache_fp16.bin"
    t0 = time.time()
    fp16_data = torch.cat([kv_cache["keys"], kv_cache["values"]], dim=0)  # (2*L, H, T, d)
    with open(fp16_path, "wb") as f:
        f.write(fp16_data.cpu().half().numpy().tobytes())
    fp16_write_time = time.time() - t0
    fp16_size = os.path.getsize(fp16_path)

    print(f"\nFP16 baseline: {fp16_size:,} bytes ({fp16_size / 1e6:.2f} MB), write time: {fp16_write_time:.3f}s")

    for bits in [3, 4]:
        for mode in ["mse"]:
            print(f"\n--- {bits}-bit TurboQuant_{mode} ---")

            # Step 1: Compress
            t_compress_start = time.time()
            quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

            all_indices = []
            all_original = []  # Store original vectors for quality verification
            for kv_type, kv_name in [(kv_cache["keys"], "keys"), (kv_cache["values"], "values")]:
                for layer in range(n_layers):
                    for head in range(n_heads):
                        vecs = kv_type[layer, head].to(device)  # (n_tokens, d)
                        indices, norms = quantizer.quantize(vecs)
                        all_indices.append((indices, norms))
                        all_original.append(vecs)

            t_compress = time.time() - t_compress_start
            print(f"  Compress time:    {t_compress:.3f}s")

            # Step 2: Serialize to disk
            filepath = OUTPUT_DIR / f"kv_cache_{bits}bit_{mode}.tqkv"
            t_write_start = time.time()

            # We serialize keys and values together: first n_layers*n_heads blocks are keys,
            # next n_layers*n_heads blocks are values
            serialize_compressed_kv(
                str(filepath),
                rotation_matrix=rotation,
                codebook=codebooks[bits],
                bits=bits,
                all_indices=all_indices,
                n_layers=n_layers * 2,  # keys + values
                n_heads=n_heads,
                n_tokens=n_tokens,
                d=d,
                mode=mode,
            )
            t_write = time.time() - t_write_start
            file_size = os.path.getsize(filepath)
            print(f"  Write time:       {t_write:.3f}s")
            print(f"  File size:        {file_size:,} bytes ({file_size / 1e6:.2f} MB)")
            print(f"  Compression:      {fp16_size / file_size:.2f}x vs FP16")

            # Step 3: Read from disk and dequantize on CPU
            t_read_start = time.time()
            data = deserialize_compressed_kv(str(filepath))
            t_read = time.time() - t_read_start

            t_decompress_start = time.time()
            reconstructed = dequantize_from_disk(data)  # (2*n_layers, n_heads, n_tokens, d)
            t_decompress = time.time() - t_decompress_start

            print(f"  Read time:        {t_read:.3f}s")
            print(f"  Decompress time:  {t_decompress:.3f}s")
            print(f"  Total round-trip: {t_compress + t_write + t_read + t_decompress:.3f}s")

            # Step 4: Verify quality
            # Compare reconstructed (CPU) vs original (GPU->CPU)
            all_orig_cat = torch.stack(
                [v.cpu().float() for v in all_original]
            )  # (2*L*H, n_tokens, d)
            reconstructed_flat = reconstructed.reshape(-1, n_tokens, d)

            mse = ((all_orig_cat - reconstructed_flat) ** 2).sum(dim=-1).mean().item()
            cos_sim = F.cosine_similarity(
                all_orig_cat.reshape(-1, d),
                reconstructed_flat.reshape(-1, d),
                dim=-1
            ).mean().item()

            print(f"  Reconstruction MSE:    {mse:.6f}")
            print(f"  Reconstruction CosSim: {cos_sim:.6f}")

    # Summary table
    print("\n" + "=" * 80)
    print("Disk Round-Trip Summary")
    print("=" * 80)
    print(f"{'Format':<25} {'Size (MB)':>10} {'Compress':>10} {'MSE':>12} {'CosSim':>10}")
    print("-" * 80)
    print(f"{'FP16 baseline':<25} {fp16_size/1e6:>10.2f} {'1.0x':>10} {'0.000000':>12} {'1.000000':>10}")

    for bits in [3, 4]:
        filepath = OUTPUT_DIR / f"kv_cache_{bits}bit_mse.tqkv"
        if filepath.exists():
            fsize = os.path.getsize(filepath)
            # Re-read and verify to get metrics for summary
            data = deserialize_compressed_kv(str(filepath))
            recon = dequantize_from_disk(data)
            orig_combined = torch.cat([kv_cache["keys"], kv_cache["values"]], dim=0).cpu().float()
            recon_flat = recon.reshape_as(orig_combined)
            s_mse = ((orig_combined - recon_flat) ** 2).sum(dim=-1).mean().item()
            s_cos = F.cosine_similarity(
                orig_combined.reshape(-1, d), recon_flat.reshape(-1, d), dim=-1
            ).mean().item()
            print(f"{'TurboQuant_MSE ' + str(bits) + '-bit':<25} {fsize/1e6:>10.2f} "
                  f"{fp16_size/fsize:>9.2f}x {s_mse:>12.6f} {s_cos:>10.6f}")


def run_large_disk_evaluation(codebooks: dict, device: str = "cuda"):
    """
    Part 2 at scale: 10K tokens, 32 layers, 8 heads, d=128.
    Demonstrates realistic disk compression scenario.
    """
    print("\n" + "=" * 80)
    print("Part 2 (Large Scale): 10K Tokens Disk Compression")
    print("=" * 80)

    n_layers, n_heads, n_tokens, d = 32, 8, 10000, DEFAULT_DIM
    print(f"Generating synthetic KV cache: {n_layers}L x {n_heads}H x {n_tokens}T x {d}d")

    # Generate in chunks to avoid OOM
    torch.manual_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FP16 baseline size calculation
    fp16_total_bytes = 2 * n_layers * n_heads * n_tokens * d * 2  # keys+values, FP16
    print(f"FP16 baseline size: {fp16_total_bytes / 1e9:.3f} GB")

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)

    for bits in [3, 4]:
        print(f"\n--- {bits}-bit TurboQuant_MSE (large scale) ---")
        quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

        filepath = OUTPUT_DIR / f"kv_cache_10k_{bits}bit_mse.tqkv"

        # Compress and serialize
        t_total_start = time.time()
        all_indices = []
        total_vecs = 0

        for kv_idx in range(2):  # keys, values
            for layer in range(n_layers):
                for head in range(n_heads):
                    # Generate on CPU first, then move to device (ensures reproducibility)
                    torch.manual_seed(42 + kv_idx * 10000 + layer * 100 + head)
                    vecs = torch.randn(n_tokens, d)
                    scale = 0.5 + 1.5 * (layer / n_layers)
                    vecs = vecs * scale
                    vecs = vecs.to(device)

                    indices, norms = quantizer.quantize(vecs)
                    all_indices.append((indices, norms))
                    total_vecs += n_tokens

        t_compress = time.time() - t_total_start
        print(f"  Compress time:  {t_compress:.2f}s ({total_vecs:,} vectors)")

        t_write_start = time.time()
        serialize_compressed_kv(
            str(filepath),
            rotation_matrix=rotation,
            codebook=codebooks[bits],
            bits=bits,
            all_indices=all_indices,
            n_layers=n_layers * 2,
            n_heads=n_heads,
            n_tokens=n_tokens,
            d=d,
            mode="mse",
        )
        t_write = time.time() - t_write_start
        file_size = os.path.getsize(filepath)

        print(f"  Write time:     {t_write:.2f}s")
        print(f"  File size:      {file_size / 1e6:.2f} MB ({file_size / 1e9:.3f} GB)")
        print(f"  Compression:    {fp16_total_bytes / file_size:.2f}x vs FP16")

        # Read back and verify a sample
        t_read_start = time.time()
        data = deserialize_compressed_kv(str(filepath))
        t_read = time.time() - t_read_start

        t_decomp_start = time.time()
        # Only dequantize a small sample (first 2 layers) to verify
        codebook_t = torch.from_numpy(data["codebook"]).float()
        rot = data["rotation"]
        sample_mse = 0.0
        sample_cos = 0.0
        n_sample = 0
        for i in range(min(2 * n_heads, len(data["blocks"]))):
            block = data["blocks"][i]
            y_hat = codebook_t[block["indices"]]
            x_hat = (rot.T @ y_hat.T).T * block["norms"].unsqueeze(-1)

            # Regenerate original for comparison
            # Block ordering matches compress loop: kv_idx -> layer -> head
            kv_idx = i // (n_layers * n_heads)
            remainder = i % (n_layers * n_heads)
            layer = remainder // n_heads
            head = remainder % n_heads
            torch.manual_seed(42 + kv_idx * 10000 + layer * 100 + head)
            orig = torch.randn(n_tokens, d)
            scale = 0.5 + 1.5 * (layer / n_layers)
            orig = orig * scale

            sample_mse += ((orig - x_hat) ** 2).sum(dim=-1).mean().item()
            sample_cos += F.cosine_similarity(orig.reshape(-1, d), x_hat.reshape(-1, d), dim=-1).mean().item()
            n_sample += 1

        t_decomp = time.time() - t_decomp_start

        print(f"  Read time:      {t_read:.2f}s")
        print(f"  Decompress (sample): {t_decomp:.2f}s")
        print(f"  Sample MSE:     {sample_mse / n_sample:.6f}")
        print(f"  Sample CosSim:  {sample_cos / n_sample:.6f}")
        print(f"  Total round-trip: {t_compress + t_write + t_read + t_decomp:.2f}s")

        # Clean up the large file
        del all_indices
        torch.cuda.empty_cache() if device == "cuda" else None

    # Final summary
    print("\n" + "=" * 80)
    print("Large-Scale Summary: GPU -> Compress -> Disk -> CPU Decompress -> Verify")
    print("=" * 80)
    print(f"  Scale: {n_layers} layers, {n_heads} heads, {n_tokens:,} tokens, d={d}")
    print(f"  FP16 size: {fp16_total_bytes / 1e9:.3f} GB")
    for bits in [3, 4]:
        fp = OUTPUT_DIR / f"kv_cache_10k_{bits}bit_mse.tqkv"
        if fp.exists():
            sz = os.path.getsize(fp)
            print(f"  {bits}-bit TurboQuant_MSE: {sz / 1e6:.2f} MB ({fp16_total_bytes / sz:.2f}x compression)")
