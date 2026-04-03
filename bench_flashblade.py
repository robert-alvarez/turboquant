#!/usr/bin/env python3
"""
FlashBlade Storage Benchmark for TurboQuant KV Cache Compression.

Demonstrates TurboQuant's value for KV cache storage on high-performance
flash storage systems. Runs four benchmarks:

  1. Storage I/O performance across context lengths (single GPU)
  2. Concurrent multi-GPU checkpoint/restore (all GPUs)
  3. Session capacity planning (sessions per TB)
  4. Session migration GPU→disk→GPU with quality verification

Usage:
  python bench_flashblade.py
  python bench_flashblade.py --output-dir /mnt/flashblade/tq_bench
  python bench_flashblade.py --context-lengths 1000,5000,10000,30000
  python bench_flashblade.py --model Qwen/Qwen2.5-7B-Instruct --gpus 0,1,2,4,5,6,7
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

import numpy as np
import torch
import torch.nn.functional as F

from turboquant import (
    compute_all_codebooks, DEFAULT_DIM, BIT_WIDTHS,
    TurboQuantMSE, generate_rotation_matrix, DEFAULT_SEED,
    serialize_compressed_kv, deserialize_compressed_kv, dequantize_from_disk,
)
from turboquant.core import (
    identify_outlier_layers, identify_outlier_channels,
    TurboQuantOutlier, OUTLIER_CONFIGS, lloyd_max_codebook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_free_gpus(min_free_mb=20000):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, free = line.split(",")
        if int(free.strip()) >= min_free_mb:
            gpus.append(int(idx.strip()))
    return sorted(gpus)


def capture_kv(model, tokenizer, device, min_tokens=0):
    """Capture KV cache at a given context length."""
    from eval.model import _PARAGRAPHS
    from transformers.cache_utils import DynamicCache

    prompt = " ".join(_PARAGRAPHS[:3])
    if min_tokens > 0:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        para_idx = 0
        while ids.shape[1] < min_tokens:
            prompt += " " + _PARAGRAPHS[para_idx % len(_PARAGRAPHS)]
            para_idx += 1
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if ids.shape[1] > min_tokens:
            ids = ids[:, :min_tokens]
            prompt = tokenizer.decode(ids[0], skip_special_tokens=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    n_tokens = input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    if isinstance(past, DynamicCache):
        n_layers = len(past.layers)
        sample = past.layers[0].keys
    else:
        n_layers = len(past)
        sample = past[0][0]

    n_heads = sample.shape[1]
    d = sample.shape[3]
    keys = torch.zeros(n_layers, n_heads, n_tokens, d, dtype=torch.float32, device=device)
    values = torch.zeros_like(keys)

    for i in range(n_layers):
        if isinstance(past, DynamicCache):
            keys[i] = past.layers[i].keys[0].float()
            values[i] = past.layers[i].values[0].float()
        else:
            keys[i] = past[i][0][0].float()
            values[i] = past[i][1][0].float()

    return {
        "keys": keys, "values": values,
        "n_layers": n_layers, "n_heads": n_heads,
        "n_tokens": n_tokens, "head_dim": d,
    }, input_ids


def drop_page_cache(filepath):
    """Hint OS to evict file from page cache."""
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (AttributeError, OSError):
        pass


def fsync_file(filepath):
    """Force file data to storage device."""
    fd = os.open(filepath, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


# ---------------------------------------------------------------------------
# Benchmark 1: I/O Performance at Each Context Length
# ---------------------------------------------------------------------------

def bench_io(kv_caches, codebooks, bit_widths, output_dir, device,
             outlier_configs=None, outlier_indices=None, subspace_codebooks=None):
    """Measure compress/write/read/decompress at each context length and bit width."""
    print("\n" + "=" * 100)
    print("Benchmark 1: Storage I/O Performance by Context Length")
    print("=" * 100)

    rotation = generate_rotation_matrix(DEFAULT_DIM, seed=DEFAULT_SEED, device=device)
    results = []

    for n_tok, kv in sorted(kv_caches.items()):
        d = kv["head_dim"]
        n_layers, n_heads = kv["n_layers"], kv["n_heads"]
        fp16_size = 2 * n_layers * n_heads * n_tok * d * 2

        # FP16 baseline write/read
        fp16_path = os.path.join(output_dir, f"baseline_{n_tok}tok_fp16.bin")
        raw = torch.cat([kv["keys"], kv["values"]], dim=0).cpu().half().numpy().tobytes()
        t0 = time.time()
        with open(fp16_path, "wb") as f:
            f.write(raw)
        fsync_file(fp16_path)
        t_fp16_write = time.time() - t0

        drop_page_cache(fp16_path)
        t0 = time.time()
        with open(fp16_path, "rb") as f:
            _ = f.read()
        t_fp16_read = time.time() - t0

        results.append({
            "n_tokens": n_tok, "format": "FP16", "bits": 16,
            "size": fp16_size, "fp16_size": fp16_size,
            "t_compress": 0, "t_write": t_fp16_write,
            "t_read": t_fp16_read, "t_decompress": 0,
            "compress_ratio": 1.0, "mse": 0.0, "cos_sim": 1.0,
        })
        os.remove(fp16_path)

        # ── Uniform bit widths ──
        for bits in bit_widths:
            quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

            # Compress
            torch.cuda.synchronize()
            t0 = time.time()
            all_indices = []
            for kv_tensor in [kv["keys"], kv["values"]]:
                for layer in range(n_layers):
                    for head in range(n_heads):
                        indices, norms = quantizer.quantize(kv_tensor[layer, head].to(device))
                        all_indices.append((indices, norms))
            torch.cuda.synchronize()
            t_compress = time.time() - t0

            # Write
            fpath = os.path.join(output_dir, f"bench_{n_tok}tok_{bits}bit.tqkv")
            t0 = time.time()
            serialize_compressed_kv(
                fpath, rotation, codebooks[bits], bits, all_indices,
                n_layers * 2, n_heads, n_tok, d, mode="mse",
            )
            fsync_file(fpath)
            t_write = time.time() - t0
            file_size = os.path.getsize(fpath)

            drop_page_cache(fpath)

            # Read
            t0 = time.time()
            data = deserialize_compressed_kv(fpath)
            t_read = time.time() - t0

            # Decompress
            t0 = time.time()
            recon = dequantize_from_disk(data)
            t_decompress = time.time() - t0

            # Verify
            orig = torch.cat([kv["keys"], kv["values"]], dim=0).cpu().float()
            recon_flat = recon.reshape_as(orig)
            mse = ((orig - recon_flat) ** 2).sum(dim=-1).mean().item()
            cos_sim = F.cosine_similarity(
                orig.reshape(-1, d), recon_flat.reshape(-1, d), dim=-1
            ).mean().item()

            results.append({
                "n_tokens": n_tok, "format": f"{bits}-bit", "bits": bits,
                "size": file_size, "fp16_size": fp16_size,
                "t_compress": t_compress, "t_write": t_write,
                "t_read": t_read, "t_decompress": t_decompress,
                "compress_ratio": fp16_size / file_size,
                "mse": mse, "cos_sim": cos_sim,
            })
            os.remove(fpath)

        # ── Outlier (mixed-precision) configs ──
        if outlier_configs and outlier_indices and subspace_codebooks:
            for name, n_out, bh, bl, eff in outlier_configs:
                d_high, d_low = n_out, d - n_out
                o_idx = outlier_indices[n_out].to(device)
                all_idx = torch.arange(d, device=device)
                mask = torch.ones(d, dtype=torch.bool, device=device)
                mask[o_idx] = False
                n_idx = all_idx[mask]

                rot_h = generate_rotation_matrix(d_high, seed=DEFAULT_SEED, device=device)
                rot_l = generate_rotation_matrix(d_low, seed=DEFAULT_SEED + 1000, device=device)
                q_h = TurboQuantMSE(d_high, bh, subspace_codebooks[(d_high, bh)], rot_h).to(device)
                q_l = TurboQuantMSE(d_low, bl, subspace_codebooks[(d_low, bl)], rot_l).to(device)

                # Compress both subspaces
                torch.cuda.synchronize()
                t0 = time.time()
                idx_high, idx_low = [], []
                for kv_tensor in [kv["keys"], kv["values"]]:
                    for layer in range(n_layers):
                        for head in range(n_heads):
                            vecs = kv_tensor[layer, head].to(device)
                            ih, nh = q_h.quantize(vecs[:, o_idx])
                            il, nl = q_l.quantize(vecs[:, n_idx])
                            idx_high.append((ih, nh))
                            idx_low.append((il, nl))
                torch.cuda.synchronize()
                t_compress = time.time() - t0

                # Write as two .tqkv files
                path_h = os.path.join(output_dir, f"bench_{n_tok}tok_{name}_high.tqkv")
                path_l = os.path.join(output_dir, f"bench_{n_tok}tok_{name}_low.tqkv")
                t0 = time.time()
                serialize_compressed_kv(
                    path_h, rot_h, subspace_codebooks[(d_high, bh)], bh,
                    idx_high, n_layers * 2, n_heads, n_tok, d_high, mode="mse",
                )
                serialize_compressed_kv(
                    path_l, rot_l, subspace_codebooks[(d_low, bl)], bl,
                    idx_low, n_layers * 2, n_heads, n_tok, d_low, mode="mse",
                )
                fsync_file(path_h)
                fsync_file(path_l)
                t_write = time.time() - t0
                file_size = os.path.getsize(path_h) + os.path.getsize(path_l)

                drop_page_cache(path_h)
                drop_page_cache(path_l)

                # Read both
                t0 = time.time()
                data_h = deserialize_compressed_kv(path_h)
                data_l = deserialize_compressed_kv(path_l)
                t_read = time.time() - t0

                # Decompress and merge channels
                t0 = time.time()
                recon_h = dequantize_from_disk(data_h)
                recon_l = dequantize_from_disk(data_l)
                recon = torch.zeros(n_layers * 2, n_heads, n_tok, d)
                recon[:, :, :, o_idx.cpu()] = recon_h
                recon[:, :, :, n_idx.cpu()] = recon_l
                t_decompress = time.time() - t0

                # Verify
                orig = torch.cat([kv["keys"], kv["values"]], dim=0).cpu().float()
                mse = ((orig - recon) ** 2).sum(dim=-1).mean().item()
                cos_sim = F.cosine_similarity(
                    orig.reshape(-1, d), recon.reshape(-1, d), dim=-1
                ).mean().item()

                results.append({
                    "n_tokens": n_tok, "format": name, "bits": eff,
                    "size": file_size, "fp16_size": fp16_size,
                    "t_compress": t_compress, "t_write": t_write,
                    "t_read": t_read, "t_decompress": t_decompress,
                    "compress_ratio": fp16_size / file_size,
                    "mse": mse, "cos_sim": cos_sim,
                })
                os.remove(path_h)
                os.remove(path_l)

    # Print table
    hdr = (f"{'Context':>8} {'Format':>8} {'Size':>10} {'Ratio':>7} "
           f"{'Compress':>10} {'Write':>10} {'Read':>10} {'Decomp':>10} "
           f"{'Wr GB/s':>8} {'Rd GB/s':>8} {'CosSim':>8}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sz_str = f"{r['size']/1e6:.1f} MB" if r['size'] < 1e9 else f"{r['size']/1e9:.2f} GB"
        wr_gbps = (r['size'] / 1e9) / r['t_write'] if r['t_write'] > 0 else 0
        rd_gbps = (r['size'] / 1e9) / r['t_read'] if r['t_read'] > 0 else 0
        comp_s = f"{r['t_compress']*1000:.0f} ms" if r['t_compress'] > 0 else "—"
        decomp_s = f"{r['t_decompress']*1000:.0f} ms" if r['t_decompress'] > 0 else "—"
        print(f"{r['n_tokens']:>7,} {r['format']:>8} {sz_str:>10} {r['compress_ratio']:>6.1f}x "
              f"{comp_s:>10} {r['t_write']*1000:>8.0f} ms {r['t_read']*1000:>8.0f} ms {decomp_s:>10} "
              f"{wr_gbps:>7.2f} {rd_gbps:>7.2f} {r['cos_sim']:>8.6f}")

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Concurrent Multi-GPU I/O
# ---------------------------------------------------------------------------

def bench_concurrent(kv_cache, codebooks, bits, gpu_ids, output_dir, shared_dir):
    """Launch parallel workers across GPUs for concurrent checkpoint/restore."""
    n_tok = kv_cache["n_tokens"]
    d = kv_cache["head_dim"]
    n_layers, n_heads = kv_cache["n_layers"], kv_cache["n_heads"]
    fp16_size = 2 * n_layers * n_heads * n_tok * d * 2
    n_workers = len(gpu_ids)

    print(f"\n{'=' * 100}")
    print(f"Benchmark 2: Concurrent Multi-GPU Checkpoint/Restore ({n_workers} GPUs, {n_tok:,} tokens, {bits}-bit)")
    print(f"{'=' * 100}")

    # Save KV cache for workers
    torch.save({
        "keys": kv_cache["keys"].cpu(), "values": kv_cache["values"].cpu(),
        "n_layers": n_layers, "n_heads": n_heads,
        "n_tokens": n_tok, "head_dim": d,
    }, os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

    worker_script = str(Path(__file__).parent / "eval" / "flashblade_worker.py")
    concurrent_dir = os.path.join(output_dir, "concurrent")
    os.makedirs(concurrent_dir, exist_ok=True)

    # Launch all workers
    t_wall_start = time.time()
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, worker_script,
            "--shared-dir", shared_dir,
            "--output-dir", concurrent_dir,
            "--worker-id", str(i),
            "--bits", str(bits),
        ]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append((i, gpu_id, proc))

    # Wait for all
    worker_results = []
    for wid, gpu_id, proc in processes:
        proc.wait()
        result_path = os.path.join(shared_dir, f"bench_{wid}.json")
        if proc.returncode == 0 and os.path.exists(result_path):
            with open(result_path) as f:
                worker_results.append(json.load(f))
        else:
            stdout = proc.stdout.read().decode()
            print(f"  Worker {wid} (GPU {gpu_id}) FAILED:\n{stdout}")

    t_wall = time.time() - t_wall_start

    if not worker_results:
        print("  All workers failed!")
        return {}

    # Aggregate metrics
    total_compressed = sum(r["file_size"] for r in worker_results)
    total_fp16 = sum(r["fp16_size"] for r in worker_results)
    max_write = max(r["t_write"] for r in worker_results)
    max_read = max(r["t_read"] for r in worker_results)
    avg_compress = sum(r["t_compress"] for r in worker_results) / len(worker_results)
    avg_decompress = sum(r["t_decompress"] for r in worker_results) / len(worker_results)
    avg_roundtrip = sum(r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
                        for r in worker_results) / len(worker_results)

    agg_wr_gbps = (total_compressed / 1e9) / max_write if max_write > 0 else 0
    agg_rd_gbps = (total_compressed / 1e9) / max_read if max_read > 0 else 0
    sessions_per_sec = len(worker_results) / t_wall

    # Per-worker table
    print(f"\n{'Worker':>7} {'GPU':>4} {'Compress':>10} {'Write':>10} {'Read':>10} {'Decomp':>10} {'Total':>10} {'MSE':>10}")
    print("-" * 82)
    for r in worker_results:
        total = r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
        print(f"{r['worker_id']:>7} {gpu_ids[r['worker_id']]:>4} "
              f"{r['t_compress']*1000:>8.0f}ms {r['t_write']*1000:>8.0f}ms "
              f"{r['t_read']*1000:>8.0f}ms {r['t_decompress']*1000:>8.0f}ms "
              f"{total*1000:>8.0f}ms {r['mse']:>10.4f}")

    # Summary
    compressed_size = worker_results[0]["file_size"]
    print(f"\n  Sessions:            {len(worker_results)}")
    print(f"  Per-session FP16:    {fp16_size/1e6:.1f} MB")
    print(f"  Per-session {bits}-bit:  {compressed_size/1e6:.1f} MB ({fp16_size/compressed_size:.1f}x compression)")
    print(f"  Wall time:           {t_wall:.2f}s")
    print(f"  Aggregate write:     {agg_wr_gbps:.2f} GB/s ({total_compressed/1e6:.0f} MB in {max_write*1000:.0f} ms)")
    print(f"  Aggregate read:      {agg_rd_gbps:.2f} GB/s ({total_compressed/1e6:.0f} MB in {max_read*1000:.0f} ms)")
    print(f"  Avg round-trip:      {avg_roundtrip*1000:.0f} ms/session")
    print(f"  Throughput:          {sessions_per_sec:.1f} sessions/sec")

    shutil.rmtree(concurrent_dir, ignore_errors=True)
    return {
        "n_workers": len(worker_results),
        "wall_time": t_wall,
        "agg_write_gbps": agg_wr_gbps,
        "agg_read_gbps": agg_rd_gbps,
        "sessions_per_sec": sessions_per_sec,
        "per_session_compressed": compressed_size,
        "per_session_fp16": fp16_size,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Session Density
# ---------------------------------------------------------------------------

def bench_density(io_results):
    """Compute sessions per TB from I/O results."""
    print(f"\n{'=' * 100}")
    print("Benchmark 3: Session Capacity Planning (sessions per TB)")
    print("=" * 100)

    TB = 1e12
    contexts = sorted(set(r["n_tokens"] for r in io_results))
    formats = sorted(set(r["format"] for r in io_results),
                     key=lambda f: (0 if f == "FP16" else float(f.split("-")[0].split()[0])))

    hdr = f"{'Context':>8}"
    for fmt in formats:
        hdr += f" {fmt:>10}"
    # Add gain column for smallest bit width
    compressed_fmts = [f for f in formats if f != "FP16"]
    if compressed_fmts:
        hdr += f" {'Gain':>8}"
    print(hdr)
    print("-" * len(hdr))

    for n_tok in contexts:
        row = f"{n_tok:>7,}"
        sizes = {}
        for fmt in formats:
            match = [r for r in io_results if r["n_tokens"] == n_tok and r["format"] == fmt]
            if match:
                sessions = int(TB / match[0]["size"])
                sizes[fmt] = sessions
                row += f" {sessions:>10,}"
            else:
                row += f" {'—':>10}"
        if compressed_fmts and "FP16" in sizes and compressed_fmts[0] in sizes:
            gain = sizes[compressed_fmts[0]] / sizes["FP16"]
            row += f" {gain:>7.1f}x"
        print(row)


# ---------------------------------------------------------------------------
# Benchmark 4: Session Migration (GPU → Disk → GPU)
# ---------------------------------------------------------------------------

def bench_migration(kv_cache, input_ids, ground_truth, codebooks, bits,
                    exempt_layers, output_dir, src_gpu, dst_gpu, n_generate, model_name):
    """Compress on src GPU, write to disk, read back, restore on dst GPU, verify generation."""
    n_tok = kv_cache["n_tokens"]
    d = kv_cache["head_dim"]
    n_layers, n_heads = kv_cache["n_layers"], kv_cache["n_heads"]
    fp16_size = 2 * n_layers * n_heads * n_tok * d * 2

    print(f"\n{'=' * 100}")
    print(f"Benchmark 4: Session Migration (GPU {src_gpu} → disk → GPU {dst_gpu})")
    print(f"  {n_tok:,} tokens, {bits}-bit TurboQuant, layer exemption: {sorted(exempt_layers) if exempt_layers else 'none'}")
    print("=" * 100)

    src_device = f"cuda:{src_gpu}"
    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=src_device)
    quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(src_device)

    # ── Step 1: Compress on source GPU ──
    torch.cuda.synchronize()
    t0 = time.time()
    all_indices = []
    for kv_tensor in [kv_cache["keys"], kv_cache["values"]]:
        for layer in range(n_layers):
            for head in range(n_heads):
                indices, norms = quantizer.quantize(kv_tensor[layer, head].to(src_device))
                all_indices.append((indices, norms))
    torch.cuda.synchronize()
    t_compress = time.time() - t0

    # ── Step 2: Write compressed .tqkv ──
    tqkv_path = os.path.join(output_dir, "migration_compressed.tqkv")
    t0 = time.time()
    serialize_compressed_kv(
        tqkv_path, rotation, codebooks[bits], bits, all_indices,
        n_layers * 2, n_heads, n_tok, d, mode="mse",
    )
    fsync_file(tqkv_path)
    t_write = time.time() - t0
    tqkv_size = os.path.getsize(tqkv_path)

    # Write exempt layer keys in FP16
    exempt_path = os.path.join(output_dir, "migration_exempt_keys.bin")
    t0 = time.time()
    if exempt_layers:
        exempt_data = kv_cache["keys"][sorted(exempt_layers)].cpu().half().numpy().tobytes()
        with open(exempt_path, "wb") as f:
            f.write(exempt_data)
        fsync_file(exempt_path)
    t_write_exempt = time.time() - t0
    exempt_size = os.path.getsize(exempt_path) if exempt_layers else 0
    total_size = tqkv_size + exempt_size

    drop_page_cache(tqkv_path)
    if exempt_layers:
        drop_page_cache(exempt_path)

    # ── Step 3: Read from disk ──
    t0 = time.time()
    data = deserialize_compressed_kv(tqkv_path)
    if exempt_layers:
        with open(exempt_path, "rb") as f:
            exempt_raw = f.read()
    t_read = time.time() - t0

    # ── Step 4: Decompress on CPU ──
    t0 = time.time()
    recon = dequantize_from_disk(data)
    # Split back into keys and values
    recon_keys = recon[:n_layers]
    recon_values = recon[n_layers:]
    # Patch exempt layer keys with FP16 originals
    if exempt_layers:
        exempt_tensor = torch.from_numpy(
            np.frombuffer(exempt_raw, dtype=np.float16).copy()
        ).float().reshape(len(exempt_layers), n_heads, n_tok, d)
        for i, layer_idx in enumerate(sorted(exempt_layers)):
            recon_keys[layer_idx] = exempt_tensor[i]
    t_decompress = time.time() - t0

    t_migration = t_compress + t_write + t_write_exempt + t_read + t_decompress

    # ── Step 5: Load model on destination GPU, verify generation ──
    print(f"\n  Loading model on GPU {dst_gpu} for verification...")
    dst_device = f"cuda:{dst_gpu}"
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=dst_device, trust_remote_code=True,
    )
    model.eval()
    t_model_load = time.time() - t0

    # Build DynamicCache on dst GPU
    t0 = time.time()
    from eval.metrics import build_dynamic_cache, teacher_forced_generate
    restored_keys = recon_keys.to(dst_device)
    restored_values = recon_values.to(dst_device)
    cache = build_dynamic_cache(
        restored_keys[:, :, :-1, :], restored_values[:, :, :-1, :],
    )
    t_build_cache = time.time() - t0

    # Teacher-forced generation
    dst_input_ids = input_ids.to(dst_device)
    dst_ground_truth = ground_truth.to(dst_device)
    t0 = time.time()
    preds, _ = teacher_forced_generate(
        model, dst_input_ids, cache, dst_ground_truth, topk_values=[1],
    )
    t_generate = time.time() - t0
    top1 = (preds == dst_ground_truth).float().mean().item()

    del model
    torch.cuda.empty_cache()

    # Clean up files
    os.remove(tqkv_path)
    if exempt_layers and os.path.exists(exempt_path):
        os.remove(exempt_path)

    # ── Report ──
    print(f"\n  {'Phase':<30} {'Time':>10} {'Notes'}")
    print(f"  {'-'*70}")
    print(f"  {'Compress (GPU)' :<30} {t_compress*1000:>8.0f} ms  {bits}-bit, {n_tok:,} tokens")
    print(f"  {'Write .tqkv':<30} {t_write*1000:>8.0f} ms  {tqkv_size/1e6:.1f} MB")
    if exempt_layers:
        print(f"  {'Write exempt keys (FP16)':<30} {t_write_exempt*1000:>8.0f} ms  {exempt_size/1e6:.1f} MB")
    print(f"  {'Read from disk':<30} {t_read*1000:>8.0f} ms  {total_size/1e6:.1f} MB total")
    print(f"  {'Decompress + patch (CPU)':<30} {t_decompress*1000:>8.0f} ms")
    print(f"  {'Build KV cache (GPU)':<30} {t_build_cache*1000:>8.0f} ms")
    print(f"  {'Resume generation':<30} {t_generate*1000:>8.0f} ms  {n_generate} tokens")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL MIGRATION':<30} {t_migration*1000:>8.0f} ms  (excl. model load)")
    print(f"  {'Model load (one-time)':<30} {t_model_load*1000:>8.0f} ms")
    print(f"  {'-'*70}")
    print(f"  {'Storage: compressed':<30} {total_size/1e6:>8.1f} MB")
    print(f"  {'Storage: FP16 baseline':<30} {fp16_size/1e6:>8.1f} MB")
    print(f"  {'Compression ratio':<30} {fp16_size/total_size:>8.1f}x")
    print(f"  {'Top-1 accuracy':<30} {top1:>8.1%}   ({int(top1*n_generate)}/{n_generate} tokens correct)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FlashBlade Storage Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="./output/flashblade_bench")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--context-lengths", default="1000,5000,10000,30000")
    parser.add_argument("--bits", default="3,4", help="Bit widths to benchmark")
    parser.add_argument("--n-generate", type=int, default=50)
    parser.add_argument("--migration-bits", type=int, default=3)
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    bit_widths = [int(x) for x in args.bits.split(",")]

    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = get_free_gpus(min_free_mb=20000)

    if not gpu_ids:
        print("ERROR: No GPUs with sufficient free memory.")
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shared_dir = tempfile.mkdtemp(prefix="tq_flashblade_")

    print("=" * 100)
    print("FlashBlade Storage Benchmark — TurboQuant KV Cache Compression")
    print("=" * 100)
    print(f"  Model:            {args.model}")
    print(f"  GPUs:             {gpu_ids}")
    print(f"  Context lengths:  {context_lengths}")
    print(f"  Bit widths:       {bit_widths}")
    print(f"  Output dir:       {output_dir}")

    setup_gpu = gpu_ids[0]
    device = f"cuda:{setup_gpu}"
    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Setup
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("Phase 1: Setup")
    print("=" * 100)

    print("Computing codebooks...")
    all_bits = sorted(set(bit_widths + [args.migration_bits]))
    codebooks = compute_all_codebooks(DEFAULT_DIM, all_bits)
    for b in all_bits:
        if b >= 2 and b - 1 not in codebooks:
            codebooks[b - 1] = lloyd_max_codebook(DEFAULT_DIM, b - 1)

    print(f"Loading model {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Capture KV caches at each context length
    kv_caches = {}
    input_ids_map = {}
    for n_tok in context_lengths:
        print(f"  Capturing KV cache at {n_tok:,} tokens...", end=" ", flush=True)
        t0 = time.time()
        try:
            kv, ids = capture_kv(model, tokenizer, device, min_tokens=n_tok)
            kv_caches[kv["n_tokens"]] = kv
            input_ids_map[kv["n_tokens"]] = ids
            fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * kv["n_tokens"] * kv["head_dim"] * 2 / 1e6
            print(f"done ({kv['n_tokens']} tokens, FP16={fp16_mb:.0f} MB, {time.time()-t0:.1f}s)")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM! Skipping {n_tok} tokens.")
            torch.cuda.empty_cache()

    if not kv_caches:
        print("ERROR: Failed to capture any KV caches.")
        sys.exit(1)

    # Detect outlier layers for migration test
    longest_ctx = max(kv_caches.keys())
    longest_kv = kv_caches[longest_ctx]
    outlier_info = identify_outlier_layers(longest_kv)
    exempt_layers = {info[0] for info in outlier_info}
    if exempt_layers:
        print(f"  Outlier layers (keys exempt): {sorted(exempt_layers)}")

    # Detect outlier channels and compute subspace codebooks for mixed-precision configs
    outlier_indices = {}  # n_outliers -> tensor of channel indices
    subspace_codebooks = {}  # (dim, bits) -> codebook
    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        if n_out not in outlier_indices:
            outlier_indices[n_out] = identify_outlier_channels(longest_kv, n_outliers=n_out)
            print(f"  Top-{n_out} outlier channels identified")
        d_high, d_low = n_out, DEFAULT_DIM - n_out
        for dim, bits in [(d_high, bh), (d_low, bl)]:
            if (dim, bits) not in subspace_codebooks:
                print(f"  Computing codebook d={dim}, {bits}-bit...", end=" ", flush=True)
                t0 = time.time()
                subspace_codebooks[(dim, bits)] = lloyd_max_codebook(dim, bits)
                print(f"done ({time.time()-t0:.1f}s)")

    # Generate ground truth for migration test
    print(f"  Generating ground truth ({args.n_generate} tokens at {longest_ctx:,} context)...")
    from eval.metrics import generate_ground_truth
    ground_truth = generate_ground_truth(
        model, input_ids_map[longest_ctx],
        longest_kv["keys"], longest_kv["values"], args.n_generate,
    )

    # Move KV caches to CPU, free model
    for kv in kv_caches.values():
        kv["keys"] = kv["keys"].cpu()
        kv["values"] = kv["values"].cpu()
    migration_input_ids = input_ids_map[longest_ctx].cpu()
    ground_truth = ground_truth.cpu()
    del model, tokenizer
    torch.cuda.empty_cache()

    print(f"Phase 1 done ({time.time()-t_total:.1f}s)")

    # ══════════════════════════════════════════════════════════════════════════
    # Benchmark 1: I/O Performance
    # ══════════════════════════════════════════════════════════════════════════
    io_results = bench_io(
        kv_caches, codebooks, bit_widths, output_dir, device,
        outlier_configs=OUTLIER_CONFIGS,
        outlier_indices=outlier_indices,
        subspace_codebooks=subspace_codebooks,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Benchmark 2: Concurrent Multi-GPU I/O
    # ══════════════════════════════════════════════════════════════════════════
    # Use the largest context that fits comfortably (10K default, or largest available)
    concurrent_ctx = min(10000, max(kv_caches.keys()))
    # Find the actual captured context closest to target
    concurrent_ctx = min(kv_caches.keys(), key=lambda k: abs(k - concurrent_ctx))
    concurrent_kv = kv_caches[concurrent_ctx]
    concurrent_bits = bit_widths[0]  # Use first (smallest) bit width

    concurrent_results = bench_concurrent(
        concurrent_kv, codebooks, concurrent_bits, gpu_ids, output_dir, shared_dir,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Benchmark 3: Session Density
    # ══════════════════════════════════════════════════════════════════════════
    bench_density(io_results)

    # ══════════════════════════════════════════════════════════════════════════
    # Benchmark 4: Session Migration
    # ══════════════════════════════════════════════════════════════════════════
    src_gpu = gpu_ids[0]
    dst_gpu = gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0]

    bench_migration(
        longest_kv, migration_input_ids, ground_truth,
        codebooks, args.migration_bits, exempt_layers,
        output_dir, src_gpu, dst_gpu, args.n_generate, args.model,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_total
    print(f"\n{'=' * 100}")
    print(f"Benchmark complete! Total time: {total_time:.0f}s")
    print(f"Output directory: {output_dir}")
    print("=" * 100)

    # Cleanup
    shutil.rmtree(shared_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
