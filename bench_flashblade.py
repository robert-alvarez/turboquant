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
import gc
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
    write_direct, read_direct,
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


def get_gpu_cpu_affinity():
    """Parse nvidia-smi topo to build GPU -> CPU core list mapping.

    Returns dict: gpu_id -> list of logical CPU core IDs (NUMA-local).
    Falls back to None if parsing fails.
    """
    import re
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        gpu_affinity = {}
        # Match CPU affinity pattern: ranges like "48-63,176-191" in tab-separated fields
        cpu_range_re = re.compile(r'^\d+(-\d+)?(,\d+(-\d+)?)+$')
        for line in result.stdout.strip().split("\n"):
            if not line.startswith("GPU"):
                continue
            parts = line.split("\t")
            gpu_label = parts[0].strip()
            if not gpu_label.startswith("GPU") or not gpu_label[3:].isdigit():
                continue
            gpu_id = int(gpu_label[3:])
            # Find the field matching a CPU range pattern (e.g. "48-63,176-191")
            affinity_str = None
            for field in parts[1:]:
                field = field.strip()
                if cpu_range_re.match(field):
                    affinity_str = field
                    break
            if not affinity_str:
                continue
            cores = _parse_cpu_range(affinity_str)
            if cores:
                gpu_affinity[gpu_id] = cores
        return gpu_affinity if gpu_affinity else None
    except (OSError, ValueError):
        return None


def _parse_cpu_range(s):
    """Parse '48-63,176-191' into sorted list of ints."""
    cores = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return sorted(cores)


def partition_cores_for_gpus(gpu_ids, affinity_map):
    """Divide NUMA-local cores among GPUs sharing the same NUMA node.

    Returns dict: gpu_id -> (cores_list, n_threads) for the requested gpu_ids.
    GPUs sharing NUMA cores get an equal split.
    """
    if not affinity_map:
        return None
    # Group requested GPUs by their NUMA core set (as a frozenset key)
    from collections import defaultdict
    numa_groups = defaultdict(list)
    for gid in gpu_ids:
        if gid not in affinity_map:
            return None  # can't partition if any GPU is unknown
        key = tuple(affinity_map[gid])
        numa_groups[key].append(gid)

    result = {}
    for cores_tuple, gids in numa_groups.items():
        cores = list(cores_tuple)
        n_gpus = len(gids)
        chunk = len(cores) // n_gpus
        for i, gid in enumerate(sorted(gids)):
            my_cores = cores[i * chunk:(i + 1) * chunk] if i < n_gpus - 1 else cores[i * chunk:]
            result[gid] = (my_cores, len(my_cores))
    return result


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
             outlier_configs=None, outlier_indices=None, subspace_codebooks=None,
             direct_io=False):
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
        if direct_io:
            write_direct(fp16_path, raw)
        else:
            with open(fp16_path, "wb") as f:
                f.write(raw)
            fsync_file(fp16_path)
        t_fp16_write = time.time() - t0

        if not direct_io:
            drop_page_cache(fp16_path)
        t0 = time.time()
        if direct_io:
            _ = read_direct(fp16_path)
        else:
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
                direct_io=direct_io,
            )
            if not direct_io:
                fsync_file(fpath)
            t_write = time.time() - t0
            file_size = os.path.getsize(fpath)

            if not direct_io:
                drop_page_cache(fpath)

            # Read
            t0 = time.time()
            data = deserialize_compressed_kv(fpath, direct_io=direct_io)
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
                    direct_io=direct_io,
                )
                serialize_compressed_kv(
                    path_l, rot_l, subspace_codebooks[(d_low, bl)], bl,
                    idx_low, n_layers * 2, n_heads, n_tok, d_low, mode="mse",
                    direct_io=direct_io,
                )
                if not direct_io:
                    fsync_file(path_h)
                    fsync_file(path_l)
                t_write = time.time() - t0
                file_size = os.path.getsize(path_h) + os.path.getsize(path_l)

                if not direct_io:
                    drop_page_cache(path_h)
                    drop_page_cache(path_l)

                # Read both
                t0 = time.time()
                data_h = deserialize_compressed_kv(path_h, direct_io=direct_io)
                data_l = deserialize_compressed_kv(path_l, direct_io=direct_io)
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
                if os.path.exists(path_h):
                    os.remove(path_h)
                if os.path.exists(path_l):
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

def bench_concurrent(kv_cache, codebooks, bits, gpu_ids, output_dir, shared_dir,
                     direct_io=False, workers_per_gpu=1, core_map=None):
    """Launch parallel workers across GPUs for concurrent checkpoint/restore."""
    n_tok = kv_cache["n_tokens"]
    d = kv_cache["head_dim"]
    n_layers, n_heads = kv_cache["n_layers"], kv_cache["n_heads"]
    fp16_size = 2 * n_layers * n_heads * n_tok * d * 2
    n_gpus = len(gpu_ids)
    n_workers = n_gpus * workers_per_gpu

    print(f"\n{'=' * 100}")
    print(f"Benchmark 2: Concurrent Multi-GPU Checkpoint/Restore "
          f"({n_workers} workers on {n_gpus} GPUs, {n_tok:,} tokens, {bits}-bit)")
    print(f"{'=' * 100}")

    # Save KV cache for workers
    torch.save({
        "keys": kv_cache["keys"].cpu(), "values": kv_cache["values"].cpu(),
        "n_layers": n_layers, "n_heads": n_heads,
        "n_tokens": n_tok, "head_dim": d,
    }, os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

    concurrent_dir = os.path.join(output_dir, "concurrent")

    # Free GPU memory from main process before spawning workers
    gc.collect()
    torch.cuda.empty_cache()

    # Launch all workers
    t_wall_start = time.time()
    worker_results, worker_gpus = _run_concurrent_workers(
        gpu_ids, shared_dir, concurrent_dir, bits,
        direct_io=direct_io, workers_per_gpu=workers_per_gpu,
        core_map=core_map,
    )
    t_wall = time.time() - t_wall_start

    if not worker_results:
        print("  All workers failed!")
        return {}

    # Aggregate metrics
    total_compressed = sum(r["file_size"] for r in worker_results)
    max_write = max(r["t_write"] for r in worker_results)
    max_read = max(r["t_read"] for r in worker_results)
    avg_roundtrip = sum(r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
                        for r in worker_results) / len(worker_results)

    agg_wr_gbps = (total_compressed / 1e9) / max_write if max_write > 0 else 0
    agg_rd_gbps = (total_compressed / 1e9) / max_read if max_read > 0 else 0
    sessions_per_sec = len(worker_results) / t_wall

    # Per-worker table
    print(f"\n{'Worker':>7} {'GPU':>4} {'Compress':>10} {'Write':>10} {'Read':>10} {'Decomp':>10} {'Total':>10} {'MSE':>10}")
    print("-" * 82)
    for r in worker_results:
        wid = r['worker_id']
        total = r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
        print(f"{wid:>7} {worker_gpus[wid]:>4} "
              f"{r['t_compress']*1000:>8.0f}ms {r['t_write']*1000:>8.0f}ms "
              f"{r['t_read']*1000:>8.0f}ms {r['t_decompress']*1000:>8.0f}ms "
              f"{total*1000:>8.0f}ms {r['mse']:>10.4f}")

    # Summary
    compressed_size = worker_results[0]["file_size"]
    print(f"\n  Workers:             {len(worker_results)} ({workers_per_gpu} per GPU x {n_gpus} GPUs)")
    print(f"  Per-session FP16:    {fp16_size/1e6:.1f} MB")
    print(f"  Per-session {bits}-bit:  {compressed_size/1e6:.1f} MB ({fp16_size/compressed_size:.1f}x compression)")
    print(f"  Wall time:           {t_wall:.2f}s")
    print(f"  Aggregate write:     {agg_wr_gbps:.2f} GB/s ({total_compressed/1e6:.0f} MB in {max_write*1000:.0f} ms)")
    print(f"  Aggregate read:      {agg_rd_gbps:.2f} GB/s ({total_compressed/1e6:.0f} MB in {max_read*1000:.0f} ms)")
    print(f"  Avg round-trip:      {avg_roundtrip*1000:.0f} ms/session")
    print(f"  Throughput:          {sessions_per_sec:.1f} sessions/sec")

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
# Benchmark 5: TurboQuant ON vs OFF — Speedup & Scale-out
# ---------------------------------------------------------------------------

def _launch_workers(gpu_ids, shared_dir, output_dir, bits, fp16=False, gds=False,
                    direct_io=False, workers_per_gpu=1, phase="all",
                    core_map=None):
    """Launch parallel workers and collect results. Returns (results, worker_gpus).

    core_map: optional dict from gpu_id -> (cores_list, n_threads) for NUMA-aware pinning.
    """
    worker_script = str(Path(__file__).parent / "eval" / "flashblade_worker.py")
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    worker_gpus = []  # maps worker_id -> gpu_id
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Pin CPU threads to NUMA-local cores for this GPU
            taskset_prefix = []
            if core_map and gpu_id in core_map:
                cores, n_threads = core_map[gpu_id]
                n_thr = str(n_threads)
                env["OMP_NUM_THREADS"] = n_thr
                env["OPENBLAS_NUM_THREADS"] = n_thr
                env["MKL_NUM_THREADS"] = n_thr
                core_str = ",".join(str(c) for c in cores)
                taskset_prefix = ["taskset", "-c", core_str]

            cmd = [
                sys.executable, worker_script,
                "--shared-dir", shared_dir,
                "--output-dir", output_dir,
                "--worker-id", str(worker_id),
                "--bits", str(bits),
                "--phase", phase,
            ]
            if fp16:
                cmd.append("--fp16")
            if gds:
                cmd.append("--gds")
            if direct_io:
                cmd.append("--direct-io")
            proc = subprocess.Popen(taskset_prefix + cmd, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            processes.append((worker_id, gpu_id, proc))
            worker_gpus.append(gpu_id)
            worker_id += 1

    results = []
    for wid, gpu_id, proc in processes:
        proc.wait()
        result_path = os.path.join(shared_dir, f"bench_{wid}.json")
        if proc.returncode == 0 and os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            # Write-only phase doesn't produce bench results, that's OK
            if phase != "write":
                stdout = proc.stdout.read().decode()
                print(f"  Worker {wid} (GPU {gpu_id}) FAILED:\n{stdout}")

    return results, worker_gpus


def _run_concurrent_workers(gpu_ids, shared_dir, output_dir, bits, fp16=False, gds=False,
                            direct_io=False, workers_per_gpu=1, core_map=None):
    """Launch workers. Uses two-phase (write then read) when workers_per_gpu > 1 to avoid contention."""
    if workers_per_gpu > 1 and gds:
        # Two-phase: write all files first, then read all concurrently
        n_total = len(gpu_ids) * workers_per_gpu
        print(f"    Phase 1: writing {n_total} files...", end=" ", flush=True)
        t0 = time.time()
        _, worker_gpus = _launch_workers(
            gpu_ids, shared_dir, output_dir, bits, fp16=fp16, gds=gds,
            direct_io=direct_io, workers_per_gpu=workers_per_gpu, phase="write",
            core_map=core_map,
        )
        t_write_wall = time.time() - t0
        print(f"done ({t_write_wall:.1f}s)")
        print(f"    Phase 2: concurrent read ({n_total} workers)...", end=" ", flush=True)
        t0 = time.time()
        results, _ = _launch_workers(
            gpu_ids, shared_dir, output_dir, bits, fp16=fp16, gds=gds,
            direct_io=direct_io, workers_per_gpu=workers_per_gpu, phase="read",
            core_map=core_map,
        )
        t_read_wall = time.time() - t0
        print(f"done ({t_read_wall:.1f}s)")
        shutil.rmtree(output_dir, ignore_errors=True)
        # Attach read-phase wall time so bench_on_vs_off can use it
        for r in results:
            r["_read_wall"] = t_read_wall
        return results, worker_gpus
    else:
        results, worker_gpus = _launch_workers(
            gpu_ids, shared_dir, output_dir, bits, fp16=fp16, gds=gds,
            direct_io=direct_io, workers_per_gpu=workers_per_gpu, phase="all",
            core_map=core_map,
        )
        shutil.rmtree(output_dir, ignore_errors=True)
        return results, worker_gpus


def bench_on_vs_off(kv_cache, codebooks, bits, gpu_ids, output_dir, shared_dir, gds=False,
                    direct_io=False, workers_per_gpu=1, core_map=None):
    """Run FP16 baseline and TurboQuant concurrently, compare. With --gds, also runs GDS variants."""
    n_tok = kv_cache["n_tokens"]
    d = kv_cache["head_dim"]
    n_layers, n_heads = kv_cache["n_layers"], kv_cache["n_heads"]
    fp16_size = 2 * n_layers * n_heads * n_tok * d * 2
    n_gpus = len(gpu_ids)
    n_workers = n_gpus * workers_per_gpu

    title = f"Benchmark 5: TurboQuant ON vs OFF ({n_workers} workers on {n_gpus} GPUs, {n_tok:,} tokens, {bits}-bit)"
    if gds:
        title += " — with GPU Direct Storage"
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)

    # Save shared data for workers
    torch.save({
        "keys": kv_cache["keys"].cpu(), "values": kv_cache["values"].cpu(),
        "n_layers": n_layers, "n_heads": n_heads,
        "n_tokens": n_tok, "head_dim": d,
    }, os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

    def agg(results):
        total_bytes = sum(r["file_size"] for r in results)
        max_write = max(r["t_write"] for r in results)
        max_read = max(r["t_read"] for r in results)
        avg_checkpoint = sum(r["t_compress"] + r["t_write"] for r in results) / len(results)
        avg_restore = sum(r["t_read"] + r["t_decompress"] for r in results) / len(results)
        avg_roundtrip = sum(
            r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
            for r in results
        ) / len(results)
        per_session = results[0]["file_size"]
        # Use read-phase wall time if available (two-phase mode)
        read_wall = results[0].get("_read_wall")
        return {
            "n": len(results),
            "per_session_bytes": per_session,
            "total_bytes": total_bytes,
            "max_write": max_write,
            "max_read": read_wall if read_wall else max_read,
            "avg_checkpoint": avg_checkpoint,
            "avg_restore": avg_restore,
            "avg_roundtrip": avg_roundtrip,
            "agg_wr_gbps": (total_bytes / 1e9) / max_write if max_write > 0 else 0,
            "agg_rd_gbps": (total_bytes / 1e9) / max_read if max_read > 0 else 0,
        }

    # Free GPU memory from main process before spawning workers
    gc.collect()
    torch.cuda.empty_cache()

    # ── Collect all configurations ──
    configs = []

    print(f"\n  Running FP16 baseline ({n_workers} concurrent sessions)...")
    t0 = time.time()
    r, wg = _run_concurrent_workers(gpu_ids, shared_dir, os.path.join(output_dir, "onoff_fp16"), bits, fp16=True, direct_io=direct_io, workers_per_gpu=workers_per_gpu, core_map=core_map)
    configs.append(("FP16", r, time.time() - t0, wg))

    if gds:
        print(f"  Running FP16 + GDS ({n_workers} concurrent sessions)...")
        t0 = time.time()
        r, wg = _run_concurrent_workers(gpu_ids, shared_dir, os.path.join(output_dir, "onoff_fp16_gds"), bits, fp16=True, gds=True, workers_per_gpu=workers_per_gpu, core_map=core_map)
        wall = r[0].get("_read_wall", time.time() - t0) if r else time.time() - t0
        configs.append(("FP16+GDS", r, wall, wg))

    # Skip non-GDS TQ when running multiple workers per GPU — it's CPU-bottlenecked
    # and each worker loads the full KV cache to GPU, causing OOM. The 1-worker-per-GPU
    # results already show this path is ~50x slower than FP16.
    if workers_per_gpu <= 1:
        print(f"  Running TurboQuant {bits}-bit ({n_workers} concurrent sessions)...")
        t0 = time.time()
        r, wg = _run_concurrent_workers(gpu_ids, shared_dir, os.path.join(output_dir, "onoff_tq"), bits, fp16=False, direct_io=direct_io, workers_per_gpu=workers_per_gpu, core_map=core_map)
        configs.append((f"TQ {bits}-bit", r, time.time() - t0, wg))
    else:
        print(f"  Skipping TQ {bits}-bit without GDS (CPU-bottlenecked, tested at 1 worker/GPU)")

    if gds:
        print(f"  Running TurboQuant {bits}-bit + GDS ({n_workers} concurrent sessions)...")
        t0 = time.time()
        r, wg = _run_concurrent_workers(gpu_ids, shared_dir, os.path.join(output_dir, "onoff_tq_gds"), bits, fp16=False, gds=True, workers_per_gpu=workers_per_gpu, core_map=core_map)
        wall = r[0].get("_read_wall", time.time() - t0) if r else time.time() - t0
        configs.append((f"TQ {bits}-bit+GDS", r, wall, wg))

    # Validate all configs ran
    valid = [(name, results, wall, wg) for name, results, wall, wg in configs if results]
    if len(valid) < 2:
        print("  ERROR: Not enough configurations completed.")
        return {}

    # ── Per-worker detail table ──
    # Use worker_gpus from first config for GPU mapping
    first_wg = valid[0][3]

    if workers_per_gpu <= 2:
        # Show all workers individually
        print(f"\n  Per-worker timings (ms):")
        col_w = 14
        header = f"  {'W':>3} {'GPU':>4}"
        for name, _, _, _ in valid:
            header += f"  {('── ' + name + ' ──'):^{col_w * 3 + 4}}"
        print(header)
        sub = f"  {'':>3} {'':>4}"
        for _ in valid:
            sub += f"  {'Comp':>{col_w}} {'Write':>{col_w}} {'Read':>{col_w}}"
        print(sub)
        print("  " + "-" * (len(sub) - 2))

        for worker_idx in range(n_workers):
            line = f"  {worker_idx:>3} {first_wg[worker_idx]:>4}"
            for _, results, _, _ in valid:
                if worker_idx < len(results):
                    r = results[worker_idx]
                    comp = r["t_compress"] * 1000
                    wr = r["t_write"] * 1000
                    rd = (r["t_read"] + r["t_decompress"]) * 1000
                    line += f"  {comp:>{col_w - 2}.0f}ms {wr:>{col_w - 2}.0f}ms {rd:>{col_w - 2}.0f}ms"
                else:
                    line += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            print(line)
    else:
        # Show per-GPU averages (too many workers for individual rows)
        print(f"\n  Per-GPU average timings (ms) — {workers_per_gpu} workers per GPU:")
        col_w = 14
        header = f"  {'GPU':>4}"
        for name, _, _, _ in valid:
            header += f"  {('── ' + name + ' ──'):^{col_w * 3 + 4}}"
        print(header)
        sub = f"  {'':>4}"
        for _ in valid:
            sub += f"  {'Comp':>{col_w}} {'Write':>{col_w}} {'Read':>{col_w}}"
        print(sub)
        print("  " + "-" * (len(sub) - 2))

        for gpu_id in gpu_ids:
            line = f"  {gpu_id:>4}"
            for _, results, _, wg in valid:
                # Find workers on this GPU
                gpu_results = [r for r, g in zip(results, wg) if g == gpu_id]
                if gpu_results:
                    comp = sum(r["t_compress"] for r in gpu_results) / len(gpu_results) * 1000
                    wr = sum(r["t_write"] for r in gpu_results) / len(gpu_results) * 1000
                    rd = sum(r["t_read"] + r["t_decompress"] for r in gpu_results) / len(gpu_results) * 1000
                    line += f"  {comp:>{col_w - 2}.0f}ms {wr:>{col_w - 2}.0f}ms {rd:>{col_w - 2}.0f}ms"
                else:
                    line += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            print(line)

    # ── Comparison table ──
    aggs = {name: agg(results) for name, results, _, _ in valid}
    walls = {name: wall for name, _, wall, _ in valid}

    # Use first config (FP16) as baseline for speedup
    base_name = valid[0][0]
    base = aggs[base_name]

    print(f"\n  {'Metric':<40}", end="")
    for name in aggs:
        print(f" {name:>16}", end="")
    print()
    print("  " + "-" * (40 + 17 * len(aggs)))

    def row(label, key_fn, fmt="time"):
        vals = {name: key_fn(a) for name, a in aggs.items()}
        print(f"  {label:<40}", end="")
        for name in aggs:
            v = vals[name]
            if fmt == "time":
                print(f" {v*1000:>14.0f}ms", end="")
            elif fmt == "size":
                print(f" {v/1e6:>13.1f}MB", end="")
            elif fmt == "gbps":
                print(f" {v:>12.2f}GB/s", end="")
        # Speedup of last vs first
        base_v = key_fn(base)
        best_name = list(aggs.keys())[-1]
        best_v = vals[best_name]
        if fmt == "time" and best_v > 0:
            print(f"  ({base_v/best_v:.1f}x)", end="")
        elif fmt == "size" and best_v > 0:
            print(f"  ({base_v/best_v:.1f}x)", end="")
        print()

    row("Per-session size",          lambda a: a["per_session_bytes"], "size")
    row("Avg checkpoint (comp+wr)",  lambda a: a["avg_checkpoint"], "time")
    row("Avg restore (rd+decomp)",   lambda a: a["avg_restore"], "time")
    row("Avg full round-trip",       lambda a: a["avg_roundtrip"], "time")

    # Wall time row (not from agg)
    sessions_label = f"Wall time ({n_workers} sessions"
    if workers_per_gpu > 1:
        sessions_label += f", {workers_per_gpu}/GPU"
    sessions_label += ")"
    print(f"  {sessions_label:<40}", end="")
    for name in aggs:
        print(f" {walls[name]*1000:>14.0f}ms", end="")
    base_wall = walls[base_name]
    best_wall = walls[list(aggs.keys())[-1]]
    if best_wall > 0:
        print(f"  ({base_wall/best_wall:.1f}x)", end="")
    print()

    row("Agg write bandwidth",       lambda a: a["agg_wr_gbps"], "gbps")
    row("Agg read bandwidth",        lambda a: a["agg_rd_gbps"], "gbps")

    # ── Scale-out projections ──
    print(f"\n  {'── Scale-out: concurrent checkpoints at bandwidth budget ──':^90}")
    print("  " + "-" * 90)

    for bw_gbps in [5, 10, 15, 20]:
        bw = bw_gbps * 1e9
        print(f"  At {bw_gbps:>2} GB/s:", end="")
        session_counts = {}
        for name, a in aggs.items():
            bw_demand = a["per_session_bytes"] / a["avg_checkpoint"] if a["avg_checkpoint"] > 0 else 1
            sessions = bw / bw_demand
            session_counts[name] = sessions
            print(f"  {name}={sessions:>6.0f}", end="")
        # Gain of last vs first
        first = session_counts[base_name]
        last = session_counts[list(aggs.keys())[-1]]
        if first > 0:
            print(f"  ({last/first:.1f}x)", end="")
        print()

    # ── Capacity ──
    TB = 1e12
    print(f"\n  {'── Storage capacity ──':^90}")
    print("  " + "-" * 90)
    print(f"  Sessions per TB:", end="")
    first_per_tb = None
    last_per_tb = None
    for name, a in aggs.items():
        per_tb = TB / a["per_session_bytes"]
        if first_per_tb is None:
            first_per_tb = per_tb
        last_per_tb = per_tb
        print(f"  {name}={per_tb:>8,.0f}", end="")
    if first_per_tb and last_per_tb:
        print(f"  ({last_per_tb/first_per_tb:.1f}x)", end="")
    print()

    return {name: {"wall": walls[name], "agg": aggs[name]} for name in aggs}


# ---------------------------------------------------------------------------
# Benchmark 4: Session Migration (GPU → Disk → GPU)
# ---------------------------------------------------------------------------

def bench_migration(kv_cache, input_ids, ground_truth, codebooks, bits,
                    exempt_layers, output_dir, src_gpu, dst_gpu, n_generate, model_name,
                    direct_io=False):
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
        direct_io=direct_io,
    )
    if not direct_io:
        fsync_file(tqkv_path)
    t_write = time.time() - t0
    tqkv_size = os.path.getsize(tqkv_path)

    # Write exempt layer keys in FP16
    exempt_path = os.path.join(output_dir, "migration_exempt_keys.bin")
    t0 = time.time()
    if exempt_layers:
        exempt_data = kv_cache["keys"][sorted(exempt_layers)].cpu().half().numpy().tobytes()
        if direct_io:
            write_direct(exempt_path, exempt_data)
        else:
            with open(exempt_path, "wb") as f:
                f.write(exempt_data)
            fsync_file(exempt_path)
    t_write_exempt = time.time() - t0
    exempt_size = os.path.getsize(exempt_path) if exempt_layers else 0
    total_size = tqkv_size + exempt_size

    if not direct_io:
        drop_page_cache(tqkv_path)
        if exempt_layers:
            drop_page_cache(exempt_path)

    # ── Step 3: Read from disk ──
    t0 = time.time()
    data = deserialize_compressed_kv(tqkv_path, direct_io=direct_io)
    if exempt_layers:
        if direct_io:
            exempt_raw = read_direct(exempt_path)
        else:
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
    parser.add_argument("--gds", action="store_true", help="Enable GPU Direct Storage (requires kvikio)")
    parser.add_argument("--direct-io", action="store_true", help="Use O_DIRECT to bypass page cache for accurate I/O timing")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of concurrent workers per GPU (default: 1)")
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
    if args.workers_per_gpu > 1:
        print(f"  Workers/GPU:      {args.workers_per_gpu} ({len(gpu_ids) * args.workers_per_gpu} total)")
    if args.direct_io:
        print(f"  Direct I/O:       ON (O_DIRECT, bypasses page cache)")

    # Build NUMA-aware CPU affinity map for worker pinning
    affinity_map = get_gpu_cpu_affinity()
    core_map = partition_cores_for_gpus(gpu_ids, affinity_map) if affinity_map else None
    if core_map:
        sample_gid = gpu_ids[0]
        cores, n_thr = core_map[sample_gid]
        print(f"  CPU pinning:      {n_thr} threads/worker (NUMA-aware)")
    else:
        print(f"  CPU pinning:      disabled (could not parse GPU topology)")

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
        direct_io=args.direct_io,
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

    if args.workers_per_gpu <= 1:
        concurrent_results = bench_concurrent(
            concurrent_kv, codebooks, concurrent_bits, gpu_ids, output_dir, shared_dir,
            direct_io=args.direct_io, core_map=core_map,
        )
    else:
        print(f"\n  Skipping Benchmark 2 (CPU-mediated TQ) — use Benchmark 5 for multi-worker results")
        concurrent_results = {}

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
        direct_io=args.direct_io,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Benchmark 5: TurboQuant ON vs OFF
    # ══════════════════════════════════════════════════════════════════════════
    bench_on_vs_off(
        concurrent_kv, codebooks, concurrent_bits, gpu_ids, output_dir, shared_dir,
        gds=args.gds, direct_io=args.direct_io, workers_per_gpu=args.workers_per_gpu,
        core_map=core_map,
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
