#!/usr/bin/env python3
"""
Generate blog-ready tables: all 4 configs x 3 context lengths.
Runs 8 NUMA-pinned workers per config, reports restore time and aggregate read bandwidth.
"""

import gc
import json
import os
import shutil
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

import torch

from turboquant import compute_all_codebooks, DEFAULT_DIM
from turboquant.core import lloyd_max_codebook
from bench_flashblade import (
    get_gpu_cpu_affinity, partition_cores_for_gpus, capture_kv,
    _run_concurrent_workers,
)


def run_config(label, gpu_ids, shared_dir, output_dir, bits, core_map,
               fp16=False, gds=False, direct_io=False):
    """Run one config, return per-worker results."""
    tag = label.replace(" ", "_").replace("+", "_").replace(",", "")
    cfg_dir = os.path.join(output_dir, tag)
    print(f"    {label}...", end=" ", flush=True)
    t0 = time.time()
    results, worker_gpus = _run_concurrent_workers(
        gpu_ids, shared_dir, cfg_dir, bits,
        fp16=fp16, gds=gds, direct_io=direct_io,
        workers_per_gpu=1, core_map=core_map,
    )
    wall = time.time() - t0
    print(f"done ({wall:.1f}s)")
    for r in results:
        r["_wall"] = wall
        r["_gpu"] = worker_gpus[r["worker_id"]]
    return results


def agg(results):
    """Aggregate stats from worker results."""
    n = len(results)
    total_bytes = sum(r["file_size"] for r in results)
    max_read = max(r["t_read"] for r in results)
    avg_restore = sum(r["t_read"] + r["t_decompress"] for r in results) / n
    avg_checkpoint = sum(r["t_compress"] + r["t_write"] for r in results) / n
    avg_roundtrip = avg_checkpoint + avg_restore
    agg_rd = (total_bytes / 1e9) / max_read if max_read > 0 else 0
    return {
        "per_session_mb": results[0]["file_size"] / 1e6,
        "fp16_mb": results[0]["fp16_size"] / 1e6,
        "avg_restore_ms": avg_restore * 1000,
        "avg_checkpoint_ms": avg_checkpoint * 1000,
        "avg_roundtrip_ms": avg_roundtrip * 1000,
        "agg_rd_gbps": agg_rd,
        "wall": results[0]["_wall"],
    }


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    gpu_ids = list(range(8))
    context_lengths = [10000, 30000, 50000]
    bits = 3

    output_dir = "./output/bench_blog"
    os.makedirs(output_dir, exist_ok=True)
    shared_dir = tempfile.mkdtemp(prefix="tq_blog_")

    affinity_map = get_gpu_cpu_affinity()
    core_map = partition_cores_for_gpus(gpu_ids, affinity_map) if affinity_map else None

    print("=" * 100)
    print("Blog Tables: 4 configs x 3 context lengths (8 NUMA-pinned GPUs)")
    print("=" * 100)
    if core_map:
        sample = core_map[gpu_ids[0]]
        print(f"  CPU pinning: {sample[1]} threads/worker")

    # Codebooks
    print("\nComputing codebooks...")
    codebooks = compute_all_codebooks(DEFAULT_DIM, [bits])
    if bits - 1 not in codebooks:
        codebooks[bits - 1] = lloyd_max_codebook(DEFAULT_DIM, bits - 1)

    # Load model
    setup_gpu = gpu_ids[0]
    device = f"cuda:{setup_gpu}"

    print(f"Loading model {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Capture KV caches
    kv_caches = {}
    for n_tok in context_lengths:
        print(f"  Capturing KV at {n_tok:,} tokens...", end=" ", flush=True)
        t0 = time.time()
        try:
            kv, _ = capture_kv(model, tokenizer, device, min_tokens=n_tok)
            actual = kv["n_tokens"]
            kv_caches[actual] = kv
            fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * actual * kv["head_dim"] * 2 / 1e6
            print(f"done ({actual:,} tokens, FP16={fp16_mb:.0f} MB, {time.time()-t0:.1f}s)")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM! Skipping.")
            torch.cuda.empty_cache()

    # Move to CPU, free model
    for kv in kv_caches.values():
        kv["keys"] = kv["keys"].cpu()
        kv["values"] = kv["values"].cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── Run all configs at all context lengths ──
    configs = [
        ("FP16, standard I/O",   dict(fp16=True,  gds=False, direct_io=True)),
        ("FP16 + GDS",           dict(fp16=True,  gds=True,  direct_io=False)),
        ("TQ 3-bit, standard I/O", dict(fp16=False, gds=False, direct_io=True)),
        ("TQ 3-bit + GDS",      dict(fp16=False, gds=True,  direct_io=False)),
    ]

    # all_results[context_len][config_name] = agg dict
    all_results = {}

    for actual_tok in sorted(kv_caches.keys()):
        kv = kv_caches[actual_tok]
        print(f"\n{'=' * 80}")
        print(f"  {actual_tok:,} tokens")
        print(f"{'=' * 80}")

        # Save shared data
        torch.save({
            "keys": kv["keys"], "values": kv["values"],
            "n_layers": kv["n_layers"], "n_heads": kv["n_heads"],
            "n_tokens": actual_tok, "head_dim": kv["head_dim"],
        }, os.path.join(shared_dir, "kv_cache.pt"))
        torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

        gc.collect()
        torch.cuda.empty_cache()

        all_results[actual_tok] = {}
        for cfg_name, cfg_kwargs in configs:
            results = run_config(
                cfg_name, gpu_ids, shared_dir,
                os.path.join(output_dir, f"{actual_tok}tok"),
                bits, core_map, **cfg_kwargs,
            )
            if results:
                all_results[actual_tok][cfg_name] = agg(results)

    # ══════════════════════════════════════════════════════════════════════════
    # Print blog-ready tables
    # ══════════════════════════════════════════════════════════════════════════
    cfg_names = [c[0] for c in configs]

    # Table 1: Avg Restore Time
    print(f"\n{'=' * 100}")
    print("TABLE 1: Average Restore Time (8 concurrent GPUs)")
    print("=" * 100)
    print(f"{'Context':>10} │", end="")
    for cfg in cfg_names:
        print(f" {cfg:>25}", end="")
    print()
    print("─" * (12 + 26 * len(cfg_names)))
    for tok in sorted(all_results.keys()):
        print(f"{tok:>9,} │", end="")
        for cfg in cfg_names:
            if cfg in all_results[tok]:
                v = all_results[tok][cfg]["avg_restore_ms"]
                print(f" {v:>22.0f} ms", end="")
            else:
                print(f" {'—':>25}", end="")
        print()

    # Table 2: Aggregate Read Bandwidth
    print(f"\n{'=' * 100}")
    print("TABLE 2: Aggregate Read Bandwidth (8 concurrent GPUs)")
    print("=" * 100)
    print(f"{'Context':>10} │", end="")
    for cfg in cfg_names:
        print(f" {cfg:>25}", end="")
    print()
    print("─" * (12 + 26 * len(cfg_names)))
    for tok in sorted(all_results.keys()):
        print(f"{tok:>9,} │", end="")
        for cfg in cfg_names:
            if cfg in all_results[tok]:
                v = all_results[tok][cfg]["agg_rd_gbps"]
                print(f" {v:>21.2f} GB/s", end="")
            else:
                print(f" {'—':>25}", end="")
        print()

    # Table 3: Per-session size (just once, same across context lengths)
    print(f"\n{'=' * 100}")
    print("TABLE 3: Per-Session Size")
    print("=" * 100)
    print(f"{'Context':>10} │", end="")
    for cfg in cfg_names:
        print(f" {cfg:>25}", end="")
    print()
    print("─" * (12 + 26 * len(cfg_names)))
    for tok in sorted(all_results.keys()):
        print(f"{tok:>9,} │", end="")
        for cfg in cfg_names:
            if cfg in all_results[tok]:
                v = all_results[tok][cfg]["per_session_mb"]
                if v >= 1000:
                    print(f" {v/1000:>21.2f} GB", end="")
                else:
                    print(f" {v:>21.1f} MB", end="")
            else:
                print(f" {'—':>25}", end="")
        print()

    # Speedup summary
    print(f"\n{'=' * 100}")
    print("SPEEDUP: TQ+GDS restore vs FP16 standard I/O restore")
    print("=" * 100)
    for tok in sorted(all_results.keys()):
        fp16 = all_results[tok].get("FP16, standard I/O", {}).get("avg_restore_ms", 0)
        tqgds = all_results[tok].get("TQ 3-bit + GDS", {}).get("avg_restore_ms", 0)
        if fp16 > 0 and tqgds > 0:
            print(f"  {tok:>6,} tokens: {fp16:.0f} ms → {tqgds:.0f} ms = {fp16/tqgds:.1f}x faster")

    shutil.rmtree(shared_dir, ignore_errors=True)
    print(f"\n{'=' * 100}")
    print("Done!")
    print("=" * 100)


if __name__ == "__main__":
    main()
