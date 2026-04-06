#!/usr/bin/env python3
"""
50K token benchmark only — retry with aggressive memory management.
"""

import gc
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
    n = len(results)
    total_bytes = sum(r["file_size"] for r in results)
    max_read = max(r["t_read"] for r in results)
    avg_restore = sum(r["t_read"] + r["t_decompress"] for r in results) / n
    agg_rd = (total_bytes / 1e9) / max_read if max_read > 0 else 0
    return {
        "per_session_mb": results[0]["file_size"] / 1e6,
        "fp16_mb": results[0]["fp16_size"] / 1e6,
        "avg_restore_ms": avg_restore * 1000,
        "agg_rd_gbps": agg_rd,
    }


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    gpu_ids = list(range(8))
    bits = 3
    context_length = 50000

    output_dir = "./output/bench_blog_50k"
    os.makedirs(output_dir, exist_ok=True)
    shared_dir = tempfile.mkdtemp(prefix="tq_blog50k_")

    affinity_map = get_gpu_cpu_affinity()
    core_map = partition_cores_for_gpus(gpu_ids, affinity_map) if affinity_map else None

    print("=" * 100)
    print("50K Token Benchmark (8 NUMA-pinned GPUs)")
    print("=" * 100)

    # Codebooks
    codebooks = compute_all_codebooks(DEFAULT_DIM, [bits])
    if bits - 1 not in codebooks:
        codebooks[bits - 1] = lloyd_max_codebook(DEFAULT_DIM, bits - 1)

    # Load model on GPU 0 with minimal footprint
    device = f"cuda:{gpu_ids[0]}"
    torch.cuda.empty_cache()
    gc.collect()

    # Check free memory
    free_mem = torch.cuda.mem_get_info(gpu_ids[0])
    print(f"  GPU 0 free memory: {free_mem[0]/1e9:.1f} GB / {free_mem[1]/1e9:.1f} GB")

    print(f"Loading model {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    free_mem = torch.cuda.mem_get_info(gpu_ids[0])
    print(f"  After model load: {free_mem[0]/1e9:.1f} GB free")

    print(f"Capturing KV at {context_length:,} tokens...", end=" ", flush=True)
    t0 = time.time()
    try:
        kv, _ = capture_kv(model, tokenizer, device, min_tokens=context_length)
        actual = kv["n_tokens"]
        fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * actual * kv["head_dim"] * 2 / 1e6
        print(f"done ({actual:,} tokens, FP16={fp16_mb:.0f} MB, {time.time()-t0:.1f}s)")
    except torch.cuda.OutOfMemoryError:
        print(f"\nOOM at 50K. Trying 40K...")
        torch.cuda.empty_cache()
        gc.collect()
        context_length = 40000
        try:
            kv, _ = capture_kv(model, tokenizer, device, min_tokens=context_length)
            actual = kv["n_tokens"]
            fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * actual * kv["head_dim"] * 2 / 1e6
            print(f"  Captured {actual:,} tokens, FP16={fp16_mb:.0f} MB")
        except torch.cuda.OutOfMemoryError:
            print("OOM at 40K too. Aborting.")
            return

    # Move to CPU, free model
    kv["keys"] = kv["keys"].cpu()
    kv["values"] = kv["values"].cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save shared data
    actual = kv["n_tokens"]
    torch.save({
        "keys": kv["keys"], "values": kv["values"],
        "n_layers": kv["n_layers"], "n_heads": kv["n_heads"],
        "n_tokens": actual, "head_dim": kv["head_dim"],
    }, os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

    configs = [
        ("FP16, standard I/O",      dict(fp16=True,  gds=False, direct_io=True)),
        ("FP16 + GDS",              dict(fp16=True,  gds=True,  direct_io=False)),
        ("TQ 3-bit, standard I/O",  dict(fp16=False, gds=False, direct_io=True)),
        ("TQ 3-bit + GDS",         dict(fp16=False, gds=True,  direct_io=False)),
    ]

    print(f"\nRunning benchmarks ({actual:,} tokens)...")
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    for cfg_name, cfg_kwargs in configs:
        r = run_config(cfg_name, gpu_ids, shared_dir,
                       os.path.join(output_dir, f"{actual}tok"),
                       bits, core_map, **cfg_kwargs)
        if r:
            results[cfg_name] = agg(r)

    print(f"\n{'=' * 100}")
    print(f"RESULTS: {actual:,} tokens")
    print(f"{'=' * 100}")
    print(f"{'Config':<30} {'Restore':>12} {'Read BW':>12} {'Size':>12}")
    print("-" * 68)
    for name, a in results.items():
        sz = f"{a['per_session_mb']:.1f} MB" if a['per_session_mb'] < 1000 else f"{a['per_session_mb']/1000:.2f} GB"
        print(f"{name:<30} {a['avg_restore_ms']:>9.0f} ms {a['agg_rd_gbps']:>8.2f} GB/s {sz:>12}")

    if "FP16, standard I/O" in results and "TQ 3-bit + GDS" in results:
        fp16 = results["FP16, standard I/O"]["avg_restore_ms"]
        tqgds = results["TQ 3-bit + GDS"]["avg_restore_ms"]
        print(f"\n  TQ+GDS restore speedup: {fp16:.0f} ms → {tqgds:.0f} ms = {fp16/tqgds:.1f}x faster")

    shutil.rmtree(shared_dir, ignore_errors=True)
    print("Done!")


if __name__ == "__main__":
    main()
