#!/usr/bin/env python3
"""
30K token benchmark: O_DIRECT vs GDS side-by-side comparison.
Runs 8 NUMA-pinned workers (one per GPU) at 30,000 tokens, 3-bit.
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
    _launch_workers, _run_concurrent_workers,
)


def run_config(label, gpu_ids, shared_dir, output_dir, bits, core_map,
               fp16=False, gds=False, direct_io=False):
    """Run one configuration and return list of per-worker result dicts."""
    tag = label.replace(" ", "_").replace("+", "_")
    cfg_dir = os.path.join(output_dir, tag)

    print(f"\n  Running {label} (8 concurrent workers)...", end=" ", flush=True)
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


def print_worker_table(label, results):
    """Print per-worker breakdown table."""
    print(f"\n  {label}")
    print(f"  {'Worker':>7} {'GPU':>4} {'Compress':>10} {'Write':>10} "
          f"{'Read':>10} {'Decomp':>10} {'Total':>10}")
    print("  " + "-" * 72)
    for r in sorted(results, key=lambda x: x["worker_id"]):
        total = r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]
        print(f"  {r['worker_id']:>7} {r['_gpu']:>4} "
              f"{r['t_compress']*1000:>8.0f}ms {r['t_write']*1000:>8.0f}ms "
              f"{r['t_read']*1000:>8.0f}ms {r['t_decompress']*1000:>8.0f}ms "
              f"{total*1000:>8.0f}ms")


def per_gpu_avg(results):
    """Compute per-GPU average timings."""
    avgs = {}
    for r in results:
        gpu = r["_gpu"]
        avgs[gpu] = {
            "compress": r["t_compress"] * 1000,
            "write": r["t_write"] * 1000,
            "read": r["t_read"] * 1000,
            "decompress": r["t_decompress"] * 1000,
            "total": (r["t_compress"] + r["t_write"] + r["t_read"] + r["t_decompress"]) * 1000,
        }
    return avgs


def agg_stats(results):
    """Compute aggregate statistics."""
    n = len(results)
    total_bytes = sum(r["file_size"] for r in results)
    max_write = max(r["t_write"] for r in results)
    max_read = max(r["t_read"] for r in results)
    avg_compress = sum(r["t_compress"] for r in results) / n * 1000
    avg_write = sum(r["t_write"] for r in results) / n * 1000
    avg_read = sum(r["t_read"] for r in results) / n * 1000
    avg_decomp = sum(r["t_decompress"] for r in results) / n * 1000
    avg_total = avg_compress + avg_write + avg_read + avg_decomp
    return {
        "n": n,
        "per_session_mb": results[0]["file_size"] / 1e6,
        "fp16_mb": results[0]["fp16_size"] / 1e6,
        "agg_wr_gbps": (total_bytes / 1e9) / max_write if max_write > 0 else 0,
        "agg_rd_gbps": (total_bytes / 1e9) / max_read if max_read > 0 else 0,
        "avg_compress": avg_compress,
        "avg_write": avg_write,
        "avg_read": avg_read,
        "avg_decomp": avg_decomp,
        "avg_total": avg_total,
        "wall": results[0]["_wall"],
        "cos_sim": sum(r["cos_sim"] for r in results) / n,
    }


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    gpu_ids = list(range(8))
    context_length = 30000
    bits = 3

    output_dir = "./output/bench_30k_comparison"
    os.makedirs(output_dir, exist_ok=True)
    shared_dir = tempfile.mkdtemp(prefix="tq_30k_")

    # NUMA topology
    affinity_map = get_gpu_cpu_affinity()
    core_map = partition_cores_for_gpus(gpu_ids, affinity_map) if affinity_map else None

    print("=" * 100)
    print("30K Token Benchmark: O_DIRECT vs GDS (NUMA-pinned, 8 GPUs)")
    print("=" * 100)
    print(f"  Model:       {model_name}")
    print(f"  GPUs:        {gpu_ids}")
    print(f"  Context:     {context_length:,} tokens")
    print(f"  Bits:        {bits}")
    if core_map:
        sample = core_map[gpu_ids[0]]
        print(f"  CPU pinning: {sample[1]} threads/worker (NUMA-aware)")

    # Codebooks
    print("\nComputing codebooks...")
    codebooks = compute_all_codebooks(DEFAULT_DIM, [bits])
    if bits - 1 not in codebooks:
        codebooks[bits - 1] = lloyd_max_codebook(DEFAULT_DIM, bits - 1)

    # Load model & capture KV
    setup_gpu = gpu_ids[0]
    device = f"cuda:{setup_gpu}"

    print(f"Loading model {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    print(f"Capturing KV cache at {context_length:,} tokens...", end=" ", flush=True)
    t0 = time.time()
    kv, _ = capture_kv(model, tokenizer, device, min_tokens=context_length)
    n_tok = kv["n_tokens"]
    fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * n_tok * kv["head_dim"] * 2 / 1e6
    print(f"done ({n_tok:,} tokens, FP16={fp16_mb:.0f} MB, {time.time()-t0:.1f}s)")

    # Move KV to CPU, free model
    kv["keys"] = kv["keys"].cpu()
    kv["values"] = kv["values"].cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Save shared data for workers
    torch.save({
        "keys": kv["keys"], "values": kv["values"],
        "n_layers": kv["n_layers"], "n_heads": kv["n_heads"],
        "n_tokens": n_tok, "head_dim": kv["head_dim"],
    }, os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))

    # ══════════════════════════════════════════════════════════════════════════
    # Run all configurations
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"Running benchmarks ({n_tok:,} tokens, {bits}-bit, 8 workers)")
    print(f"{'=' * 100}")

    r_direct = run_config("TQ 3-bit + O_DIRECT", gpu_ids, shared_dir, output_dir, bits,
                          core_map, direct_io=True)
    r_gds = run_config("TQ 3-bit + GDS", gpu_ids, shared_dir, output_dir, bits,
                       core_map, gds=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════════════

    # Per-worker tables
    print(f"\n{'=' * 100}")
    print(f"RESULTS: {n_tok:,} tokens, {bits}-bit, 8 NUMA-pinned workers")
    print(f"{'=' * 100}")

    print_worker_table("── O_DIRECT (CPU-mediated) ──", r_direct)
    print_worker_table("── GDS (GPU Direct Storage) ──", r_gds)

    # Aggregate per-GPU average
    avg_direct = per_gpu_avg(r_direct)
    avg_gds = per_gpu_avg(r_gds)

    print(f"\n  ── Per-GPU Average Comparison ──")
    print(f"  {'GPU':>4} │ {'── O_DIRECT ──':^42} │ {'── GDS ──':^42} │ {'Speedup':>8}")
    print(f"  {'':>4} │ {'Comp':>8} {'Write':>8} {'Read':>8} {'Decomp':>8} {'Total':>8}"
          f" │ {'Comp':>8} {'Write':>8} {'Read':>8} {'Decomp':>8} {'Total':>8}"
          f" │ {'':>8}")
    print("  " + "─" * 103)
    for gpu in sorted(avg_direct.keys()):
        d = avg_direct[gpu]
        g = avg_gds[gpu]
        speedup = d["total"] / g["total"] if g["total"] > 0 else 0
        print(f"  {gpu:>4} │ {d['compress']:>6.0f}ms {d['write']:>6.0f}ms "
              f"{d['read']:>6.0f}ms {d['decompress']:>6.0f}ms {d['total']:>6.0f}ms"
              f" │ {g['compress']:>6.0f}ms {g['write']:>6.0f}ms "
              f"{g['read']:>6.0f}ms {g['decompress']:>6.0f}ms {g['total']:>6.0f}ms"
              f" │ {speedup:>6.1f}x")

    # Grand averages
    all_d = list(avg_direct.values())
    all_g = list(avg_gds.values())
    avg_d_total = sum(v["total"] for v in all_d) / len(all_d)
    avg_g_total = sum(v["total"] for v in all_g) / len(all_g)
    avg_d_comp = sum(v["compress"] for v in all_d) / len(all_d)
    avg_g_comp = sum(v["compress"] for v in all_g) / len(all_g)
    avg_d_wr = sum(v["write"] for v in all_d) / len(all_d)
    avg_g_wr = sum(v["write"] for v in all_g) / len(all_g)
    avg_d_rd = sum(v["read"] for v in all_d) / len(all_d)
    avg_g_rd = sum(v["read"] for v in all_g) / len(all_g)
    avg_d_dc = sum(v["decompress"] for v in all_d) / len(all_d)
    avg_g_dc = sum(v["decompress"] for v in all_g) / len(all_g)
    print("  " + "─" * 103)
    print(f"  {'AVG':>4} │ {avg_d_comp:>6.0f}ms {avg_d_wr:>6.0f}ms "
          f"{avg_d_rd:>6.0f}ms {avg_d_dc:>6.0f}ms {avg_d_total:>6.0f}ms"
          f" │ {avg_g_comp:>6.0f}ms {avg_g_wr:>6.0f}ms "
          f"{avg_g_rd:>6.0f}ms {avg_g_dc:>6.0f}ms {avg_g_total:>6.0f}ms"
          f" │ {avg_d_total/avg_g_total:>6.1f}x")

    # Summary stats
    s_direct = agg_stats(r_direct)
    s_gds = agg_stats(r_gds)

    print(f"\n  ── Aggregate Summary ──")
    print(f"  {'Metric':<35} {'O_DIRECT':>15} {'GDS':>15} {'Speedup':>10}")
    print("  " + "-" * 78)
    print(f"  {'Per-session FP16 size':<35} {s_direct['fp16_mb']:>12.1f} MB {s_gds['fp16_mb']:>12.1f} MB {'':>10}")
    print(f"  {'Per-session compressed':<35} {s_direct['per_session_mb']:>12.1f} MB {s_gds['per_session_mb']:>12.1f} MB {'':>10}")
    print(f"  {'Avg compress':<35} {s_direct['avg_compress']:>11.0f} ms {s_gds['avg_compress']:>11.0f} ms"
          f" {s_direct['avg_compress']/s_gds['avg_compress']:>8.1f}x" if s_gds['avg_compress'] > 0 else "")
    print(f"  {'Avg write':<35} {s_direct['avg_write']:>11.0f} ms {s_gds['avg_write']:>11.0f} ms"
          f" {s_direct['avg_write']/s_gds['avg_write']:>8.1f}x" if s_gds['avg_write'] > 0 else "")
    print(f"  {'Avg read':<35} {s_direct['avg_read']:>11.0f} ms {s_gds['avg_read']:>11.0f} ms"
          f" {s_direct['avg_read']/s_gds['avg_read']:>8.1f}x" if s_gds['avg_read'] > 0 else "")
    print(f"  {'Avg decompress':<35} {s_direct['avg_decomp']:>11.0f} ms {s_gds['avg_decomp']:>11.0f} ms"
          f" {s_direct['avg_decomp']/s_gds['avg_decomp']:>8.1f}x" if s_gds['avg_decomp'] > 0 else "")
    print(f"  {'Avg total round-trip':<35} {s_direct['avg_total']:>11.0f} ms {s_gds['avg_total']:>11.0f} ms"
          f" {s_direct['avg_total']/s_gds['avg_total']:>8.1f}x" if s_gds['avg_total'] > 0 else "")
    print(f"  {'Aggregate write bandwidth':<35} {s_direct['agg_wr_gbps']:>10.2f} GB/s {s_gds['agg_wr_gbps']:>10.2f} GB/s {'':>10}")
    print(f"  {'Aggregate read bandwidth':<35} {s_direct['agg_rd_gbps']:>10.2f} GB/s {s_gds['agg_rd_gbps']:>10.2f} GB/s {'':>10}")
    print(f"  {'Wall time (8 sessions)':<35} {s_direct['wall']:>11.1f} s {s_gds['wall']:>11.1f} s"
          f" {s_direct['wall']/s_gds['wall']:>8.1f}x" if s_gds['wall'] > 0 else "")

    if s_gds.get('cos_sim', 0) > 0:
        print(f"  {'Cosine similarity':<35} {s_direct['cos_sim']:>15.6f} {s_gds['cos_sim']:>15.6f}")

    # Cleanup
    shutil.rmtree(shared_dir, ignore_errors=True)
    print(f"\n{'=' * 100}")
    print("Done!")
    print("=" * 100)


if __name__ == "__main__":
    main()
