#!/usr/bin/env python3
"""
Focused re-run of Benchmark 2 (concurrent NUMA-pinned checkpoint/restore)
with O_DIRECT to bypass page cache. Compares standard I/O vs O_DIRECT.
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
    bench_concurrent,
)


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    gpu_ids = list(range(8))
    context_length = 10000
    bits = 3

    output_dir = "./output/bench_direct_io"
    os.makedirs(output_dir, exist_ok=True)
    shared_dir = tempfile.mkdtemp(prefix="tq_direct_io_")

    # NUMA topology
    affinity_map = get_gpu_cpu_affinity()
    core_map = partition_cores_for_gpus(gpu_ids, affinity_map) if affinity_map else None

    print("=" * 100)
    print("Benchmark 2 Re-run: O_DIRECT vs Standard I/O (NUMA-pinned)")
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
    fp16_mb = 2 * kv["n_layers"] * kv["n_heads"] * kv["n_tokens"] * kv["head_dim"] * 2 / 1e6
    print(f"done ({kv['n_tokens']} tokens, FP16={fp16_mb:.0f} MB, {time.time()-t0:.1f}s)")

    # Move KV to CPU, free model
    kv["keys"] = kv["keys"].cpu()
    kv["values"] = kv["values"].cpu()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ── Run 1: Standard I/O (no O_DIRECT) ──
    print("\n" + "=" * 100)
    print("RUN 1: Standard I/O (buffered, with fsync + POSIX_FADV_DONTNEED)")
    print("=" * 100)
    shared_dir_1 = tempfile.mkdtemp(prefix="tq_run1_")
    results_std = bench_concurrent(
        kv, codebooks, bits, gpu_ids, output_dir, shared_dir_1,
        direct_io=False, core_map=core_map,
    )
    shutil.rmtree(shared_dir_1, ignore_errors=True)

    # ── Run 2: O_DIRECT ──
    print("\n" + "=" * 100)
    print("RUN 2: O_DIRECT (bypasses page cache entirely)")
    print("=" * 100)
    shared_dir_2 = tempfile.mkdtemp(prefix="tq_run2_")
    results_direct = bench_concurrent(
        kv, codebooks, bits, gpu_ids, output_dir, shared_dir_2,
        direct_io=True, core_map=core_map,
    )
    shutil.rmtree(shared_dir_2, ignore_errors=True)

    # ── Comparison ──
    print("\n" + "=" * 100)
    print("COMPARISON: Standard I/O vs O_DIRECT")
    print("=" * 100)
    for label, r in [("Standard I/O", results_std), ("O_DIRECT", results_direct)]:
        if r:
            print(f"\n  {label}:")
            print(f"    Wall time:        {r['wall_time']:.2f}s")
            print(f"    Aggregate write:  {r['agg_write_gbps']:.2f} GB/s")
            print(f"    Aggregate read:   {r['agg_read_gbps']:.2f} GB/s")
            print(f"    Sessions/sec:     {r['sessions_per_sec']:.1f}")

    # Cleanup
    shutil.rmtree(shared_dir, ignore_errors=True)
    print("\nDone!")


if __name__ == "__main__":
    main()
