#!/usr/bin/env python3
"""
Parallel TurboQuant evaluation across multiple GPUs.

Distributes evaluation workloads across available GPUs using subprocesses
with CUDA_VISIBLE_DEVICES isolation. Each worker loads its own model copy.

Phase 1: Coordinator loads model, captures KV cache, generates ground truth
Phase 2: Workers run evaluation tasks in parallel on separate GPUs
Phase 3: Coordinator collects results and prints unified tables

Usage:
  python run_parallel.py --model Qwen/Qwen2.5-7B-Instruct --eval-top1
  python run_parallel.py --model Qwen/Qwen2.5-7B-Instruct --eval-top1 --min-tokens 1000
  python run_parallel.py --model Qwen/Qwen2.5-7B-Instruct --eval-top1 --residual-window 128
  python run_parallel.py --model Qwen/Qwen2.5-7B-Instruct --eval-top1 --gpus 0,1,2,4,5,6,7
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

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

import torch

from turboquant import compute_all_codebooks, DEFAULT_DIM, BIT_WIDTHS
from turboquant.core import (
    lloyd_max_codebook,
    identify_outlier_channels,
    identify_outlier_layers,
    OUTLIER_CONFIGS,
)


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def get_free_gpus(min_free_mb=20000):
    """Return sorted list of GPU indices with at least min_free_mb free memory."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, free = line.split(",")
        idx, free = int(idx.strip()), int(free.strip())
        if free >= min_free_mb:
            gpus.append(idx)
    return sorted(gpus)


# ---------------------------------------------------------------------------
# Task definition and distribution
# ---------------------------------------------------------------------------

def define_tasks(eval_top1, residual_window):
    """Create the list of atomic evaluation tasks."""
    tasks = []
    for bits in BIT_WIDTHS:
        tasks.append({"type": "mse", "bits": bits})
    for bits in BIT_WIDTHS:
        tasks.append({"type": "prod", "bits": bits})
    for i in range(len(OUTLIER_CONFIGS)):
        tasks.append({"type": "outlier", "config_idx": i})
    if eval_top1:
        tasks.append({"type": "topk", "bits_list": [2, 3, 4, 5]})
    if residual_window > 0 and eval_top1:
        tasks.append({"type": "window", "bits_list": [2, 3, 4], "window": residual_window})
    return tasks


def _task_cost(task, eval_top1, has_exempt):
    """Estimate relative cost (units ~ eval_topk_match calls)."""
    if not eval_top1:
        return 1
    if task["type"] in ("mse", "prod"):
        return 2 if has_exempt else 1
    if task["type"] == "outlier":
        return 1
    if task["type"] == "topk":
        return len(task["bits_list"])
    if task["type"] == "window":
        return len(task["bits_list"])
    return 1


def distribute_tasks(tasks, n_gpus, eval_top1, has_exempt):
    """Greedy load-balanced assignment of tasks to GPUs."""
    ordered = sorted(tasks, key=lambda t: _task_cost(t, eval_top1, has_exempt), reverse=True)
    loads = [0] * n_gpus
    groups = [[] for _ in range(n_gpus)]
    for task in ordered:
        i = min(range(n_gpus), key=lambda i: loads[i])
        groups[i].append(task)
        loads[i] += _task_cost(task, eval_top1, has_exempt)
    return groups, loads


# ---------------------------------------------------------------------------
# Result printing (mirrors gpu_eval.py output format)
# ---------------------------------------------------------------------------

def print_results(R, meta, eval_top1, has_exempt, n_generate, residual_window):
    """Reconstruct and print the standard output tables from worker results."""
    d = meta["head_dim"]
    n_tokens = meta["n_tokens"]
    fp16 = 2 * d

    # ── MSE ──
    print("\n" + "=" * 80)
    print("TurboQuant_MSE Results")
    print("=" * 80)
    hdr = f"{'Bits':>5} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Bytes/Vec':>12} {'Compress':>10}"
    if eval_top1:
        hdr += f" {'Top-1':>8}"
        if has_exempt:
            hdr += f" {'Top-1+LE':>9}"
    print(hdr)
    print("-" * len(hdr))

    mse_r = {}
    for bits in BIT_WIDTHS:
        k = f"mse_{bits}"
        if k not in R:
            continue
        r, m = R[k], R[k]["metrics"]
        mse_r[bits] = dict(m)
        bpv = bits * d / 8 + 4
        row = (f"{bits:>5} {m['mse']:>12.6f} {m['cosine_similarity']:>12.6f} "
               f"{m['inner_product_correlation']:>12.6f} {bpv:>12.1f} {fp16/bpv:>9.1f}x")
        if eval_top1 and "top1" in r:
            mse_r[bits]["top1_match_rate"] = r["top1"]
            row += f" {r['top1']:>7.1%}"
            if "top1_le" in r:
                mse_r[bits]["top1_layer_exempt"] = r["top1_le"]
                row += f" {r['top1_le']:>8.1%}"
        print(row)

    # ── Prod ──
    print("\n" + "=" * 80)
    print("TurboQuant_prod (MSE + QJL) Results")
    print("=" * 80)
    hdr = f"{'Bits':>5} {'MSE(b-1)':>8} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Bytes/Vec':>12} {'Compress':>10}"
    if eval_top1:
        hdr += f" {'Top-1':>8}"
        if has_exempt:
            hdr += f" {'Top-1+LE':>9}"
    print(hdr)
    print("-" * len(hdr))

    prod_r = {}
    for bits in BIT_WIDTHS:
        k = f"prod_{bits}"
        if k not in R:
            continue
        r, m = R[k], R[k]["metrics"]
        prod_r[bits] = dict(m)
        bpv = bits * d / 8 + 8
        row = (f"{bits:>5} {bits-1:>8} {m['mse']:>12.6f} {m['cosine_similarity']:>12.6f} "
               f"{m['inner_product_correlation']:>12.6f} {bpv:>12.1f} {fp16/bpv:>9.1f}x")
        if eval_top1 and "top1" in r:
            prod_r[bits]["top1_match_rate"] = r["top1"]
            row += f" {r['top1']:>7.1%}"
            if "top1_le" in r:
                prod_r[bits]["top1_layer_exempt"] = r["top1_le"]
                row += f" {r['top1_le']:>8.1%}"
        print(row)

    # ── MSE vs Prod comparison ──
    print("\n" + "=" * 80)
    print("Comparison: MSE-only vs MSE+QJL at Same Total Bit Budget")
    print("=" * 80)
    hdr = f"{'Budget':>7} {'Method':>15} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12}"
    if eval_top1:
        hdr += f" {'Top-1':>8}"
    print(hdr)
    print("-" * (80 + (9 if eval_top1 else 0)))

    for bits in [3, 4, 5]:
        if bits in mse_r and bits in prod_r:
            for label, src in [("MSE-only", mse_r[bits]), (f"MSE({bits-1}b)+QJL", prod_r[bits])]:
                row = (f"{bits:>5}b {label:>15} {src['mse']:>12.6f} "
                       f"{src['cosine_similarity']:>12.6f} {src['inner_product_correlation']:>12.6f}")
                if eval_top1 and "top1_match_rate" in src:
                    row += f" {src['top1_match_rate']:>7.1%}"
                print(row)

    # ── Outlier channel handling ──
    print("\n" + "=" * 80)
    print("TurboQuant_MSE with Outlier Channel Handling")
    print("=" * 80)
    hdr = f"{'Config':<18} {'Eff.Bits':>8} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Compress':>10}"
    if eval_top1:
        hdr += f" {'Top-1':>8}"
    print(hdr)
    print("-" * (80 + (9 if eval_top1 else 0)))

    outlier_r = {}
    for i, (name, n_out, bh, bl, eff) in enumerate(OUTLIER_CONFIGS):
        k = f"outlier_{i}"
        if k not in R:
            continue
        r, m = R[k], R[k]["metrics"]
        outlier_r[name] = dict(m)
        bpv = (n_out * bh + (d - n_out) * bl) / 8 + 8
        row = (f"{name:<18} {eff:>8.1f} {m['mse']:>12.6f} {m['cosine_similarity']:>12.6f} "
               f"{m['inner_product_correlation']:>12.6f} {fp16/bpv:>9.1f}x")
        if eval_top1 and "top1" in r:
            outlier_r[name]["top1_match_rate"] = r["top1"]
            row += f" {r['top1']:>7.1%}"
        print(row)

    # ── Outlier vs uniform comparison ──
    print("\n" + "=" * 80)
    print("Comparison: Outlier Handling vs Uniform Bit Width")
    print("=" * 80)
    hdr = f"{'Method':<25} {'Eff.Bits':>8} {'MSE':>12} {'Cos Sim':>12} {'Compress':>10}"
    if eval_top1:
        hdr += f" {'Top-1':>8}"
    print(hdr)
    print("-" * (75 + (9 if eval_top1 else 0)))

    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        if name not in outlier_r:
            continue
        lo, hi = int(eff), int(eff) + 1
        for label, rd, bk in [
            (f"Uniform {lo}-bit", mse_r, lo),
            (f"Outlier {name}", outlier_r, name),
            (f"Uniform {hi}-bit", mse_r, hi),
        ]:
            if bk not in rd:
                continue
            src = rd[bk]
            if isinstance(bk, int):
                bpv, eb = bk * d / 8 + 4, float(bk)
            else:
                bpv, eb = (n_out * bh + (d - n_out) * bl) / 8 + 8, eff
            row = (f"{label:<25} {eb:>8.1f} {src['mse']:>12.6f} "
                   f"{src['cosine_similarity']:>12.6f} {fp16/bpv:>9.1f}x")
            if eval_top1 and "top1_match_rate" in src:
                row += f" {src['top1_match_rate']:>7.1%}"
            print(row)
        print()

    # ── Residual window ──
    window_keys = sorted(k for k in R if k.startswith("window_"))
    if window_keys:
        W = R[window_keys[0]]["window"]
        print("\n" + "=" * 80)
        print(f"Residual Window Evaluation (last {W} of {n_tokens} tokens in FP16)")
        print("=" * 80)
        print(f"{'Bits':>5} {'Window':>8} {'Top-1':>8} {'Top-1+W':>8} {'Eff.Compress':>12}")
        print("-" * 55)
        for bits in [2, 3, 4]:
            wk = f"window_{bits}"
            if wk not in R:
                continue
            wr = R[wk]
            no_win = mse_r.get(bits, {}).get("top1_match_rate")
            no_str = f"{no_win:>7.1%}" if no_win is not None else "    N/A"
            print(f"{bits:>5} {W:>8} {no_str} {wr['top1']:>7.1%} {wr['eff_compress']:>11.2f}x")

    # ── Top-k summary ──
    topk_keys = sorted(k for k in R if k.startswith("topk_"))
    if topk_keys:
        ks = [1, 2, 4, 8, 16]
        le_tag = " (with layer exemption)" if has_exempt else ""
        print("\n" + "=" * 80)
        print(f"Top-k Token Match Summary (teacher-forced, {n_generate} tokens)")
        print("=" * 80)
        hdr = f"{'Bits':>5}" + "".join(f" {'Top-'+str(k):>8}" for k in ks)
        print(hdr + le_tag)
        print("-" * (5 + 9 * len(ks)))
        for bits in [2, 3, 4, 5]:
            tk = f"topk_{bits}"
            if tk not in R:
                continue
            rates = R[tk]["topk_match_rates"]
            row = f"{bits:>5}" + "".join(f" {rates[str(k)]:>7.1%}" for k in ks)
            print(row)


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel TurboQuant Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--eval-top1", action="store_true")
    parser.add_argument("--n-generate", type=int, default=50)
    parser.add_argument("--residual-window", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (default: auto-detect)")
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    # ── Discover GPUs ──
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = get_free_gpus(min_free_mb=20000)

    if not gpu_ids:
        print("ERROR: No GPUs with sufficient free memory (>=20 GB).")
        sys.exit(1)

    print(f"Available GPUs: {gpu_ids} ({len(gpu_ids)} total)")

    setup_gpu = gpu_ids[0]
    device = f"cuda:{setup_gpu}"
    output_dir = Path(__file__).parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Setup ──
    t_total = time.time()
    print("\n" + "=" * 80)
    print("Phase 1: Setup (codebooks, model loading, KV capture, ground truth)")
    print("=" * 80)

    codebooks = compute_all_codebooks(DEFAULT_DIM, BIT_WIDTHS)
    for bits in BIT_WIDTHS:
        if bits >= 2 and bits - 1 not in codebooks:
            codebooks[bits - 1] = lloyd_max_codebook(DEFAULT_DIM, bits - 1)

    from eval.model import load_model_and_capture_kv
    kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
        args.model, device=device, min_tokens=args.min_tokens, image_path=args.image,
    )

    d = kv_cache["head_dim"]
    if d != DEFAULT_DIM:
        print(f"Note: head_dim={d}, recomputing codebooks...")
        codebooks = compute_all_codebooks(d, BIT_WIDTHS)
        for bits in BIT_WIDTHS:
            if bits >= 2 and bits - 1 not in codebooks:
                codebooks[bits - 1] = lloyd_max_codebook(d, bits - 1)

    # Outlier detection
    outlier_layers_info = identify_outlier_layers(kv_cache)
    exempt_layers = [info[0] for info in outlier_layers_info]
    has_exempt = len(exempt_layers) > 0

    if has_exempt:
        print("Layer-level key norm outliers (keys kept in FP16):")
        for layer_idx, mean_norm, median_norm in outlier_layers_info:
            print(f"  Layer {layer_idx}: mean_key_norm={mean_norm:.1f} "
                  f"({mean_norm/median_norm:.1f}x median={median_norm:.1f})")

    outlier_idx_32 = identify_outlier_channels(kv_cache, n_outliers=32)
    outlier_idx_64 = identify_outlier_channels(kv_cache, n_outliers=64)
    print(f"Top-32 outlier channels: {outlier_idx_32.cpu().tolist()}")
    print(f"Top-64 outlier channels: {outlier_idx_64.cpu().tolist()}")

    # Subspace codebooks for outlier configs
    subspace_codebooks = {}
    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        for dim, bits in [(n_out, bh), (d - n_out, bl)]:
            key = f"{dim}_{bits}"
            if key not in subspace_codebooks:
                print(f"  Computing codebook d={dim}, {bits}-bit...", end=" ", flush=True)
                t0 = time.time()
                subspace_codebooks[key] = lloyd_max_codebook(dim, bits)
                print(f"done ({time.time()-t0:.1f}s)")

    # Ground truth
    ground_truth = None
    if args.eval_top1:
        print("Generating ground truth tokens...")
        from eval.metrics import generate_ground_truth
        ground_truth = generate_ground_truth(
            model, input_ids, kv_cache["keys"], kv_cache["values"], args.n_generate,
        )
        print(f"  Generated {len(ground_truth)} tokens")

    # Save shared state
    shared_dir = tempfile.mkdtemp(prefix="tq_parallel_")
    print(f"Shared state: {shared_dir}")

    kv_meta = {
        "n_layers": kv_cache["n_layers"],
        "n_heads": kv_cache["n_heads"],
        "n_tokens": kv_cache["n_tokens"],
        "head_dim": kv_cache["head_dim"],
    }
    torch.save({**kv_meta, "keys": kv_cache["keys"].cpu(), "values": kv_cache["values"].cpu()},
               os.path.join(shared_dir, "kv_cache.pt"))
    torch.save(input_ids.cpu(), os.path.join(shared_dir, "input_ids.pt"))
    if ground_truth is not None:
        torch.save(ground_truth.cpu(), os.path.join(shared_dir, "ground_truth.pt"))
    torch.save(codebooks, os.path.join(shared_dir, "codebooks.pt"))
    torch.save({
        "exempt_layers": exempt_layers,
        "outlier_idx_32": outlier_idx_32.cpu(),
        "outlier_idx_64": outlier_idx_64.cpu(),
        "subspace_codebooks": subspace_codebooks,
    }, os.path.join(shared_dir, "outlier_info.pt"))

    # Free model — GPU is now available for a worker
    del model, tokenizer, kv_cache
    torch.cuda.empty_cache()
    print(f"Phase 1 done ({time.time()-t_total:.1f}s)")

    # ── Phase 2: Parallel evaluation ──
    print("\n" + "=" * 80)
    print("Phase 2: Parallel Evaluation")
    print("=" * 80)

    tasks = define_tasks(args.eval_top1, args.residual_window)
    n_gpus = len(gpu_ids)
    task_groups, loads = distribute_tasks(tasks, n_gpus, args.eval_top1, has_exempt)

    # Drop empty groups
    work = [(gpu_ids[i], i, task_groups[i], loads[i])
            for i in range(n_gpus) if task_groups[i]]

    print(f"Distributing {len(tasks)} tasks across {len(work)} GPUs:")
    for gpu_id, wid, tg, load in work:
        names = ", ".join(f"{t['type']}({t.get('bits', t.get('config_idx', t.get('bits_list', '')))})"
                          for t in tg)
        print(f"  GPU {gpu_id} (worker {wid}): cost={load}  [{names}]")

    worker_script = str(Path(__file__).parent / "eval" / "parallel_worker.py")
    processes = []
    t_phase2 = time.time()

    for gpu_id, wid, tg, _ in work:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, worker_script,
            "--shared-dir", shared_dir,
            "--tasks", json.dumps(tg),
            "--worker-id", str(wid),
            "--model", args.model,
            "--n-generate", str(args.n_generate),
        ]
        if args.eval_top1:
            cmd.append("--eval-top1")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append((wid, gpu_id, proc))
        print(f"  Launched worker {wid} on GPU {gpu_id} (PID {proc.pid})")

    # Poll for completion
    remaining = list(processes)
    all_results = {}
    failed = []

    while remaining:
        for item in remaining[:]:
            wid, gpu_id, proc = item
            ret = proc.poll()
            if ret is not None:
                stdout = proc.stdout.read().decode()
                if ret != 0:
                    print(f"\n  Worker {wid} (GPU {gpu_id}) FAILED (exit {ret}):")
                    for line in stdout.strip().split("\n"):
                        print(f"    {line}")
                    failed.append(wid)
                else:
                    for line in stdout.strip().split("\n"):
                        if line:
                            print(f"  {line}")
                    result_path = os.path.join(shared_dir, f"results_{wid}.json")
                    if os.path.exists(result_path):
                        with open(result_path) as f:
                            all_results.update(json.load(f))
                remaining.remove(item)
        if remaining:
            time.sleep(2)

    elapsed = time.time() - t_phase2
    print(f"\nPhase 2 done: {len(all_results)} results in {elapsed:.1f}s "
          f"({len(failed)} failed)")

    if failed:
        print(f"WARNING: Workers {failed} failed. Results may be incomplete.")

    # ── Phase 3: Print results ──
    print("\n" + "=" * 80)
    print("Phase 3: Results")
    print("=" * 80)

    n_vecs = kv_meta["n_layers"] * kv_meta["n_heads"] * kv_meta["n_tokens"] * 2
    print(f"\nEvaluating on {n_vecs} vectors (keys + values), d={kv_meta['head_dim']}")
    if args.eval_top1:
        print(f"Top-1 evaluation: {args.n_generate} tokens per config")

    print_results(all_results, kv_meta, args.eval_top1, has_exempt,
                  args.n_generate, args.residual_window)

    # Cleanup
    shutil.rmtree(shared_dir, ignore_errors=True)

    total = time.time() - t_total
    print("\n" + "=" * 80)
    print(f"All done! Total wall time: {total:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
