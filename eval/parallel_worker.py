#!/usr/bin/env python3
"""
Worker process for parallel TurboQuant evaluation.

Invoked by run_parallel.py with CUDA_VISIBLE_DEVICES set to a single GPU.
Loads shared state (KV cache, codebooks, ground truth) from a temp directory,
loads the model, runs assigned evaluation tasks, and writes results to JSON.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for HF_TOKEN
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

import torch

from turboquant.core import (
    TurboQuantMSE,
    TurboQuantProd,
    TurboQuantOutlier,
    generate_rotation_matrix,
    DEFAULT_SEED,
    OUTLIER_CONFIGS,
)
from eval.metrics import compute_metrics, eval_topk_match


def _log(worker_id, msg):
    print(f"[Worker {worker_id}] {msg}", flush=True)


def clean_metrics(m):
    """Extract JSON-serializable metrics (drop numpy arrays)."""
    return {
        "mse": float(m["mse"]),
        "cosine_similarity": float(m["cosine_similarity"]),
        "inner_product_correlation": float(m["inner_product_correlation"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-generate", type=int, default=50)
    parser.add_argument("--eval-top1", action="store_true")
    args = parser.parse_args()

    wid = args.worker_id
    device = "cuda"
    tasks = json.loads(args.tasks)
    shared_dir = args.shared_dir

    _log(wid, f"Starting with {len(tasks)} task(s)")
    t0 = time.time()

    # ── Load shared state ──
    kv_data = torch.load(os.path.join(shared_dir, "kv_cache.pt"), weights_only=False)
    kv_cache = {
        "keys": kv_data["keys"].to(device),
        "values": kv_data["values"].to(device),
        "n_layers": kv_data["n_layers"],
        "n_heads": kv_data["n_heads"],
        "n_tokens": kv_data["n_tokens"],
        "head_dim": kv_data["head_dim"],
    }
    input_ids = torch.load(os.path.join(shared_dir, "input_ids.pt"), weights_only=False).to(device)
    codebooks = torch.load(os.path.join(shared_dir, "codebooks.pt"), weights_only=False)
    outlier_info = torch.load(os.path.join(shared_dir, "outlier_info.pt"), weights_only=False)

    gt_path = os.path.join(shared_dir, "ground_truth.pt")
    ground_truth = torch.load(gt_path, weights_only=False).to(device) if os.path.exists(gt_path) else None

    exempt_layers = set(outlier_info["exempt_layers"])
    outlier_idx_32 = outlier_info["outlier_idx_32"].to(device)
    outlier_idx_64 = outlier_info["outlier_idx_64"].to(device)
    subspace_codebooks = outlier_info["subspace_codebooks"]

    d = kv_cache["head_dim"]
    n_layers = kv_cache["n_layers"]
    n_heads = kv_cache["n_heads"]
    n_tokens = kv_cache["n_tokens"]

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)
    all_vecs = torch.cat([
        kv_cache["keys"].reshape(-1, d),
        kv_cache["values"].reshape(-1, d),
    ], dim=0)

    _log(wid, f"Shared state loaded ({time.time() - t0:.1f}s)")

    # ── Load model if needed ──
    model = None
    if args.eval_top1:
        from transformers import AutoModelForCausalLM
        _log(wid, f"Loading model {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
        model.eval()
        _log(wid, f"Model loaded ({time.time() - t0:.1f}s)")

    # ── Helpers ──
    def quantize_kv(quantizer, skip_key_layers=None):
        skl = skip_key_layers or set()
        q_keys = kv_cache["keys"].clone() if skl else torch.zeros_like(kv_cache["keys"])
        q_values = torch.zeros_like(kv_cache["values"])
        for layer in range(n_layers):
            for head in range(n_heads):
                if layer not in skl:
                    q_keys[layer, head] = quantizer.quantize_dequantize(kv_cache["keys"][layer, head])
                q_values[layer, head] = quantizer.quantize_dequantize(kv_cache["values"][layer, head])
        return q_keys, q_values

    def run_top1(q_keys, q_values):
        return eval_topk_match(
            model, input_ids,
            kv_cache["keys"], kv_cache["values"],
            q_keys, q_values,
            n_generate=args.n_generate,
            ground_truth_tokens=ground_truth,
        )["top1_match_rate"]

    # ── Execute tasks ──
    results = {}

    for task in tasks:
        ttype = task["type"]
        task_t0 = time.time()

        if ttype == "mse":
            bits = task["bits"]
            _log(wid, f"MSE {bits}-bit")
            quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)
            metrics = clean_metrics(compute_metrics(all_vecs, quantizer.quantize_dequantize(all_vecs)))
            result = {"bits": bits, "metrics": metrics}

            if model is not None:
                q_keys, q_values = quantize_kv(quantizer)
                result["top1"] = run_top1(q_keys, q_values)
                if exempt_layers:
                    q_keys_le, q_values_le = quantize_kv(quantizer, skip_key_layers=exempt_layers)
                    result["top1_le"] = run_top1(q_keys_le, q_values_le)

            results[f"mse_{bits}"] = result

        elif ttype == "prod":
            bits = task["bits"]
            _log(wid, f"Prod {bits}-bit")
            quantizer = TurboQuantProd(d, bits, codebooks, rotation).to(device)
            metrics = clean_metrics(compute_metrics(all_vecs, quantizer.quantize_dequantize(all_vecs)))
            result = {"bits": bits, "metrics": metrics}

            if model is not None:
                q_keys, q_values = quantize_kv(quantizer)
                result["top1"] = run_top1(q_keys, q_values)
                if exempt_layers:
                    q_keys_le, q_values_le = quantize_kv(quantizer, skip_key_layers=exempt_layers)
                    result["top1_le"] = run_top1(q_keys_le, q_values_le)

            results[f"prod_{bits}"] = result

        elif ttype == "outlier":
            idx = task["config_idx"]
            name, n_out, bh, bl, eff = OUTLIER_CONFIGS[idx]
            _log(wid, f"Outlier {name}")

            d_high, d_low = n_out, d - n_out
            outlier_idx = outlier_idx_32 if n_out == 32 else outlier_idx_64
            cb_high = {bh: subspace_codebooks[f"{d_high}_{bh}"]}
            cb_low = {bl: subspace_codebooks[f"{d_low}_{bl}"]}

            quantizer = TurboQuantOutlier(d, bh, bl, outlier_idx, cb_high, cb_low).to(device)
            metrics = clean_metrics(compute_metrics(all_vecs, quantizer.quantize_dequantize(all_vecs)))
            result = {"name": name, "n_out": n_out, "bh": bh, "bl": bl, "eff": eff, "metrics": metrics}

            if model is not None:
                q_keys = kv_cache["keys"].clone() if exempt_layers else torch.zeros_like(kv_cache["keys"])
                q_values = torch.zeros_like(kv_cache["values"])
                for layer in range(n_layers):
                    for head in range(n_heads):
                        if layer not in exempt_layers:
                            q_keys[layer, head] = quantizer.quantize_dequantize(
                                kv_cache["keys"][layer, head])
                        q_values[layer, head] = quantizer.quantize_dequantize(
                            kv_cache["values"][layer, head])
                result["top1"] = run_top1(q_keys, q_values)

            results[f"outlier_{idx}"] = result

        elif ttype == "topk":
            topk_values = [1, 2, 4, 8, 16]
            for bits in task["bits_list"]:
                _log(wid, f"Top-k {bits}-bit")
                if bits not in codebooks:
                    continue
                quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)
                q_keys, q_values = quantize_kv(quantizer, skip_key_layers=exempt_layers)
                topk_result = eval_topk_match(
                    model, input_ids,
                    kv_cache["keys"], kv_cache["values"],
                    q_keys, q_values,
                    n_generate=args.n_generate,
                    topk_values=topk_values,
                    ground_truth_tokens=ground_truth,
                )
                results[f"topk_{bits}"] = {
                    "bits": bits,
                    "topk_match_rates": {str(k): v for k, v in topk_result["topk_match_rates"].items()},
                }

        elif ttype == "window":
            W = task["window"]
            fp16_bytes = 2 * d
            for bits in task["bits_list"]:
                _log(wid, f"Window {bits}-bit (W={W})")
                quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)
                q_keys = kv_cache["keys"].clone()
                q_values = kv_cache["values"].clone()
                cutoff = max(0, n_tokens - W)
                if cutoff > 0:
                    for layer in range(n_layers):
                        for head in range(n_heads):
                            q_keys[layer, head, :cutoff] = quantizer.quantize_dequantize(
                                kv_cache["keys"][layer, head, :cutoff])
                            q_values[layer, head, :cutoff] = quantizer.quantize_dequantize(
                                kv_cache["values"][layer, head, :cutoff])
                top1 = eval_topk_match(
                    model, input_ids,
                    kv_cache["keys"], kv_cache["values"],
                    q_keys, q_values,
                    n_generate=args.n_generate,
                    ground_truth_tokens=ground_truth,
                )
                n_compressed = n_tokens - W
                eff_bytes = n_compressed * (bits * d / 8 + 4) + W * fp16_bytes
                baseline_bytes = n_tokens * fp16_bytes
                results[f"window_{bits}"] = {
                    "bits": bits, "window": W,
                    "top1": top1["top1_match_rate"],
                    "eff_compress": baseline_bytes / eff_bytes,
                }

        _log(wid, f"  {ttype} done ({time.time() - task_t0:.1f}s)")

    # ── Save results ──
    out_path = os.path.join(shared_dir, f"results_{wid}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(wid, f"All done, {len(results)} results saved ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
