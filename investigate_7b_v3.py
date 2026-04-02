"""Investigate 7B — v3: Full 50-token teacher-forced eval with selective layer quantization.

v2 only measured single-token logit divergence. This runs the actual 50-token
teacher-forced Top-1 evaluation for each layer isolation scenario.
"""

import torch
import sys, os, gc

sys.path.insert(0, os.path.dirname(__file__))

from turboquant.core import TurboQuantMSE, lloyd_max_codebook, generate_rotation_matrix
from eval.model import load_model_and_capture_kv
from eval.metrics import generate_ground_truth, eval_topk_match


def quantize_kv_selective(keys, values, quantizer, skip_key_layers=None, skip_val_layers=None,
                          device="cuda"):
    if skip_key_layers is None:
        skip_key_layers = set()
    if skip_val_layers is None:
        skip_val_layers = set()

    n_layers, n_heads, n_tokens, d = keys.shape
    q_keys = keys.clone()
    q_values = values.clone()

    for layer in range(n_layers):
        for head in range(n_heads):
            if layer not in skip_key_layers:
                q_keys[layer, head] = quantizer.quantize_dequantize(
                    keys[layer, head].to(device)).cpu()
            if layer not in skip_val_layers:
                q_values[layer, head] = quantizer.quantize_dequantize(
                    values[layer, head].to(device)).cpu()
    return q_keys, q_values


def run_top1(model, input_ids, keys, values, q_keys, q_values, ground_truth, n_generate=50):
    """Run full teacher-forced Top-1 evaluation."""
    result = eval_topk_match(
        model, input_ids, keys, values, q_keys, q_values,
        n_generate=n_generate, topk_values=[1],
        ground_truth_tokens=ground_truth,
    )
    return result["top1_match_rate"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_generate = 50

    kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
        model_name="Qwen/Qwen2.5-7B-Instruct", device=device, min_tokens=1000,
    )

    keys = kv_cache["keys"]
    values = kv_cache["values"]
    n_layers = kv_cache["n_layers"]
    d = kv_cache["head_dim"]
    all_layers = set(range(n_layers))

    print(f"KV cache: {n_layers}L, generating ground truth ({n_generate} tokens)...")
    ground_truth = generate_ground_truth(model, input_ids, keys, values, n_generate)
    print(f"Ground truth: {ground_truth.tolist()}")

    bit_widths = [2, 3, 4, 5, 6, 8]
    codebooks = {b: lloyd_max_codebook(d, b) for b in bit_widths}
    rotation = generate_rotation_matrix(d, seed=42, device=device)

    scenarios = [
        ("Full quantization",              lambda: ({},          {})),
        ("Only L0 keys quantized",         lambda: (all_layers - {0}, all_layers)),
        ("Only L27 keys quantized",        lambda: (all_layers - {27}, all_layers)),
        ("Only L0+L27 keys quantized",     lambda: (all_layers - {0, 27}, all_layers)),
        ("Skip L0 keys only",              lambda: ({0},         {})),
        ("Skip L27 keys only",             lambda: ({27},        {})),
        ("Skip L0+L27 keys (full LE)",     lambda: ({0, 27},     {})),
        ("Only L0 values quantized",       lambda: (all_layers,  all_layers - {0})),
        ("Only L27 values quantized",      lambda: (all_layers,  all_layers - {27})),
    ]

    # Header
    print(f"\n{'Scenario':<35} ", end="")
    for b in bit_widths:
        print(f" {b:>5}b", end="")
    print()
    print("-" * (35 + 7 * len(bit_widths)))

    for name, skip_fn in scenarios:
        print(f"{name:<35} ", end="", flush=True)
        for b in bit_widths:
            quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
            quantizer.to(device)
            skip_keys, skip_vals = skip_fn()
            q_keys, q_values = quantize_kv_selective(
                keys, values, quantizer,
                skip_key_layers=skip_keys, skip_val_layers=skip_vals,
                device=device)
            acc = run_top1(model, input_ids, keys, values, q_keys, q_values, ground_truth, n_generate)
            print(f" {acc*100:5.1f}%", end="", flush=True)
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
