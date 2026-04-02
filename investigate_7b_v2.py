"""Investigate 7B non-monotonic Top-1 — v2 with corrected methodology.

v1 issues:
  - Used keys as queries (wrong — model uses separate Q projections)
  - Per-channel variance was circular (consequence of high norms)
  - Layer 27 KL was constant even at 8-bit, undermining the narrative

v2 approach:
  - Use the ACTUAL model to compute attention distributions with original vs quantized keys
  - Measure the downstream effect: logit divergence at the output, not attention KL
  - Check if Layer 0 vs Layer 27 contribution can be isolated
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys, os, gc, copy

sys.path.insert(0, os.path.dirname(__file__))

from turboquant.core import TurboQuantMSE, lloyd_max_codebook, generate_rotation_matrix
from eval.model import load_model_and_capture_kv
from eval.metrics import build_dynamic_cache


def measure_logit_divergence(model, input_ids, original_keys, original_values,
                             quantized_keys, quantized_values):
    """Run model with original vs quantized KV cache and measure output logit divergence."""
    device = input_ids.device

    # Build caches (exclude last token — it's in input_ids)
    orig_cache = build_dynamic_cache(
        original_keys[:, :, :-1, :], original_values[:, :, :-1, :]
    )
    quant_cache = build_dynamic_cache(
        quantized_keys[:, :, :-1, :], quantized_values[:, :, :-1, :]
    )

    with torch.no_grad():
        token = input_ids[:, -1:]

        orig_out = model(token, past_key_values=copy.deepcopy(orig_cache), use_cache=False)
        quant_out = model(token, past_key_values=copy.deepcopy(quant_cache), use_cache=False)

        orig_logits = orig_out.logits[:, -1, :]   # (1, vocab_size)
        quant_logits = quant_out.logits[:, -1, :]

        # Logit divergence metrics
        orig_probs = F.softmax(orig_logits, dim=-1)
        quant_probs = F.softmax(quant_logits, dim=-1)

        kl = F.kl_div(quant_probs.log().clamp(min=-100), orig_probs,
                       reduction='batchmean').item()

        # Top-1 match
        orig_top1 = orig_logits.argmax(dim=-1)
        quant_top1 = quant_logits.argmax(dim=-1)
        top1_match = (orig_top1 == quant_top1).item()

        # Top-1 probability in quantized distribution
        orig_top1_prob = quant_probs[0, orig_top1[0]].item()

        # Rank of correct token in quantized distribution
        rank = (quant_probs[0] >= quant_probs[0, orig_top1[0]]).sum().item()

    return {
        "kl": kl,
        "top1_match": top1_match,
        "orig_top1_prob": orig_top1_prob,
        "orig_top1_rank": rank,
    }


def quantize_kv_selective(keys, values, quantizer, skip_key_layers=None, skip_val_layers=None,
                          device="cuda"):
    """Quantize KV cache with optional per-layer skipping."""
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
        model_name="Qwen/Qwen2.5-7B-Instruct", device=device, min_tokens=1000,
    )

    keys = kv_cache["keys"]
    values = kv_cache["values"]
    n_layers, n_heads, n_tokens, d = keys.shape
    print(f"KV cache: {n_layers}L x {n_heads}H x {n_tokens}T x {d}d")

    # Setup
    bit_widths = [2, 3, 4, 5, 6, 8]
    codebooks = {b: lloyd_max_codebook(d, b) for b in bit_widths}
    rotation = generate_rotation_matrix(d, seed=42, device=device)

    # =========================================================================
    # Test 1: Full quantization — logit divergence at each bit width
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 1: Full Quantization — Output Logit Divergence")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        q_keys, q_values = quantize_kv_selective(keys, values, quantizer, device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    # =========================================================================
    # Test 2: Only quantize Layer 0 keys (everything else FP16)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 2: Only Quantize Layer 0 Keys (all else FP16)")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    all_layers = set(range(n_layers))
    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        # Only quantize layer 0 keys, skip all other keys and all values
        q_keys, q_values = quantize_kv_selective(
            keys, values, quantizer,
            skip_key_layers=all_layers - {0},
            skip_val_layers=all_layers,
            device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    # =========================================================================
    # Test 3: Only quantize Layer 27 keys (everything else FP16)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 3: Only Quantize Layer 27 Keys (all else FP16)")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        q_keys, q_values = quantize_kv_selective(
            keys, values, quantizer,
            skip_key_layers=all_layers - {27},
            skip_val_layers=all_layers,
            device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    # =========================================================================
    # Test 4: Quantize everything EXCEPT Layer 0 keys
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 4: Quantize Everything EXCEPT Layer 0 Keys")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        q_keys, q_values = quantize_kv_selective(
            keys, values, quantizer,
            skip_key_layers={0},
            device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    # =========================================================================
    # Test 5: Quantize everything EXCEPT Layer 27 keys
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 5: Quantize Everything EXCEPT Layer 27 Keys")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        q_keys, q_values = quantize_kv_selective(
            keys, values, quantizer,
            skip_key_layers={27},
            device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    # =========================================================================
    # Test 6: Quantize everything EXCEPT Layers 0+27 keys (= layer exemption)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test 6: Quantize Everything EXCEPT Layers 0+27 Keys (layer exemption)")
    print("=" * 80)
    print(f"{'Bits':>4}  {'KL Div':>10}  {'Top1 Match':>10}  {'GT Prob':>8}  {'GT Rank':>8}")
    print("-" * 50)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)
        q_keys, q_values = quantize_kv_selective(
            keys, values, quantizer,
            skip_key_layers={0, 27},
            device=device)
        result = measure_logit_divergence(model, input_ids, keys, values, q_keys, q_values)
        print(f"  {b:2d}   {result['kl']:10.4f}  {'YES' if result['top1_match'] else 'NO':>10}"
              f"  {result['orig_top1_prob']:8.4f}  {result['orig_top1_rank']:8d}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
