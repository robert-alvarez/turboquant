"""Investigate the 7B non-monotonic Top-1 pattern.

At 1000 tokens without layer exemption:
  2-bit: 96%, 3-bit: 80%, 4-bit: 12%, 5-bit: 48%, 6-bit: 98%, 8-bit: 100%

Hypothesis: Outlier layers (0, 27) have key norms 13x median. Quantization error
is proportional to norm, so these layers get huge absolute error. At moderate bit
widths (3-5), the error is large enough to systematically distort attention patterns
but structured enough to be "confidently wrong". At 2-bit, quantization is so coarse
it effectively randomizes attention for those layers (harmless noise). At 6+ bits,
quantization is accurate enough that attention patterns are preserved.

This script measures:
1. Per-layer attention divergence (KL) between original and quantized KV
2. Per-layer MSE for keys at each bit width
3. Attention entropy for outlier vs normal layers
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from turboquant.core import TurboQuantMSE, lloyd_max_codebook, generate_rotation_matrix

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and capture KV cache
    from eval.model import load_model_and_capture_kv
    kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device,
        min_tokens=1000,
    )
    # Free model — we only need the KV cache
    del model, tokenizer
    torch.cuda.empty_cache()

    keys = kv_cache["keys"]      # (n_layers, n_heads, n_tokens, d)
    values = kv_cache["values"]
    n_layers, n_heads, n_tokens, d = keys.shape
    print(f"KV cache: {n_layers}L x {n_heads}H x {n_tokens}T x {d}d")

    # Compute per-layer key norms
    print("\n=== Per-Layer Key Norms ===")
    layer_norms = []
    for layer in range(n_layers):
        norms = keys[layer].reshape(-1, d).norm(dim=-1)
        layer_norms.append(norms.mean().item())
    median_norm = float(sorted(layer_norms)[n_layers // 2])
    for layer in range(n_layers):
        ratio = layer_norms[layer] / median_norm
        flag = " *** OUTLIER" if ratio > 4 else ""
        if ratio > 2 or layer in [0, 27]:
            print(f"  Layer {layer:2d}: mean_key_norm={layer_norms[layer]:.1f} ({ratio:.1f}x median){flag}")

    # Compute codebooks and rotation
    bit_widths = [2, 3, 4, 5, 6, 8]
    codebooks = {}
    for b in bit_widths:
        codebooks[b] = lloyd_max_codebook(d, b)
    rotation = generate_rotation_matrix(d, seed=42, device=device)

    # For each bit width, measure per-layer key MSE and attention divergence
    print("\n=== Per-Layer Key MSE (Outlier Layers 0, 27 vs Median Layer) ===")
    print(f"{'Bits':>4}  {'Layer 0 MSE':>12}  {'Layer 27 MSE':>13}  {'Median MSE':>11}  {'L0/Med':>7}  {'L27/Med':>8}")
    print("-" * 70)

    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)

        layer_mses = []
        for layer in range(n_layers):
            mse_sum = 0
            for head in range(n_heads):
                vecs = keys[layer, head].to(device)
                vecs_hat = quantizer.quantize_dequantize(vecs)
                mse_sum += (vecs - vecs_hat).pow(2).sum(dim=-1).mean().item()
            layer_mses.append(mse_sum / n_heads)

        med_mse = float(sorted(layer_mses)[n_layers // 2])
        l0_mse = layer_mses[0]
        l27_mse = layer_mses[27]
        print(f"  {b:2d}   {l0_mse:12.2f}  {l27_mse:13.2f}  {med_mse:11.4f}  {l0_mse/med_mse:6.0f}x  {l27_mse/med_mse:7.0f}x")

    # Measure attention pattern divergence
    # Use a random query to compute attention over original vs quantized keys
    print("\n=== Attention Pattern Analysis ===")
    print("Computing attention KL divergence (original vs quantized) for outlier layers...")
    print(f"{'Bits':>4}  {'Layer 0 KL':>11}  {'Layer 27 KL':>12}  {'Median KL':>10}  {'L0 Entropy Orig':>16}  {'L0 Entropy Quant':>17}")
    print("-" * 80)

    # Use actual keys as queries (first head, sample of query positions)
    sample_queries = 10  # Use last 10 token positions as query points
    for b in bit_widths:
        quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
        quantizer.to(device)

        layer_kls = []
        l0_ent_orig = l0_ent_quant = 0
        l27_ent_orig = l27_ent_quant = 0

        for layer in range(n_layers):
            kl_sum = 0
            ent_orig_sum = 0
            ent_quant_sum = 0

            for head in range(n_heads):
                k_orig = keys[layer, head].to(device)   # (n_tokens, d)
                k_quant = quantizer.quantize_dequantize(k_orig)

                # Use last sample_queries tokens as queries
                queries = k_orig[-sample_queries:]  # (sample_queries, d)

                # Attention scores: q @ k^T / sqrt(d)
                scale = d ** 0.5
                attn_orig = (queries @ k_orig.T) / scale    # (sample_queries, n_tokens)
                attn_quant = (queries @ k_quant.T) / scale

                # Softmax
                p = F.softmax(attn_orig, dim=-1)
                q = F.softmax(attn_quant, dim=-1)

                # KL divergence: sum over tokens, mean over queries
                kl = F.kl_div(q.log().clamp(min=-100), p, reduction='none').sum(dim=-1).mean().item()
                kl_sum += kl

                # Entropy
                ent_orig = -(p * p.log().clamp(min=-100)).sum(dim=-1).mean().item()
                ent_quant = -(q * q.log().clamp(min=-100)).sum(dim=-1).mean().item()
                ent_orig_sum += ent_orig
                ent_quant_sum += ent_quant

            layer_kls.append(kl_sum / n_heads)
            if layer == 0:
                l0_ent_orig = ent_orig_sum / n_heads
                l0_ent_quant = ent_quant_sum / n_heads
            if layer == 27:
                l27_ent_orig = ent_orig_sum / n_heads
                l27_ent_quant = ent_quant_sum / n_heads

        med_kl = float(sorted(layer_kls)[n_layers // 2])
        print(f"  {b:2d}   {layer_kls[0]:11.4f}  {layer_kls[27]:12.4f}  {med_kl:10.6f}  {l0_ent_orig:16.4f}  {l0_ent_quant:17.4f}")

    # Summary of the hypothesis
    print("\n=== Hypothesis ===")
    print("If the KL divergence for layers 0/27 is NON-MONOTONIC across bit widths")
    print("(high at 2-bit, lower at 4-bit, then lower at 6-bit), it would suggest")
    print("the attention distortion explanation. If KL is monotonically decreasing,")
    print("the mechanism must be more subtle — perhaps related to how the attention")
    print("error at outlier layers interacts with the value aggregation.")

if __name__ == "__main__":
    main()
