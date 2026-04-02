"""
GPU-based evaluation of TurboQuant quantizers on real KV cache tensors.

Runs three quantizer families (MSE uniform, prod, outlier handling),
optional residual window evaluation, and top-k summary.
"""

import time

import torch

from turboquant.core import (
    TurboQuantMSE,
    TurboQuantProd,
    TurboQuantOutlier,
    identify_outlier_channels,
    identify_outlier_layers,
    OUTLIER_CONFIGS,
    BIT_WIDTHS,
    generate_rotation_matrix,
    lloyd_max_codebook,
    DEFAULT_SEED,
)
from eval.metrics import compute_metrics, eval_topk_match, generate_ground_truth


def run_gpu_evaluation(kv_cache: dict, codebooks: dict, device: str = "cuda",
                       model=None, input_ids=None, eval_top1: bool = False,
                       n_generate: int = 50, residual_window: int = 0):
    """
    Apply TurboQuant to real KV cache tensors and measure quality at all bit widths.

    Compares MSE-only vs MSE+QJL. When eval_top1=True and a model is provided,
    also measures Top-1 token match rate by generating tokens with the original
    vs quantized KV cache.

    When residual_window > 0, also runs a comparison showing the benefit of
    keeping the last W tokens in FP16 (from KIVI).
    """
    d = kv_cache["head_dim"]
    n_layers = kv_cache["n_layers"]
    n_heads = kv_cache["n_heads"]
    n_tokens = kv_cache["n_tokens"]

    do_top1 = eval_top1 and model is not None and input_ids is not None
    fp16_bytes = 2 * d

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)

    # Flatten all key vectors for evaluation: (n_layers * n_heads * n_tokens, d)
    all_keys = kv_cache["keys"].reshape(-1, d).to(device)
    all_values = kv_cache["values"].reshape(-1, d).to(device)
    # Combine keys and values for comprehensive evaluation
    all_vecs = torch.cat([all_keys, all_values], dim=0)
    print(f"\nEvaluating on {all_vecs.shape[0]} vectors (keys + values), d={d}")
    if do_top1:
        print(f"Top-1 evaluation enabled: generating {n_generate} tokens per bit width")

    # Generate ground truth ONCE to avoid CUDA non-determinism across bit widths
    ground_truth = None
    if do_top1:
        ground_truth = generate_ground_truth(
            model, input_ids, kv_cache["keys"], kv_cache["values"], n_generate
        )

    mse_results = {}
    prod_results = {}

    # --- Detect layers with anomalously large key norms ---
    outlier_layers_info = identify_outlier_layers(kv_cache)
    exempt_layers = {info[0] for info in outlier_layers_info}
    if exempt_layers:
        print(f"\nLayer-level key norm outliers (kept in FP16 for keys):")
        for layer_idx, mean_norm, median_norm in outlier_layers_info:
            print(f"  Layer {layer_idx}: mean_key_norm={mean_norm:.1f} "
                  f"({mean_norm/median_norm:.1f}x median={median_norm:.1f})")

    # --- Helper to quantize the full KV cache (not flattened) for top-1 eval ---
    def quantize_kv_cache_mse(quantizer, skip_key_layers=None):
        """Quantize keys and values preserving (n_layers, n_heads, n_tokens, d) shape.

        skip_key_layers: set of layer indices whose keys are kept in FP16.
        Values are always quantized (they are robust to quantization).
        """
        if skip_key_layers is None:
            skip_key_layers = set()
        q_keys = kv_cache["keys"].clone() if skip_key_layers else torch.zeros_like(kv_cache["keys"])
        q_values = torch.zeros_like(kv_cache["values"])
        for layer in range(n_layers):
            for head in range(n_heads):
                if layer not in skip_key_layers:
                    q_keys[layer, head] = quantizer.quantize_dequantize(
                        kv_cache["keys"][layer, head].to(device)
                    )
                q_values[layer, head] = quantizer.quantize_dequantize(
                    kv_cache["values"][layer, head].to(device)
                )
        return q_keys, q_values

    def quantize_kv_cache_prod(quantizer, skip_key_layers=None):
        """Same but for TurboQuantProd."""
        if skip_key_layers is None:
            skip_key_layers = set()
        q_keys = kv_cache["keys"].clone() if skip_key_layers else torch.zeros_like(kv_cache["keys"])
        q_values = torch.zeros_like(kv_cache["values"])
        for layer in range(n_layers):
            for head in range(n_heads):
                if layer not in skip_key_layers:
                    q_keys[layer, head] = quantizer.quantize_dequantize(
                        kv_cache["keys"][layer, head].to(device)
                    )
                q_values[layer, head] = quantizer.quantize_dequantize(
                    kv_cache["values"][layer, head].to(device)
                )
        return q_keys, q_values

    def quantize_kv_with_window(quantizer, window: int):
        """
        Quantize the KV cache but keep the last `window` tokens in FP16.

        Tokens [0, n_tokens - window) are quantized.
        Tokens [n_tokens - window, n_tokens) are kept as-is.

        This is the "residual window" pattern from KIVI: recent tokens
        are most important for generation quality and are cheap to keep
        in full precision since the window is fixed-size.
        """
        q_keys = kv_cache["keys"].clone()
        q_values = kv_cache["values"].clone()
        cutoff = max(0, n_tokens - window)
        if cutoff > 0:
            for layer in range(n_layers):
                for head in range(n_heads):
                    q_keys[layer, head, :cutoff] = quantizer.quantize_dequantize(
                        kv_cache["keys"][layer, head, :cutoff].to(device)
                    )
                    q_values[layer, head, :cutoff] = quantizer.quantize_dequantize(
                        kv_cache["values"][layer, head, :cutoff].to(device)
                    )
        return q_keys, q_values

    # ---- TurboQuant_MSE ----
    print("\n" + "=" * 80)
    print("TurboQuant_MSE Results")
    print("=" * 80)
    header = f"{'Bits':>5} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Bytes/Vec':>12} {'Compress':>10}"
    if do_top1:
        header += f" {'Top-1':>8}"
        if exempt_layers:
            header += f" {'Top-1+LE':>9}"
    print(header)
    print("-" * len(header))

    for bits in BIT_WIDTHS:
        quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)
        x_hat = quantizer.quantize_dequantize(all_vecs)
        metrics = compute_metrics(all_vecs, x_hat)
        mse_results[bits] = metrics

        bytes_per_vec = bits * d / 8 + 4  # +4 for the float32 norm
        fp16_bytes = 2 * d
        compress = fp16_bytes / bytes_per_vec

        row = (f"{bits:>5} {metrics['mse']:>12.6f} {metrics['cosine_similarity']:>12.6f} "
               f"{metrics['inner_product_correlation']:>12.6f} {bytes_per_vec:>12.1f} {compress:>9.1f}x")

        if do_top1:
            q_keys, q_values = quantize_kv_cache_mse(quantizer)
            top1 = eval_topk_match(
                model, input_ids,
                kv_cache["keys"], kv_cache["values"],
                q_keys, q_values,
                n_generate=n_generate,
                ground_truth_tokens=ground_truth,
            )
            mse_results[bits]["top1_match_rate"] = top1["top1_match_rate"]
            row += f" {top1['top1_match_rate']:>7.1%}"

            # Layer-exempt evaluation: keep outlier layers' keys in FP16
            if exempt_layers:
                q_keys_le, q_values_le = quantize_kv_cache_mse(quantizer, skip_key_layers=exempt_layers)
                top1_le = eval_topk_match(
                    model, input_ids,
                    kv_cache["keys"], kv_cache["values"],
                    q_keys_le, q_values_le,
                    n_generate=n_generate,
                    ground_truth_tokens=ground_truth,
                )
                mse_results[bits]["top1_layer_exempt"] = top1_le["top1_match_rate"]
                row += f" {top1_le['top1_match_rate']:>8.1%}"

        print(row)

    # ---- TurboQuant_prod ----
    print("\n" + "=" * 80)
    print("TurboQuant_prod (MSE + QJL) Results")
    print("=" * 80)
    header = f"{'Bits':>5} {'MSE(b-1)':>8} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Bytes/Vec':>12} {'Compress':>10}"
    if do_top1:
        header += f" {'Top-1':>8}"
        if exempt_layers:
            header += f" {'Top-1+LE':>9}"
    print(header)
    print("-" * len(header))

    # Ensure we have codebooks for (bits-1) needed by prod mode
    for bits in BIT_WIDTHS:
        if bits - 1 not in codebooks and bits >= 2:
            print(f"  Computing {bits-1}-bit codebook for prod mode...", end=" ", flush=True)
            codebooks[bits - 1] = lloyd_max_codebook(d, bits - 1)
            print("done")

    for bits in BIT_WIDTHS:
        if bits < 2:
            continue  # Need at least 2 bits for prod (1 MSE + 1 QJL)
        quantizer = TurboQuantProd(d, bits, codebooks, rotation).to(device)
        x_hat = quantizer.quantize_dequantize(all_vecs)
        metrics = compute_metrics(all_vecs, x_hat)
        prod_results[bits] = metrics

        # Total: (bits-1)*d bits for MSE + d bits for QJL signs + 4 bytes norm + 4 bytes residual norm
        bytes_per_vec = bits * d / 8 + 8  # +4 norm, +4 residual norm
        fp16_bytes = 2 * d
        compress = fp16_bytes / bytes_per_vec

        row = (f"{bits:>5} {bits-1:>8} {metrics['mse']:>12.6f} {metrics['cosine_similarity']:>12.6f} "
               f"{metrics['inner_product_correlation']:>12.6f} {bytes_per_vec:>12.1f} {compress:>9.1f}x")

        if do_top1:
            q_keys, q_values = quantize_kv_cache_prod(quantizer)
            top1 = eval_topk_match(
                model, input_ids,
                kv_cache["keys"], kv_cache["values"],
                q_keys, q_values,
                n_generate=n_generate,
                ground_truth_tokens=ground_truth,
            )
            prod_results[bits]["top1_match_rate"] = top1["top1_match_rate"]
            row += f" {top1['top1_match_rate']:>7.1%}"

            if exempt_layers:
                q_keys_le, q_values_le = quantize_kv_cache_prod(quantizer, skip_key_layers=exempt_layers)
                top1_le = eval_topk_match(
                    model, input_ids,
                    kv_cache["keys"], kv_cache["values"],
                    q_keys_le, q_values_le,
                    n_generate=n_generate,
                    ground_truth_tokens=ground_truth,
                )
                prod_results[bits]["top1_layer_exempt"] = top1_le["top1_match_rate"]
                row += f" {top1_le['top1_match_rate']:>8.1%}"

        print(row)

    # --- MSE-only vs MSE+QJL at same total bit budget ---
    print("\n" + "=" * 80)
    print("Comparison: MSE-only vs MSE+QJL at Same Total Bit Budget")
    print("=" * 80)
    header = f"{'Budget':>7} {'Method':>15} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12}"
    if do_top1:
        header += f" {'Top-1':>8}"
    print(header)
    print("-" * (80 + (9 if do_top1 else 0)))

    for bits in [3, 4, 5]:
        if bits in mse_results and bits in prod_results:
            m = mse_results[bits]
            p = prod_results[bits]
            row_m = (f"{bits:>5}b {'MSE-only':>15} {m['mse']:>12.6f} {m['cosine_similarity']:>12.6f} "
                     f"{m['inner_product_correlation']:>12.6f}")
            row_p = (f"{bits:>5}b {f'MSE({bits-1}b)+QJL':>15} {p['mse']:>12.6f} {p['cosine_similarity']:>12.6f} "
                     f"{p['inner_product_correlation']:>12.6f}")
            if do_top1 and "top1_match_rate" in m:
                row_m += f" {m['top1_match_rate']:>7.1%}"
            if do_top1 and "top1_match_rate" in p:
                row_p += f" {p['top1_match_rate']:>7.1%}"
            print(row_m)
            print(row_p)

    # ---- TurboQuant with Outlier Channel Handling ----
    outlier_results = {}

    print("\n" + "=" * 80)
    print("TurboQuant_MSE with Outlier Channel Handling")
    print("=" * 80)

    # Identify outlier channels
    outlier_idx_32 = identify_outlier_channels(kv_cache, n_outliers=32)
    outlier_idx_64 = identify_outlier_channels(kv_cache, n_outliers=64)
    print(f"  Top-32 outlier channels (by variance): {outlier_idx_32.cpu().tolist()}")
    print(f"  Top-64 outlier channels (by variance): {outlier_idx_64.cpu().tolist()}")

    # Precompute codebooks for the outlier/non-outlier subspace dimensions
    needed_dims = {}
    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        d_high = n_out
        d_low = d - n_out
        if d_high not in needed_dims:
            needed_dims[d_high] = set()
        if d_low not in needed_dims:
            needed_dims[d_low] = set()
        needed_dims[d_high].add(bh)
        needed_dims[d_low].add(bl)

    subspace_codebooks = {}  # (dim, bits) -> codebook
    for dim, bits_set in needed_dims.items():
        for bits in bits_set:
            key = (dim, bits)
            if key not in subspace_codebooks:
                print(f"  Computing Lloyd-Max codebook for d={dim}, {bits}-bit...", end=" ", flush=True)
                t0 = time.time()
                subspace_codebooks[key] = lloyd_max_codebook(dim, bits)
                print(f"done in {time.time() - t0:.2f}s")

    header = f"{'Config':<18} {'Eff.Bits':>8} {'MSE':>12} {'Cos Sim':>12} {'IP Corr':>12} {'Compress':>10}"
    if do_top1:
        header += f" {'Top-1':>8}"
    print(header)
    print("-" * (80 + (9 if do_top1 else 0)))

    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        d_high = n_out
        d_low = d - n_out
        outlier_idx = outlier_idx_32 if n_out == 32 else outlier_idx_64

        # Build per-subspace codebook dicts
        cb_high = {bh: subspace_codebooks[(d_high, bh)]}
        cb_low = {bl: subspace_codebooks[(d_low, bl)]}

        quantizer = TurboQuantOutlier(
            d, bh, bl, outlier_idx, cb_high, cb_low
        ).to(device)

        x_hat = quantizer.quantize_dequantize(all_vecs)
        metrics = compute_metrics(all_vecs, x_hat)
        outlier_results[name] = metrics

        # Storage: each subspace stores its own norms
        bytes_per_vec = (d_high * bh + d_low * bl) / 8 + 8  # +4 per norm (two norms)
        fp16_bytes = 2 * d
        compress = fp16_bytes / bytes_per_vec

        row = (f"{name:<18} {eff:>8.1f} {metrics['mse']:>12.6f} {metrics['cosine_similarity']:>12.6f} "
               f"{metrics['inner_product_correlation']:>12.6f} {compress:>9.1f}x")

        if do_top1:
            q_keys = kv_cache["keys"].clone() if exempt_layers else torch.zeros_like(kv_cache["keys"])
            q_values = torch.zeros_like(kv_cache["values"])
            for layer in range(n_layers):
                for head in range(n_heads):
                    if layer not in exempt_layers:
                        q_keys[layer, head] = quantizer.quantize_dequantize(
                            kv_cache["keys"][layer, head].to(device))
                    q_values[layer, head] = quantizer.quantize_dequantize(
                        kv_cache["values"][layer, head].to(device))
            top1 = eval_topk_match(
                model, input_ids,
                kv_cache["keys"], kv_cache["values"],
                q_keys, q_values,
                n_generate=n_generate,
                ground_truth_tokens=ground_truth,
            )
            outlier_results[name]["top1_match_rate"] = top1["top1_match_rate"]
            row += f" {top1['top1_match_rate']:>7.1%}"

        print(row)

    # --- Outlier vs uniform comparison ---
    print("\n" + "=" * 80)
    print("Comparison: Outlier Handling vs Uniform Bit Width")
    print("=" * 80)
    header = f"{'Method':<25} {'Eff.Bits':>8} {'MSE':>12} {'Cos Sim':>12} {'Compress':>10}"
    if do_top1:
        header += f" {'Top-1':>8}"
    print(header)
    print("-" * (75 + (9 if do_top1 else 0)))

    # For each outlier config, compare against the nearest uniform bit widths
    for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
        if name not in outlier_results:
            continue
        o = outlier_results[name]
        # Find the uniform results that bracket this effective bit rate
        lower_b = int(eff)
        upper_b = lower_b + 1

        for label, result_dict, bits_key in [
            (f"Uniform {lower_b}-bit", mse_results, lower_b),
            (f"Outlier {name}", outlier_results, name),
            (f"Uniform {upper_b}-bit", mse_results, upper_b),
        ]:
            if bits_key in result_dict:
                r = result_dict[bits_key]
                if isinstance(bits_key, int):
                    bpv = bits_key * d / 8 + 4
                    eb = float(bits_key)
                else:
                    bpv = (n_out * bh + (d - n_out) * bl) / 8 + 8
                    eb = eff
                comp = (2 * d) / bpv
                row = (f"{label:<25} {eb:>8.1f} {r['mse']:>12.6f} {r['cosine_similarity']:>12.6f} "
                       f"{comp:>9.1f}x")
                if do_top1 and "top1_match_rate" in r:
                    row += f" {r['top1_match_rate']:>7.1%}"
                print(row)
        print()

    # ---- Residual Window Evaluation ----
    window_results = {}
    if residual_window > 0 and do_top1:
        W = residual_window
        print("\n" + "=" * 80)
        print(f"Residual Window Evaluation (last {W} of {n_tokens} tokens in FP16)")
        print("=" * 80)

        if W >= n_tokens:
            print(f"  Window ({W}) >= sequence length ({n_tokens}), skipping -- "
                  "entire cache would be FP16.")
        else:
            n_compressed = n_tokens - W
            print(f"  Compressed tokens: {n_compressed}, FP16 tokens: {W}")

            header = (f"{'Bits':>5} {'Window':>8} {'Top-1':>8} {'Top-1+W':>8} "
                      f"{'Eff.Compress':>12}")
            print(header)
            print("-" * 55)

            for bits in [2, 3, 4]:
                quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

                # Without window (already computed)
                top1_no_win = mse_results[bits].get("top1_match_rate", None)

                # With window
                q_keys_w, q_values_w = quantize_kv_with_window(quantizer, W)
                top1_w = eval_topk_match(
                    model, input_ids,
                    kv_cache["keys"], kv_cache["values"],
                    q_keys_w, q_values_w,
                    n_generate=n_generate,
                    ground_truth_tokens=ground_truth,
                )

                # Effective compression: W tokens at 16 bits + rest at b bits
                eff_bytes = (n_compressed * (bits * d / 8 + 4) + W * fp16_bytes)
                baseline_bytes = n_tokens * fp16_bytes
                eff_compress = baseline_bytes / eff_bytes

                no_win_str = f"{top1_no_win:>7.1%}" if top1_no_win is not None else "    N/A"
                row = (f"{bits:>5} {W:>8} {no_win_str} "
                       f"{top1_w['top1_match_rate']:>7.1%} {eff_compress:>11.2f}x")
                print(row)

                window_results[bits] = {
                    "top1_no_window": top1_no_win,
                    "top1_with_window": top1_w["top1_match_rate"],
                    "effective_compression": eff_compress,
                    "window_size": W,
                }

    elif residual_window > 0 and not do_top1:
        print(f"\n  Note: --residual-window={residual_window} requires --eval-top1 to measure quality.")

    # ---- Top-k Summary Table ----
    topk_values = [1, 2, 4, 8, 16]
    if do_top1:
        print("\n" + "=" * 80)
        print(f"Top-k Token Match Summary (teacher-forced, {n_generate} tokens)")
        print("=" * 80)

        # Run Top-k for key bit widths (with layer exemption if applicable)
        topk_bits = [2, 3, 4, 5]
        le_suffix = " (with layer exemption)" if exempt_layers else ""
        header = f"{'Bits':>5}"
        for k in topk_values:
            header += f" {'Top-'+str(k):>8}"
        print(header + le_suffix)
        print("-" * (5 + 9 * len(topk_values)))

        for bits in topk_bits:
            if bits not in codebooks:
                continue
            quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)
            q_keys, q_values = quantize_kv_cache_mse(quantizer, skip_key_layers=exempt_layers)
            topk_result = eval_topk_match(
                model, input_ids,
                kv_cache["keys"], kv_cache["values"],
                q_keys, q_values,
                n_generate=n_generate,
                topk_values=topk_values,
                ground_truth_tokens=ground_truth,
            )
            row = f"{bits:>5}"
            for k in topk_values:
                row += f" {topk_result['topk_match_rates'][k]:>7.1%}"
            print(row)

    return mse_results, prod_results, outlier_results, window_results, rotation
