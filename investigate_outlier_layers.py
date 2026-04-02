"""Why does only the 7B model have extreme outlier key layers?

Known facts:
  3B:  Layer 0 = 172 (8.4x median)  — mild
  7B:  Layer 0 = 274 (13.2x), Layer 27 = 243 (11.8x) — severe
  14B: No outliers
  32B: No outliers

This script profiles all 4 models to understand the structural cause.
"""

import torch
import numpy as np
import sys, os, gc

sys.path.insert(0, os.path.dirname(__file__))

def profile_model(model_name, device="cuda", min_tokens=1000):
    """Load model, capture KV cache, return per-layer key/value norm stats."""
    from eval.model import load_model_and_capture_kv

    kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
        model_name=model_name, device=device, min_tokens=min_tokens,
    )

    keys = kv_cache["keys"]      # (n_layers, n_heads, n_tokens, d)
    values = kv_cache["values"]
    n_layers, n_heads, n_tokens, d = keys.shape

    # Per-layer key norms
    key_norms = []
    key_stds = []
    val_norms = []
    val_stds = []
    # Per-layer key norm variance across tokens (do norms vary a lot or are they uniform?)
    key_norm_cv = []  # coefficient of variation

    for layer in range(n_layers):
        k = keys[layer].reshape(-1, d)
        v = values[layer].reshape(-1, d)
        kn = k.norm(dim=-1)
        vn = v.norm(dim=-1)
        key_norms.append(kn.mean().item())
        key_stds.append(kn.std().item())
        val_norms.append(vn.mean().item())
        val_stds.append(vn.std().item())
        key_norm_cv.append((kn.std() / kn.mean()).item())

    # Per-channel variance of keys (which channels are outliers?)
    # Flatten across layers/heads/tokens to get per-channel stats
    all_keys = keys.reshape(-1, d)  # (n_layers * n_heads * n_tokens, d)
    channel_var = all_keys.var(dim=0).cpu().numpy()  # (d,)
    channel_mean = all_keys.mean(dim=0).cpu().numpy()

    # Check if specific channels dominate
    sorted_var_idx = np.argsort(channel_var)[::-1]
    top10_var = channel_var[sorted_var_idx[:10]]
    median_var = np.median(channel_var)

    # Per-layer: which channels carry the most energy in outlier layers?
    layer_channel_vars = []
    for layer in range(n_layers):
        lk = keys[layer].reshape(-1, d)
        lv = lk.var(dim=0).cpu().numpy()
        layer_channel_vars.append(lv)

    # Check the key projection weight norms if accessible
    key_proj_norms = []
    try:
        for layer in range(n_layers):
            # Qwen2.5 architecture
            k_proj = model.model.layers[layer].self_attn.k_proj.weight
            key_proj_norms.append(k_proj.norm().item())
    except Exception as e:
        print(f"  Could not access key projection weights: {e}")

    # Free memory
    del model, tokenizer, kv_cache, keys, values, all_keys, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "d": d,
        "key_norms": key_norms,
        "key_stds": key_stds,
        "val_norms": val_norms,
        "val_stds": val_stds,
        "key_norm_cv": key_norm_cv,
        "channel_var": channel_var,
        "channel_mean": channel_mean,
        "layer_channel_vars": layer_channel_vars,
        "key_proj_norms": key_proj_norms,
    }


def print_profile(p):
    n = p["n_layers"]
    kn = p["key_norms"]
    vn = p["val_norms"]
    cv = p["key_norm_cv"]
    median_kn = float(sorted(kn)[n // 2])

    short_name = p["model"].split("/")[-1]
    print(f"\n{'='*80}")
    print(f"{short_name}: {n}L x {p['n_heads']}H x {p['n_tokens']}T x {p['d']}d")
    print(f"{'='*80}")

    # Key norm profile
    print(f"\n  Per-Layer Key Norms (median={median_kn:.1f}):")
    print(f"  {'Layer':>5}  {'Mean Norm':>10}  {'Std':>8}  {'CV':>6}  {'Ratio':>6}  {'Flag':>8}")
    print(f"  {'-'*55}")
    for i in range(n):
        ratio = kn[i] / median_kn
        flag = "OUTLIER" if ratio > 4 else ("high" if ratio > 2 else "")
        if ratio > 1.5 or i == 0 or i == n - 1:
            print(f"  {i:5d}  {kn[i]:10.1f}  {p['key_stds'][i]:8.1f}  {cv[i]:6.3f}  {ratio:5.1f}x  {flag:>8}")

    # Value norm profile (for comparison)
    median_vn = float(sorted(vn)[n // 2])
    print(f"\n  Per-Layer Value Norms (median={median_vn:.1f}):")
    max_vn_ratio = max(v / median_vn for v in vn)
    max_vn_layer = max(range(n), key=lambda i: vn[i])
    print(f"  Max value norm: Layer {max_vn_layer} = {vn[max_vn_layer]:.1f} ({vn[max_vn_layer]/median_vn:.1f}x median)")
    print(f"  Value norms are {'stable' if max_vn_ratio < 2 else 'VARIABLE'} (max ratio: {max_vn_ratio:.1f}x)")

    # Channel variance
    cv_arr = p["channel_var"]
    sorted_idx = np.argsort(cv_arr)[::-1]
    median_cv = np.median(cv_arr)
    print(f"\n  Per-Channel Variance (global, median={median_cv:.2f}):")
    print(f"  Top 5 channels: {sorted_idx[:5].tolist()} with var {cv_arr[sorted_idx[:5]].tolist()}")
    print(f"  Top channel / median: {cv_arr[sorted_idx[0]] / median_cv:.1f}x")

    # Key projection weight norms
    if p["key_proj_norms"]:
        kpn = p["key_proj_norms"]
        median_kpn = float(sorted(kpn)[n // 2])
        print(f"\n  Key Projection Weight Norms (median={median_kpn:.1f}):")
        for i in range(n):
            ratio = kpn[i] / median_kpn
            if ratio > 1.5 or i == 0 or i == n - 1:
                flag = "OUTLIER" if ratio > 2 else ("high" if ratio > 1.5 else "")
                print(f"  Layer {i:3d}: ||W_k|| = {kpn[i]:.1f} ({ratio:.2f}x median) {flag}")

    # Per-layer channel variance for outlier layers
    print(f"\n  Channel Variance in Outlier vs Normal Layers:")
    outlier_layers = [i for i in range(n) if kn[i] / median_kn > 4]
    normal_layer = n // 2  # pick middle layer as normal reference

    if outlier_layers:
        for ol in outlier_layers:
            ol_var = p["layer_channel_vars"][ol]
            nl_var = p["layer_channel_vars"][normal_layer]
            ol_top5 = np.argsort(ol_var)[::-1][:5]
            nl_top5 = np.argsort(nl_var)[::-1][:5]
            print(f"  Layer {ol} (outlier): max_ch_var={ol_var.max():.1f}, median_ch_var={np.median(ol_var):.2f}, top channels={ol_top5.tolist()}")
            print(f"  Layer {normal_layer} (normal):  max_ch_var={nl_var.max():.1f}, median_ch_var={np.median(nl_var):.2f}, top channels={nl_top5.tolist()}")
    else:
        # Show first and middle layer for models without outliers
        for li in [0, normal_layer, n - 1]:
            lv = p["layer_channel_vars"][li]
            print(f"  Layer {li}: max_ch_var={lv.max():.1f}, median_ch_var={np.median(lv):.2f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    profiles = []
    for m in models:
        print(f"\n{'#'*80}")
        print(f"# Profiling {m}")
        print(f"{'#'*80}")
        p = profile_model(m, device=device)
        profiles.append(p)
        print_profile(p)

    # Cross-model comparison
    print(f"\n{'='*80}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*80}")

    print(f"\n  Key Norm Outlier Summary:")
    print(f"  {'Model':<30}  {'Layers':>6}  {'Outlier Layers':>40}  {'Max Ratio':>10}")
    print(f"  {'-'*90}")
    for p in profiles:
        n = p["n_layers"]
        kn = p["key_norms"]
        median_kn = float(sorted(kn)[n // 2])
        outliers = [(i, kn[i] / median_kn) for i in range(n) if kn[i] / median_kn > 4]
        max_ratio = max(kn[i] / median_kn for i in range(n))
        outlier_str = ", ".join(f"L{i}({r:.1f}x)" for i, r in outliers) if outliers else "None"
        short = p["model"].split("/")[-1]
        print(f"  {short:<30}  {n:6d}  {outlier_str:>40}  {max_ratio:9.1f}x")

    # Compare Layer 0 and last layer key norms across models
    print(f"\n  Layer 0 Key Norms (the 'embedding' layer):")
    for p in profiles:
        short = p["model"].split("/")[-1]
        median_kn = float(sorted(p["key_norms"])[p["n_layers"] // 2])
        print(f"  {short:<30}  L0={p['key_norms'][0]:.1f} ({p['key_norms'][0]/median_kn:.1f}x)  median={median_kn:.1f}")

    print(f"\n  Last Layer Key Norms:")
    for p in profiles:
        short = p["model"].split("/")[-1]
        n = p["n_layers"]
        median_kn = float(sorted(p["key_norms"])[n // 2])
        print(f"  {short:<30}  L{n-1}={p['key_norms'][n-1]:.1f} ({p['key_norms'][n-1]/median_kn:.1f}x)  median={median_kn:.1f}")

    # Key projection weight comparison
    print(f"\n  Key Projection Weight Norms — Layer 0 vs Median:")
    for p in profiles:
        if p["key_proj_norms"]:
            short = p["model"].split("/")[-1]
            kpn = p["key_proj_norms"]
            n = p["n_layers"]
            median_kpn = float(sorted(kpn)[n // 2])
            print(f"  {short:<30}  L0={kpn[0]:.1f} ({kpn[0]/median_kpn:.2f}x)  median={median_kpn:.1f}")
            # For 7B, also show Layer 27
            if "7B" in short:
                print(f"  {'':<30}  L27={kpn[27]:.1f} ({kpn[27]/median_kpn:.2f}x)")

if __name__ == "__main__":
    main()
