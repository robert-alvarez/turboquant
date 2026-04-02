"""
Matplotlib visualizations for TurboQuant evaluation results.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from turboquant.core import beta_pdf_sphere, OUTLIER_CONFIGS

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def plot_all(kv_cache: dict, codebooks: dict, rotation: torch.Tensor,
             mse_results: dict, prod_results: dict,
             outlier_results: dict = None, device: str = "cuda"):
    """Generate all four visualization panels and save as PNGs."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    d = kv_cache["head_dim"]
    bits_list = sorted(mse_results.keys())
    fp16_bytes = 2 * d

    # ---- (b) Cosine similarity vs bit width (with outlier configs) ----
    fig, ax = plt.subplots(figsize=(10, 6))

    cos_mse = [mse_results[b]["cosine_similarity"] for b in bits_list]
    ax.plot(bits_list, cos_mse, "o-", linewidth=2, markersize=8, color="#1f77b4",
            label="MSE (uniform)", zorder=3)

    if prod_results:
        bits_prod = sorted(prod_results.keys())
        cos_prod = [prod_results[b]["cosine_similarity"] for b in bits_prod]
        ax.plot(bits_prod, cos_prod, "s--", linewidth=2, markersize=8, color="#ff7f0e",
                label="MSE+QJL (uniform)", zorder=3)

    if outlier_results:
        for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
            if name in outlier_results:
                cs = outlier_results[name]["cosine_similarity"]
                ax.plot(eff, cs, "D", markersize=10, color="#2ca02c", zorder=4,
                        markeredgecolor="black", markeredgewidth=0.8)
                ax.annotate(name, (eff, cs), textcoords="offset points",
                            xytext=(6, -12), fontsize=8, color="#2ca02c")
        ax.plot([], [], "D", markersize=10, color="#2ca02c", markeredgecolor="black",
                markeredgewidth=0.8, label="MSE (outlier mixed-prec.)")

    ax.set_xlabel("Effective Bit Width", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Reconstruction Quality: Cosine Similarity vs Bit Width", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    all_cos = cos_mse + (cos_prod if prod_results else [])
    if outlier_results:
        all_cos += [outlier_results[n]["cosine_similarity"]
                    for n, *_ in OUTLIER_CONFIGS if n in outlier_results]
    ax.set_ylim(bottom=max(0.9, min(all_cos) - 0.02), top=1.002)
    ax.set_xlim(1.5, 8.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "b_cosine_vs_bits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'b_cosine_vs_bits.png'}")

    # ---- (d) Scatter plot of true vs estimated inner products ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rep_bits_mse = 3 if 3 in mse_results else bits_list[1]
    true_ip = mse_results[rep_bits_mse]["true_inner_products"]
    est_ip = mse_results[rep_bits_mse]["estimated_inner_products"]
    corr = mse_results[rep_bits_mse]["inner_product_correlation"]

    def robust_lims(a, b, pct=99):
        combined = np.concatenate([a, b])
        lo = np.percentile(combined, (100 - pct) / 2)
        hi = np.percentile(combined, 100 - (100 - pct) / 2)
        margin = (hi - lo) * 0.05
        return lo - margin, hi + margin

    lo, hi = robust_lims(true_ip, est_ip)
    axes[0].scatter(true_ip, est_ip, alpha=0.2, s=6, c="#1f77b4", rasterized=True)
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y = x")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("True Inner Product", fontsize=11)
    axes[0].set_ylabel("Estimated Inner Product", fontsize=11)
    axes[0].set_title(f"TurboQuant_MSE ({rep_bits_mse}-bit)\nr = {corr:.4f}", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    if prod_results:
        rep_bits_prod = 3 if 3 in prod_results else sorted(prod_results.keys())[0]
        true_ip_p = prod_results[rep_bits_prod]["true_inner_products"]
        est_ip_p = prod_results[rep_bits_prod]["estimated_inner_products"]
        corr_p = prod_results[rep_bits_prod]["inner_product_correlation"]

        lo_p, hi_p = robust_lims(true_ip_p, est_ip_p)
        axes[1].scatter(true_ip_p, est_ip_p, alpha=0.2, s=6, c="#ff7f0e", rasterized=True)
        axes[1].plot([lo_p, hi_p], [lo_p, hi_p], "r--", linewidth=1.2, label="y = x")
        axes[1].set_xlim(lo_p, hi_p)
        axes[1].set_ylim(lo_p, hi_p)
        axes[1].set_xlabel("True Inner Product", fontsize=11)
        axes[1].set_ylabel("Estimated Inner Product", fontsize=11)
        axes[1].set_title(f"TurboQuant_prod ({rep_bits_prod}-bit)\nr = {corr_p:.4f}", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].set_aspect("equal")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No prod results", ha="center", va="center",
                     transform=axes[1].transAxes)

    fig.suptitle("True vs Estimated Inner Products (bias vs variance)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "d_inner_product_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'd_inner_product_scatter.png'}")

    # ---- (a) Per-channel variance profile + coordinate distribution ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_kv = torch.cat([
        kv_cache["keys"].reshape(-1, d).cpu(),
        kv_cache["values"].reshape(-1, d).cpu(),
    ], dim=0)
    norms_all = all_kv.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    all_unit = all_kv / norms_all
    n_vecs = all_unit.shape[0]

    rot = rotation.cpu()
    all_rotated = (rot @ all_unit.T).T

    std_before = all_unit.std(dim=0).numpy()
    std_after = all_rotated.std(dim=0).numpy()
    theoretical_std = 1.0 / np.sqrt(d)

    axes[0, 0].bar(range(d), std_before, width=1.0, color="#1f77b4", alpha=0.8)
    axes[0, 0].axhline(y=theoretical_std, color="red", linestyle="--", linewidth=1.5,
                        label=f"Theoretical 1/sqrt(d) = {theoretical_std:.4f}")
    axes[0, 0].set_title("Before Rotation: Per-Channel Std Dev", fontsize=12)
    axes[0, 0].set_xlabel("Channel index", fontsize=11)
    axes[0, 0].set_ylabel("Std dev", fontsize=11)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xlim(-1, d)

    axes[0, 1].bar(range(d), std_after, width=1.0, color="#2ca02c", alpha=0.8)
    axes[0, 1].axhline(y=theoretical_std, color="red", linestyle="--", linewidth=1.5,
                        label=f"Theoretical 1/sqrt(d) = {theoretical_std:.4f}")
    axes[0, 1].set_title("After Rotation: Per-Channel Std Dev", fontsize=12)
    axes[0, 1].set_xlabel("Channel index", fontsize=11)
    axes[0, 1].set_ylabel("Std dev", fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlim(-1, d)
    ymax = max(std_before.max(), std_after.max()) * 1.1
    axes[0, 0].set_ylim(0, ymax)
    axes[0, 1].set_ylim(0, ymax)

    x_range = np.linspace(-0.4, 0.4, 500)

    coords_before = all_unit.flatten().numpy()
    axes[1, 0].hist(coords_before, bins=150, density=True, alpha=0.7, color="#1f77b4",
                     label=f"Empirical ({n_vecs} vectors)")
    axes[1, 0].plot(x_range, beta_pdf_sphere(x_range, d), "r-", linewidth=2.5,
                     label=f"Beta PDF (d={d})")
    axes[1, 0].set_title("Before Rotation: Aggregate Histogram", fontsize=12)
    axes[1, 0].set_xlabel("Coordinate value", fontsize=11)
    axes[1, 0].set_ylabel("Density", fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xlim(-0.35, 0.35)

    coords_after = all_rotated.flatten().numpy()
    axes[1, 1].hist(coords_after, bins=150, density=True, alpha=0.7, color="#2ca02c",
                     label=f"Empirical ({n_vecs} vectors)")
    axes[1, 1].plot(x_range, beta_pdf_sphere(x_range, d), "r-", linewidth=2.5,
                     label=f"Beta PDF (d={d})")
    axes[1, 1].set_title("After Rotation: Aggregate Histogram", fontsize=12)
    axes[1, 1].set_xlabel("Coordinate value", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_xlim(-0.35, 0.35)

    fig.suptitle("Effect of Random Orthogonal Rotation on KV Cache Vectors", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "a_coordinate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'a_coordinate_distribution.png'}")

    # ---- (c) Quality-Compression tradeoff ----
    fig, ax = plt.subplots(figsize=(10, 6))

    for b in bits_list:
        bpv = b * d / 8 + 4
        comp = fp16_bytes / bpv
        cs = mse_results[b]["cosine_similarity"]
        ax.plot(comp, cs, "o", markersize=9, color="#1f77b4", zorder=3)
        ax.annotate(f"{b}b", (comp, cs), textcoords="offset points",
                    xytext=(-5, 8), fontsize=9, color="#1f77b4", fontweight="bold")
    mse_comps = [fp16_bytes / (b * d / 8 + 4) for b in bits_list]
    mse_coss = [mse_results[b]["cosine_similarity"] for b in bits_list]
    ax.plot(mse_comps, mse_coss, "-", linewidth=1.5, color="#1f77b4", alpha=0.5)
    ax.plot([], [], "o-", color="#1f77b4", markersize=9, label="MSE (uniform)")

    if prod_results:
        bits_prod = sorted(prod_results.keys())
        for b in bits_prod:
            bpv = b * d / 8 + 8
            comp = fp16_bytes / bpv
            cs = prod_results[b]["cosine_similarity"]
            ax.plot(comp, cs, "s", markersize=8, color="#ff7f0e", zorder=3)
        prod_comps = [fp16_bytes / (b * d / 8 + 8) for b in bits_prod]
        prod_coss = [prod_results[b]["cosine_similarity"] for b in bits_prod]
        ax.plot(prod_comps, prod_coss, "--", linewidth=1.5, color="#ff7f0e", alpha=0.5)
        ax.plot([], [], "s--", color="#ff7f0e", markersize=8, label="MSE+QJL (uniform)")

    if outlier_results:
        for name, n_out, bh, bl, eff in OUTLIER_CONFIGS:
            if name in outlier_results:
                d_high, d_low = n_out, d - n_out
                bpv = (d_high * bh + d_low * bl) / 8 + 8
                comp = fp16_bytes / bpv
                cs = outlier_results[name]["cosine_similarity"]
                ax.plot(comp, cs, "D", markersize=10, color="#2ca02c", zorder=4,
                        markeredgecolor="black", markeredgewidth=0.8)
                ax.annotate(name, (comp, cs), textcoords="offset points",
                            xytext=(6, -12), fontsize=8, color="#2ca02c")
        ax.plot([], [], "D", markersize=10, color="#2ca02c", markeredgecolor="black",
                markeredgewidth=0.8, label="MSE (outlier mixed-prec.)")

    ax.set_xlabel("Compression Ratio (vs FP16)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Quality-Compression Tradeoff", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.9, min(all_cos) - 0.02), top=1.002)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "c_compression_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'c_compression_ratio.png'}")
