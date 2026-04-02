#!/usr/bin/env python3
"""
TurboQuant evaluation CLI.

Usage:
  python run_eval.py                          # Default: Qwen2.5-3B, 168 tokens
  python run_eval.py --eval-top1              # With Top-k token accuracy
  python run_eval.py --model Qwen/Qwen2.5-7B-Instruct --min-tokens 1000 --eval-top1
  python run_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct --image path/to/img.png
  python run_eval.py --skip-model --device cpu # Synthetic data, CPU only
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env file if present (for HF_TOKEN etc.)
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


def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV Cache Compression Evaluation")
    parser.add_argument("--skip-model", action="store_true",
                        help="Use synthetic data instead of loading a real model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model to load for KV cache capture")
    parser.add_argument("--eval-top1", action="store_true",
                        help="Evaluate Top-k token match rate (keeps model in memory)")
    parser.add_argument("--n-generate", type=int, default=50,
                        help="Number of tokens to generate for Top-k evaluation")
    parser.add_argument("--residual-window", type=int, default=0,
                        help="Keep the last W tokens in FP16 (0 = disabled, typical: 128)")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Extend the prompt to at least N tokens (0 = use default prompt)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path or URL to an image for VLM evaluation (requires a VL model)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(__file__).parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Compute codebooks ---
    print("\n" + "=" * 80)
    print("Step 1: Computing Lloyd-Max Codebooks")
    print("=" * 80)
    codebooks = compute_all_codebooks(DEFAULT_DIM, BIT_WIDTHS)

    print("\nCodebook Summary:")
    for b in BIT_WIDTHS:
        cb = codebooks[b]
        print(f"  {b}-bit: {len(cb)} levels, range [{cb[0]:.6f}, {cb[-1]:.6f}]")

    # --- Step 2: Load model or generate synthetic data ---
    print("\n" + "=" * 80)
    print("Step 2: Capturing KV Cache")
    print("=" * 80)

    model, tokenizer, input_ids = None, None, None
    if args.skip_model:
        from eval.model import generate_synthetic_kv
        kv_cache = generate_synthetic_kv(n_layers=24, n_heads=4, n_tokens=200, d=DEFAULT_DIM, device=device)
    else:
        from eval.model import load_model_and_capture_kv
        kv_cache, model, tokenizer, input_ids = load_model_and_capture_kv(
            args.model, device=device, min_tokens=args.min_tokens,
            image_path=args.image,
        )
        if kv_cache["head_dim"] != DEFAULT_DIM:
            print(f"\nNote: model head_dim={kv_cache['head_dim']}, recomputing codebooks...")
            codebooks = compute_all_codebooks(kv_cache["head_dim"], BIT_WIDTHS)

    # --- Step 3: GPU Evaluation ---
    print("\n" + "=" * 80)
    print("Step 3: GPU Evaluation")
    print("=" * 80)
    from eval.gpu_eval import run_gpu_evaluation
    mse_results, prod_results, outlier_results, window_results, rotation = run_gpu_evaluation(
        kv_cache, codebooks, device,
        model=model, input_ids=input_ids,
        eval_top1=args.eval_top1, n_generate=args.n_generate,
        residual_window=args.residual_window,
    )

    # Free model after evaluation
    if model is not None:
        del model, tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None

    # --- Step 4: Visualizations ---
    print("\n" + "=" * 80)
    print("Step 4: Generating Visualizations")
    print("=" * 80)
    if args.skip_model:
        print("  Skipping plots (synthetic data has no outlier structure — plots would be misleading)")
    else:
        from eval.visualize import plot_all
        plot_all(kv_cache, codebooks, rotation, mse_results, prod_results,
                 outlier_results, device)

    # --- Step 5: Disk-based compression ---
    print("\n" + "=" * 80)
    print("Step 5: Disk-Based Compression (Model Scale)")
    print("=" * 80)
    from eval.disk_eval import run_disk_evaluation
    run_disk_evaluation(kv_cache, codebooks, device)

    # Free KV cache before large-scale test
    del kv_cache
    torch.cuda.empty_cache() if device == "cuda" else None

    # --- Step 6: Large-scale disk evaluation ---
    if DEFAULT_DIM not in [128]:
        codebooks_128 = compute_all_codebooks(128, [3, 4])
    else:
        codebooks_128 = codebooks
    from eval.disk_eval import run_large_disk_evaluation
    run_large_disk_evaluation(codebooks_128, device)

    print("\n" + "=" * 80)
    print("All done! Output files in:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
