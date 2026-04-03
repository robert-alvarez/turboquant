"""Verify 7B analysis: higher n_generate and multiple context lengths.

Tests the key claims from analysis_why_7b_is_different.md:
1. Layer 0 is the sole cause (L0-only reproduces, skip-L0 fixes)
2. The non-monotonic pattern holds with higher statistical power (n=200)
3. Context length affects the failure pattern

Runs 3 scenarios x 6 bit widths x 3 context lengths.
"""

import torch
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from turboquant.core import TurboQuantMSE, lloyd_max_codebook, generate_rotation_matrix
from eval.metrics import generate_ground_truth, eval_topk_match
from eval.model import _PARAGRAPHS


def capture_kv(model, tokenizer, prompt, device, min_tokens=0):
    """Run forward pass and capture KV cache without reloading the model."""
    from transformers.cache_utils import DynamicCache

    if min_tokens > 0:
        test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        current_tokens = test_ids.shape[1]
        para_idx = 0
        while current_tokens < min_tokens:
            prompt += " " + _PARAGRAPHS[para_idx % len(_PARAGRAPHS)]
            para_idx += 1
            test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            current_tokens = test_ids.shape[1]
        test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if test_ids.shape[1] > min_tokens:
            test_ids = test_ids[:, :min_tokens]
            prompt = tokenizer.decode(test_ids[0], skip_special_tokens=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    n_tokens = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values

    if isinstance(past_kv, DynamicCache):
        n_layers = len(past_kv.layers)
        sample_k = past_kv.layers[0].keys
    else:
        n_layers = len(past_kv)
        sample_k = past_kv[0][0]

    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[3]

    keys = torch.zeros(n_layers, n_kv_heads, n_tokens, head_dim,
                       dtype=torch.float32, device=device)
    values = torch.zeros_like(keys)

    for i in range(n_layers):
        if isinstance(past_kv, DynamicCache):
            keys[i] = past_kv.layers[i].keys[0].float()
            values[i] = past_kv.layers[i].values[0].float()
        else:
            keys[i] = past_kv[i][0][0].float()
            values[i] = past_kv[i][1][0].float()

    return keys, values, input_ids, n_layers, n_tokens


def quantize_kv_selective(keys, values, quantizer, skip_key_layers=None, device="cuda"):
    """Quantize all keys/values, skipping specified layers for keys only."""
    if skip_key_layers is None:
        skip_key_layers = set()

    n_layers, n_heads, n_tokens, d = keys.shape
    q_keys = keys.clone()
    q_values = values.clone()

    for layer in range(n_layers):
        for head in range(n_heads):
            if layer not in skip_key_layers:
                q_keys[layer, head] = quantizer.quantize_dequantize(
                    keys[layer, head].to(device)).cpu()
            q_values[layer, head] = quantizer.quantize_dequantize(
                values[layer, head].to(device)).cpu()
    return q_keys, q_values


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_generate = 200
    d = 128
    bit_widths = [2, 3, 4, 5, 6, 8]

    print("Computing codebooks...")
    codebooks = {b: lloyd_max_codebook(d, b) for b in bit_widths}
    rotation = generate_rotation_matrix(d, seed=42, device=device)

    # Load model once
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    base_prompt = " ".join(_PARAGRAPHS[:3])
    context_lengths = [0, 1000, 5000]  # 0 = default (~168 tokens)
    all_results = {}

    for min_tok in context_lengths:
        keys, values, input_ids, n_layers, n_tokens = capture_kv(
            model, tokenizer, base_prompt, device, min_tokens=min_tok)
        all_layers = set(range(n_layers))

        print(f"\n{'='*80}")
        print(f"Context: {n_tokens} tokens | Generating: {n_generate} tokens")
        print(f"{'='*80}")

        ground_truth = generate_ground_truth(model, input_ids, keys, values, n_generate)

        scenarios = [
            ("Full quantization", lambda: set()),
            ("Only L0 keys quant", lambda: all_layers - {0}),
            ("Skip L0 keys",      lambda: {0}),
        ]

        print(f"\n{'Scenario':<22}", end="")
        for b in bit_widths:
            print(f"  {b:>5}b", end="")
        print()
        print("-" * (22 + 7 * len(bit_widths)))

        results = {}
        for name, skip_fn in scenarios:
            print(f"{name:<22}", end="", flush=True)
            row = []
            for b in bit_widths:
                quantizer = TurboQuantMSE(d, b, codebooks[b], rotation)
                quantizer.to(device)
                skip_keys = skip_fn()
                q_keys, q_values = quantize_kv_selective(
                    keys, values, quantizer, skip_key_layers=skip_keys, device=device)
                result = eval_topk_match(
                    model, input_ids, keys, values, q_keys, q_values,
                    n_generate=n_generate, topk_values=[1],
                    ground_truth_tokens=ground_truth,
                )
                acc = result["top1_match_rate"]
                row.append(acc)
                print(f"  {acc*100:5.1f}%", end="", flush=True)
            print()
            results[name] = row
        all_results[n_tokens] = results

    # Summary tables
    print(f"\n{'='*80}")
    print(f"SUMMARY: Full quantization Top-1 (n_generate={n_generate})")
    print(f"{'='*80}")
    print(f"{'Context':<12}", end="")
    for b in bit_widths:
        print(f"  {b:>5}b", end="")
    print()
    print("-" * (12 + 7 * len(bit_widths)))
    for n_tok in sorted(all_results):
        row = all_results[n_tok]["Full quantization"]
        print(f"{n_tok:>6} tok ", end="")
        for acc in row:
            print(f"  {acc*100:5.1f}%", end="")
        print()

    print(f"\n{'='*80}")
    print(f"SUMMARY: Only L0 keys quantized (n_generate={n_generate})")
    print(f"{'='*80}")
    print(f"{'Context':<12}", end="")
    for b in bit_widths:
        print(f"  {b:>5}b", end="")
    print()
    print("-" * (12 + 7 * len(bit_widths)))
    for n_tok in sorted(all_results):
        row = all_results[n_tok]["Only L0 keys quant"]
        print(f"{n_tok:>6} tok ", end="")
        for acc in row:
            print(f"  {acc*100:5.1f}%", end="")
        print()

    print(f"\n{'='*80}")
    print(f"SUMMARY: Skip L0 keys (should be ~100%)")
    print(f"{'='*80}")
    print(f"{'Context':<12}", end="")
    for b in bit_widths:
        print(f"  {b:>5}b", end="")
    print()
    print("-" * (12 + 7 * len(bit_widths)))
    for n_tok in sorted(all_results):
        row = all_results[n_tok]["Skip L0 keys"]
        print(f"{n_tok:>6} tok ", end="")
        for acc in row:
            print(f"  {acc*100:5.1f}%", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
