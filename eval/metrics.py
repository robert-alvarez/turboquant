"""
Quality metrics and Top-k teacher-forced evaluation.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F


def compute_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    """
    Compute reconstruction quality metrics between original and reconstructed vectors.

    Returns dict with mse, cosine_similarity, inner_product_correlation,
    true_inner_products, estimated_inner_products.
    """
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    cos_sim = F.cosine_similarity(x, x_hat, dim=-1).mean().item()

    batch = x.shape[0]
    if batch > 1:
        n_pairs = min(5000, batch * (batch - 1) // 2)
        rng = np.random.RandomState(42)
        idx_i = rng.randint(0, batch, size=n_pairs)
        idx_j = rng.randint(0, batch, size=n_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        true_ips = (x[idx_i] * x[idx_j]).sum(dim=-1).cpu().numpy()
        est_ips = (x_hat[idx_i] * x_hat[idx_j]).sum(dim=-1).cpu().numpy()
        ip_corr = np.corrcoef(true_ips, est_ips)[0, 1] if len(true_ips) > 1 else 1.0
    else:
        true_ips = np.array([0.0])
        est_ips = np.array([0.0])
        ip_corr = 1.0

    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "inner_product_correlation": ip_corr,
        "true_inner_products": true_ips,
        "estimated_inner_products": est_ips,
    }


def build_dynamic_cache(keys: torch.Tensor, values: torch.Tensor, dtype=torch.float16):
    """
    Build a DynamicCache from key/value tensors.

    Args:
        keys: (n_layers, n_heads, n_tokens, head_dim)
        values: same shape
        dtype: target dtype for the cache tensors (model expects float16)
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer

    cache = DynamicCache()
    n_layers = keys.shape[0]
    device = keys.device

    for i in range(n_layers):
        layer = DynamicLayer()
        layer.keys = keys[i].unsqueeze(0).to(dtype=dtype)
        layer.values = values[i].unsqueeze(0).to(dtype=dtype)
        layer.is_initialized = True
        layer.dtype = dtype
        layer.device = device
        cache.layers.append(layer)

    return cache


def teacher_forced_generate(model, input_ids: torch.Tensor, past_kv,
                            ground_truth_tokens: torch.Tensor,
                            topk_values: list = None) -> tuple:
    """
    Teacher-forced generation: at each step, feed the ground truth token
    (not the model's own prediction) and record the model's top-k predictions.

    Returns:
        (predictions, topk_sets) where:
          predictions: (n,) tensor of argmax predicted token IDs
          topk_sets: dict mapping k -> (n, k) tensor of top-k token IDs per position
    """
    if topk_values is None:
        topk_values = [1]
    max_k = max(topk_values)

    device = input_ids.device
    n = len(ground_truth_tokens)
    predictions = []
    all_topk = []

    with torch.no_grad():
        cache = copy.deepcopy(past_kv)

        token = input_ids[:, -1:]
        out = model(token, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        topk_ids = logits.topk(max_k, dim=-1).indices[0]
        predictions.append(topk_ids[0].item())
        all_topk.append(topk_ids)
        cache = out.past_key_values

        for i in range(n - 1):
            token = ground_truth_tokens[i].view(1, 1).to(device)
            out = model(token, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :]
            topk_ids = logits.topk(max_k, dim=-1).indices[0]
            predictions.append(topk_ids[0].item())
            all_topk.append(topk_ids)
            cache = out.past_key_values

    predictions = torch.tensor(predictions, device=device)
    all_topk = torch.stack(all_topk)

    topk_sets = {}
    for k in topk_values:
        topk_sets[k] = all_topk[:, :k]

    return predictions, topk_sets


def generate_ground_truth(model, input_ids: torch.Tensor,
                          original_keys: torch.Tensor, original_values: torch.Tensor,
                          n_generate: int = 50) -> torch.Tensor:
    """
    Generate ground truth tokens once from the original (unquantized) KV cache.

    Must be called once and the result reused for all quantized evaluations
    to avoid CUDA non-determinism causing different ground truths per bit width.
    """
    device = input_ids.device
    orig_cache = build_dynamic_cache(
        original_keys[:, :, :-1, :], original_values[:, :, :-1, :]
    )

    ground_truth = []
    with torch.no_grad():
        token = input_ids[:, -1:]
        cache = orig_cache
        for _ in range(n_generate):
            out = model(token, past_key_values=cache, use_cache=True)
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ground_truth.append(token.item())
            cache = out.past_key_values
    return torch.tensor(ground_truth, device=device)


def eval_topk_match(model, input_ids: torch.Tensor,
                    original_keys: torch.Tensor, original_values: torch.Tensor,
                    quantized_keys: torch.Tensor, quantized_values: torch.Tensor,
                    n_generate: int = 50,
                    topk_values: list = None,
                    ground_truth_tokens: torch.Tensor = None) -> dict:
    """
    Measure Top-k token match rate using teacher-forced evaluation.

    For each k in topk_values, checks whether the ground truth token at each
    position appears in the quantized model's top-k predictions.

    If ground_truth_tokens is provided, skips regeneration and uses those directly.
    This is important: CUDA non-determinism means regenerating ground truth for
    each bit width produces different targets, causing nonsensical comparisons.
    """
    if topk_values is None:
        topk_values = [1, 2, 4, 8, 16]

    device = input_ids.device

    if ground_truth_tokens is None:
        ground_truth = generate_ground_truth(
            model, input_ids, original_keys, original_values, n_generate
        )
    else:
        ground_truth = ground_truth_tokens

    quant_cache = build_dynamic_cache(
        quantized_keys[:, :, :-1, :], quantized_values[:, :, :-1, :]
    )
    quant_preds, quant_topk = teacher_forced_generate(
        model, input_ids, quant_cache, ground_truth, topk_values=topk_values
    )

    topk_match_rates = {}
    for k in topk_values:
        topk_set = quant_topk[k]
        gt_expanded = ground_truth.unsqueeze(-1).expand_as(topk_set)
        matches = (topk_set == gt_expanded).any(dim=-1).float()
        topk_match_rates[k] = matches.mean().item()

    return {
        "top1_match_rate": topk_match_rates.get(1, topk_match_rates[topk_values[0]]),
        "topk_match_rates": topk_match_rates,
        "ground_truth": ground_truth,
        "quantized_predictions": quant_preds,
        "n_matched": int((quant_preds == ground_truth).sum().item()),
        "n_total": n_generate,
    }
