#!/usr/bin/env python3
"""
Worker for concurrent FlashBlade I/O benchmark.

Invoked by bench_flashblade.py with CUDA_VISIBLE_DEVICES set.
Performs: compress (GPU) → write (disk) → sync → read (disk) → decompress (CPU) → verify.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

import torch
import torch.nn.functional as F

from turboquant.core import TurboQuantMSE, generate_rotation_matrix, DEFAULT_SEED
from turboquant import serialize_compressed_kv, deserialize_compressed_kv, dequantize_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--bits", type=int, required=True)
    args = parser.parse_args()

    wid = args.worker_id
    device = "cuda"

    # Load shared data
    kv_data = torch.load(os.path.join(args.shared_dir, "kv_cache.pt"), weights_only=False)
    codebooks = torch.load(os.path.join(args.shared_dir, "codebooks.pt"), weights_only=False)

    keys = kv_data["keys"].to(device)
    values = kv_data["values"].to(device)
    d = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]
    n_tokens = kv_data["n_tokens"]
    bits = args.bits

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)
    quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

    # FP16 baseline size
    fp16_size = 2 * n_layers * n_heads * n_tokens * d * 2

    # ── Compress (GPU) ──
    torch.cuda.synchronize()
    t0 = time.time()
    all_indices = []
    for kv_tensor in [keys, values]:
        for layer in range(n_layers):
            for head in range(n_heads):
                indices, norms = quantizer.quantize(kv_tensor[layer, head])
                all_indices.append((indices, norms))
    torch.cuda.synchronize()
    t_compress = time.time() - t0

    # ── Write (disk) ──
    filepath = os.path.join(args.output_dir, f"worker_{wid}_{bits}bit.tqkv")
    t0 = time.time()
    serialize_compressed_kv(
        filepath, rotation, codebooks[bits], bits, all_indices,
        n_layers * 2, n_heads, n_tokens, d, mode="mse",
    )
    # Force flush to storage device
    fd = os.open(filepath, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    t_write = time.time() - t0
    file_size = os.path.getsize(filepath)

    # ── Hint OS to drop page cache before read ──
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (AttributeError, OSError):
        pass

    # ── Read (disk) ──
    t0 = time.time()
    data = deserialize_compressed_kv(filepath)
    t_read = time.time() - t0

    # ── Decompress (CPU) ──
    t0 = time.time()
    reconstructed = dequantize_from_disk(data)
    t_decompress = time.time() - t0

    # ── Verify ──
    orig = torch.cat([keys, values], dim=0).cpu().float()
    recon = reconstructed.reshape_as(orig)
    mse = ((orig - recon) ** 2).sum(dim=-1).mean().item()
    cos_sim = F.cosine_similarity(orig.reshape(-1, d), recon.reshape(-1, d), dim=-1).mean().item()

    results = {
        "worker_id": wid, "bits": bits, "n_tokens": n_tokens,
        "fp16_size": fp16_size, "file_size": file_size,
        "t_compress": t_compress, "t_write": t_write,
        "t_read": t_read, "t_decompress": t_decompress,
        "mse": mse, "cos_sim": cos_sim,
    }

    with open(os.path.join(args.shared_dir, f"bench_{wid}.json"), "w") as f:
        json.dump(results, f)

    os.remove(filepath)


if __name__ == "__main__":
    main()
