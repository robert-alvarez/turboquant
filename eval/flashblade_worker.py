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

# Respect OMP_NUM_THREADS set by parent for NUMA-aware CPU pinning
_n_threads = os.environ.get("OMP_NUM_THREADS")
if _n_threads:
    torch.set_num_threads(int(_n_threads))
    torch.set_num_interop_threads(int(_n_threads))

from turboquant.core import TurboQuantMSE, generate_rotation_matrix, DEFAULT_SEED
from turboquant import (
    serialize_compressed_kv, deserialize_compressed_kv, dequantize_from_disk,
    write_direct, read_direct,
)


# ---------------------------------------------------------------------------
# GPU-side bit packing/unpacking for GDS path
# ---------------------------------------------------------------------------

def gpu_pack(indices, bits):
    """Pack quantized indices on GPU. indices: (N, D) long tensor, returns (N, packed_D) uint8."""
    idx = indices.to(torch.uint8)
    if bits == 2:
        idx = idx.reshape(idx.shape[0], -1, 4)
        return ((idx[..., 0] << 6) | (idx[..., 1] << 4) | (idx[..., 2] << 2) | idx[..., 3])
    elif bits == 3:
        idx = idx.reshape(idx.shape[0], -1, 8)
        b0 = (idx[..., 0] << 5) | (idx[..., 1] << 2) | (idx[..., 2] >> 1)
        b1 = ((idx[..., 2] & 1) << 7) | (idx[..., 3] << 4) | (idx[..., 4] << 1) | (idx[..., 5] >> 2)
        b2 = ((idx[..., 5] & 3) << 6) | (idx[..., 6] << 3) | idx[..., 7]
        return torch.stack([b0, b1, b2], dim=-1).reshape(idx.shape[0], -1)
    elif bits == 4:
        idx = idx.reshape(idx.shape[0], -1, 2)
        return (idx[..., 0] << 4) | idx[..., 1]
    elif bits == 5:
        idx = idx.reshape(idx.shape[0], -1, 8)
        b0 = (idx[..., 0] << 3) | (idx[..., 1] >> 2)
        b1 = ((idx[..., 1] & 3) << 6) | (idx[..., 2] << 1) | (idx[..., 3] >> 4)
        b2 = ((idx[..., 3] & 0xF) << 4) | (idx[..., 4] >> 1)
        b3 = ((idx[..., 4] & 1) << 7) | (idx[..., 5] << 2) | (idx[..., 6] >> 3)
        b4 = ((idx[..., 6] & 7) << 5) | idx[..., 7]
        return torch.stack([b0, b1, b2, b3, b4], dim=-1).reshape(idx.shape[0], -1)
    elif bits == 6:
        idx = idx.reshape(idx.shape[0], -1, 4)
        b0 = (idx[..., 0] << 2) | (idx[..., 1] >> 4)
        b1 = ((idx[..., 1] & 0xF) << 4) | (idx[..., 2] >> 2)
        b2 = ((idx[..., 2] & 3) << 6) | idx[..., 3]
        return torch.stack([b0, b1, b2], dim=-1).reshape(idx.shape[0], -1)
    elif bits == 8:
        return idx
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def gpu_unpack(packed, bits, d):
    """Unpack GPU-packed indices. packed: (N, packed_D) uint8, returns (N, D) long tensor."""
    n = packed.shape[0]
    if bits == 2:
        p = packed.unsqueeze(-1)
        v = torch.stack([(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], dim=-1)
        return v.reshape(n, -1)[:, :d].long()
    elif bits == 3:
        p = packed.reshape(n, -1, 3)
        b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
        v = torch.stack([
            (b0 >> 5) & 7, (b0 >> 2) & 7, ((b0 & 3) << 1) | (b1 >> 7),
            (b1 >> 4) & 7, (b1 >> 1) & 7, ((b1 & 1) << 2) | (b2 >> 6),
            (b2 >> 3) & 7, b2 & 7,
        ], dim=-1)
        return v.reshape(n, -1)[:, :d].long()
    elif bits == 4:
        p = packed.unsqueeze(-1)
        v = torch.stack([(p >> 4) & 0xF, p & 0xF], dim=-1)
        return v.reshape(n, -1)[:, :d].long()
    elif bits == 5:
        p = packed.reshape(n, -1, 5)
        b0, b1, b2, b3, b4 = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
        v = torch.stack([
            (b0 >> 3) & 0x1F, ((b0 & 7) << 2) | (b1 >> 6),
            (b1 >> 1) & 0x1F, ((b1 & 1) << 4) | (b2 >> 4),
            ((b2 & 0xF) << 1) | (b3 >> 7), (b3 >> 2) & 0x1F,
            ((b3 & 3) << 3) | (b4 >> 5), b4 & 0x1F,
        ], dim=-1)
        return v.reshape(n, -1)[:, :d].long()
    elif bits == 6:
        p = packed.reshape(n, -1, 3)
        b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
        v = torch.stack([
            ((b0 >> 2) & 0x3F), ((b0 & 3) << 4) | (b1 >> 4),
            ((b1 & 0xF) << 2) | (b2 >> 6), b2 & 0x3F,
        ], dim=-1)
        return v.reshape(n, -1)[:, :d].long()
    elif bits == 8:
        return packed.long()
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


# ---------------------------------------------------------------------------
# Worker modes
# ---------------------------------------------------------------------------

def run_fp16(args):
    """FP16 baseline: raw write/read with no compression."""
    wid = args.worker_id

    kv_data = torch.load(os.path.join(args.shared_dir, "kv_cache.pt"), weights_only=False)
    keys = kv_data["keys"]      # already CPU float32
    values = kv_data["values"]
    d = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]
    n_tokens = kv_data["n_tokens"]
    fp16_size = 2 * n_layers * n_heads * n_tokens * d * 2

    # Convert to FP16 bytes
    raw = torch.cat([keys, values], dim=0).half().numpy().tobytes()

    # ── Write ──
    filepath = os.path.join(args.output_dir, f"worker_{wid}_fp16.bin")
    t0 = time.time()
    if args.direct_io:
        write_direct(filepath, raw)
    else:
        with open(filepath, "wb") as f:
            f.write(raw)
        fd = os.open(filepath, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    t_write = time.time() - t0
    file_size = os.path.getsize(filepath)

    if not args.direct_io:
        # Drop page cache
        try:
            fd = os.open(filepath, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass

    # ── Read ──
    t0 = time.time()
    if args.direct_io:
        _ = read_direct(filepath)
    else:
        with open(filepath, "rb") as f:
            _ = f.read()
    t_read = time.time() - t0

    results = {
        "worker_id": wid, "mode": "fp16", "bits": 16, "n_tokens": n_tokens,
        "fp16_size": fp16_size, "file_size": file_size,
        "t_compress": 0.0, "t_write": t_write,
        "t_read": t_read, "t_decompress": 0.0,
        "mse": 0.0, "cos_sim": 1.0,
    }

    with open(os.path.join(args.shared_dir, f"bench_{wid}.json"), "w") as f:
        json.dump(results, f)

    os.remove(filepath)


def run_turboquant(args):
    """TurboQuant compressed checkpoint/restore."""
    wid = args.worker_id
    device = "cuda"

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
        direct_io=args.direct_io,
    )
    if not args.direct_io:
        fd = os.open(filepath, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    t_write = time.time() - t0
    file_size = os.path.getsize(filepath)

    if not args.direct_io:
        # Drop page cache
        try:
            fd = os.open(filepath, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass

    # ── Read (disk) ──
    t0 = time.time()
    data = deserialize_compressed_kv(filepath, direct_io=args.direct_io)
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
        "worker_id": wid, "mode": "turboquant", "bits": bits, "n_tokens": n_tokens,
        "fp16_size": fp16_size, "file_size": file_size,
        "t_compress": t_compress, "t_write": t_write,
        "t_read": t_read, "t_decompress": t_decompress,
        "mse": mse, "cos_sim": cos_sim,
    }

    with open(os.path.join(args.shared_dir, f"bench_{wid}.json"), "w") as f:
        json.dump(results, f)

    os.remove(filepath)


def run_fp16_gds(args):
    """FP16 via GPU Direct Storage — direct GPU↔FlashBlade, no CPU bounce."""
    import kvikio
    wid = args.worker_id
    device = "cuda"
    phase = args.phase

    kv_data = torch.load(os.path.join(args.shared_dir, "kv_cache.pt"), weights_only=False)
    keys = kv_data["keys"].to(device)
    values = kv_data["values"].to(device)
    d = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]
    n_tokens = kv_data["n_tokens"]
    fp16_size = 2 * n_layers * n_heads * n_tokens * d * 2

    data = torch.cat([keys, values], dim=0).half().contiguous()

    filepath = os.path.join(args.output_dir, f"worker_{wid}_fp16_gds.bin")
    t_write = 0.0
    t_read = 0.0
    file_size = fp16_size

    # ── Write via GDS ──
    if phase in ("all", "write"):
        torch.cuda.synchronize()
        t0 = time.time()
        with kvikio.CuFile(filepath, "w") as f:
            f.write(data)
        torch.cuda.synchronize()
        t_write = time.time() - t0
        file_size = os.path.getsize(filepath)

    if phase == "write":
        # Save metadata and exit — don't delete file
        json.dump({"file_size": file_size, "t_write": t_write},
                  open(os.path.join(args.shared_dir, f"meta_{wid}.json"), "w"))
        return

    # ── Read via GDS ──
    if phase == "read":
        meta = json.load(open(os.path.join(args.shared_dir, f"meta_{wid}.json")))
        file_size = meta["file_size"]
        t_write = meta["t_write"]

    buf = torch.empty_like(data)
    torch.cuda.synchronize()
    t0 = time.time()
    with kvikio.CuFile(filepath, "r") as f:
        f.read(buf)
    torch.cuda.synchronize()
    t_read = time.time() - t0

    results = {
        "worker_id": wid, "mode": "fp16_gds", "bits": 16, "n_tokens": n_tokens,
        "fp16_size": fp16_size, "file_size": file_size,
        "t_compress": 0.0, "t_write": t_write,
        "t_read": t_read, "t_decompress": 0.0,
        "mse": 0.0, "cos_sim": 1.0,
    }

    with open(os.path.join(args.shared_dir, f"bench_{wid}.json"), "w") as f:
        json.dump(results, f)

    os.remove(filepath)


def run_turboquant_gds(args):
    """TurboQuant with GPU Direct Storage — compress, pack, write, read, unpack, decompress all on GPU."""
    import kvikio
    wid = args.worker_id
    device = "cuda"
    phase = args.phase

    kv_data = torch.load(os.path.join(args.shared_dir, "kv_cache.pt"), weights_only=False)
    codebooks = torch.load(os.path.join(args.shared_dir, "codebooks.pt"), weights_only=False)

    keys = kv_data["keys"].to(device)
    values = kv_data["values"].to(device)
    d = kv_data["head_dim"]
    n_layers = kv_data["n_layers"]
    n_heads = kv_data["n_heads"]
    n_tokens = kv_data["n_tokens"]
    bits = args.bits
    n_blocks = n_layers * 2 * n_heads

    rotation = generate_rotation_matrix(d, seed=DEFAULT_SEED, device=device)
    quantizer = TurboQuantMSE(d, bits, codebooks[bits], rotation).to(device)

    fp16_size = 2 * n_layers * n_heads * n_tokens * d * 2
    idx_path = os.path.join(args.output_dir, f"worker_{wid}_{bits}bit_gds_idx.bin")
    norm_path = os.path.join(args.output_dir, f"worker_{wid}_{bits}bit_gds_norms.bin")

    t_compress = 0.0
    t_write = 0.0
    t_read = 0.0
    t_decompress = 0.0

    if phase in ("all", "write"):
        # ── Compress on GPU ──
        torch.cuda.synchronize()
        t0 = time.time()
        all_indices = torch.empty(n_blocks, n_tokens, d, dtype=torch.long, device=device)
        all_norms = torch.empty(n_blocks, n_tokens, device=device)
        block = 0
        for kv_tensor in [keys, values]:
            for layer in range(n_layers):
                for head in range(n_heads):
                    indices, norms = quantizer.quantize(kv_tensor[layer, head])
                    all_indices[block] = indices
                    all_norms[block] = norms
                    block += 1
        torch.cuda.synchronize()
        t_compress = time.time() - t0

        # ── Pack indices on GPU ──
        torch.cuda.synchronize()
        t0 = time.time()
        flat_indices = all_indices.reshape(n_blocks * n_tokens, d)
        packed = gpu_pack(flat_indices, bits).contiguous()
        norms_buf = all_norms.float().contiguous()
        torch.cuda.synchronize()
        t_pack = time.time() - t0
        t_compress += t_pack
        packed_shape = list(packed.shape)
        norms_shape = list(norms_buf.shape)

        # ── Write via GDS ──
        torch.cuda.synchronize()
        t0 = time.time()
        with kvikio.CuFile(idx_path, "w") as f:
            f.write(packed)
        with kvikio.CuFile(norm_path, "w") as f:
            f.write(norms_buf)
        torch.cuda.synchronize()
        t_write = time.time() - t0
        file_size = os.path.getsize(idx_path) + os.path.getsize(norm_path)

        del all_indices, all_norms, flat_indices, packed, norms_buf
        torch.cuda.empty_cache()

        if phase == "write":
            # Save metadata for read phase, don't delete files
            json.dump({
                "file_size": file_size, "t_compress": t_compress, "t_write": t_write,
                "packed_shape": packed_shape, "norms_shape": norms_shape,
            }, open(os.path.join(args.shared_dir, f"meta_{wid}.json"), "w"))
            return

    if phase in ("all", "read"):
        if phase == "read":
            meta = json.load(open(os.path.join(args.shared_dir, f"meta_{wid}.json")))
            file_size = meta["file_size"]
            t_compress = meta["t_compress"]
            t_write = meta["t_write"]
            packed_shape = meta["packed_shape"]
            norms_shape = meta["norms_shape"]

        # Free source data — keep orig on CPU for verification
        orig_cpu = torch.cat([keys, values], dim=0).reshape(n_blocks, n_tokens, d).cpu().float()
        del keys, values
        torch.cuda.empty_cache()

        # ── Read via GDS ──
        packed_read = torch.empty(packed_shape, dtype=torch.uint8, device=device)
        norms_read = torch.empty(norms_shape, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        with kvikio.CuFile(idx_path, "r") as f:
            f.read(packed_read)
        with kvikio.CuFile(norm_path, "r") as f:
            f.read(norms_read)
        torch.cuda.synchronize()
        t_read = time.time() - t0

        # ── Unpack + decompress on GPU ──
        torch.cuda.synchronize()
        t0 = time.time()
        unpacked = gpu_unpack(packed_read, bits, d)
        del packed_read
        unpacked = unpacked.reshape(n_blocks, n_tokens, d)
        restored_norms = norms_read.reshape(n_blocks, n_tokens)
        del norms_read
        reconstructed = torch.empty(n_blocks, n_tokens, d, device=device)
        for b in range(n_blocks):
            reconstructed[b] = quantizer.dequantize(unpacked[b], restored_norms[b])
        torch.cuda.synchronize()
        t_decompress = time.time() - t0

        # ── Verify ──
        recon = reconstructed.cpu().float()
        del reconstructed, unpacked, restored_norms
        mse = ((orig_cpu - recon) ** 2).sum(dim=-1).mean().item()
        cos_sim = F.cosine_similarity(orig_cpu.reshape(-1, d), recon.reshape(-1, d), dim=-1).mean().item()

    results = {
        "worker_id": wid, "mode": "turboquant_gds", "bits": bits, "n_tokens": n_tokens,
        "fp16_size": fp16_size, "file_size": file_size,
        "t_compress": t_compress, "t_write": t_write,
        "t_read": t_read, "t_decompress": t_decompress,
        "mse": mse, "cos_sim": cos_sim,
    }

    with open(os.path.join(args.shared_dir, f"bench_{wid}.json"), "w") as f:
        json.dump(results, f)

    os.remove(idx_path)
    os.remove(norm_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--fp16", action="store_true", help="Run FP16 baseline (no compression)")
    parser.add_argument("--gds", action="store_true", help="Use GPU Direct Storage")
    parser.add_argument("--direct-io", action="store_true", help="Use O_DIRECT to bypass page cache")
    parser.add_argument("--phase", choices=["all", "write", "read"], default="all",
                        help="Run only write or read phase (for two-phase benchmarking)")
    args = parser.parse_args()

    if args.fp16 and args.gds:
        run_fp16_gds(args)
    elif args.fp16:
        run_fp16(args)
    elif args.gds:
        run_turboquant_gds(args)
    else:
        run_turboquant(args)


if __name__ == "__main__":
    main()
