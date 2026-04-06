"""
Binary disk format (.tqkv) for serializing and deserializing compressed KV caches.
"""

import io
import math
import mmap
import os
import struct

import numpy as np
import torch

from .bitpack import pack_indices_fast, unpack_indices_fast, pack_signs_fast, unpack_signs_fast

MAGIC = b"TQKV"
VERSION = 1

# ---------------------------------------------------------------------------
# O_DIRECT I/O helpers (bypass Linux page cache)
# ---------------------------------------------------------------------------

_ALIGN = 4096  # O_DIRECT alignment requirement


def write_direct(filepath, data):
    """Write bytes using O_DIRECT to bypass the page cache entirely.

    Uses mmap for page-aligned buffers as required by O_DIRECT. Includes
    fsync so the caller doesn't need a separate sync step.
    """
    size = len(data)
    padded = (size + _ALIGN - 1) & ~(_ALIGN - 1)
    padded = max(padded, _ALIGN)

    buf = mmap.mmap(-1, padded)
    buf[:size] = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    if padded > size:
        buf[size:padded] = b'\x00' * (padded - size)

    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        mv = memoryview(buf)
        offset = 0
        while offset < padded:
            chunk = min(padded - offset, 16 * 1024 * 1024)
            n = os.write(fd, mv[offset:offset + chunk])
            offset += n
        del mv
        os.fsync(fd)
    finally:
        os.close(fd)
    buf.close()

    if padded > size:
        os.truncate(filepath, size)


def read_direct(filepath):
    """Read a file using O_DIRECT to bypass the page cache entirely.

    Returns bytes. Uses mmap for page-aligned receive buffers.
    """
    size = os.path.getsize(filepath)
    if size == 0:
        return b''
    padded = (size + _ALIGN - 1) & ~(_ALIGN - 1)

    buf = mmap.mmap(-1, padded)
    fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
    try:
        mv = memoryview(buf)
        offset = 0
        while offset < padded:
            chunk = min(padded - offset, 16 * 1024 * 1024)
            n = os.preadv(fd, [mv[offset:offset + chunk]], offset)
            if n == 0:
                break
            offset += n
        del mv
    finally:
        os.close(fd)

    result = bytes(buf[:size])
    buf.close()
    return result


def serialize_compressed_kv(
    filepath: str,
    rotation_matrix: torch.Tensor,
    codebook: np.ndarray,
    bits: int,
    all_indices: list,
    n_layers: int,
    n_heads: int,
    n_tokens: int,
    d: int,
    mode: str = "mse",
    qjl_data: list = None,
    qjl_matrix: torch.Tensor = None,
    direct_io: bool = False,
):
    """
    Serialize a compressed KV cache to a binary file.

    Format:
      Header (32 bytes): magic, version, mode, bits, d, n_layers, n_heads, n_tokens
      Rotation matrix: d*d float32
      Codebook: 2^b float64
      [If prod] QJL matrix: d*d float32
      Per-layer per-head: packed indices + norms [+ signs + residual_norms]
    """
    mode_int = 0 if mode == "mse" else 1
    n_levels = 1 << bits

    # Build the entire file in memory first, then write once.
    # This avoids hundreds of small writes and lets the OS do a single large I/O.
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<I", VERSION)
    buf += struct.pack("<I", mode_int)
    buf += struct.pack("<I", bits)
    buf += struct.pack("<I", d)
    buf += struct.pack("<I", n_layers)
    buf += struct.pack("<I", n_heads)
    buf += struct.pack("<I", n_tokens)

    buf += rotation_matrix.cpu().float().numpy().tobytes()
    buf += codebook.astype(np.float64).tobytes()

    if mode_int == 1 and qjl_matrix is not None:
        buf += qjl_matrix.cpu().float().numpy().tobytes()

    block_idx = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            indices, norms = all_indices[block_idx]
            buf += pack_indices_fast(indices, bits)
            buf += norms.cpu().float().numpy().tobytes()

            if mode_int == 1 and qjl_data is not None:
                qjl_signs, res_norms = qjl_data[block_idx]
                buf += pack_signs_fast(qjl_signs)
                buf += res_norms.cpu().float().numpy().tobytes()

            block_idx += 1

    if direct_io:
        write_direct(filepath, buf)
    else:
        with open(filepath, "wb") as f:
            f.write(buf)


def _parse_tqkv(f) -> dict:
    """Parse .tqkv from a file-like object (file handle or BytesIO)."""
    magic = f.read(4)
    assert magic == MAGIC, f"Invalid magic: {magic}"
    version = struct.unpack("<I", f.read(4))[0]
    assert version == VERSION, f"Unsupported version: {version}"
    mode_int = struct.unpack("<I", f.read(4))[0]
    bits = struct.unpack("<I", f.read(4))[0]
    d = struct.unpack("<I", f.read(4))[0]
    n_layers = struct.unpack("<I", f.read(4))[0]
    n_heads = struct.unpack("<I", f.read(4))[0]
    n_tokens = struct.unpack("<I", f.read(4))[0]

    mode = "mse" if mode_int == 0 else "prod"
    n_levels = 1 << bits

    rot_data = f.read(d * d * 4)
    rotation = torch.from_numpy(
        np.frombuffer(rot_data, dtype=np.float32).reshape(d, d).copy()
    )

    cb_data = f.read(n_levels * 8)
    codebook = np.frombuffer(cb_data, dtype=np.float64).copy()

    qjl_matrix = None
    if mode_int == 1:
        qjl_data_raw = f.read(d * d * 4)
        qjl_matrix = torch.from_numpy(
            np.frombuffer(qjl_data_raw, dtype=np.float32).reshape(d, d).copy()
        )

    indices_size = (n_tokens * d * bits + 7) // 8
    norms_size = n_tokens * 4
    signs_size = (n_tokens * d + 7) // 8 if mode_int == 1 else 0
    res_norms_size = n_tokens * 4 if mode_int == 1 else 0

    blocks = []
    for layer in range(n_layers):
        for head in range(n_heads):
            packed_idx = f.read(indices_size)
            indices = unpack_indices_fast(packed_idx, bits, n_tokens * d)
            indices = torch.from_numpy(indices.reshape(n_tokens, d).copy()).long()

            norms_data = f.read(norms_size)
            norms = torch.from_numpy(np.frombuffer(norms_data, dtype=np.float32).copy())

            qjl_signs = None
            res_norms = None
            if mode_int == 1:
                signs_data = f.read(signs_size)
                qjl_signs_np = unpack_signs_fast(signs_data, n_tokens * d)
                qjl_signs = torch.from_numpy(qjl_signs_np.reshape(n_tokens, d).copy())

                res_norms_data = f.read(res_norms_size)
                res_norms = torch.from_numpy(
                    np.frombuffer(res_norms_data, dtype=np.float32).copy()
                )

            blocks.append({
                "indices": indices, "norms": norms,
                "qjl_signs": qjl_signs, "residual_norms": res_norms,
            })

    return {
        "mode": mode, "bits": bits, "d": d,
        "n_layers": n_layers, "n_heads": n_heads, "n_tokens": n_tokens,
        "rotation": rotation, "codebook": codebook, "qjl_matrix": qjl_matrix,
        "blocks": blocks,
    }


def deserialize_compressed_kv(filepath: str, direct_io: bool = False) -> dict:
    """Deserialize a compressed KV cache from a binary file on CPU."""
    if direct_io:
        return _parse_tqkv(io.BytesIO(read_direct(filepath)))
    with open(filepath, "rb") as f:
        return _parse_tqkv(f)


def dequantize_from_disk(data: dict) -> torch.Tensor:
    """
    Dequantize an entire KV cache from the deserialized disk representation.
    Runs entirely on CPU.

    Returns:
        Tensor of shape (n_layers, n_heads, n_tokens, d) in float32.
    """
    d = data["d"]
    n_layers = data["n_layers"]
    n_heads = data["n_heads"]
    n_tokens = data["n_tokens"]
    rotation = data["rotation"]
    codebook = torch.from_numpy(data["codebook"]).float()
    mode = data["mode"]

    result = torch.zeros(n_layers, n_heads, n_tokens, d, dtype=torch.float32)

    for layer in range(n_layers):
        for head in range(n_heads):
            block = data["blocks"][layer * n_heads + head]
            indices = block["indices"]
            norms = block["norms"]

            y_hat = codebook[indices]
            x_hat = (rotation.T @ y_hat.T).T

            if mode == "prod" and block["qjl_signs"] is not None:
                qjl_matrix = data["qjl_matrix"]
                qjl_signs = block["qjl_signs"]
                res_norms = block["residual_norms"]
                scale = math.sqrt(math.pi / 2.0) / d
                x_qjl = (qjl_matrix.T @ qjl_signs.T).T
                x_qjl = scale * res_norms.unsqueeze(-1) * x_qjl
                x_hat = x_hat + x_qjl

            x_hat = x_hat * norms.unsqueeze(-1)
            result[layer, head] = x_hat

    return result
