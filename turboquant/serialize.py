"""
Binary disk format (.tqkv) for serializing and deserializing compressed KV caches.
"""

import math
import struct

import numpy as np
import torch

from .bitpack import pack_indices_fast, unpack_indices_fast, pack_signs_fast, unpack_signs_fast

MAGIC = b"TQKV"
VERSION = 1


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

    with open(filepath, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", mode_int))
        f.write(struct.pack("<I", bits))
        f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", n_layers))
        f.write(struct.pack("<I", n_heads))
        f.write(struct.pack("<I", n_tokens))

        f.write(rotation_matrix.cpu().float().numpy().tobytes())
        f.write(codebook.astype(np.float64).tobytes())

        if mode_int == 1 and qjl_matrix is not None:
            f.write(qjl_matrix.cpu().float().numpy().tobytes())

        block_idx = 0
        for layer in range(n_layers):
            for head in range(n_heads):
                indices, norms = all_indices[block_idx]
                f.write(pack_indices_fast(indices, bits))
                f.write(norms.cpu().float().numpy().tobytes())

                if mode_int == 1 and qjl_data is not None:
                    qjl_signs, res_norms = qjl_data[block_idx]
                    f.write(pack_signs_fast(qjl_signs))
                    f.write(res_norms.cpu().float().numpy().tobytes())

                block_idx += 1


def deserialize_compressed_kv(filepath: str) -> dict:
    """
    Deserialize a compressed KV cache from a binary file on CPU.

    Returns a dict with all the data needed for dequantization.
    """
    with open(filepath, "rb") as f:
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
