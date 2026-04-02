"""
Bit packing utilities for quantized indices and sign bits.

Provides both loop-based (reference) and vectorized (fast) implementations.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loop-based (reference implementations)
# ---------------------------------------------------------------------------

def pack_indices(indices: torch.Tensor, bits: int) -> bytes:
    """Pack a tensor of quantization indices into a compact byte string."""
    flat = indices.flatten().cpu().numpy().astype(np.uint32)
    n = len(flat)
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    packed = np.zeros(total_bytes, dtype=np.uint8)

    bit_pos = 0
    for val in flat:
        for b in range(bits):
            if val & (1 << b):
                byte_idx = bit_pos >> 3
                bit_idx = bit_pos & 7
                packed[byte_idx] |= (1 << bit_idx)
            bit_pos += 1

    return packed.tobytes()


def unpack_indices(data: bytes, bits: int, n_elements: int) -> np.ndarray:
    """Unpack a byte string into an array of quantization indices."""
    packed = np.frombuffer(data, dtype=np.uint8)
    result = np.zeros(n_elements, dtype=np.uint32)

    bit_pos = 0
    for i in range(n_elements):
        val = 0
        for b in range(bits):
            byte_idx = bit_pos >> 3
            bit_idx = bit_pos & 7
            if packed[byte_idx] & (1 << bit_idx):
                val |= (1 << b)
            bit_pos += 1
        result[i] = val

    return result


def pack_signs(signs: torch.Tensor) -> bytes:
    """Pack +1/-1 sign tensor into bits (1 = positive, 0 = negative)."""
    flat = signs.flatten().cpu().numpy()
    bits_arr = (flat > 0).astype(np.uint8)
    n = len(bits_arr)
    total_bytes = (n + 7) // 8
    packed = np.zeros(total_bytes, dtype=np.uint8)
    for i in range(n):
        if bits_arr[i]:
            packed[i >> 3] |= (1 << (i & 7))
    return packed.tobytes()


def unpack_signs(data: bytes, n_elements: int) -> np.ndarray:
    """Unpack bytes into +1/-1 sign array."""
    packed = np.frombuffer(data, dtype=np.uint8)
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        byte_idx = i >> 3
        bit_idx = i & 7
        result[i] = 1.0 if (packed[byte_idx] & (1 << bit_idx)) else -1.0
    return result


# ---------------------------------------------------------------------------
# Vectorized (fast path for disk serialization)
# ---------------------------------------------------------------------------

def pack_indices_fast(indices: torch.Tensor, bits: int) -> bytes:
    """Vectorized bit packing using numpy."""
    flat = indices.flatten().cpu().numpy().astype(np.uint64)
    n = len(flat)
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    packed = np.zeros(total_bytes, dtype=np.uint8)

    for b in range(bits):
        bit_vals = ((flat >> b) & 1).astype(np.uint8)
        bit_positions = np.arange(n, dtype=np.uint64) * bits + b
        byte_indices = (bit_positions >> 3).astype(np.intp)
        bit_offsets = (bit_positions & 7).astype(np.uint8)
        np.add.at(packed, byte_indices, bit_vals << bit_offsets)

    return packed.tobytes()


def unpack_indices_fast(data: bytes, bits: int, n_elements: int) -> np.ndarray:
    """Vectorized bit unpacking."""
    packed = np.frombuffer(data, dtype=np.uint8)
    result = np.zeros(n_elements, dtype=np.uint32)

    for b in range(bits):
        bit_positions = np.arange(n_elements, dtype=np.uint64) * bits + b
        byte_indices = (bit_positions >> 3).astype(np.intp)
        bit_offsets = (bit_positions & 7).astype(np.uint8)
        bit_vals = ((packed[byte_indices] >> bit_offsets) & 1).astype(np.uint32)
        result |= (bit_vals << b)

    return result


def pack_signs_fast(signs: torch.Tensor) -> bytes:
    """Vectorized sign packing."""
    flat = (signs.flatten().cpu().numpy() > 0).astype(np.uint8)
    n = len(flat)
    total_bytes = (n + 7) // 8
    packed = np.zeros(total_bytes, dtype=np.uint8)
    indices = np.arange(n)
    byte_indices = (indices >> 3).astype(np.intp)
    bit_offsets = (indices & 7).astype(np.uint8)
    np.add.at(packed, byte_indices, flat << bit_offsets)
    return packed.tobytes()


def unpack_signs_fast(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized sign unpacking."""
    packed = np.frombuffer(data, dtype=np.uint8)
    indices = np.arange(n_elements, dtype=np.uint64)
    byte_indices = (indices >> 3).astype(np.intp)
    bit_offsets = (indices & 7).astype(np.uint8)
    bit_vals = ((packed[byte_indices] >> bit_offsets) & 1).astype(np.float32)
    return bit_vals * 2.0 - 1.0
