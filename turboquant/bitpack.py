"""
Bit packing utilities for quantized indices and sign bits.

Provides both loop-based (reference) and vectorized (fast) implementations.
The fast path uses group-aligned packing: elements are processed in groups
that align to byte boundaries, avoiding np.add.at entirely.
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
# Vectorized (fast path) — group-aligned packing
#
# For each bit width b, we process elements in groups where
# LCM(b, 8) / b elements map to LCM(b, 8) / 8 bytes.
# This uses pure shift-and-OR with no np.add.at.
# ---------------------------------------------------------------------------

def _pad_to(arr, multiple, dtype=None):
    """Pad array to a multiple of `multiple` with zeros."""
    rem = len(arr) % multiple
    if rem == 0:
        return arr
    pad = np.zeros(multiple - rem, dtype=arr.dtype if dtype is None else dtype)
    return np.concatenate([arr, pad])


def _to_flat(indices, dtype=np.uint16):
    """Convert indices (tensor or array) to flat numpy array."""
    if isinstance(indices, torch.Tensor):
        return indices.flatten().cpu().numpy().astype(dtype)
    return np.asarray(indices, dtype=dtype).ravel()


# ── 2-bit: 4 elements → 1 byte ──

def _pack_2bit(flat, total_bytes):
    v = _pad_to(flat, 4).reshape(-1, 4)
    packed = v[:, 0] | (v[:, 1] << 2) | (v[:, 2] << 4) | (v[:, 3] << 6)
    return packed.astype(np.uint8)[:total_bytes]


def _unpack_2bit(packed, n):
    p = packed
    e0 = p & 3
    e1 = (p >> 2) & 3
    e2 = (p >> 4) & 3
    e3 = (p >> 6) & 3
    return np.column_stack([e0, e1, e2, e3]).ravel()[:n].astype(np.uint32)


# ── 3-bit: 8 elements → 3 bytes ──

def _pack_3bit(flat, total_bytes):
    v = _pad_to(flat, 8).reshape(-1, 8).astype(np.uint16)
    b0 = (v[:, 0] | (v[:, 1] << 3) | (v[:, 2] << 6)).astype(np.uint8)
    b1 = ((v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | (v[:, 5] << 7)).astype(np.uint8)
    b2 = ((v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)).astype(np.uint8)
    return np.column_stack([b0, b1, b2]).ravel()[:total_bytes]


def _unpack_3bit(packed, n):
    p = _pad_to(packed, 3).reshape(-1, 3).astype(np.uint16)
    b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]
    e0 = b0 & 7
    e1 = (b0 >> 3) & 7
    e2 = ((b0 >> 6) | (b1 << 2)) & 7
    e3 = (b1 >> 1) & 7
    e4 = (b1 >> 4) & 7
    e5 = ((b1 >> 7) | (b2 << 1)) & 7
    e6 = (b2 >> 2) & 7
    e7 = (b2 >> 5) & 7
    return np.column_stack([e0, e1, e2, e3, e4, e5, e6, e7]).ravel()[:n].astype(np.uint32)


# ── 4-bit: 2 elements → 1 byte ──

def _pack_4bit(flat, total_bytes):
    v = _pad_to(flat, 2).reshape(-1, 2)
    packed = v[:, 0] | (v[:, 1] << 4)
    return packed.astype(np.uint8)[:total_bytes]


def _unpack_4bit(packed, n):
    e0 = packed & 0xF
    e1 = (packed >> 4) & 0xF
    return np.column_stack([e0, e1]).ravel()[:n].astype(np.uint32)


# ── 5-bit: 8 elements → 5 bytes ──

def _pack_5bit(flat, total_bytes):
    v = _pad_to(flat, 8).reshape(-1, 8).astype(np.uint16)
    b0 = (v[:, 0] | (v[:, 1] << 5)).astype(np.uint8)
    b1 = ((v[:, 1] >> 3) | (v[:, 2] << 2) | (v[:, 3] << 7)).astype(np.uint8)
    b2 = ((v[:, 3] >> 1) | (v[:, 4] << 4)).astype(np.uint8)
    b3 = ((v[:, 4] >> 4) | (v[:, 5] << 1) | (v[:, 6] << 6)).astype(np.uint8)
    b4 = ((v[:, 6] >> 2) | (v[:, 7] << 3)).astype(np.uint8)
    return np.column_stack([b0, b1, b2, b3, b4]).ravel()[:total_bytes]


def _unpack_5bit(packed, n):
    p = _pad_to(packed, 5).reshape(-1, 5).astype(np.uint16)
    b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    e0 = b0 & 31
    e1 = ((b0 >> 5) | (b1 << 3)) & 31
    e2 = (b1 >> 2) & 31
    e3 = ((b1 >> 7) | (b2 << 1)) & 31
    e4 = ((b2 >> 4) | (b3 << 4)) & 31
    e5 = (b3 >> 1) & 31
    e6 = ((b3 >> 6) | (b4 << 2)) & 31
    e7 = (b4 >> 3) & 31
    return np.column_stack([e0, e1, e2, e3, e4, e5, e6, e7]).ravel()[:n].astype(np.uint32)


# ── 6-bit: 4 elements → 3 bytes ──

def _pack_6bit(flat, total_bytes):
    v = _pad_to(flat, 4).reshape(-1, 4).astype(np.uint16)
    b0 = (v[:, 0] | (v[:, 1] << 6)).astype(np.uint8)
    b1 = ((v[:, 1] >> 2) | (v[:, 2] << 4)).astype(np.uint8)
    b2 = ((v[:, 2] >> 4) | (v[:, 3] << 2)).astype(np.uint8)
    return np.column_stack([b0, b1, b2]).ravel()[:total_bytes]


def _unpack_6bit(packed, n):
    p = _pad_to(packed, 3).reshape(-1, 3).astype(np.uint16)
    b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]
    e0 = b0 & 63
    e1 = ((b0 >> 6) | (b1 << 2)) & 63
    e2 = ((b1 >> 4) | (b2 << 4)) & 63
    e3 = (b2 >> 2) & 63
    return np.column_stack([e0, e1, e2, e3]).ravel()[:n].astype(np.uint32)


# ── 8-bit: 1 element → 1 byte ──

def _pack_8bit(flat, total_bytes):
    return flat.astype(np.uint8)[:total_bytes]


def _unpack_8bit(packed, n):
    return packed[:n].astype(np.uint32)


# ── Dispatch tables ──

_PACKERS = {2: _pack_2bit, 3: _pack_3bit, 4: _pack_4bit,
            5: _pack_5bit, 6: _pack_6bit, 8: _pack_8bit}
_UNPACKERS = {2: _unpack_2bit, 3: _unpack_3bit, 4: _unpack_4bit,
              5: _unpack_5bit, 6: _unpack_6bit, 8: _unpack_8bit}


# ── Fallback for arbitrary bit widths (uses np.add.at) ──

def _pack_generic(flat, bits, total_bytes):
    flat64 = flat.astype(np.uint64)
    n = len(flat64)
    packed = np.zeros(total_bytes, dtype=np.uint8)
    for b in range(bits):
        bit_vals = ((flat64 >> b) & 1).astype(np.uint8)
        bit_positions = np.arange(n, dtype=np.uint64) * bits + b
        byte_indices = (bit_positions >> 3).astype(np.intp)
        bit_offsets = (bit_positions & 7).astype(np.uint8)
        np.add.at(packed, byte_indices, bit_vals << bit_offsets)
    return packed


def _unpack_generic(packed, bits, n):
    result = np.zeros(n, dtype=np.uint32)
    for b in range(bits):
        bit_positions = np.arange(n, dtype=np.uint64) * bits + b
        byte_indices = (bit_positions >> 3).astype(np.intp)
        bit_offsets = (bit_positions & 7).astype(np.uint8)
        bit_vals = ((packed[byte_indices] >> bit_offsets) & 1).astype(np.uint32)
        result |= (bit_vals << b)
    return result


# ---------------------------------------------------------------------------
# Public fast API
# ---------------------------------------------------------------------------

def pack_indices_fast(indices, bits: int) -> bytes:
    """Vectorized bit packing — group-aligned for known bit widths."""
    flat = _to_flat(indices, dtype=np.uint16)
    n = len(flat)
    total_bytes = (n * bits + 7) // 8
    packer = _PACKERS.get(bits)
    if packer is not None:
        return packer(flat, total_bytes).tobytes()
    return _pack_generic(flat, bits, total_bytes).tobytes()


def unpack_indices_fast(data: bytes, bits: int, n_elements: int) -> np.ndarray:
    """Vectorized bit unpacking — group-aligned for known bit widths."""
    packed = np.frombuffer(data, dtype=np.uint8)
    unpacker = _UNPACKERS.get(bits)
    if unpacker is not None:
        return unpacker(packed, n_elements)
    return _unpack_generic(packed, bits, n_elements)


def pack_signs_fast(signs: torch.Tensor) -> bytes:
    """Vectorized sign packing (1-bit per element)."""
    flat = (signs.flatten().cpu().numpy() > 0).astype(np.uint8)
    n = len(flat)
    total_bytes = (n + 7) // 8
    # Use 8-element groups → 1 byte each
    flat = _pad_to(flat, 8).reshape(-1, 8)
    packed = flat[:, 0]
    for i in range(1, 8):
        packed = packed | (flat[:, i] << i)
    return packed.astype(np.uint8)[:total_bytes].tobytes()


def unpack_signs_fast(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized sign unpacking."""
    packed = np.frombuffer(data, dtype=np.uint8)
    indices = np.arange(n_elements, dtype=np.uint64)
    byte_indices = (indices >> 3).astype(np.intp)
    bit_offsets = (indices & 7).astype(np.uint8)
    bit_vals = ((packed[byte_indices] >> bit_offsets) & 1).astype(np.float32)
    return bit_vals * 2.0 - 1.0
