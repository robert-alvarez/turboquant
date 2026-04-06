"""
TurboQuant: Near-optimal KV cache compression via random orthogonal rotation
and Lloyd-Max scalar quantization.

Paper: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh et al., Google Research, arXiv:2504.19874v1
"""

from .core import (
    DEFAULT_DIM,
    DEFAULT_SEED,
    BIT_WIDTHS,
    OUTLIER_CONFIGS,
    beta_pdf_sphere,
    lloyd_max_codebook,
    compute_all_codebooks,
    generate_rotation_matrix,
    identify_outlier_channels,
    identify_outlier_layers,
    TurboQuantMSE,
    TurboQuantProd,
    TurboQuantOutlier,
)

from .bitpack import (
    pack_indices,
    unpack_indices,
    pack_signs,
    unpack_signs,
    pack_indices_fast,
    unpack_indices_fast,
    pack_signs_fast,
    unpack_signs_fast,
)

from .serialize import (
    MAGIC,
    VERSION,
    write_direct,
    read_direct,
    serialize_compressed_kv,
    deserialize_compressed_kv,
    dequantize_from_disk,
)

__all__ = [
    "DEFAULT_DIM", "DEFAULT_SEED", "BIT_WIDTHS", "OUTLIER_CONFIGS",
    "beta_pdf_sphere", "lloyd_max_codebook", "compute_all_codebooks",
    "generate_rotation_matrix", "identify_outlier_channels", "identify_outlier_layers",
    "TurboQuantMSE", "TurboQuantProd", "TurboQuantOutlier",
    "pack_indices", "unpack_indices", "pack_signs", "unpack_signs",
    "pack_indices_fast", "unpack_indices_fast", "pack_signs_fast", "unpack_signs_fast",
    "MAGIC", "VERSION", "write_direct", "read_direct",
    "serialize_compressed_kv", "deserialize_compressed_kv", "dequantize_from_disk",
]
