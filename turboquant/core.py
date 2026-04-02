"""
Core TurboQuant algorithm: codebooks, rotation, and quantizer classes.

Dependencies: torch, numpy, scipy (no transformers, no matplotlib).
"""

import math
import time

import numpy as np
import torch
from scipy import special, integrate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM = 128          # head dimension for KV cache
DEFAULT_SEED = 42          # fixed seed for reproducible rotation matrix
BIT_WIDTHS = [2, 3, 4, 5, 6, 8]

# Predefined outlier configurations matching the paper's operating points
OUTLIER_CONFIGS = [
    # (name, n_outliers, bits_high, bits_low, effective_bits)
    ("2.5-bit", 32, 4, 2, 2.5),
    ("3.5-bit", 32, 5, 3, 3.5),
    ("3.0-bit (mixed)", 64, 4, 2, 3.0),
    ("4.0-bit (mixed)", 64, 5, 3, 4.0),
]


# ---------------------------------------------------------------------------
# Lloyd-Max Codebook Computation for the Beta Distribution
# ---------------------------------------------------------------------------

def beta_pdf_sphere(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a point uniformly distributed on S^{d-1}.

    After rotating a unit vector by a Haar-random orthogonal matrix, each
    coordinate follows this distribution on [-1, 1]:

        f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    This is equivalent to a scaled Beta((d-1)/2, (d-1)/2) distribution shifted
    from [0,1] to [-1,1].

    Reference: TurboQuant paper, Lemma 1.
    """
    if d < 3:
        raise ValueError("d must be >= 3 for the Beta distribution to be well-defined")
    log_norm = (
        special.gammaln(d / 2.0)
        - 0.5 * np.log(np.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    mask = np.abs(x) < 1.0
    result = np.zeros_like(x, dtype=np.float64)
    result[mask] = np.exp(
        log_norm + ((d - 3) / 2.0) * np.log(1.0 - x[mask] ** 2)
    )
    return result


def lloyd_max_codebook(d: int, bits: int, max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """
    Compute the optimal Lloyd-Max scalar quantization codebook for the coordinate
    distribution on S^{d-1} (a symmetric Beta distribution on [-1, 1]).

    The Lloyd-Max algorithm alternates:
      1. Update boundaries as midpoints of consecutive centroids.
      2. Update centroids as conditional expectations E[X | X in cell_i].

    Returns:
        centroids: array of shape (2^bits,) sorted in ascending order.

    Reference: TurboQuant paper, Section 2.1 and Appendix A.
    """
    n_levels = 1 << bits

    spread = min(3.0 / np.sqrt(d), 0.99)
    centroids = np.linspace(-spread, spread, n_levels)

    for iteration in range(max_iter):
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n_levels):
            boundaries[i] = 0.5 * (centroids[i - 1] + centroids[i])

        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = 0.5 * (lo + hi)
                continue

            num, _ = integrate.quad(lambda x: x * beta_pdf_sphere(np.array([x]), d)[0], lo, hi)
            den, _ = integrate.quad(lambda x: beta_pdf_sphere(np.array([x]), d)[0], lo, hi)
            if den > 1e-20:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = 0.5 * (lo + hi)

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return np.sort(centroids).astype(np.float64)


def compute_all_codebooks(d: int = DEFAULT_DIM, bit_widths: list = None) -> dict:
    """
    Precompute Lloyd-Max codebooks for all requested bit widths.

    Returns:
        dict mapping bit_width -> np.ndarray of centroids (2^b values).
    """
    if bit_widths is None:
        bit_widths = BIT_WIDTHS
    codebooks = {}
    for b in bit_widths:
        print(f"  Computing Lloyd-Max codebook for {b}-bit ({1 << b} levels), d={d}...", end=" ", flush=True)
        t0 = time.time()
        codebooks[b] = lloyd_max_codebook(d, b)
        print(f"done in {time.time() - t0:.2f}s")
    return codebooks


# ---------------------------------------------------------------------------
# Random Orthogonal Rotation Matrix
# ---------------------------------------------------------------------------

def generate_rotation_matrix(d: int, seed: int = DEFAULT_SEED, device: str = "cpu") -> torch.Tensor:
    """
    Generate a Haar-random orthogonal matrix by QR decomposition of a Gaussian matrix.

    This is generated ONCE from a fixed seed and reused for every vector. It is
    data-oblivious: it does not depend on the model or the KV cache contents.

    Reference: TurboQuant paper, Algorithm 1, setup step.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


# ---------------------------------------------------------------------------
# TurboQuant_MSE (Algorithm 1)
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    TurboQuant_MSE quantizer: random orthogonal rotation + per-coordinate
    Lloyd-Max scalar quantization.

    For a unit vector x in S^{d-1}:
      1. Rotate: y = Pi @ x
      2. Quantize each coordinate y_j to its nearest centroid
      3. Store: d indices (each b bits) + the original norm

    Dequantization reverses the process:
      1. Look up centroids from indices
      2. Inverse rotate: x_hat = Pi^T @ y_hat
      3. Rescale by the stored norm

    Reference: TurboQuant paper, Algorithm 1.
    """

    def __init__(self, d: int, bits: int, codebook: np.ndarray, rotation: torch.Tensor):
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.codebook = torch.from_numpy(codebook).float()
        self.rotation = rotation

    def to(self, device):
        self.codebook = self.codebook.to(device)
        self.rotation = self.rotation.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> tuple:
        """Quantize a batch of (batch, d) vectors. Returns (indices, norms)."""
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_unit = x / norms
        y = (self.rotation @ x_unit.T).T
        n_vecs, d = y.shape
        n_levels = self.codebook.shape[0]
        # Batch to avoid huge intermediate tensor: (n_vecs, d, n_levels)
        # Limit each batch to ~2 GiB of float32
        max_batch = max(1, (2 * 1024**3) // (d * n_levels * 4))
        if n_vecs <= max_batch:
            dists = (y.unsqueeze(-1) - self.codebook.view(1, 1, -1)).abs()
            indices = dists.argmin(dim=-1)
        else:
            indices = torch.empty(n_vecs, d, dtype=torch.long, device=y.device)
            for start in range(0, n_vecs, max_batch):
                end = min(start + max_batch, n_vecs)
                dists = (y[start:end].unsqueeze(-1) - self.codebook.view(1, 1, -1)).abs()
                indices[start:end] = dists.argmin(dim=-1)
        return indices, norms.squeeze(-1)

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize from (indices, norms) back to (batch, d) vectors."""
        y_hat = self.codebook[indices.long()]
        x_hat = (self.rotation.T @ y_hat.T).T
        x_hat = x_hat * norms.unsqueeze(-1)
        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: quantize then immediately dequantize."""
        indices, norms = self.quantize(x)
        return self.dequantize(indices, norms)


# ---------------------------------------------------------------------------
# TurboQuant_prod (Algorithm 2 - MSE + QJL)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    TurboQuant_prod: MSE quantization at (b-1) bits + 1-bit QJL on the residual.

    The QJL component makes inner product estimation unbiased.

    Reference: TurboQuant paper, Algorithm 2.
    """

    def __init__(self, d: int, bits: int, codebooks: dict, rotation: torch.Tensor,
                 qjl_seed: int = 137):
        self.d = d
        self.bits = bits
        self.mse_bits = bits - 1

        if self.mse_bits < 1:
            raise ValueError("TurboQuant_prod requires bits >= 2 (1 bit for MSE + 1 bit for QJL)")

        self.mse_quantizer = TurboQuantMSE(d, self.mse_bits, codebooks[self.mse_bits], rotation)

        rng = torch.Generator(device="cpu").manual_seed(qjl_seed)
        self.S = torch.randn(d, d, generator=rng, dtype=torch.float32)

    def to(self, device):
        self.mse_quantizer.to(device)
        self.S = self.S.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> tuple:
        """Quantize using MSE + QJL. Returns (mse_indices, norms, qjl_signs, residual_norms)."""
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_unit = x / norms

        mse_indices, unit_norms = self.mse_quantizer.quantize(x_unit)
        x_mse = self.mse_quantizer.dequantize(mse_indices, unit_norms)

        residual = x_unit - x_mse
        residual_norms = residual.norm(dim=-1)

        proj = (self.S @ residual.T).T
        qjl_signs = torch.sign(proj)
        qjl_signs[qjl_signs == 0] = 1.0

        return mse_indices, norms.squeeze(-1), qjl_signs, residual_norms

    def dequantize(self, mse_indices: torch.Tensor, norms: torch.Tensor,
                   qjl_signs: torch.Tensor, residual_norms: torch.Tensor) -> torch.Tensor:
        """Dequantize from MSE + QJL representation."""
        ones = torch.ones_like(norms)
        x_mse = self.mse_quantizer.dequantize(mse_indices, ones)

        scale = math.sqrt(math.pi / 2.0) / self.d
        x_qjl = (self.S.T @ qjl_signs.T).T
        x_qjl = scale * residual_norms.unsqueeze(-1) * x_qjl

        x_hat = (x_mse + x_qjl) * norms.unsqueeze(-1)
        return x_hat

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: quantize then immediately dequantize."""
        mse_indices, norms, qjl_signs, residual_norms = self.quantize(x)
        return self.dequantize(mse_indices, norms, qjl_signs, residual_norms)


# ---------------------------------------------------------------------------
# TurboQuant with Outlier Channel Handling
# ---------------------------------------------------------------------------

class TurboQuantOutlier:
    """
    Mixed-precision TurboQuant: quantizes outlier channels at higher precision
    and non-outlier channels at lower precision.

    Reference: TurboQuant paper, Section 4.3.
    """

    def __init__(self, d: int, bits_high: int, bits_low: int,
                 outlier_indices: torch.Tensor,
                 codebooks_high: dict, codebooks_low: dict,
                 seed: int = DEFAULT_SEED):
        self.d = d
        self.bits_high = bits_high
        self.bits_low = bits_low

        self.outlier_idx = outlier_indices.long()
        all_idx = torch.arange(d)
        mask = torch.ones(d, dtype=torch.bool)
        mask[self.outlier_idx] = False
        self.normal_idx = all_idx[mask].long()

        self.d_high = len(self.outlier_idx)
        self.d_low = len(self.normal_idx)
        assert self.d_high + self.d_low == d

        self.effective_bits = (self.d_high * bits_high + self.d_low * bits_low) / d

        rot_high = generate_rotation_matrix(self.d_high, seed=seed, device="cpu")
        rot_low = generate_rotation_matrix(self.d_low, seed=seed + 1000, device="cpu")

        self.quantizer_high = TurboQuantMSE(
            self.d_high, bits_high, codebooks_high[bits_high], rot_high
        )
        self.quantizer_low = TurboQuantMSE(
            self.d_low, bits_low, codebooks_low[bits_low], rot_low
        )

    def to(self, device):
        self.quantizer_high.to(device)
        self.quantizer_low.to(device)
        self.outlier_idx = self.outlier_idx.to(device)
        self.normal_idx = self.normal_idx.to(device)
        return self

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize using mixed precision."""
        x_high = x[:, self.outlier_idx]
        x_low = x[:, self.normal_idx]

        x_high_hat = self.quantizer_high.quantize_dequantize(x_high)
        x_low_hat = self.quantizer_low.quantize_dequantize(x_low)

        x_hat = torch.empty_like(x)
        x_hat[:, self.outlier_idx] = x_high_hat
        x_hat[:, self.normal_idx] = x_low_hat
        return x_hat


def identify_outlier_layers(kv_cache: dict, threshold: float = 4.0) -> list:
    """
    Identify layers with anomalously large key norms that should be kept in FP16.

    Some models (e.g., Qwen2.5-7B) have layers (typically Layer 0 and the last
    layer) where key vector norms are 10-15x larger than typical layers. At
    moderate bit widths (3-6 bit), quantizing these layers creates systematically
    wrong attention patterns — the error is large enough to mislead the model
    but structured enough that it doesn't get averaged out like random noise.

    The fix: keep these layers' keys in FP16 (skip quantization). Values are
    not affected since the diagnostic shows value quantization is robust.

    Args:
        kv_cache: dict with "keys" tensor of shape (L, H, T, d)
        threshold: a layer is flagged if its mean key norm > threshold * median

    Returns:
        list of (layer_index, mean_norm, median_norm) tuples for exempt layers
    """
    d = kv_cache["head_dim"]
    n_layers = kv_cache["n_layers"]
    layer_norms = []
    for layer in range(n_layers):
        norms = kv_cache["keys"][layer].reshape(-1, d).norm(dim=-1)
        layer_norms.append(norms.mean().item())

    median_norm = float(sorted(layer_norms)[n_layers // 2])
    outliers = []
    for layer, mean_norm in enumerate(layer_norms):
        if mean_norm > threshold * median_norm:
            outliers.append((layer, mean_norm, median_norm))
    return outliers


def identify_outlier_channels(kv_cache: dict, n_outliers: int = 32) -> torch.Tensor:
    """
    Identify the n_outliers channels with highest variance across the KV cache.

    Args:
        kv_cache: dict with "keys" and "values" tensors of shape (L, H, T, d)
        n_outliers: number of outlier channels to identify

    Returns:
        outlier_indices: 1-D tensor of shape (n_outliers,)
    """
    keys_flat = kv_cache["keys"].reshape(-1, kv_cache["head_dim"])
    vals_flat = kv_cache["values"].reshape(-1, kv_cache["head_dim"])
    all_vecs = torch.cat([keys_flat, vals_flat], dim=0)

    channel_var = all_vecs.var(dim=0)
    _, outlier_idx = channel_var.topk(n_outliers)
    return outlier_idx.sort().values
