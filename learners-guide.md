# TurboQuant Learner's Guide

A technical guide to what this implementation measures and why, written for readers comfortable with linear algebra and probability but not necessarily with PyTorch internals.

---

## Table of Contents

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [The TurboQuant algorithm in plain math](#2-the-turboquant-algorithm-in-plain-math)
3. [Why random rotation works](#3-why-random-rotation-works)
4. [Lloyd-Max codebooks and the Beta distribution](#4-lloyd-max-codebooks-and-the-beta-distribution)
5. [The four quality metrics](#5-the-four-quality-metrics)
6. [MSE-only vs MSE+QJL: the variance-bias tradeoff](#6-mse-only-vs-msejql-the-variance-bias-tradeoff)
7. [Top-1 token accuracy and the teacher forcing problem](#7-top-1-token-accuracy-and-the-teacher-forcing-problem)
8. [Outlier channel handling and fractional bit rates](#8-outlier-channel-handling-and-fractional-bit-rates)
9. [The residual window](#9-the-residual-window)
10. [The disk round-trip test](#10-the-disk-round-trip-test)
11. [Reading the output tables](#11-reading-the-output-tables)
12. [What the visualizations show](#12-what-the-visualizations-show)

---

## 1. What problem are we solving?

Large language models (LLMs) use an **attention mechanism** that, at each generation step, compares a new query vector against all previous key vectors and produces a weighted sum of value vectors. The keys and values from previous tokens are stored in a **KV cache** to avoid recomputation.

For a model like Qwen2.5-3B with 36 layers, 2 KV heads, and head dimension 128, a 10,000-token context requires storing:

    2 (keys + values) x 36 layers x 2 heads x 10,000 tokens x 128 dims x 2 bytes (FP16) = 1.31 GB

This grows linearly with context length and becomes the dominant memory bottleneck. TurboQuant compresses these vectors from 16 bits per coordinate to 2-8 bits, achieving 2-8x compression with minimal quality loss.

The critical question is: **how much does this compression degrade the model's output?** That is what this implementation measures.

---

## 2. The TurboQuant algorithm in plain math

### TurboQuant_MSE (Algorithm 1)

Given a vector **x** in R^d:

**Setup (once, independent of data):**
1. Sample M in R^{d x d} with i.i.d. entries from N(0,1)
2. Compute QR decomposition: M = QR
3. Set Pi = Q (a Haar-random orthogonal matrix)
4. Compute optimal Lloyd-Max codebook C = {c_1, ..., c_{2^b}} for the Beta distribution on [-1, 1]

**Quantize:**
1. Store the norm: alpha = ||**x**||_2
2. Normalize: **u** = **x** / alpha
3. Rotate: **y** = Pi **u**
4. For each coordinate j: idx_j = argmin_k |y_j - c_k|

**Dequantize:**
1. For each j: y_hat_j = c_{idx_j}
2. Inverse rotate: **u_hat** = Pi^T **y_hat**
3. Rescale: **x_hat** = alpha * **u_hat**

**Storage per vector:** b * d bits (indices) + 32 bits (norm) = b*d + 32 bits

### TurboQuant_prod (Algorithm 2)

Allocates (b-1) bits to MSE quantization and 1 bit to a QJL correction on the residual:

**Quantize:**
1. Run TurboQuant_MSE at (b-1) bits to get **x_hat_mse**
2. Compute residual on the unit sphere: **r** = **u** - **u_hat_mse**
3. Store gamma = ||**r**||_2
4. Compute sign bits: **s** = sign(S **r**), where S is a fixed random Gaussian matrix

**Dequantize:**
1. Dequantize the MSE part: **u_hat_mse**
2. Dequantize the QJL part: **u_hat_qjl** = (sqrt(pi/2) / d) * gamma * S^T **s**
3. Combine: **x_hat** = alpha * (**u_hat_mse** + **u_hat_qjl**)

The sqrt(pi/2)/d factor is not arbitrary. It arises from E[|z|] = sqrt(2/pi) for z ~ N(0,1), and is the exact scaling needed to make the inner product estimator unbiased: E[<**q**, **x_hat**>] = <**q**, **x**>.

---

## 3. Why random rotation works

Real KV cache vectors are far from uniformly distributed on the sphere. Some coordinates carry much more information than others (the "outlier channel" phenomenon). If you quantize each coordinate independently, the high-variance coordinates get crushed by a codebook optimized for average coordinates.

Multiplying by a random orthogonal matrix Pi **redistributes information uniformly** across all coordinates. After rotation, each coordinate of the unit-normalized vector follows the **same** marginal distribution, regardless of the original structure of the data.

The key theorem: if **u** is any fixed unit vector and Pi is a Haar-random orthogonal matrix, then the coordinates of Pi**u** are marginally distributed as the coordinates of a uniformly random point on S^{d-1}. This is not an approximation -- it is exact, for any **u**.

This means a single precomputed codebook (optimized for this known marginal distribution) works optimally for **every** input vector. No calibration data, no per-layer tuning, no online adaptation.

The rotation matrix is generated once from a fixed random seed and stored alongside the compressed data. It is applied as a dense matrix-vector multiply: O(d^2) per vector. For d = 128, this is 16,384 multiply-adds, which is negligible compared to the attention computation itself.

### Does TurboQuant interfere with RoPE?

A common concern: most modern LLMs use Rotary Position Embeddings (RoPE), which apply a position-dependent rotation R(n) to the key vector at position n before it is stored in the KV cache. Does TurboQuant's own random rotation Pi interact badly with RoPE's rotation?

No, for a simple reason: **TurboQuant never sees the pre-RoPE vectors.** The KV cache stores keys *after* RoPE has been applied:

    k_cached = R(n) * k_n

By the time TurboQuant operates, k_cached is just a vector in R^d. TurboQuant quantizes it, reconstructs it with bounded MSE, and gives it back. The two rotations operate at different stages of the pipeline:

1. **RoPE** runs during the forward pass, before caching. It produces k_cached.
2. **TurboQuant's Pi** is applied to k_cached during quantization, then inverted by Pi^T during dequantization. The reconstructed vector k_hat is back in the original (post-RoPE) basis.

The attention computation then uses <q, k_hat> instead of <q, k_cached>. The error |<q, k_cached> - <q, k_hat>| <= ||q|| * ||k_cached - k_hat|| by Cauchy-Schwarz, and the MSE bound on ||k_cached - k_hat||^2 holds for *any* input vector, regardless of how it was produced. This is the data-obliviousness property: the MSE guarantee is universal over all inputs on S^{d-1}, which includes any vector that RoPE could produce.

This is not something that needs empirical testing -- it is a mathematical property of the algorithm. There is no mechanism by which RoPE's structure could degrade TurboQuant's quantization quality.

---

## 4. Lloyd-Max codebooks and the Beta distribution

### The marginal distribution

A single coordinate of a point uniformly distributed on S^{d-1} follows:

    f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2},  x in [-1, 1]

This is a symmetric Beta distribution, rescaled from [0,1] to [-1,1]. For d = 128, it is very concentrated around zero (std ~ 1/sqrt(d) ~ 0.088) and approximately Gaussian in shape, but with bounded support.

### Lloyd-Max quantization

The Lloyd-Max algorithm finds the optimal scalar quantizer for a given distribution. "Optimal" means minimizing the expected squared error E[(X - Q(X))^2] over all 2^b-level quantizers.

It alternates two steps until convergence:
1. **Nearest-neighbor assignment:** Each input value is assigned to its closest centroid
2. **Centroid update:** Each centroid becomes the conditional mean E[X | X is in this cell]

The resulting codebook is symmetric around zero (because the distribution is symmetric). For b = 3 bits (8 levels), the centroids at d = 128 are approximately:

    {-0.188, -0.117, -0.044, -0.015, +0.015, +0.044, +0.117, +0.188}

The implementation computes these by numerically integrating the exact PDF, not by sampling. This gives machine-precision codebooks.

### Connection to the Panter-Dite formula

For large b, the optimal quantization distortion satisfies:

    D(b) <= (sqrt(3) * pi / 2) * 4^{-b}

The information-theoretic lower bound (for ANY quantizer) is:

    D(b) >= 4^{-b}

TurboQuant is therefore within a factor of sqrt(3)*pi/2 ~ 2.72 of optimal. This gap is a property of scalar quantization (quantizing each coordinate independently) versus vector quantization (quantizing all coordinates jointly). The gap is modest and worth it because scalar quantization is dramatically simpler and faster.

---

## 5. The four quality metrics

We measure four quantities, each capturing a different aspect of reconstruction quality:

### 5a. Mean Squared Error (MSE)

    MSE = (1/N) * sum_i ||x_i - x_hat_i||_2^2

The average squared L2 distance between original and reconstructed vectors, averaged over all N vectors in the KV cache (keys and values combined, across all layers and heads).

**What it tells you:** The raw magnitude of the reconstruction error. Note that this scales with the norm of the vectors. For the Qwen2.5-3B model, key/value vectors have norms ranging from ~5 to ~50, so the MSE values (e.g., 23.9 at 3-bit) must be interpreted relative to the typical squared norm (~400).

**Theoretical prediction:** For unit vectors, the paper predicts MSE <= sqrt(3)*pi/2 * 4^{-b}. Our vectors are not unit-normalized (norms are stored separately), so our measured MSE is scaled by the mean squared norm.

### 5b. Cosine Similarity

    CosSim = (1/N) * sum_i (x_i . x_hat_i) / (||x_i|| * ||x_hat_i||)

The average cosine of the angle between original and reconstructed vectors.

**What it tells you:** Whether the quantized vector points in the same direction as the original, regardless of magnitude. This is arguably more relevant than MSE for attention, because attention weights are computed from dot products of normalized queries and keys.

**Interpretation:** A cosine similarity of 0.983 (3-bit) means the average angular error is arccos(0.983) ~ 10.5 degrees. At 0.9995 (6-bit), it is ~1.8 degrees.

### 5c. Inner Product Correlation

    IP_Corr = Pearson_r({<x_i, x_j>}, {<x_hat_i, x_hat_j>})

For a random sample of 5,000 pairs (i, j), we compute the true inner product <x_i, x_j> and the estimated inner product <x_hat_i, x_hat_j>, then measure the Pearson correlation.

**What it tells you:** Whether the relative ordering and magnitudes of pairwise inner products are preserved. This is the most directly relevant metric for attention, because attention scores are computed as softmax over inner products. If the correlation is 0.999, then the attention distribution is very close to the original.

**Why pairs from different vectors?** In attention, a query vector from position t is dotted with key vectors from positions 1, ..., t-1. We need the quantized keys to produce similar inner products with the (unquantized) query. Our evaluation uses pairs of quantized vectors as a proxy, which is a slightly harder test.

**Subtle point:** TurboQuant_MSE introduces a systematic **multiplicative bias** in inner products. At 1 bit, E[<**q**, **x_hat_mse**>] = (2/pi) * <**q**, **x**> ~ 0.637 * <**q**, **x**>. This bias shrinks at higher bit widths but never fully vanishes. The IP correlation metric captures rank-order preservation (which is what softmax cares about) even in the presence of this bias.

### 5d. Top-1 Token Match Rate

    Top1 = (1/N) * sum_{t=1}^{N} 1[argmax p_orig(t) == argmax p_quant(t)]

The fraction of generation steps where the model's highest-probability next token is the same when using the original vs quantized KV cache.

**What it tells you:** The ultimate end-to-end metric. MSE, cosine similarity, and IP correlation are all proxies for "does the model produce the same output?" Top-1 match directly answers this question.

**Why this is the metric that matters:** A cosine similarity of 0.995 sounds excellent, but the attention mechanism applies softmax, which is highly sensitive to the relative ordering of logits. A small change in attention weights can flip which token has the highest probability. Conversely, large MSE can be tolerable if it doesn't change the argmax. Top-1 match captures these nonlinear effects that the other metrics miss.

---

## 6. MSE-only vs MSE+QJL: the variance-bias tradeoff

At the same total bit budget (say 3 bits per coordinate), you can either:

- **MSE-only:** Use all 3 bits for scalar quantization. Lower MSE, higher cosine similarity, but inner product estimates are **biased** (systematically shrunk toward zero).

- **MSE+QJL:** Use 2 bits for scalar quantization + 1 bit for the QJL residual correction. Inner product estimates are **unbiased** (E[<**q**, **x_hat**>] = <**q**, **x**>), but MSE and cosine similarity are worse because you have fewer bits for reconstruction.

The QJL correction achieves unbiasedness by adding a noise term with the right expected magnitude. But this noise increases variance, and variance at every position in the attention computation compounds through softmax into degraded token predictions.

Our results on Qwen2.5-3B confirm this dramatically:

| Bit budget | Method | MSE | CosSim | Top-1 |
|-----------|--------|-----|--------|-------|
| 3 bits | MSE-only | 23.9 | 0.983 | 90% |
| 3 bits | MSE(2b)+QJL | 146.2 | 0.919 | 8% |
| 4 bits | MSE-only | 6.5 | 0.995 | 92% |
| 4 bits | MSE(3b)+QJL | 38.8 | 0.974 | 12% |
| 5 bits | MSE-only | 1.8 | 0.999 | 100% |
| 5 bits | MSE(4b)+QJL | 10.4 | 0.993 | 84% |

The lesson: **unbiasedness is a theoretical virtue that doesn't help in practice for KV cache quantization.** The softmax in attention is invariant to global scaling of the logits (which is what the MSE bias amounts to), so the bias is mostly harmless. But the increased variance from QJL is not harmless -- it flips token predictions.

This is consistent with findings from the llama.cpp community and has led to the practical recommendation: **use MSE-only (Algorithm 1) for KV cache compression.**

### Where QJL does matter: vector search and MIPS

The paper is not only about KV caches. Section 4.4 evaluates TurboQuant on **Maximum Inner Product Search (MIPS)**: given a query q, find argmax_i <q, x_i> over a database of millions of stored vectors. This is the core operation in vector databases (Pinecone, Weaviate, FAISS), RAG retrieval, and recommendation systems.

In MIPS, the downstream computation is fundamentally different from attention:

- **No softmax.** Raw inner products are compared directly to rank results. A multiplicative bias that systematically shrinks all scores by different amounts *can* change which vector is the nearest neighbor. QJL's unbiasedness guarantee (E[<q, x_hat>] = <q, x>) directly preserves expected rankings.

- **Variance is tolerable.** In retrieval you return the top-k results. The top-k set is robust to variance in individual scores -- you need the variance to be large enough to push a true top-k result below the (k+1)-th result, which is unlikely when the gap is large. This is the opposite of the softmax case, where variance at every position independently flips token predictions.

The indexing speed advantage is also dramatic. From the paper's benchmarks on 100K vectors at d=1536:

| Method | Indexing time | Notes |
|--------|--------------|-------|
| Product Quantization (PQ) | 240 seconds | Requires k-means training on the data |
| RabitQ | 2,268 seconds | Requires extensive per-vector optimization |
| **TurboQuant** | **0.001 seconds** | Data-oblivious: just rotate and quantize |

TurboQuant is 5-6 orders of magnitude faster to index because its quantization is data-oblivious -- no codebook training, no clustering, no data-dependent optimization. You apply the fixed rotation matrix and quantize against the precomputed Lloyd-Max codebook. A new vector can be indexed in microseconds.

So the two algorithms map to different applications:

| Application | Downstream op | Bias tolerance | Variance tolerance | Winner |
|-------------|--------------|----------------|-------------------|--------|
| KV cache compression | softmax(QK^T) | High (softmax is scale-invariant) | Low (flips argmax) | MSE-only |
| Vector DB / MIPS | argmax <q, x_i> | Low (changes rankings) | Moderate (top-k is robust) | MSE+QJL |
| RAG retrieval | argmax <q, x_i> | Low | Moderate | MSE+QJL |
| Embedding search | argmax <q, x_i> | Low | Moderate | MSE+QJL |

The paper needs both algorithms because they are near-optimal for different objectives -- MSE for reconstruction, prod for inner product distortion -- and those objectives map to genuinely different real-world applications. Our implementation focuses on KV cache compression (where MSE dominates), but TurboQuant's larger practical impact may end up being in vector search, where the combination of QJL's unbiasedness and data-oblivious O(d^2) indexing is compelling.

---

## 7. Top-1 token accuracy and the teacher forcing problem

### The naive approach (and why it fails)

The obvious way to measure Top-1 accuracy is:

1. Generate N tokens with the original KV cache: t_1, t_2, ..., t_N
2. Generate N tokens with the quantized KV cache: t'_1, t'_2, ..., t'_N
3. Report the fraction where t_i = t'_i

The problem is that autoregressive generation is a **chaotic dynamical system**. Each generated token becomes the input for the next step. If t'_1 != t_1 (a single token mismatch at position 1), then position 2 receives different input context, and the entire subsequent sequence diverges. In our initial testing, this produced absurd results:

- 4-bit (CosSim 0.995): 0% match
- 5-bit (CosSim 0.999): 100% match
- 8-bit (CosSim 0.99998): 12% match

The metric was dominated by whether the very first token happened to match, not by the quality of the quantization.

### Teacher-forced evaluation

The fix is to **decouple the positions**. At each step, instead of feeding the model its own previous prediction, we feed the ground truth token (from the original cache's generation). This way, the context at position i is the same for both the original and quantized evaluations, and each position is assessed independently.

The procedure:

1. **Generate ground truth:** Using the original KV cache, greedily generate N tokens: g_1, g_2, ..., g_N. This is ordinary autoregressive generation.

2. **Evaluate the quantized cache (teacher-forced):** Starting from the quantized KV cache:
   - Feed the last prompt token. Record the model's prediction p_1.
   - Feed g_1 (the ground truth, NOT p_1). Record prediction p_2.
   - Feed g_2. Record prediction p_3.
   - ... and so on for N steps.

3. **Score:** Top-1 match = fraction of positions where p_i = g_i.

Concretely, at position i, the model's context is:
- The quantized KV cache for the original prompt (tokens 0 through seq_len-1)
- Ground truth continuation tokens g_1 through g_{i-1} (generated fresh, not from the quantized cache)

So the only source of error is the quantized prompt cache, and each position independently tests whether this error is enough to flip the argmax.

### Why "teacher forcing?"

The term comes from sequence-to-sequence training in NLP: during training, you feed the ground truth at each step rather than the model's own predictions. The same idea applies here -- we force the "correct" context so we can evaluate each position in isolation.

### A subtlety: the continuation tokens are unquantized

In positions beyond the original prompt, the KV cache entries come from the model's own forward pass (processing g_1, g_2, etc.) and are stored in full precision. Only the prompt portion of the KV cache is quantized. This is the realistic deployment scenario: you quantize the prefilled cache to save memory, then continue generating with fresh (unquantized) KV entries.

---

## 8. Outlier channel handling and fractional bit rates

### The problem with uniform bit widths

With uniform quantization at b bits per coordinate, the only operating points are b = 2, 3, 4, 5, ... This is a coarse grid. The paper's headline results are at 2.5-bit and 3.5-bit, which uniform quantization cannot reach.

More fundamentally, not all channels are equally important. In real LLMs, certain channels of the key/value embeddings carry disproportionately more information (higher variance, larger magnitudes). These are called **outlier channels**. Spending the same number of bits on a high-variance channel and a low-variance channel is suboptimal.

### The rotation paradox

This raises a natural question: doesn't the random rotation already equalize all channels? Yes, after rotation, each coordinate of the unit-normalized vector follows the same marginal distribution. So why do outlier channels matter?

The answer is that the rotation operates on the **unit-normalized** vector, but the channels of the **original** (pre-normalization) vector still have different variances. When we split the original vector into two channel groups, normalize each group independently, and apply separate rotations, we are effectively:

1. Giving more bits to the channels that contribute more to the overall vector magnitude
2. Using a rotation matrix whose dimension matches the subspace, which changes the Beta distribution parameters (Beta((K-1)/2, (K-1)/2) for a K-dimensional subspace, with different concentration than the full d-dimensional case)

### The mixed-precision scheme

Split the d = 128 channels into two groups:

- **Outlier group**: K channels with the highest per-channel variance (typically K = 32 or 64)
- **Normal group**: the remaining d - K channels

Each group gets its **own** TurboQuant instance:
- Its own rotation matrix (dimension K or d-K)
- Its own Lloyd-Max codebook (computed for the Beta distribution at dimension K or d-K)
- Its own bit width (b_high for outliers, b_low for normals)
- Its own stored norm (one float32 per group per vector)

The effective bit rate is:

    effective_bits = (K * b_high + (d - K) * b_low) / d

For example, with K = 32 outlier channels at 4 bits and 96 normal channels at 2 bits:

    effective = (32 * 4 + 96 * 2) / 128 = (128 + 192) / 128 = 2.5 bits

### How outlier channels are identified

We compute the per-channel variance across the entire KV cache:

    var_j = Var({x_{i,j} : i = 1, ..., N})  for each channel j in {0, ..., d-1}

where x_{i,j} is channel j of the i-th vector, pooled across all layers, heads, tokens, and both keys and values. The top-K channels by variance are designated as outliers.

This is a data-dependent step: the outlier indices depend on the actual KV cache from the model. However, in practice, outlier channels are highly consistent across different prompts for a given model (they are a property of the trained weight matrices, not the input).

In Qwen2.5-3B, the top-32 outlier channels cluster in two contiguous blocks (channels 49-63 and 109-127), reflecting the model's internal structure.

### Results on Qwen2.5-3B

The comparison table shows outlier handling interpolating between uniform bit widths:

| Method | Eff. Bits | CosSim | Top-1 |
|--------|-----------|--------|-------|
| Uniform 2-bit | 2.0 | 0.941 | 76% |
| **Outlier 2.5-bit** | **2.5** | **0.963** | **82%** |
| Uniform 3-bit | 3.0 | 0.983 | 90% |
| **Outlier 3.5-bit** | **3.5** | **0.990** | **92%** |
| Uniform 4-bit | 4.0 | 0.995 | 92% |

The 2.5-bit outlier configuration achieves 82% Top-1 at 5.3x compression — a significant jump from uniform 2-bit (76% at 7.1x) while using only half a bit more per coordinate. The 3.5-bit configuration matches uniform 4-bit Top-1 accuracy (92%) at a better compression ratio (4.0x vs 3.8x).

### A subtlety: the codebook changes with subspace dimension

The Beta distribution's shape depends on dimension: Beta((d-1)/2, (d-1)/2). At d = 128, the standard deviation is ~1/sqrt(128) = 0.088. At d = 32, it is ~1/sqrt(32) = 0.177 — nearly twice as wide. This means the codebook centroids are different for each subspace dimension, and **cannot** be reused from the d = 128 codebooks. The implementation precomputes separate codebooks for each (dimension, bit-width) pair needed.

---

## 9. The residual window

### The idea

The residual window is a practical technique from KIVI (not part of the TurboQuant paper) that keeps the most recent W tokens in full FP16 precision while compressing all older tokens. The KV cache is split into two regions:

```
Token positions:   [0 ............... n-W-1] [n-W ........... n-1]
                   |---- compressed ----|  |---- FP16 ----|
                        (TurboQuant)         (residual window)
```

### Why recent tokens matter disproportionately

In autoregressive generation, the model's attention at position t computes softmax over scores {<q_t, k_i> : i = 1, ..., t-1}. In practice, attention scores are not uniform -- they are heavily skewed toward recent tokens. This is partly architectural (RoPE biases attention toward nearby positions) and partly statistical (recent context is more relevant for predicting the next token).

A quantization error in a high-attention-weight token has more impact on the output than the same error in a low-weight token. Since recent tokens receive the highest attention weights, keeping them in full precision gives the most quality improvement per byte of uncompressed storage.

### The compression cost is small at scale

The window is a fixed size (typically W = 128) regardless of sequence length. Its cost as a fraction of total storage diminishes with context length:

| Sequence length | Window (W=128) | Compressed tokens | Effective 3-bit compression |
|----------------|---------------|-------------------|---------------------------|
| 168 tokens | 76% FP16 | 40 | 1.23x (not useful) |
| 1,000 tokens | 13% FP16 | 872 | 3.98x |
| 10,000 tokens | 1.3% FP16 | 9,872 | 4.69x |
| 100,000 tokens | 0.13% FP16 | 99,872 | 4.89x |

At 10K+ tokens, the window costs less than 5% of the compression ratio but protects the tokens that matter most for generation quality. The benefit is most visible in long-context workflows (document QA, multi-turn conversations, code generation with large contexts) where the compressed prefix is large and the recent window is a small fraction.

### What our evaluation measures

The `--residual-window W` flag runs a comparison at 2, 3, and 4-bit:

- **Top-1 (no window):** Every token in the KV cache is quantized. This is the number from the main MSE table.
- **Top-1+W:** Tokens [0, n_tokens-W) are quantized, tokens [n_tokens-W, n_tokens) are kept in FP16. This is the residual window result.

The comparison uses the same teacher-forced evaluation as the main Top-1 metric, so the numbers are directly comparable.

### Limitations of our test

Our default prompt is 168 tokens with a 128-token window, meaning only 40 tokens are compressed. This makes the effective compression ratio poor (~1.2x) and limits the statistical power of the comparison. For a proper demonstration, use a prompt that produces 1000+ tokens of KV cache. The implementation is correct at any sequence length -- the limitation is the test prompt, not the algorithm.

---

## 10. The disk round-trip test

The disk evaluation tests a practical deployment scenario: **warm-tier KV cache storage.** The idea is that after processing a long document, you compress and serialize the KV cache to disk (SSD/NVMe). When the user returns, you load and decompress it instead of re-running the expensive prefill.

The test measures the full pipeline:

1. **GPU quantization:** Apply TurboQuant to the KV cache tensors on GPU.
2. **Bit packing and serialization:** Pack the b-bit indices into a contiguous bitstream and write to disk along with the rotation matrix (once), codebook (once), and per-block norms.
3. **Deserialization on CPU:** Read the binary file and unpack indices.
4. **CPU dequantization:** Look up codebook values, apply the inverse rotation, and rescale by norms. No GPU required.
5. **Quality verification:** Compare the CPU-reconstructed tensors against the originals.

### The binary file format (.tqkv)

```
[Header: 32 bytes]
  Magic "TQKV" | version | mode | bits | d | n_layers | n_heads | n_tokens

[Rotation matrix: d*d*4 bytes]    <- stored once, shared by all vectors

[Codebook: 2^b * 8 bytes]         <- stored once (float64 for precision)

[Per layer, per head:]
  Packed indices: ceil(n_tokens * d * b / 8) bytes
  Norms: n_tokens * 4 bytes (float32)
```

The rotation matrix (128x128 float32 = 64 KB) and codebook (e.g., 8 float64 values = 64 bytes for 3-bit) are negligible overhead amortized over the cache.

### Compression arithmetic

For a vector of dimension d = 128:

| Representation | Bits per coordinate | Bytes per vector | Overhead |
|---------------|-------------------|-----------------|----------|
| FP16 | 16 | 256 | - |
| 3-bit TurboQuant | 3 | 48 + 4 (norm) = 52 | rotation + codebook (shared) |
| 4-bit TurboQuant | 4 | 64 + 4 (norm) = 68 | same |

The "+4 bytes" norm overhead is per vector (one float32 scalar). The rotation matrix and codebook are shared across all vectors and stored once in the file header.

### Why verify on CPU?

The point of disk serialization is that you might load the cache on a different machine, or on the CPU of a machine whose GPU is busy with another request. Verifying that CPU dequantization produces the exact same tensors as GPU dequantization (within floating-point tolerance) confirms the format is self-contained.

---

## 11. Reading the output tables

### The MSE results table

```
 Bits          MSE      Cos Sim      IP Corr    Bytes/Vec   Compress    Top-1
    2    86.861328     0.940694     0.998940         36.0       7.1x   76.0%
    3    23.888739     0.983144     0.999736         52.0       4.9x   90.0%
    ...
```

Each row is one bit width. All metrics are computed over the entire KV cache: all layers, all heads, both keys and values combined. The vectors are evaluated at their natural scale (not unit-normalized), so MSE values reflect the actual norms of the cache entries.

**Bytes/Vec** includes the per-vector norm storage (4 bytes). **Compress** is FP16 bytes (256) divided by Bytes/Vec.

**Top-1** only appears when `--eval-top1` is used. It is evaluated on 50 generated tokens by default.

### The comparison table

```
 Budget          Method          MSE      Cos Sim      IP Corr    Top-1
    3b        MSE-only    23.888739     0.983144     0.999736   90.0%
    3b     MSE(2b)+QJL   146.154449     0.919248     0.997648    8.0%
```

This is the key table. At the same total bit budget, MSE-only consistently outperforms MSE+QJL on every metric. The Top-1 column makes the case decisive.

### The outlier handling table

```
Config             Eff.Bits          MSE      Cos Sim      IP Corr   Compress    Top-1
2.5-bit                 2.5    22.710089     0.963274     0.999713       5.3x   82.0%
3.5-bit                 3.5     6.617840     0.989528     0.999914       4.0x   92.0%
3.0-bit (mixed)         3.0    16.992500     0.976017     0.999721       4.6x   84.0%
4.0-bit (mixed)         4.0     4.590285     0.993196     0.999938       3.6x   90.0%
```

Each row is one mixed-precision configuration. **Eff.Bits** is the weighted average: (K * b_high + (d-K) * b_low) / d. The compression ratio accounts for two norms per vector (one per subspace) instead of one.

The "Outlier vs Uniform" comparison table that follows groups each outlier config between the two uniform bit widths that bracket it. This is the key table for judging whether outlier handling is worth the added complexity. Look for cases where the outlier config matches or exceeds the higher uniform bit width at the lower one's compression ratio.

### The disk round-trip table

```
Format                     Size (MB)   Compress          MSE     CosSim
FP16 baseline                   6.19       1.0x     0.000000   1.000000
TurboQuant_MSE 3-bit            1.32      4.68x    23.888737   0.983144
```

The MSE and CosSim here should match the GPU evaluation exactly (modulo float32 rounding from the serialization round-trip). If they diverge, there is a bug in the binary format.

---

## 12. What the visualizations show

### (a) Coordinate distribution before and after rotation

**Left panel:** Histogram of all coordinates of the unit-normalized KV cache vectors (before rotation). This is typically non-uniform -- in a real model, some coordinates carry much more information than others.

**Right panel:** The same vectors after applying the random orthogonal rotation. The histogram should closely match the red curve, which is the theoretical Beta((d-1)/2, (d-1)/2) PDF. This confirms that the rotation successfully isotropizes the data.

If the right panel does NOT match the Beta PDF, something is wrong -- either the rotation matrix is not orthogonal, or the normalization step has a bug.

### (b) Cosine similarity vs bit width

Two curves: MSE-only (solid) and MSE+QJL (dashed). Both should increase monotonically toward 1.0. The MSE-only curve should be consistently above the MSE+QJL curve, confirming MSE-only's superiority at every bit width.

The y-axis starts at 0.9 (not 0) to make the differences visible. At 6+ bits, both methods are essentially perfect.

### (c) Compression ratio and storage

Bar chart showing compression ratio (left axis) with a line showing bytes per vector (right axis). The gray dashed line marks the FP16 baseline (256 bytes per vector). This is a practical engineering chart: "how much memory do I save?"

### (d) True vs estimated inner products

**Left panel (MSE):** Scatter plot of <x_i, x_j> vs <x_hat_i, x_hat_j> for 5,000 random pairs. For MSE quantization, the cloud should cluster tightly around the y=x line but with a slight **downward bias** (points below the red line), reflecting the systematic shrinkage of inner products.

**Right panel (MSE+QJL):** Same scatter plot for the prod quantizer. The cloud should be centered on the y=x line (unbiased) but with more **scatter** (higher variance). This is the visual representation of the bias-variance tradeoff discussed in Section 6.

---

## Further reading

- **TurboQuant paper:** [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **QJL paper** (the 1-bit sign quantization used in Algorithm 2): [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **PolarQuant** (alternative approach using polar coordinates, NOT random rotation): [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **KIVI** (the residual window technique, not implemented here): [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
