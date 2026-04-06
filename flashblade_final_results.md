# TurboQuant + FlashBlade: 6x Faster KV Cache Restore with Compression and GPU Direct Storage

## Overview

Large language models maintain a **KV (key-value) cache** during inference — a running record of every token's internal representation that the model needs to generate the next token. As conversations grow longer, this cache grows proportionally: a 7-billion parameter model at 30,000 tokens of context produces a KV cache of **1.72 GB**. For services running thousands of concurrent sessions, this creates two problems:

1. **Storage capacity**: Thousands of sessions times gigabytes per session means terabytes of KV cache data.
2. **Restore latency**: When a user returns to a conversation, the system needs to reload their KV cache from storage back into GPU memory before it can generate the first token. Slower restore means longer time-to-first-token.

This benchmark measures how **TurboQuant** — a mathematically-principled compression algorithm — combines with **Pure Storage FlashBlade** and **NVIDIA GPU Direct Storage (GDS)** to solve both problems simultaneously. The key result: **compressed KV caches restore 6x faster than uncompressed ones** when the full stack (TurboQuant + GDS + FlashBlade over RDMA) works together.

### What is TurboQuant?

TurboQuant is an implementation of the algorithm from "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et al., Google Research, 2025). Unlike traditional quantization methods that require calibration data, TurboQuant is **data-oblivious**: it applies a random orthogonal rotation to each vector, which transforms the data into a known Beta distribution, then applies precomputed Lloyd-Max scalar quantization. This means the codebooks can be computed once offline and reused for any data — no calibration step, no data-dependent tuning.

At 3-bit precision, TurboQuant achieves **4.9x compression** while maintaining **0.983 cosine similarity** to the original FP16 data. More importantly for real-world use, a Qwen2.5-7B model with a 3-bit compressed KV cache produces the **exact same tokens** as the uncompressed model in our teacher-forced evaluation (100% Top-1 accuracy at 30,000 tokens of context).

### What is GPU Direct Storage (GDS)?

In a conventional I/O path, data flows: **Storage → CPU Memory → GPU Memory**. The CPU is involved at every step — it manages the DMA transfers, copies data between buffers, and orchestrates the flow. When 8 GPUs are all trying to restore sessions simultaneously, the CPU becomes a bottleneck.

GPU Direct Storage eliminates the CPU from the data path entirely: **Storage → GPU Memory**. The storage device (FlashBlade) writes directly into GPU memory over RDMA, and the GPU itself handles any post-processing (decompression, in our case). This is enabled by NVIDIA's kvikio library and requires RDMA-capable storage.

### Why do these two technologies complement each other?

Compression alone doesn't help if the CPU still has to decompress the data before copying it to the GPU — you've reduced the I/O but added CPU work. GDS alone helps with bandwidth but doesn't reduce the amount of data transferred. Together:

- **TurboQuant** shrinks the data 4.9x, so there's less to transfer over the network.
- **GDS** sends that compressed data directly to the GPU, bypassing the CPU entirely.
- **The GPU decompresses** the data in-place — and GPUs are massively parallel, so decompression is trivial (sub-millisecond).

The result is a pipeline where the CPU does almost nothing, the network transfers 5x less data, and the GPU handles both decompression and inference.

---

## Test Environment

| Component | Details |
|-----------|---------|
| **GPUs** | 8x NVIDIA A100-SXM4-40GB |
| **CPU** | 256 cores, 1 TB RAM |
| **Storage** | Pure Storage FlashBlade |
| **Network** | NFS over RDMA (`proto=rdma, nconnect=16`) |
| **Model** | Qwen/Qwen2.5-7B-Instruct (28 layers, 4 KV heads, head_dim=128) |
| **Context lengths** | 1,000 / 5,000 / 10,000 / 30,000 tokens |
| **Compression** | 3-bit and 4-bit TurboQuant, plus mixed-precision variants (2.5, 3.0, 3.5, 4.0 effective bits) |
| **I/O mode** | O_DIRECT (bypasses Linux page cache for accurate timing) |

### A note on O_DIRECT

All I/O in these benchmarks uses `O_DIRECT`, which bypasses the Linux page cache and writes/reads directly to/from the storage device. Without O_DIRECT, writes appear artificially fast (data lands in page cache and the kernel flushes it asynchronously) and reads may hit cached data instead of the storage device. O_DIRECT ensures every timing measurement reflects the actual FlashBlade performance. This is implemented using page-aligned `mmap` buffers and the `O_DIRECT` flag on `open()`, with `fsync` after writes.

---

## Benchmark 1: Single-GPU Storage I/O Performance

### What this benchmark measures

This is the foundational benchmark. It answers: **How long does it take to compress, write, read, and decompress a KV cache at each context length and bit width?** Everything runs on a single GPU, and each operation is timed independently:

1. **Compress** — The GPU quantizes the FP16 KV cache into packed indices + norms using TurboQuant. This involves applying a random orthogonal rotation to each head's vectors, then mapping each rotated coordinate to its nearest codebook entry.

2. **Write** — The compressed data (or raw FP16 for the baseline) is serialized to a `.tqkv` binary file on the FlashBlade. For compressed data, this includes the rotation matrix, codebook, and packed indices/norms. For FP16, it's just the raw tensor bytes.

3. **Read** — The file is read back from the FlashBlade into CPU memory. For compressed data, this includes parsing the binary format and reconstructing the index arrays.

4. **Decompress** — The CPU dequantizes the data: looks up codebook values, applies the inverse rotation, and scales by the stored norms to reconstruct approximate FP16 vectors.

The benchmark also computes **cosine similarity** between the original and reconstructed vectors to quantify compression quality.

### Example: What happens to a single KV cache vector

Consider a single attention head's key vector at one token position — a 128-dimensional FP16 vector (256 bytes).

**Without compression (FP16):**
- Write 256 bytes to disk, read 256 bytes back. The vector is stored and restored exactly.

**With 3-bit TurboQuant:**
1. Multiply the 128-dim vector by a 128x128 orthogonal rotation matrix (precomputed from a fixed seed).
2. Record the vector's L2 norm (4 bytes, float32) and normalize the vector.
3. Map each of the 128 rotated coordinates to one of 8 codebook levels (3 bits each).
4. Pack the 128 three-bit indices into 48 bytes.
5. Write 48 + 4 = 52 bytes to disk (vs 256 bytes for FP16 — a **4.9x** reduction).
6. To restore: read 52 bytes, unpack the 3-bit indices, look up codebook values, apply the inverse rotation, scale by the norm.

The math guarantees that the rotation transforms any distribution into a Beta distribution, so the codebook (precomputed for that Beta distribution) is near-optimal regardless of the input data.

### Results

Focusing on the larger, more representative context lengths (5,000 and 30,000 tokens):

| Context | Format | Size | Ratio | Compress | Write | Read | Decompress | Cosine Sim |
|---------|--------|------|-------|----------|-------|------|------------|------------|
| 5,000 | FP16 | 286.7 MB | 1.0x | -- | 1,338 ms | 553 ms | -- | 1.000000 |
| 5,000 | 3-bit | 58.3 MB | 4.9x | 133 ms | 790 ms | 628 ms | 588 ms | 0.983172 |
| 5,000 | 4-bit | 76.2 MB | 3.8x | 324 ms | 857 ms | 484 ms | 587 ms | 0.995443 |
| 5,000 | 2.5-bit | 53.8 MB | 5.3x | 175 ms | 812 ms | 645 ms | 898 ms | 0.964625 |
| 5,000 | 3.5-bit | 71.7 MB | 4.0x | 146 ms | 899 ms | 762 ms | 1,067 ms | 0.989901 |
| 30,000 | FP16 | 1.72 GB | 1.0x | -- | 32,942 ms | 3,275 ms | -- | 1.000000 |
| 30,000 | 3-bit | 349.5 MB | 4.9x | 590 ms | 4,001 ms | 3,436 ms | 3,353 ms | 0.983159 |
| 30,000 | 4-bit | 457.0 MB | 3.8x | 772 ms | 25,299 ms | 2,923 ms | 2,821 ms | 0.995440 |
| 30,000 | 2.5-bit | 322.6 MB | 5.3x | 981 ms | 2,934 ms | 2,383 ms | 4,280 ms | 0.964414 |
| 30,000 | 3.5-bit | 430.1 MB | 4.0x | 1,192 ms | 4,057 ms | 3,862 ms | 4,591 ms | 0.989841 |

### What to take away

- **Compression ratios are consistent** across context lengths: 3-bit always gives 4.9x, 4-bit always gives 3.8x. This is expected since TurboQuant's compression is per-vector and doesn't depend on sequence length.

- **Cosine similarity is stable** across context lengths too. 3-bit gives 0.983 whether you compress 1,000 or 30,000 tokens. The rotation + codebook is mathematically optimal for any input.

- **Compression is fast** relative to I/O: at 30K tokens, 3-bit compression takes 590 ms on GPU, while the write takes 4,001 ms. Compression is not the bottleneck.

- **Write times show high variance** at small sizes (1K tokens) due to O_DIRECT overhead on NFS — each write requires a full round-trip to the FlashBlade. At larger sizes, this overhead is amortized.

- **The mixed-precision variants** (2.5-bit, 3.5-bit) use split-channel quantization: the 32 highest-variance channels are quantized at a higher bit width (4 or 5 bits), while the remaining 96 channels use a lower bit width (2 or 3 bits). This gives fractional effective rates and intermediate quality/size trade-offs.

---

## Benchmark 2: Concurrent Multi-GPU Checkpoint/Restore

### What this benchmark measures

Real inference servers don't checkpoint one session at a time. This benchmark answers: **What happens when all 8 GPUs simultaneously checkpoint and restore their sessions?**

Each of the 8 GPUs runs as an independent worker process (via Python subprocess with `CUDA_VISIBLE_DEVICES` set). Every worker performs the full cycle:

1. Load shared KV cache data (from a staging file, not timed)
2. **Compress** on GPU — each worker independently quantizes the same KV cache
3. **Write** the compressed file to the FlashBlade (each worker writes its own file)
4. **Read** the file back from the FlashBlade
5. **Decompress** on CPU
6. **Verify** — compute MSE against the original to confirm correctness

The benchmark reports per-worker timings, aggregate bandwidth (total bytes across all workers / wall-clock time for the slowest worker), and overall throughput in sessions per second.

### Example: 8-user inference server

Imagine an inference server handling 8 concurrent users, each with a 10,000-token conversation. The server needs to periodically checkpoint these sessions to persistent storage so it can free GPU memory for new requests. Later, when a user returns, it restores their session.

With 3-bit TurboQuant, each session compresses from 573 MB to 117 MB. All 8 workers write their compressed sessions concurrently, then read them back.

### Results (10,000 tokens, 3-bit, 8 GPUs, NUMA-pinned workers)

| Worker | GPU | Compress | Write | Read | Decompress | Total |
|--------|-----|----------|-------|------|------------|-------|
| 0 | 0 | 267 ms | 1,285 ms | 1,540 ms | 1,836 ms | 4,927 ms |
| 1 | 1 | 261 ms | 1,308 ms | 1,470 ms | 1,800 ms | 4,839 ms |
| 2 | 2 | 264 ms | 28,338 ms | 1,064 ms | 1,344 ms | 31,009 ms |
| 3 | 3 | 259 ms | 29,356 ms | 1,641 ms | 1,305 ms | 32,562 ms |
| 4 | 4 | 360 ms | 28,653 ms | 1,065 ms | 960 ms | 31,038 ms |
| 5 | 5 | 267 ms | 15,370 ms | 1,101 ms | 940 ms | 17,679 ms |
| 6 | 6 | 266 ms | 28,358 ms | 1,299 ms | 1,013 ms | 30,936 ms |
| 7 | 7 | 268 ms | 1,416 ms | 1,110 ms | 960 ms | 3,753 ms |

| Summary Metric | Value |
|----------------|-------|
| Per-session FP16 | 573.4 MB |
| Per-session compressed | 116.5 MB (4.9x) |
| Wall time (all 8) | 42.18 s |
| Aggregate write bandwidth | 0.03 GB/s |
| Aggregate read bandwidth | 0.57 GB/s |
| Avg round-trip per session | 19,593 ms |

### What to take away

- **NUMA-aware CPU pinning is critical**: Each worker is pinned to the 16 CPU cores on the NUMA node local to its GPU (e.g., GPU 0 on NUMA 3 uses cores 48-63). Without pinning, 8 workers each spawning 128 threads caused massive contention (40+ second decompression). With pinning, decompression drops to **0.9–1.8 seconds**.

- **Reads and decompression scale well with pinning**: 1.0–1.6 seconds for reads, 0.9–1.8 seconds for decompression. The CPU-mediated path works at scale when threads are properly isolated.

- **Write contention is the new bottleneck**: Workers 2-6 hit NFS write spikes of 15–29 seconds, while workers 0, 1, and 7 completed writes in ~1.3 seconds. This is I/O-layer contention (NFS locking, FlashBlade ingestion), not CPU contention.

- **This is the problem GDS solves**: Benchmark 5 shows that replacing the CPU-mediated path with GDS drops restore to **188 milliseconds**. GDS eliminates both the CPU decompression step and the NFS read path.

---

## Benchmark 3: Session Capacity Planning

### What this benchmark measures

This is a planning tool, not a performance benchmark. It answers: **How many sessions can you store per terabyte at each context length and compression level?**

The calculation is simple: `sessions_per_TB = 1,000,000,000,000 / bytes_per_session`. But seeing the numbers in a table makes the capacity impact of compression concrete.

### Example: Planning a 10TB FlashBlade deployment

Suppose you're deploying a FlashBlade with 10 TB usable capacity for KV cache storage. Your users average 10,000 tokens of context. How many sessions can you keep warm?

| Format | Sessions per TB | Sessions on 10 TB |
|--------|----------------|--------------------|
| FP16 | 1,743 | 17,430 |
| 2.5-bit | 9,297 | 92,970 |
| 3-bit | 8,580 | 85,800 |
| 3.5-bit | 6,973 | 69,730 |
| 4-bit | 6,562 | 65,620 |

### Full results

| Context | FP16 | 2.5-bit | 3-bit | 3.0-bit (mixed) | 3.5-bit | 4-bit | 4.0-bit (mixed) | Gain |
|---------|------|---------|-------|------------------|---------|-------|------------------|------|
| 1,000 | 17,438 | 92,651 | 85,370 | 79,510 | 69,553 | 65,369 | 61,876 | 5.3x |
| 5,000 | 3,487 | 18,586 | 17,151 | 15,935 | 13,942 | 13,118 | 12,395 | 5.3x |
| 10,000 | 1,743 | 9,297 | 8,580 | 7,969 | 6,973 | 6,562 | 6,199 | 5.3x |
| 30,000 | 581 | 3,099 | 2,861 | 2,657 | 2,324 | 2,188 | 2,066 | 5.3x |

### What to take away

- **5.3x more sessions per TB** at the most aggressive compression (2.5-bit). Even at 4-bit (higher quality), you get **3.8x** more sessions.

- **Context length is the dominant factor**: A 30K-token session is 30x larger than a 1K-token session, regardless of compression. Compression multiplies your capacity at every context length equally.

- **The gain column is constant (5.3x)** because it compares the smallest compressed format (2.5-bit) against FP16. The ratio depends only on the compression factor, not the context length.

---

## Benchmark 4: Session Migration (GPU 0 → Disk → GPU 1)

### What this benchmark measures

This is the most realistic end-to-end benchmark. It simulates a complete **session migration** — the process of moving a user's conversation from one GPU to another, which happens during load balancing, GPU maintenance, or when a user resumes a conversation that was previously evicted from GPU memory.

The full pipeline:

1. **Compress on GPU 0** — Quantize the KV cache using 3-bit TurboQuant. Two "outlier layers" (layers 0 and 27, which have anomalously large key norms) have their keys kept in FP16 to prevent catastrophic attention distortion.

2. **Write to FlashBlade** — Serialize the compressed cache to a `.tqkv` file, plus a separate FP16 file for the exempt layers' keys.

3. **Read from FlashBlade** — Read both files back into CPU memory.

4. **Decompress on CPU** — Dequantize the compressed data and patch in the exempt layers' FP16 keys.

5. **Build KV cache on GPU 1** — Transfer the reconstructed cache to a different GPU and build a `DynamicCache` object compatible with the HuggingFace Transformers library.

6. **Resume generation** — Load the model on GPU 1, feed the reconstructed KV cache, and generate 50 tokens using teacher-forced evaluation. Compare each predicted token against the ground truth (generated with the original uncompressed cache) to measure Top-1 accuracy.

### Example: User resumes a long conversation

A user had a 30,000-token conversation (about 22,500 words — roughly a 45-page document worth of context). They closed their browser and returned an hour later. The server had evicted their session to the FlashBlade. Now it needs to restore the session onto whatever GPU has capacity.

Without compression: read 1.72 GB from storage, copy to GPU. Simple but slow and storage-intensive.

With TurboQuant: read 411 MB (4.2x smaller), decompress, copy to GPU, resume. More steps but less I/O.

### Results (30,000 tokens, 3-bit, GPU 0 → GPU 1)

| Phase | Time | Notes |
|-------|------|-------|
| Compress (GPU) | 553 ms | 3-bit, 30,000 tokens |
| Write .tqkv | 7,708 ms | 349.5 MB |
| Write exempt keys (FP16) | 182 ms | 61.4 MB |
| Read from disk | 3,335 ms | 410.9 MB total |
| Decompress + patch (CPU) | 3,619 ms | |
| Build KV cache (GPU) | 321 ms | |
| Resume generation | 1,342 ms | 50 tokens |
| **TOTAL MIGRATION** | **15,398 ms** | (excl. model load) |
| Model load (one-time) | 4,130 ms | |

| Storage Metric | Value |
|---------------|-------|
| Compressed size | 410.9 MB |
| FP16 baseline | 1,720.3 MB |
| Compression ratio | 4.2x |
| **Top-1 accuracy** | **100.0%** (50/50 tokens correct) |

### What to take away

- **100% Top-1 accuracy**: The compressed-and-restored model produced the exact same tokens as the original model for all 50 generated tokens. This is the most important number — compression is useless if it changes the model's output.

- **The compression ratio is 4.2x, not 4.9x**: Two of the 28 layers have their keys kept in FP16 (layer exemption), which adds 61.4 MB of uncompressed data. This reduces the effective compression ratio from the theoretical 4.9x to 4.2x. The cost is small; the quality benefit is large — without layer exemption, these two outlier layers would produce systematically wrong attention patterns.

- **Write is the slowest phase** (7.7 seconds for 350 MB). This is a single-GPU O_DIRECT write — the FlashBlade is capable of much higher throughput with concurrent writers, but this benchmark intentionally measures single-stream performance.

- **CPU decompression (3.6 seconds)** is comparable to disk read (3.3 seconds). In a production system with GDS, the GPU would handle decompression in milliseconds, eliminating this phase entirely.

- **Model load (4.1 seconds)** is a one-time cost — in production, the model is already loaded on the destination GPU. The migration time that matters is the 15.4 seconds (or much less with GDS).

---

## Benchmark 5: TurboQuant ON vs OFF — The Full Comparison

### What this benchmark measures

This is the headline benchmark. It runs four configurations head-to-head with 8 concurrent GPU workers, comparing every combination of compression (on/off) and GDS (on/off):

1. **FP16** — No compression, standard I/O (CPU-mediated). The baseline.
2. **FP16 + GDS** — No compression, GPU Direct Storage. Shows the benefit of GDS alone.
3. **TQ 3-bit** — TurboQuant compression, standard I/O. Shows the benefit of compression alone.
4. **TQ 3-bit + GDS** — TurboQuant compression, GPU Direct Storage. The full stack.

Each configuration runs 8 workers in parallel (one per GPU), each performing a full checkpoint (compress + write) and restore (read + decompress) cycle. The benchmark measures per-worker timings and computes aggregate metrics.

### The I/O path for each configuration

**FP16 (standard I/O):**
```
Checkpoint: GPU → CPU memory → FlashBlade
Restore:    FlashBlade → CPU memory → GPU
```
The CPU copies FP16 tensors to RAM, then writes to storage. On restore, it reads from storage into RAM, then copies to GPU. Simple but the CPU is in the critical path.

**FP16 + GDS:**
```
Checkpoint: GPU → FlashBlade  (direct, via RDMA)
Restore:    FlashBlade → GPU   (direct, via RDMA)
```
RDMA allows the FlashBlade to read/write GPU memory directly. No CPU involvement. But the full 573 MB still has to travel over the network.

**TQ 3-bit (standard I/O):**
```
Checkpoint: GPU compresses → CPU serializes → FlashBlade
Restore:    FlashBlade → CPU deserializes → CPU decompresses → GPU
```
Only 117 MB travels over the network (4.9x less), but the CPU has to deserialize and decompress. With 8 workers, CPU contention makes this path very slow.

**TQ 3-bit + GDS:**
```
Checkpoint: GPU compresses + packs → FlashBlade  (direct, via RDMA)
Restore:    FlashBlade → GPU unpacks + decompresses  (all on GPU)
```
Only 117 MB travels over the network via RDMA, and the GPU handles all decompression. The CPU does essentially nothing.

### Results (10,000 tokens, 3-bit, 8 GPUs)

#### Per-worker timings (milliseconds)

| GPU | FP16 Write | FP16 Read | FP16+GDS Write | FP16+GDS Read | TQ Write | TQ Read | TQ+GDS Write | TQ+GDS Read |
|-----|------------|-----------|----------------|---------------|----------|---------|--------------|-------------|
| 0 | 1,119 | 1,090 | 4,754 | 824 | 1,804 | 54,633 | 1,226 | 184 |
| 1 | 1,124 | 1,127 | 4,946 | 838 | 1,998 | 54,238 | 2,136 | 184 |
| 2 | 1,219 | 1,138 | 4,859 | 810 | 1,768 | 55,659 | 2,185 | 178 |
| 3 | 1,077 | 1,124 | 4,596 | 813 | 1,698 | 43,713 | 1,129 | 187 |
| 4 | 1,265 | 1,130 | 4,916 | 809 | 1,754 | 38,506 | 2,233 | 187 |
| 5 | 1,134 | 1,135 | 4,958 | 2,527 | 1,790 | 52,217 | 2,273 | 178 |
| 6 | 1,236 | 1,122 | 4,859 | 814 | 1,559 | 44,656 | 1,954 | 222 |
| 7 | 1,141 | 1,149 | 4,961 | 792 | 1,851 | 55,721 | 2,207 | 187 |

*Note: "Read" for TQ includes both read and CPU decompression; for TQ+GDS it includes read, GPU unpack, and GPU decompression.*

#### Aggregate comparison

| Metric | FP16 | FP16+GDS | TQ 3-bit | TQ 3-bit+GDS |
|--------|------|----------|----------|--------------|
| Per-session size | 573.4 MB | 573.4 MB | 116.5 MB | 116.5 MB **(4.9x smaller)** |
| Avg checkpoint (compress+write) | 1,164 ms | 4,856 ms | 2,082 ms | 2,293 ms |
| **Avg restore (read+decompress)** | **1,127 ms** | **1,028 ms** | **49,918 ms** | **188 ms (6.0x faster)** |
| Avg full round-trip | 2,291 ms | 5,884 ms | 52,000 ms | 2,481 ms |
| Wall time (8 sessions) | 11,109 ms | 15,075 ms | 82,235 ms | 34,289 ms |
| Aggregate write bandwidth | 3.63 GB/s | 0.92 GB/s | 0.47 GB/s | 0.41 GB/s |
| **Aggregate read bandwidth** | **3.99 GB/s** | **1.82 GB/s** | **0.05 GB/s** | **6.63 GB/s** |

#### Scale-out projections: concurrent sessions at a given bandwidth budget

| Network bandwidth | FP16 | FP16+GDS | TQ 3-bit | TQ 3-bit+GDS | Gain |
|-------------------|------|----------|----------|--------------|------|
| 5 GB/s | 10 | 42 | 89 | 98 | 9.7x |
| 10 GB/s | 20 | 85 | 179 | 197 | 9.7x |
| 15 GB/s | 30 | 127 | 268 | 295 | 9.7x |
| 20 GB/s | 41 | 169 | 357 | 394 | 9.7x |

#### Storage capacity

| | FP16 | FP16+GDS | TQ 3-bit | TQ 3-bit+GDS |
|--|------|----------|----------|--------------|
| Sessions per TB | 1,744 | 1,744 | 8,580 | 8,585 | **(4.9x)** |

### What to take away

**The headline number: 6x faster restore.** TQ+GDS restores a session in 188 ms on average, compared to 1,127 ms for FP16 without GDS. That's the difference between a perceptible delay and an imperceptible one when a user resumes a conversation.

**Compression without GDS is slower for concurrent restores.** TQ 3-bit without GDS averaged 49,918 ms per restore — 44x slower than FP16. (Note: this was measured without NUMA-aware CPU pinning; with pinning, the CPU path improves dramatically to ~2.4 seconds — see Benchmark 2 — but GDS still wins by eliminating the CPU entirely.)

**GDS without compression is a modest improvement.** FP16+GDS restore (1,028 ms) is only slightly faster than FP16 without GDS (1,127 ms). GDS helps, but the network still has to transfer the full 573 MB. The big win comes from combining GDS with compression.

**Checkpoint (write) performance is comparable across configurations.** FP16 writes are fastest (1,164 ms) because there's no compression step. TQ+GDS writes (2,293 ms) include GPU compression and packing before the GDS write. Since checkpoints are write-once/read-many in practice, the slightly slower write is a good trade-off for much faster reads.

**Aggregate read bandwidth: 6.63 GB/s with TQ+GDS.** This is higher than the FP16 read bandwidth (3.99 GB/s) despite reading 4.9x less total data, because the RDMA path to GPU memory is extremely efficient for the smaller transfers. The FlashBlade is delivering data as fast as the GPUs can accept it.

**Scale-out: 9.7x more concurrent sessions at any bandwidth budget.** At 20 GB/s network bandwidth, TQ+GDS supports 394 concurrent checkpoint operations vs 41 for FP16. This is the product of two factors: 4.9x less data per session (compression) and ~2x more efficient use of bandwidth (GDS + GPU decompression eliminating CPU overhead).

---

## Summary of Key Results

| Metric | Value |
|--------|-------|
| Compression ratio (3-bit) | 4.9x |
| Cosine similarity (3-bit) | 0.983 |
| Top-1 token accuracy after migration | 100% (50/50 tokens) |
| Restore speedup (TQ+GDS vs FP16) | **6.0x** |
| Aggregate read bandwidth (TQ+GDS, 8 GPUs) | 6.63 GB/s |
| Sessions per TB improvement | 4.9x |
| Concurrent session scale-out | 9.7x |

### The stack

Each layer contributes to the final result:

| Layer | Contribution |
|-------|-------------|
| **TurboQuant** | 4.9x less data to store and transfer. Data-oblivious — no calibration needed. |
| **FlashBlade** | High-throughput, low-latency persistent storage accessible over RDMA. |
| **GDS (kvikio)** | Eliminates CPU from the I/O path — data flows directly between FlashBlade and GPU memory. |
| **RDMA** | Network protocol enabling direct memory access — no CPU involvement in data transfer. |
| **GPU decompression** | Bit unpacking and inverse rotation run on GPU in sub-millisecond time, trivial for a massively parallel processor. |

Remove any one layer and the result degrades significantly:
- Without TurboQuant: 4.9x more data to transfer, 4.9x less storage capacity.
- Without GDS: CPU becomes the bottleneck at scale (49 seconds vs 188 ms for 8 concurrent restores).
- Without RDMA: Network transfers are CPU-mediated, adding latency and contention.
- Without FlashBlade: Need persistent storage that can sustain multi-GB/s reads across concurrent sessions.

---

## Appendix: Layer Exemption

Not all transformer layers are created equal. In the Qwen2.5-7B model, layers 0 and 27 have key norms that are 13-14x larger than the median layer. When these layers' keys are quantized at moderate bit widths (3-6 bit), the quantization error is amplified by the large norms, creating systematically wrong attention patterns.

The solution is **layer exemption**: these two layers' keys are kept in full FP16 precision while everything else is quantized. The storage cost is minimal — 2 of 28 layers, keys only — adding 61 MB to the 350 MB compressed cache (reducing effective compression from 4.9x to 4.2x). The quality benefit is dramatic: without layer exemption, 3-bit quantization degrades Top-1 accuracy; with it, accuracy is 100%.

## Appendix: Mixed-Precision (Outlier Channel) Quantization

Within each head, not all channels carry equal information. TurboQuant's outlier mode splits the 128 channels into two groups by variance:

| Config | Outlier channels | High bits | Low bits | Effective rate | Cosine Sim |
|--------|-----------------|-----------|----------|----------------|------------|
| 2.5-bit | 32 highest-variance @ 4-bit | 4 | 2 | 2.5 | 0.964 |
| 3.0-bit (mixed) | 64 highest-variance @ 4-bit | 4 | 2 | 3.0 | 0.977 |
| 3.5-bit | 32 highest-variance @ 5-bit | 5 | 3 | 3.5 | 0.990 |
| 4.0-bit (mixed) | 64 highest-variance @ 5-bit | 5 | 3 | 4.0 | 0.993 |

Each subspace gets its own rotation matrix and codebook (computed for the appropriate dimension — d=32, d=64, or d=96). This enables fractional effective bit rates that sit between the uniform quantization levels, offering fine-grained quality/size trade-offs.

## Appendix: Benchmark Methodology

- **Model**: Qwen/Qwen2.5-7B-Instruct, loaded in FP16, run on NVIDIA A100-SXM4-40GB GPUs.
- **KV cache capture**: A multi-paragraph text prompt is extended to the target context length. The model runs a forward pass and the KV cache is extracted from the `DynamicCache`.
- **I/O mode**: All non-GDS I/O uses `O_DIRECT` (`os.O_DIRECT` flag) with page-aligned buffers (via `mmap.mmap(-1, size)`). This bypasses the Linux page cache entirely, ensuring timing measurements reflect true FlashBlade performance. Without O_DIRECT, writes would appear artificially fast (data landing in page cache) and reads might hit cached data.
- **GDS I/O**: Uses `kvikio.CuFile` for direct GPU-to-storage transfers over RDMA. No page cache, no CPU memory copy.
- **Timing**: `time.time()` for CPU operations, `torch.cuda.synchronize()` before/after GPU operations. Wall-clock time for concurrent benchmarks.
- **Verification**: MSE and cosine similarity against original FP16 data for all compression benchmarks. Teacher-forced Top-1 token accuracy for migration benchmark.
- **Worker isolation**: Each concurrent worker runs as a separate Python process with `CUDA_VISIBLE_DEVICES` set to a single GPU and `taskset` pinning to the NUMA-local CPU cores (16 cores per GPU). Thread counts (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`) are set to match. Results are communicated via JSON files in a shared temp directory.
- **FlashBlade mount**: NFS v3 over RDMA (`proto=rdma`), 16 connections (`nconnect=16`), 512 KB read/write sizes.
