#!/usr/bin/env python3
"""
Plot system metrics collected by monitor.py.

Reads a JSONL metrics file and generates one PNG per metric category.

Usage:
    python plot_monitor.py metrics.jsonl
    python plot_monitor.py metrics.jsonl --output-dir ./plots
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_metrics(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("{"):
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def plot_cpu_per_numa(samples, output_dir):
    """CPU utilization per NUMA node: user + system + iowait stacked."""
    if not samples or "cpu_numa" not in samples[0]:
        return
    nodes = sorted(samples[0]["cpu_numa"].keys(), key=lambda x: int(x))
    n_nodes = len(nodes)
    if n_nodes == 0:
        return

    fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 2.5 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    ts = [s["t"] for s in samples]

    for i, nid in enumerate(nodes):
        ax = axes[i]
        user = [s["cpu_numa"].get(nid, {}).get("user", 0) for s in samples]
        sys_ = [s["cpu_numa"].get(nid, {}).get("sys", 0) for s in samples]
        iow = [s["cpu_numa"].get(nid, {}).get("iowait", 0) for s in samples]

        ax.fill_between(ts, 0, user, alpha=0.7, label="user", color="#2196F3")
        ax.fill_between(ts, user, [u + s for u, s in zip(user, sys_)],
                        alpha=0.7, label="system", color="#FF9800")
        ax.fill_between(ts, [u + s for u, s in zip(user, sys_)],
                        [u + s + w for u, s, w in zip(user, sys_, iow)],
                        alpha=0.7, label="iowait", color="#F44336")
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"NUMA {nid} %")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("CPU Utilization per NUMA Node", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cpu_per_numa.png"), dpi=150)
    plt.close(fig)
    print(f"  cpu_per_numa.png")


def plot_iowait(samples, output_dir):
    """System-wide I/O wait percentage."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ts = [s["t"] for s in samples]
    iow = [s.get("iowait_pct", 0) for s in samples]
    ax.fill_between(ts, 0, iow, alpha=0.7, color="#F44336")
    ax.set_ylabel("I/O Wait %")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, max(max(iow) * 1.2, 10))
    ax.set_title("System-wide I/O Wait")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iowait.png"), dpi=150)
    plt.close(fig)
    print(f"  iowait.png")


def plot_gpu(samples, output_dir):
    """GPU utilization and memory per GPU."""
    if not samples or "gpu" not in samples[0] or not samples[0]["gpu"]:
        return
    gpu_ids = sorted(samples[0]["gpu"].keys(), key=lambda x: int(x))
    ts = [s["t"] for s in samples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for gid in gpu_ids:
        util = [s.get("gpu", {}).get(gid, {}).get("util_pct", 0) for s in samples]
        mem = [s.get("gpu", {}).get(gid, {}).get("mem_used_mb", 0) for s in samples]
        ax1.plot(ts, util, label=f"GPU {gid}", alpha=0.8, linewidth=0.8)
        ax2.plot(ts, mem, label=f"GPU {gid}", alpha=0.8, linewidth=0.8)

    ax1.set_ylabel("Utilization %")
    ax1.set_ylim(0, 105)
    ax1.set_title("GPU Utilization")
    ax1.legend(loc="upper right", fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Memory Used (MB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("GPU Memory Used")
    ax2.legend(loc="upper right", fontsize=7, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gpu.png"), dpi=150)
    plt.close(fig)
    print(f"  gpu.png")


def plot_network(samples, output_dir):
    """Network throughput (rx/tx) per interface."""
    if not samples or "net" not in samples[0] or not samples[0]["net"]:
        return
    ifaces = sorted(samples[0]["net"].keys())
    ts = [s["t"] for s in samples]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for iface in ifaces:
        rx = [s.get("net", {}).get(iface, {}).get("rx_mbps", 0) for s in samples]
        tx = [s.get("net", {}).get(iface, {}).get("tx_mbps", 0) for s in samples]
        ax1.plot(ts, rx, label=f"{iface} RX", alpha=0.8, linewidth=0.8)
        ax2.plot(ts, tx, label=f"{iface} TX", alpha=0.8, linewidth=0.8)

    ax1.set_ylabel("RX (MB/s)")
    ax1.set_title("Network Receive Throughput")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("TX (MB/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Network Transmit Throughput")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "network.png"), dpi=150)
    plt.close(fig)
    print(f"  network.png")


def plot_nfs(samples, output_dir):
    """NFS read/write byte rates and op rates."""
    ts = [s["t"] for s in samples]

    # Byte rates
    has_bytes = any("nfs" in s and "bytes_read_rate" in s.get("nfs", {}) for s in samples)
    has_ops = any("nfs" in s and "nfs_read_ops_rate" in s.get("nfs", {}) for s in samples)

    if not has_bytes and not has_ops:
        return

    n_plots = (1 if has_bytes else 0) + (1 if has_ops else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    if has_bytes:
        ax = axes[ax_idx]; ax_idx += 1
        rd = [s.get("nfs", {}).get("bytes_read_rate", 0) / 1e6 for s in samples]
        wr = [s.get("nfs", {}).get("bytes_written_rate", 0) / 1e6 for s in samples]
        srv_rd = [s.get("nfs", {}).get("server_read_rate", 0) / 1e6 for s in samples]
        srv_wr = [s.get("nfs", {}).get("server_write_rate", 0) / 1e6 for s in samples]
        ax.plot(ts, rd, label="Read (app)", alpha=0.8)
        ax.plot(ts, wr, label="Write (app)", alpha=0.8)
        ax.plot(ts, srv_rd, label="Read (server)", alpha=0.8, linestyle="--")
        ax.plot(ts, srv_wr, label="Write (server)", alpha=0.8, linestyle="--")
        ax.set_ylabel("MB/s")
        ax.set_title("NFS Byte Throughput")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    if has_ops:
        ax = axes[ax_idx]; ax_idx += 1
        rd_ops = [s.get("nfs", {}).get("nfs_read_ops_rate", 0) for s in samples]
        wr_ops = [s.get("nfs", {}).get("nfs_write_ops_rate", 0) for s in samples]
        rd_rtt = [s.get("nfs", {}).get("nfs_read_rtt_ms_rate", 0) for s in samples]
        wr_rtt = [s.get("nfs", {}).get("nfs_write_rtt_ms_rate", 0) for s in samples]
        ax.plot(ts, rd_ops, label="Read ops/s", alpha=0.8)
        ax.plot(ts, wr_ops, label="Write ops/s", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(ts, rd_rtt, label="Read RTT (ms/s)", alpha=0.5, linestyle="--", color="red")
        ax2.plot(ts, wr_rtt, label="Write RTT (ms/s)", alpha=0.5, linestyle="--", color="orange")
        ax.set_ylabel("Ops/s")
        ax2.set_ylabel("Cumulative RTT (ms/s)")
        ax.set_title("NFS Operations")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nfs.png"), dpi=150)
    plt.close(fig)
    print(f"  nfs.png")


def plot_mem_bandwidth(samples, output_dir):
    """Memory bandwidth: pgpgin/pgpgout rates."""
    ts = [s["t"] for s in samples]

    fig, ax = plt.subplots(figsize=(16, 4))
    pgin = [s.get("mem_bw", {}).get("pgpgin_mbps", 0) for s in samples]
    pgout = [s.get("mem_bw", {}).get("pgpgout_mbps", 0) for s in samples]
    ax.plot(ts, pgin, label="Page In (MB/s)", alpha=0.8, color="#2196F3")
    ax.plot(ts, pgout, label="Page Out (MB/s)", alpha=0.8, color="#FF9800")
    ax.set_ylabel("MB/s")
    ax.set_xlabel("Time (s)")
    ax.set_title("Memory Bandwidth (Page I/O)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mem_bandwidth.png"), dpi=150)
    plt.close(fig)
    print(f"  mem_bandwidth.png")


def plot_numa_bandwidth(samples, output_dir):
    """NUMA memory allocation rates per node — proxy for memory bandwidth."""
    if not samples or "numa_membw" not in samples[0] or not samples[0]["numa_membw"]:
        return
    nodes = sorted(samples[0]["numa_membw"].keys(), key=lambda x: int(x))
    ts = [s["t"] for s in samples]

    n_nodes = len(nodes)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 2.5 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for i, nid in enumerate(nodes):
        ax = axes[i]
        local = [s.get("numa_membw", {}).get(nid, {}).get("local_kpages_s", 0) for s in samples]
        other = [s.get("numa_membw", {}).get(nid, {}).get("other_kpages_s", 0) for s in samples]
        hit = [s.get("numa_membw", {}).get(nid, {}).get("hit_kpages_s", 0) for s in samples]
        miss = [s.get("numa_membw", {}).get(nid, {}).get("miss_kpages_s", 0) for s in samples]

        ax.plot(ts, local, label="local", alpha=0.8, color="#4CAF50")
        ax.plot(ts, other, label="other (cross-node)", alpha=0.8, color="#F44336")
        ax.plot(ts, hit, label="hit", alpha=0.5, linestyle="--", color="#2196F3")
        ax.plot(ts, miss, label="miss", alpha=0.5, linestyle="--", color="#FF9800")
        ax.set_ylabel(f"NUMA {nid}\nkpages/s")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("NUMA Memory Allocation Rates (proxy for memory bandwidth)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "numa_bandwidth.png"), dpi=150)
    plt.close(fig)
    print(f"  numa_bandwidth.png")


def plot_meminfo(samples, output_dir):
    """Dirty pages and writeback — memory pressure indicators."""
    ts = [s["t"] for s in samples]
    dirty = [s.get("meminfo", {}).get("dirty_mb", 0) for s in samples]
    wb = [s.get("meminfo", {}).get("writeback_mb", 0) for s in samples]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.fill_between(ts, 0, dirty, alpha=0.7, label="Dirty (MB)", color="#FF9800")
    ax.fill_between(ts, dirty, [d + w for d, w in zip(dirty, wb)],
                    alpha=0.7, label="Writeback (MB)", color="#F44336")
    ax.set_ylabel("MB")
    ax.set_xlabel("Time (s)")
    ax.set_title("Memory: Dirty Pages and Writeback")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "meminfo.png"), dpi=150)
    plt.close(fig)
    print(f"  meminfo.png")


def plot_workers(samples, output_dir):
    """Number of active worker processes over time."""
    ts = [s["t"] for s in samples]
    nw = [s.get("n_workers", 0) for s in samples]

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.fill_between(ts, 0, nw, alpha=0.7, color="#9C27B0", step="post")
    ax.set_ylabel("Workers")
    ax.set_xlabel("Time (s)")
    ax.set_title("Active Benchmark Workers")
    ax.set_ylim(0, max(max(nw) + 1, 1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "workers.png"), dpi=150)
    plt.close(fig)
    print(f"  workers.png")


def main():
    parser = argparse.ArgumentParser(description="Plot monitor metrics")
    parser.add_argument("input", help="JSONL metrics file from monitor.py")
    parser.add_argument("--output-dir", "-o", default="./output/monitor_plots",
                        help="Directory for output PNGs")
    args = parser.parse_args()

    samples = load_metrics(args.input)
    if not samples:
        print("No samples found!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    duration = samples[-1]["t"] - samples[0]["t"]
    print(f"Loaded {len(samples)} samples over {duration:.1f}s")
    print(f"Generating plots in {args.output_dir}/:")

    plot_cpu_per_numa(samples, args.output_dir)
    plot_iowait(samples, args.output_dir)
    plot_gpu(samples, args.output_dir)
    plot_network(samples, args.output_dir)
    plot_nfs(samples, args.output_dir)
    plot_mem_bandwidth(samples, args.output_dir)
    plot_numa_bandwidth(samples, args.output_dir)
    plot_meminfo(samples, args.output_dir)
    plot_workers(samples, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
