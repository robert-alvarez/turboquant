#!/usr/bin/env python3
"""
System monitor for FlashBlade benchmarks.

Samples CPU, iowait, GPU, network, NFS, and memory bandwidth metrics
at regular intervals and writes JSON-lines to an output file.

Usage:
    python monitor.py --output metrics.jsonl --interval 0.25
    # In another terminal: python bench_flashblade.py ...
    # Then: python plot_monitor.py metrics.jsonl
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time


# ── NUMA topology ──────────────────────────────────────────────────────────

def parse_numa_topology():
    """Return dict: node_id -> list of CPU core IDs."""
    nodes = {}
    node_dir = "/sys/devices/system/node"
    if not os.path.isdir(node_dir):
        return nodes
    for entry in os.listdir(node_dir):
        m = re.match(r"node(\d+)", entry)
        if not m:
            continue
        nid = int(m.group(1))
        cpulist_path = os.path.join(node_dir, entry, "cpulist")
        if os.path.exists(cpulist_path):
            with open(cpulist_path) as f:
                nodes[nid] = _parse_range(f.read().strip())
    return nodes


def _parse_range(s):
    cores = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.extend(range(int(lo), int(hi) + 1))
        elif part:
            cores.append(int(part))
    return sorted(cores)


# ── CPU stats ──────────────────────────────────────────────────────────────

def read_cpu_stats():
    """Read per-CPU jiffies from /proc/stat. Returns dict: cpu_id -> (user, nice, system, idle, iowait, irq, softirq, steal)."""
    stats = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3] != " ":
                parts = line.split()
                cpu_id = int(parts[0][3:])
                vals = tuple(int(x) for x in parts[1:9])
                stats[cpu_id] = vals
            elif line.startswith("cpu "):
                parts = line.split()
                vals = tuple(int(x) for x in parts[1:9])
                stats["total"] = vals
    return stats


def compute_cpu_delta(prev, curr, cores):
    """Compute CPU utilization for a set of cores. Returns (user%, system%, iowait%, idle%)."""
    total_user = total_sys = total_iowait = total_idle = total_all = 0
    for c in cores:
        if c not in prev or c not in curr:
            continue
        p, q = prev[c], curr[c]
        d = tuple(q[i] - p[i] for i in range(len(p)))
        s = sum(d) or 1
        total_user += d[0] + d[1]  # user + nice
        total_sys += d[2]          # system
        total_idle += d[3]         # idle
        total_iowait += d[4]       # iowait
        total_all += s
    if total_all == 0:
        return 0, 0, 0, 100
    return (
        100 * total_user / total_all,
        100 * total_sys / total_all,
        100 * total_iowait / total_all,
        100 * total_idle / total_all,
    )


# ── GPU stats ──────────────────────────────────────────────────────────────

def read_gpu_stats():
    """Query nvidia-smi for per-GPU utilization and memory."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = {}
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            gid = int(parts[0])
            gpus[gid] = {
                "util_pct": int(parts[1]),
                "mem_used_mb": int(parts[2]),
                "mem_total_mb": int(parts[3]),
            }
        return gpus
    except Exception:
        return {}


# ── Network stats ──────────────────────────────────────────────────────────

def read_net_stats(interfaces=None):
    """Read bytes rx/tx from /proc/net/dev."""
    stats = {}
    with open("/proc/net/dev") as f:
        for line in f:
            if ":" not in line:
                continue
            iface, data = line.split(":", 1)
            iface = iface.strip()
            if interfaces and iface not in interfaces:
                continue
            parts = data.split()
            stats[iface] = {
                "rx_bytes": int(parts[0]),
                "tx_bytes": int(parts[8]),
            }
    return stats


# ── NFS mountstats ─────────────────────────────────────────────────────────

def read_nfs_stats(mount_point="/mnt/data"):
    """Read NFS per-op stats from /proc/self/mountstats."""
    stats = {}
    in_mount = False
    in_ops = False
    with open("/proc/self/mountstats") as f:
        for line in f:
            if f"mounted on {mount_point}" in line and "nfs" in line:
                in_mount = True
                continue
            if in_mount and line.startswith("device "):
                break  # next mount
            if in_mount and "bytes:" in line:
                parts = line.split()
                # bytes: normal_read normal_write direct_read direct_write server_read server_write pages_read pages_written
                if len(parts) >= 9:
                    stats["bytes_read"] = int(parts[1])
                    stats["bytes_written"] = int(parts[2])
                    stats["direct_read"] = int(parts[3])
                    stats["direct_write"] = int(parts[4])
                    stats["server_read"] = int(parts[5])
                    stats["server_write"] = int(parts[6])
            if in_mount and "per-op statistics" in line:
                in_ops = True
                continue
            if in_mount and in_ops:
                parts = line.split()
                if len(parts) >= 9 and parts[0].rstrip(":") in ("READ", "WRITE"):
                    op = parts[0].rstrip(":")
                    stats[f"nfs_{op.lower()}_ops"] = int(parts[1])
                    stats[f"nfs_{op.lower()}_bytes"] = int(parts[4] if op == "READ" else parts[4])
                    stats[f"nfs_{op.lower()}_rtt_ms"] = int(parts[6])
                    stats[f"nfs_{op.lower()}_exe_ms"] = int(parts[7])
    return stats


# ── Memory / NUMA stats ───────────────────────────────────────────────────

def read_vmstat():
    """Read key counters from /proc/vmstat."""
    keys = {"pgpgin", "pgpgout", "numa_hit", "numa_miss", "numa_local", "numa_other"}
    stats = {}
    with open("/proc/vmstat") as f:
        for line in f:
            parts = line.split()
            if parts[0] in keys:
                stats[parts[0]] = int(parts[1])
    return stats


def read_numastat():
    """Read per-node numa_hit/miss/local/other from /sys."""
    nodes = {}
    node_dir = "/sys/devices/system/node"
    if not os.path.isdir(node_dir):
        return nodes
    for entry in sorted(os.listdir(node_dir)):
        m = re.match(r"node(\d+)", entry)
        if not m:
            continue
        nid = int(m.group(1))
        path = os.path.join(node_dir, entry, "numastat")
        if not os.path.exists(path):
            continue
        info = {}
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    info[parts[0]] = int(parts[1])
        nodes[nid] = info
    return nodes


def read_meminfo():
    """Read MemFree, Dirty, Writeback from /proc/meminfo (MB)."""
    keys = {"MemFree", "Dirty", "Writeback", "Buffers", "Cached"}
    stats = {}
    with open("/proc/meminfo") as f:
        for line in f:
            for k in keys:
                if line.startswith(k + ":"):
                    stats[k.lower() + "_mb"] = int(line.split()[1]) / 1024
    return stats


# ── Process stats ──────────────────────────────────────────────────────────

def find_worker_pids():
    """Find flashblade_worker.py PIDs and their GPU assignments."""
    try:
        r = subprocess.run(
            ["pgrep", "-af", "flashblade_worker"],
            capture_output=True, text=True, timeout=5,
        )
        workers = []
        my_pid = os.getpid()
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            pid = int(parts[0])
            if pid == my_pid:
                continue
            cmd = parts[1] if len(parts) > 1 else ""
            # Skip pgrep itself
            if "pgrep" in cmd:
                continue
            wid_m = re.search(r"--worker-id\s+(\d+)", cmd)
            wid = int(wid_m.group(1)) if wid_m else -1
            workers.append({"pid": pid, "worker_id": wid})
        return workers
    except Exception:
        return []


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="System monitor for FlashBlade benchmarks")
    parser.add_argument("--output", "-o", default="metrics.jsonl", help="Output JSONL file")
    parser.add_argument("--interval", type=float, default=0.25, help="Sample interval in seconds")
    parser.add_argument("--net-iface", default=None,
                        help="Network interface to monitor (auto-detect if not set)")
    args = parser.parse_args()

    # Auto-detect RDMA-capable interface (MTU 9000 = jumbo frames)
    net_ifaces = None
    if args.net_iface:
        net_ifaces = [args.net_iface]
    else:
        # Look for interfaces with large MTU (RDMA/jumbo)
        try:
            with open("/proc/net/dev") as f:
                for line in f:
                    if ":" in line:
                        iface = line.split(":")[0].strip()
                        if iface.startswith("enp"):
                            if net_ifaces is None:
                                net_ifaces = []
                            net_ifaces.append(iface)
        except Exception:
            pass

    # Parse NUMA topology
    numa_topo = parse_numa_topology()
    all_cores = []
    for cores in numa_topo.values():
        all_cores.extend(cores)

    print(f"Monitor: writing to {args.output}, interval={args.interval}s", file=sys.stderr)
    print(f"  NUMA nodes: {len(numa_topo)}, total cores: {len(all_cores)}", file=sys.stderr)
    if net_ifaces:
        print(f"  Network interfaces: {net_ifaces}", file=sys.stderr)

    # Initialize previous readings for delta computation
    prev_cpu = read_cpu_stats()
    prev_net = read_net_stats(net_ifaces)
    prev_vmstat = read_vmstat()
    prev_nfs = read_nfs_stats()
    prev_numa = read_numastat()
    t_prev = time.time()

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with open(args.output, "w") as out:
        sample_idx = 0
        t_start = time.time()

        while running:
            time.sleep(args.interval)
            t_now = time.time()
            dt = t_now - t_prev

            # ── CPU ──
            curr_cpu = read_cpu_stats()
            cpu_per_numa = {}
            for nid, cores in sorted(numa_topo.items()):
                u, s, io, idle = compute_cpu_delta(prev_cpu, curr_cpu, cores)
                cpu_per_numa[nid] = {"user": round(u, 1), "sys": round(s, 1),
                                     "iowait": round(io, 1), "idle": round(idle, 1)}

            # System-wide iowait
            if "total" in prev_cpu and "total" in curr_cpu:
                p, q = prev_cpu["total"], curr_cpu["total"]
                d = tuple(q[i] - p[i] for i in range(len(p)))
                total = sum(d) or 1
                sys_iowait = round(100 * d[4] / total, 1)
            else:
                sys_iowait = 0

            # ── GPU ──
            gpu = read_gpu_stats()

            # ── Network ──
            curr_net = read_net_stats(net_ifaces)
            net_delta = {}
            for iface in curr_net:
                if iface in prev_net:
                    rx = (curr_net[iface]["rx_bytes"] - prev_net[iface]["rx_bytes"]) / dt
                    tx = (curr_net[iface]["tx_bytes"] - prev_net[iface]["tx_bytes"]) / dt
                    net_delta[iface] = {
                        "rx_mbps": round(rx / 1e6, 1),
                        "tx_mbps": round(tx / 1e6, 1),
                    }

            # ── NFS ──
            curr_nfs = read_nfs_stats()
            nfs_delta = {}
            for k in curr_nfs:
                if k in prev_nfs:
                    nfs_delta[k + "_rate"] = round((curr_nfs[k] - prev_nfs[k]) / dt, 1)

            # ── Memory bandwidth (vmstat page I/O) ──
            curr_vmstat = read_vmstat()
            mem_bw = {}
            for k in ("pgpgin", "pgpgout"):
                if k in curr_vmstat and k in prev_vmstat:
                    # pgpgin/pgpgout are in KB
                    mem_bw[k + "_mbps"] = round(
                        (curr_vmstat[k] - prev_vmstat[k]) / dt / 1024, 1
                    )

            # ── NUMA memory bandwidth (page allocations as proxy) ──
            curr_numa = read_numastat()
            numa_membw = {}
            for nid in curr_numa:
                if nid in prev_numa:
                    hit_rate = (curr_numa[nid].get("numa_hit", 0) - prev_numa[nid].get("numa_hit", 0)) / dt
                    miss_rate = (curr_numa[nid].get("numa_miss", 0) - prev_numa[nid].get("numa_miss", 0)) / dt
                    local_rate = (curr_numa[nid].get("local_node", 0) - prev_numa[nid].get("local_node", 0)) / dt
                    other_rate = (curr_numa[nid].get("other_node", 0) - prev_numa[nid].get("other_node", 0)) / dt
                    numa_membw[nid] = {
                        "hit_kpages_s": round(hit_rate / 1000, 1),
                        "miss_kpages_s": round(miss_rate / 1000, 1),
                        "local_kpages_s": round(local_rate / 1000, 1),
                        "other_kpages_s": round(other_rate / 1000, 1),
                    }

            # ── Meminfo ──
            meminfo = read_meminfo()

            # ── Workers ──
            workers = find_worker_pids()

            # ── Assemble sample ──
            sample = {
                "t": round(t_now - t_start, 3),
                "dt": round(dt, 4),
                "cpu_numa": cpu_per_numa,
                "iowait_pct": sys_iowait,
                "gpu": gpu,
                "net": net_delta,
                "nfs": nfs_delta,
                "mem_bw": mem_bw,
                "numa_membw": numa_membw,
                "meminfo": meminfo,
                "n_workers": len(workers),
            }

            out.write(json.dumps(sample) + "\n")
            out.flush()

            prev_cpu = curr_cpu
            prev_net = curr_net
            prev_vmstat = curr_vmstat
            prev_nfs = curr_nfs
            prev_numa = curr_numa
            t_prev = t_now
            sample_idx += 1

    print(f"Monitor: stopped after {sample_idx} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
