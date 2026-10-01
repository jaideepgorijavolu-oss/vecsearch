"""perf stat counters per query for the single-threaded search loop (bench/ann/profile_search).

For each configuration, runs profile_search twice under `perf stat`: once with a 0-second
search loop (load + warm-up only) and once with a long loop; the difference, divided by the
number of queries searched, is the cost of one query.

Needs Linux perf with hardware counters (in Docker: --privileged).
Usage: python3 bench/ann/profile.py [--ef 64] [--seconds 20] > bench/ann/results/profile.md
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess

EVENTS = ["cycles", "instructions", "L1-dcache-loads", "L1-dcache-load-misses",
          "cache-references", "cache-misses", "branch-misses"]


def perf_binary() -> str:
    found = sorted(glob.glob("/usr/lib/linux-tools/*/perf"))
    return found[0] if found else "perf"


def run(tool: list[str]) -> tuple[dict[str, float], int]:
    cmd = [perf_binary(), "stat", "-x", ",", "-e", ",".join(EVENTS), "--"] + tool
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    counters = {}
    for line in p.stderr.splitlines():
        parts = line.split(",")
        if len(parts) > 2 and parts[2] in EVENTS and parts[0] not in ("<not supported>", ""):
            counters[parts[2]] = float(parts[0])
    m = re.search(r": (\d+) queries", p.stdout)
    return counters, int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--build", default="build/bench")
    ap.add_argument("--data", default=os.environ.get("VECSEARCH_DATA", "data"))
    args = ap.parse_args()

    tool = os.path.join(args.build, "bench/ann/profile_search")
    files = [f"{args.data}/sift1m.base.fbin", f"{args.data}/sift1m.query.fbin",
             f"{args.data}/sift1m.hnsw"]
    print(f"SIFT1M, M=16, ef_construction=200, ef_search={args.ef}, 1 thread, "
          f"{args.seconds:.0f} s loop\n")
    print("| prefetch | QPS | cycles/query | instructions/query | IPC | L1d misses/query "
          "| cache misses/query (LLC) | branch misses/query |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for prefetch in (0, 1):
        base, _ = run([tool, *files, str(args.ef), str(prefetch), "0"])
        full, nq = run([tool, *files, str(args.ef), str(prefetch), str(args.seconds)])
        d = {e: (full.get(e, 0) - base.get(e, 0)) / nq for e in EVENTS}
        qps = subprocess.run([tool, *files, str(args.ef), str(prefetch), "5"], capture_output=True,
                             text=True, check=True).stdout
        qps = float(re.search(r"= (\d+) QPS", qps).group(1))
        print(f"| {'on' if prefetch else 'off'} | {qps:.0f} | {d['cycles']:.0f} | "
              f"{d['instructions']:.0f} | {d['instructions'] / d['cycles']:.2f} | "
              f"{d['L1-dcache-load-misses']:.0f} | {d['cache-misses']:.0f} | "
              f"{d['branch-misses']:.0f} |")


if __name__ == "__main__":
    main()
