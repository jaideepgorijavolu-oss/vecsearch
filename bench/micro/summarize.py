"""Turn Google Benchmark JSON output into the Markdown tables used in docs/RESULTS.md.

Usage: python3 bench/micro/summarize.py bench/micro/results/distance.json
"""

import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    ctx = data["context"]
    print(f"Run on {ctx.get('num_cpus')} CPUs @ {ctx.get('mhz_per_cpu')} MHz, "
          f"build type {ctx.get('library_build_type')}, {ctx.get('date')}\n")

    # name looks like "hot/l2/avx2/128_median" when run with repetitions
    rows = defaultdict(dict)  # (mode, metric, dim) -> kernel -> (ns, GB/s)
    kernels = []
    for b in data["benchmarks"]:
        if b.get("run_type") == "aggregate" and b.get("aggregate_name") != "median":
            continue
        name = b.get("run_name", b["name"])
        mode, metric, kernel, dim = name.split("/")
        if kernel not in kernels:
            kernels.append(kernel)
        ns = b["real_time"] if b["time_unit"] == "ns" else b["real_time"] * 1e3
        rows[(mode, metric, int(dim))][kernel] = (ns, b["bytes_per_second"] / 1e9)

    for mode in ("hot", "scan"):
        for metric in ("l2", "dot"):
            print(f"### {mode} / {metric}: ns per call (GB/s)\n")
            print("| dim | " + " | ".join(kernels) + " |")
            print("|---:|" + "---:|" * len(kernels))
            for dim in sorted({k[2] for k in rows if k[0] == mode and k[1] == metric}):
                cells = []
                for kern in kernels:
                    ns, gbs = rows[(mode, metric, dim)].get(kern, (float("nan"), float("nan")))
                    cells.append(f"{ns:.1f} ({gbs:.1f})")
                print(f"| {dim} | " + " | ".join(cells) + " |")
            print()


if __name__ == "__main__":
    main(sys.argv[1])
