"""Plots and Markdown tables from bench/ann/results/*.json.

Writes bench/ann/results/<dataset>_recall_qps.png (all threads and 1 thread) and prints the
tables pasted into docs/RESULTS.md.
Usage: python3 bench/ann/plot.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
ORDER = ["vecsearch", "vecsearch-noprefetch", "hnswlib", "faiss"]
STYLE = {
    "vecsearch": dict(color="#d6336c", marker="o"),
    "vecsearch-noprefetch": dict(color="#d6336c", marker="o", linestyle="--", alpha=0.6),
    "hnswlib": dict(color="#1c7ed6", marker="s"),
    "faiss": dict(color="#2b8a3e", marker="^"),
}
TITLES = {"sift": "SIFT1M (128-d, L2)", "glove": "GloVe-100 (100-d, angular)"}


def load():
    runs = {}
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["dataset"], {})[r["engine"]] = r
    return runs


def qps_at(run, target, key):
    """QPS at the first ef reaching `target` recall (None if never reached)."""
    for s in run["sweep"]:
        if s["recall"] >= target:
            return s[key], s["ef"]
    return None, None


def main() -> None:
    runs = load()
    for ds, engines in runs.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for ax, key, label in zip(axes, ["qps_all_threads", "qps_1thread"],
                                  ["all threads", "1 thread"]):
            for name in ORDER:
                if name not in engines:
                    continue
                r = engines[name]
                xs = [s["recall"] for s in r["sweep"]]
                ys = [s[key] for s in r["sweep"]]
                ax.plot(xs, ys, label=name, **STYLE[name])
            threads = next(iter(engines.values()))["threads"]
            ax.set_title(f"{TITLES.get(ds, ds)}: {label}" + (f" ({threads})" if key ==
                                                                "qps_all_threads" else ""))
            ax.set_xlabel("recall@10")
            ax.set_ylabel("queries per second (log)")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(left=max(0.5, min(s["recall"] for e in engines.values()
                                          for s in e["sweep"]) - 0.02), right=1.0)
            ax.legend()
        out = RESULTS / f"{ds}_recall_qps.png"
        fig.savefig(out, dpi=130)
        print(f"wrote {out}\n")

        print(f"### {TITLES.get(ds, ds)}\n")
        print("| engine | build (s) | memory: RSS growth (MiB) | QPS @0.95, all threads (ef) "
              "| QPS @0.95, 1 thread | QPS @0.99, 1 thread | p50 / p99 latency @0.95 (ms) |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name in ORDER:
            if name not in engines:
                continue
            r = engines[name]
            qa, efa = qps_at(r, 0.95, "qps_all_threads")
            q1, _ = qps_at(r, 0.95, "qps_1thread")
            q99, _ = qps_at(r, 0.99, "qps_1thread")
            lat = r["latency"]
            fmt = lambda v: "n/a" if v is None else f"{v:,.0f}"  # noqa: E731
            print(f"| {name} | {r['build_seconds']:.1f} | {r['rss_growth_bytes'] / 2**20:,.0f} | "
                  f"{fmt(qa)} ({efa}) | {fmt(q1)} | {fmt(q99)} | "
                  + (f"{lat['p50_ms']:.3f} / {lat['p99_ms']:.3f}" if lat else "n/a") + " |")
        print()
        print("Full sweep (recall@10 / QPS all threads / QPS 1 thread):\n")
        names = [n for n in ORDER if n in engines]
        print("| ef | " + " | ".join(names) + " |")
        print("|---:|" + "---|" * len(names))
        for i, ef in enumerate([s["ef"] for s in engines[names[0]]["sweep"]]):
            cells = []
            for n in names:
                s = engines[n]["sweep"][i]
                cells.append(f"{s['recall']:.4f} / {s['qps_all_threads']:,.0f} / "
                             f"{s['qps_1thread']:,.0f}")
            print(f"| {ef} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
