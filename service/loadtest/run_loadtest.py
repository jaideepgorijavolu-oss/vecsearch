"""Load the SIFT 100k subset into a running service, then run Locust at several concurrency
levels and print a Markdown table (client-side vs server-side latency).

Usage: python3 service/loadtest/run_loadtest.py --url http://vecsearch:8000 [--users 1,4,16,64]
Needs data/sift100k.{base,query}.fbin (bench/ann/make_subset.py) and locust.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).parent


def read_fbin(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    n, dim = raw[:2].view(np.uint32)
    return raw[2:].reshape(int(n), int(dim))


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def wait_healthy(url: str) -> dict:
    for _ in range(60):
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    raise SystemExit(f"service at {url} is not healthy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--data", default=os.environ.get("VECSEARCH_DATA", "data"))
    ap.add_argument("--users", default="1,4,16,64")
    ap.add_argument("--duration", default="30s")
    ap.add_argument("--ef", type=int, default=64)
    args = ap.parse_args()

    print(f"service: {wait_healthy(args.url)}")
    base = read_fbin(f"{args.data}/sift100k.base.fbin")
    try:
        post(args.url + "/collections", {"name": "sift", "dim": base.shape[1], "metric": "l2"})
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise
    t0 = time.perf_counter()
    for start in range(0, len(base), 5000):
        chunk = base[start:start + 5000]
        post(args.url + "/collections/sift/vectors", {"vectors": [
            {"id": start + i, "vector": v.tolist(), "tags": {"shard": (start + i) % 10}}
            for i, v in enumerate(chunk)]})
    print(f"loaded {len(base)} vectors over HTTP in {time.perf_counter() - t0:.1f} s")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for users in [int(u) for u in args.users.split(",")]:
            prefix = f"{tmp}/u{users}"
            env = dict(os.environ, SERVER_TIMES_OUT=f"{prefix}_server.json", EF=str(args.ef),
                       QUERIES_FBIN=f"{args.data}/sift100k.query.fbin")
            subprocess.run(["locust", "-f", str(HERE / "locustfile.py"), "--headless",
                            "-u", str(users), "-r", str(users), "-t", args.duration,
                            "--host", args.url, "--csv", prefix, "--only-summary"],
                           env=env, check=True, capture_output=True)
            with open(f"{prefix}_stats.csv") as f:
                agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
            server = json.loads(pathlib.Path(f"{prefix}_server.json").read_text())
            rows.append((users, agg, server))
            print(f"users {users}: {float(agg['Requests/s']):.0f} req/s, "
                  f"p50 {agg['50%']} ms, p99 {agg['99%']} ms")

    print(f"\nSIFT 100k subset, k = 10, ef = {args.ef}, {args.duration} per level\n")
    print("| concurrent users | req/s | failures | client p50 (ms) | client p99 (ms) "
          "| handler p50 / p99 (ms) | engine p50 / p99 (ms) | engine share of client p50 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for users, agg, s in rows:
        p50 = float(agg["50%"])
        print(f"| {users} | {float(agg['Requests/s']):,.0f} | {agg['Failure Count']} | {agg['50%']} | "
              f"{agg['99%']} | {s['handler_50']:.2f} / {s['handler_99']:.2f} | "
              f"{s['engine_50']:.2f} / {s['engine_99']:.2f} | {s['engine_50'] / p50:.0%} |")


if __name__ == "__main__":
    main()
