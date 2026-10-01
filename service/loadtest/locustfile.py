"""Locust load test: k=10 searches with random SIFT query vectors against the 'sift' collection.

Each simulated user sends requests back to back (closed loop, no think time), so the number of
users is the number of concurrent in-flight requests. Besides Locust's own client-side latency,
we record the server's X-Engine-Time-Ms and X-Handler-Time-Ms headers and dump their
percentiles to $SERVER_TIMES_OUT when the test stops.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
from locust import FastHttpUser, constant, events, task

QUERIES = np.fromfile(os.environ.get("QUERIES_FBIN", "/data/sift100k.query.fbin"), dtype=np.float32)
DIM = int(QUERIES[:2].view(np.uint32)[1])
QUERIES = QUERIES[2:].reshape(-1, DIM)
BODIES = [json.dumps({"vector": q.tolist(), "k": 10, "ef": int(os.environ.get("EF", "64"))})
          for q in QUERIES[:2000]]
ENGINE_MS: list[float] = []
HANDLER_MS: list[float] = []


class Searcher(FastHttpUser):
    wait_time = constant(0)

    @task
    def search(self):
        with self.client.post("/collections/sift/search", data=random.choice(BODIES),
                              headers={"Content-Type": "application/json"},
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")
                return
            ENGINE_MS.append(float(r.headers["X-Engine-Time-Ms"]))
            HANDLER_MS.append(float(r.headers["X-Handler-Time-Ms"]))


@events.test_stop.add_listener
def dump(**_):
    out = os.environ.get("SERVER_TIMES_OUT")
    if not out or not ENGINE_MS:
        return
    e, h = np.array(ENGINE_MS), np.array(HANDLER_MS)
    stats = {f"{name}_{p}": float(np.percentile(arr, p)) for name, arr in (("engine", e), ("handler", h))
             for p in (50, 99)}
    stats.update(engine_mean=float(e.mean()), handler_mean=float(h.mean()), samples=len(e))
    with open(out, "w") as f:
        json.dump(stats, f)
