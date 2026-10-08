"""Adaptive-search study (docs/adaptive_search/PROTOCOL.md). Run in the dev container.

  DATASET=sift|glove python3 bench/ann/adaptive/study.py <stage> <run_dir>

The run id is the name of <run_dir> (e.g. bench/ann/results/adaptive_search/sift1m_v1). Stages:
  traces   unbudgeted search traces for learn / val / test at every ef_max and checkpoint
  select   train models on learn; pick every method's setting on val (simulation + real runs)
  test     single-thread timing runs on the final test queries (needs a quiet machine)
  bundle   seal the raw test outputs into an audit bundle ($VECSEARCH_DATA/adaptive/bundles)
  report   tables, plots and summary.json, regenerated only from the audit bundle
  stress   (SIFT) the selected policies on a rebuilt graph, no retraining
  mt       16-thread throughput of the matched configs
  adopt    one-off: bring a run made before run isolation existed under these rules

Isolation rules (added after review; see PROTOCOL.md A4):
  * Raw outputs of a run live in $VECSEARCH_DATA/adaptive/runs_by_id/<run id>/, never shared.
  * inputs.json in the run dir pins the sha256 of the index, query, base and ground-truth files.
    Every stage checks them; a rebuilt index or changed data stops the run with an error.
  * Traces are cached by a key over (index, query file, search-code fingerprint, ef, checkpoint)
    and carry a manifest with their own sha256; damaged or mismatched cache files are rejected.
    traces.json in the run dir lists exactly which traces the run used.
  * select, test and bundle refuse to overwrite their own completed outputs. Use a new run id.
  * The measurement environment is recorded by the test stage; reports keep it and record their
    own environment separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

K = 10
DS = os.environ.get("DATASET", "sift")
# Grids per dataset (PROTOCOL.md A2 for SIFT, A3 for GloVe). GloVe needs ~10x more work per query.
GRIDS = {
    "sift": dict(
        metric="l2", ef_max=[128, 256, 512], checkpoints=[100, 200, 400], targets=[0.90, 0.95, 0.99],
        fixed_ef=[10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256,
                  320, 384, 512],
        patience_ef=[64, 128, 256, 512],
        patience_n=[2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 65, 80], passes=5),
    "glove": dict(
        metric="cosine", ef_max=[512, 1024, 2048], checkpoints=[250, 500, 1000],
        targets=[0.80, 0.90, 0.95],
        fixed_ef=[32, 48, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048],
        patience_ef=[256, 512, 1024, 2048],
        patience_n=[5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320], passes=3),
}[DS]
METRIC = GRIDS["metric"]
EF_MAX, CHECKPOINTS, TARGETS = GRIDS["ef_max"], GRIDS["checkpoints"], GRIDS["targets"]
FIXED_EF, PATIENCE_EF, PATIENCE_N = GRIDS["fixed_ef"], GRIDS["patience_ef"], GRIDS["patience_n"]
PASSES = GRIDS["passes"]
F = 9  # kNumTerminationFeatures
BUNDLE_FORMAT = 1

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("VECSEARCH_DATA", "/data")) / "adaptive"
TOOL = os.environ.get("VECSEARCH_ADAPTIVE_EVAL", str(ROOT / "build/bench/bench/ann/adaptive_eval"))
INDEX = DATA / f"{DS}_m16_efc200_s100.hnsw"
# Sources whose behavior determines a trace. Any edit (even formatting) changes the fingerprint
# and therefore the cache key: conservative, and traces are cheap to regenerate.
CODE_FILES = ["src/index/hnsw_index.cpp", "include/vecsearch/hnsw_index.hpp",
              "src/index/termination_model.cpp", "include/vecsearch/termination_model.hpp",
              "src/distance/avx2.cpp", "src/distance/dispatch.cpp", "src/distance/scalar.cpp",
              "src/distance/neon.cpp", "include/vecsearch/distance.hpp",
              "include/vecsearch/visited_list.hpp", "bench/ann/adaptive_eval.cpp"]


class StudyError(RuntimeError):
    pass


# ---------------------------------------------------------------- io and fingerprints

def fbin(name):
    return DATA / f"{DS}_{name}.fbin"


def gt_path(split):
    return DATA / f"{DS}_{split}.gt"


def read_fbin(path):
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(n, d)


def model_queries(split):
    """Query vectors exactly as the C++ model sees them: normalized for cosine (the search
    normalizes each query before the layer-0 search)."""
    q = read_fbin(fbin(split))
    if METRIC == "cosine":
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    return q.astype(np.float32)


def read_gt(split):
    path = gt_path(split)
    nq, k = np.fromfile(path, dtype=np.uint64, count=2).astype(int)
    ids = np.fromfile(path, dtype=np.int64, offset=16, count=nq * k).reshape(nq, k)
    dists = np.fromfile(path, dtype=np.float32, offset=16 + 8 * nq * k).reshape(nq, k)
    return ids[:, :K], dists[:, :K]


def read_trace(path):
    with open(path, "rb") as f:
        nq, nf, ne = np.fromfile(f, dtype=np.uint64, count=3).astype(int)
        t = {"evals": np.fromfile(f, np.uint32, nq), "expansions": np.fromfile(f, np.uint32, nq),
             "valid": np.fromfile(f, np.uint8, nq).astype(bool),
             "features": np.fromfile(f, np.float32, nq * nf).reshape(nq, nf),
             "offsets": np.fromfile(f, np.uint64, nq + 1).astype(np.int64),
             "ev_eval": np.fromfile(f, np.uint32, ne), "ev_id": np.fromfile(f, np.int64, ne),
             "ev_dist": np.fromfile(f, np.float32, ne)}
    return t


_SHA_MEMO: dict = {}


def sha256(path):
    """sha256 of a file, memoized on (path, size, mtime) so large inputs are hashed once."""
    path = pathlib.Path(path)
    st = path.stat()
    memo_key = (str(path), st.st_size, st.st_mtime_ns)
    if memo_key not in _SHA_MEMO:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _SHA_MEMO[memo_key] = h.hexdigest()
    return _SHA_MEMO[memo_key]


def code_fingerprint():
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(rel.encode() + b"\0" + (ROOT / rel).read_bytes() + b"\0")
    return h.hexdigest()


def write_json(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=1, default=float))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def sh(cmd, log):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(json.dumps({"cmd": cmd, "rc": r.returncode, "seconds": time.time() - t0}) + "\n")
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise StudyError(f"{cmd} failed: {r.stderr[-2000:]}")
    return r.stdout


# ---------------------------------------------------------------- run isolation

def runs_dir(run_dir):
    return DATA / "runs_by_id" / pathlib.Path(run_dir).name


def current_inputs():
    return {"dataset": DS, "index": {INDEX.name: sha256(INDEX)},
            "fbin": {s: sha256(fbin(s)) for s in ["base", "learn", "val", "test"]},
            "gt": {s: sha256(gt_path(s)) for s in ["learn", "val", "test"]}}


def pin_inputs(run_dir, note=None):
    """Record the run's input fingerprints (first stage), or check them (every later stage)."""
    p = pathlib.Path(run_dir) / "inputs.json"
    cur = current_inputs()
    if not p.exists():
        rec = {**cur, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "code_fingerprint_at_pin": code_fingerprint()}
        if note:
            rec["note"] = note
        write_json(p, rec)
        return
    rec = read_json(p)
    for key in ["dataset", "index", "fbin", "gt"]:
        if rec.get(key) != cur[key]:
            raise StudyError(f"{p}: {key} differs from the files this run was made with "
                             f"(recorded {rec.get(key)}, now {cur[key]}). Use a new run id.")


def refuse_overwrite(path):
    if pathlib.Path(path).exists():
        raise StudyError(f"{path} exists: this stage already completed for this run. "
                         "Results are never overwritten; use a new run directory.")


def trace_key(index, queries, ef, c):
    spec = {"index": sha256(index), "queries": sha256(queries), "code": code_fingerprint(),
            "ef": int(ef), "checkpoint": int(c), "k": K}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:32], spec


def cache_paths(key):
    d = DATA / "trace_cache"
    return d / f"{key}.trace", d / f"{key}.json"


def verify_cached_trace(key):
    tpath, mpath = cache_paths(key)
    if not (tpath.exists() and mpath.exists()):
        raise StudyError(f"trace {key} missing from the cache")
    man = read_json(mpath)
    if man.get("key") != key or sha256(tpath) != man.get("trace_sha256"):
        raise StudyError(f"trace cache entry {key} is damaged or does not match its manifest")
    return tpath, man


def ensure_trace(index, queries, ef, c, log):
    """Path of the trace for these exact inputs, generating it if the cache has no valid entry."""
    key, spec = trace_key(index, queries, ef, c)
    tpath, mpath = cache_paths(key)
    if tpath.exists() or mpath.exists():
        verify_cached_trace(key)  # raises on a damaged or mismatched entry; never regenerates over it
        return key
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = tpath.with_suffix(".part")
    sh([TOOL, "trace", str(index), str(queries), str(ef), str(c), str(tmp)], log)
    tmp.rename(tpath)
    write_json(mpath, {"key": key, "inputs": spec, "trace_sha256": sha256(tpath),
                       "index_file": str(index), "queries_file": str(queries),
                       "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return key


def record_trace(run_dir, name, key):
    p = pathlib.Path(run_dir) / "traces.json"
    rec = read_json(p) if p.exists() else {}
    _, man = verify_cached_trace(key)
    if name in rec and rec[name]["key"] != key:
        raise StudyError(f"{p}: {name} was made from different inputs or code "
                         f"({rec[name]['key']} vs {key}). Use a new run id.")
    rec[name] = {"key": key, "sha256": man["trace_sha256"]}
    write_json(p, rec)


def load_trace(run_dir, name):
    rec = read_json(pathlib.Path(run_dir) / "traces.json")
    if name not in rec:
        raise StudyError(f"trace {name} is not part of run {pathlib.Path(run_dir).name}")
    tpath, man = verify_cached_trace(rec[name]["key"])
    if man["trace_sha256"] != rec[name]["sha256"]:
        raise StudyError(f"trace {name} changed since the run recorded it")
    return read_trace(tpath)


def tname(split, ef, c):
    return f"{split}/ef{ef}/c{c}"


# ---------------------------------------------------------------- labels and simulation

def entry_evals(trace, gt_ids):
    """entry[q, j] = evaluation at which the j-th true neighbor entered the top-k (inf if never).

    A true top-k neighbor, once in the running top-k, can only be pushed out by a closer node,
    i.e. another true neighbor; so (ties aside) recall after B evaluations is
    #{j : entry[q, j] <= B} / k. Checked against real budgeted runs (Adaptive.BudgetIsPrefixOfFullRun
    and the val runs in `select`)."""
    nq = len(gt_ids)
    entry = np.full((nq, K), np.inf)
    off, ev_eval, ev_id = trace["offsets"], trace["ev_eval"], trace["ev_id"]
    for q in range(nq):
        a, b = off[q], off[q + 1]
        first = dict(zip(ev_id[a:b][::-1].tolist(), ev_eval[a:b][::-1].tolist()))
        for j, g in enumerate(gt_ids[q].tolist()):
            e = first.get(g)
            if e is not None:
                entry[q, j] = e
    return entry


def label_evals(entry, total, c):
    """T_q: evaluations after which the top-k holds every true neighbor the full run finds."""
    found = np.where(np.isfinite(entry), entry, 0).max(axis=1)
    return np.clip(np.maximum(found, 1), 1, total).astype(np.float64)


def simulate(entry, total, budget):
    """Mean recall@k and per-query evaluations when stopping each query at `budget` (array)."""
    ev = np.minimum(budget, total)
    rec = (entry <= ev[:, None]).sum(axis=1) / K
    return rec, ev


def learned_budget(pred_log, valid, total, c, mult):
    b = np.maximum(c, np.ceil(np.minimum(mult * np.exp(pred_log), 4e9)))
    use = valid & (total > c)
    return np.where(use, b, total).astype(np.float64)


def fit_multiplier(entry, total, pred_log, valid, c, target):
    lo, hi = 0.05, 64.0
    if simulate(entry, total, learned_budget(pred_log, valid, total, c, hi))[0].mean() < target:
        return None
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        r = simulate(entry, total, learned_budget(pred_log, valid, total, c, mid))[0].mean()
        lo, hi = (lo, mid) if r >= target else (mid, hi)
    return hi


def fit_cap(entry, total, target):
    lo, hi = 1, int(total.max())
    if simulate(entry, total, np.full(len(total), hi))[0].mean() < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if simulate(entry, total, np.full(len(total), mid))[0].mean() >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def greedy_oracle(entry, target):
    """Diagnostic only (uses ground truth): per-query stopping points chosen greedily, cheapest
    additional true neighbor first, until mean recall reaches the target. Not deployable."""
    import heapq
    nq = len(entry)
    steps = np.sort(np.where(np.isfinite(entry), entry, np.inf), axis=1)
    pos = np.zeros(nq, dtype=int)
    cur = np.zeros(nq)
    need = int(np.ceil(target * nq * K))
    have = 0
    heap = [(steps[q, 0], q) for q in range(nq) if np.isfinite(steps[q, 0])]
    heapq.heapify(heap)
    while have < need and heap:
        _, q = heapq.heappop(heap)
        cur[q] = steps[q, pos[q]]
        pos[q] += 1
        have += 1
        if pos[q] < K and np.isfinite(steps[q, pos[q]]):
            heapq.heappush(heap, (steps[q, pos[q]] - cur[q], q))
    return have / (nq * K), cur.mean()


# ---------------------------------------------------------------- models

def train_models(X, y, use_query, Q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    Xf = np.hstack([X, Q]) if use_query else X
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
    ridge = Ridge(alpha=1.0).fit((Xf - mu) / sd, y)
    w = ridge.coef_ / sd
    lin = {"kind": "linear", "bias": float(ridge.intercept_ - (ridge.coef_ * mu / sd).sum()),
           "weights": w.tolist()}
    gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                                       early_stopping=False, random_state=0).fit(Xf, y)
    trees = []
    for (pred,) in gb._predictors:
        n = pred.nodes
        trees.append([(-1 if n["is_leaf"][i] else int(n["feature_idx"][i]),
                       float(n["num_threshold"][i]), int(n["left"][i]), int(n["right"][i]),
                       float(n["value"][i])) for i in range(len(n))])
    gbdt = {"kind": "gbdt", "bias": float(np.ravel(gb._baseline_prediction)[0]), "trees": trees}
    # The exported trees must reproduce sklearn's predictions (C++ mirrors predict()).
    sample = Xf[:2000]
    assert np.abs(predict(gbdt, sample) - gb.predict(sample)).max() < 1e-6, "tree export mismatch"
    return {"linear": lin, "gbdt": gbdt}


def predict(model, X, Q=None):
    """Mirror of TerminationModel::predict_log_evals (used for simulation and export checks)."""
    Xf = (np.hstack([X, Q]) if Q is not None else X).astype(np.float32).astype(np.float64)
    if model["kind"] == "linear":
        return model["bias"] + Xf @ np.array(model["weights"])
    out = np.full(len(Xf), model["bias"])
    for tree in model["trees"]:
        node = np.zeros(len(Xf), dtype=int)
        feat = np.array([t[0] for t in tree]); thr = np.array([t[1] for t in tree])
        left = np.array([t[2] for t in tree]); right = np.array([t[3] for t in tree])
        val = np.array([t[4] for t in tree])
        for _ in range(64):
            inner = feat[node] >= 0
            if not inner.any():
                break
            idx = np.where(inner)[0]
            go_left = Xf[idx, feat[node[idx]]] <= thr[node[idx]]
            node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
        out += val[node]
    return out


def write_model(model, path, checkpoint, query_dim):
    lines = ["vecsearch-termination-model 1", f"kind {model['kind']}", f"checkpoint {checkpoint}",
             f"query_dim {query_dim}", f"bias {model['bias']!r}"]
    if model["kind"] == "linear":
        lines.append(f"weights {len(model['weights'])} " + " ".join(repr(w) for w in model["weights"]))
    else:
        lines.append(f"trees {len(model['trees'])}")
        for tree in model["trees"]:
            lines.append(f"nodes {len(tree)}")
            lines += [f"{f} {t!r} {l} {r} {v!r}" for f, t, l, r, v in tree]
    lines.append("end")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def parse_model(tok):
    i = tok.index("bias")
    kind = tok[tok.index("kind") + 1]
    m = {"kind": kind, "bias": float(tok[i + 1])}
    if kind == "linear":
        n = int(tok[i + 3])
        m["weights"] = [float(x) for x in tok[i + 4:i + 4 + n]]
        return m
    j, trees = i + 4, []
    for _ in range(int(tok[i + 3])):
        n = int(tok[j + 1]); j += 2
        trees.append([(int(tok[j + 5 * a]), float(tok[j + 5 * a + 1]), int(tok[j + 5 * a + 2]),
                       int(tok[j + 5 * a + 3]), float(tok[j + 5 * a + 4])) for a in range(n)])
        j += 5 * n
    m["trees"] = trees
    return m


def model_hashes(run_dir):
    return {p.name: sha256(p) for p in sorted((pathlib.Path(run_dir) / "models").glob("*.txt"))}


def verify_models(run_dir, names):
    rec = read_json(pathlib.Path(run_dir) / "models.sha256.json")
    for n in names:
        p = pathlib.Path(run_dir) / "models" / n
        if rec.get(n) != sha256(p):
            raise StudyError(f"model {p} differs from the one selection recorded")


# ---------------------------------------------------------------- runs

STATS_DTYPE = np.dtype([("evals", "<u4"), ("exp", "<u4"), ("budget", "<u4"), ("stop", "u1"),
                        ("model_used", "u1")])


def run_configs(run_dir, name, split, configs, passes, log):
    """configs: list of (name, kind, ef, max_evals, patience, model_path, mult). Raw outputs go
    to this run's own directory; an existing stage directory is never reused or overwritten."""
    out = runs_dir(run_dir) / name
    if out.exists():
        raise StudyError(f"{out} exists: raw outputs of this run are never overwritten")
    out.mkdir(parents=True)
    cfg = out / "configs.txt"
    cfg.write_text("".join(" ".join(str(x) for x in c) + "\n" for c in configs))
    cmd = [TOOL, "run", str(INDEX), str(fbin(split)), str(cfg), str(passes), str(out)]
    if passes > 0:
        cmd = ["taskset", "-c", "3"] + cmd
    sh(cmd, log)
    return load_runs(out, [c[0] for c in configs], passes)


def load_runs(out, names, passes):
    res = {}
    for n in names:
        ids = np.fromfile(out / f"{n}.ids", dtype=np.int64).reshape(-1, K)
        st = np.fromfile(out / f"{n}.stats", dtype=STATS_DTYPE)
        lat = np.fromfile(out / f"{n}.lat", dtype=np.int64)
        res[n] = {"ids": ids, "stats": st, "lat": lat.reshape(passes, -1) if passes > 0 else None}
    return res


def returned_dists(ids, base, queries):
    if METRIC == "cosine":
        unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)  # noqa: E731
        return (1 - (unit(base[ids]) * unit(queries)[:, None, :]).sum(-1)).astype(np.float32)
    return ((base[ids] - queries[:, None, :]) ** 2).sum(-1).astype(np.float32)


def recall_from(ids, dists, gt_ids, gt_d):
    by_id = np.array([len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt_ids.tolist())]) / K
    # Tie-aware: a returned point counts if it is no further than the true k-th neighbor.
    tie = (dists <= gt_d[:, -1:] * (1 + 1e-6) + 1e-6).sum(1) / K
    return by_id, tie


def recall_rows(ids, gt_ids, gt_d, base, queries):
    return recall_from(ids, returned_dists(ids, base, queries), gt_ids, gt_d)


def environment():
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, shell=True, cwd=ROOT).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return str(e)
    import sklearn
    return {"captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cpu": cmd("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
            "nproc": os.cpu_count(), "kernel": platform.platform(),
            "compiler": cmd("g++ --version | head -1"),
            "build": "CMake preset 'bench' (Release, -O3), AVX2 kernels",
            "tool_sha256": sha256(TOOL), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "git_commit": cmd("git rev-parse HEAD"), "git_dirty_files": cmd("git status --porcelain | wc -l"),
            "code_fingerprint": code_fingerprint(), "index_sha256": sha256(INDEX)}


# ---------------------------------------------------------------- stages

def stage_traces(run_dir):
    pin_inputs(run_dir)
    log = run_dir / "traces.log"
    for split in ["learn", "val", "test"]:
        for ef in EF_MAX:
            for c in CHECKPOINTS:
                record_trace(run_dir, tname(split, ef, c), ensure_trace(INDEX, fbin(split), ef, c, log))


def stage_select(run_dir):
    pin_inputs(run_dir)
    refuse_overwrite(run_dir / "selection.json")
    log = run_dir / "select.log"
    learn_gt, _ = read_gt("learn")
    val_gt, val_gt_d = read_gt("val")
    Ql, Qv = model_queries("learn"), model_queries("val")
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    sim, chosen = [], {}

    for ef in EF_MAX:
        tl = load_trace(run_dir, tname("learn", ef, CHECKPOINTS[0]))
        tv = load_trace(run_dir, tname("val", ef, CHECKPOINTS[0]))
        el, ev = entry_evals(tl, learn_gt), entry_evals(tv, val_gt)
        tot_l, tot_v = tl["evals"].astype(float), tv["evals"].astype(float)
        full_rec = simulate(ev, tot_v, tot_v)[0].mean()
        sim.append({"method": "full", "ef_max": ef, "recall": full_rec, "evals": tot_v.mean()})
        for r in TARGETS:
            cap = fit_cap(ev, tot_v, r)
            if cap is not None:
                rec, e = simulate(ev, tot_v, np.full(len(tot_v), cap))
                sim.append({"method": "cap", "ef_max": ef, "target": r, "max_evals": cap,
                            "recall": rec.mean(), "evals": e.mean()})
        for c in CHECKPOINTS:
            tl, tv = load_trace(run_dir, tname("learn", ef, c)), load_trace(run_dir, tname("val", ef, c))
            y = np.log(label_evals(el, tot_l, c))
            train = tl["valid"] & (tot_l > c)
            for use_q in [False, True]:
                ms = train_models(tl["features"][train], y[train], use_q, Ql[train])
                for kind, m in ms.items():
                    tag = f"{kind}{'_q' if use_q else ''}_ef{ef}_c{c}"
                    pv = predict(m, tv["features"], Qv if use_q else None)
                    pv = np.where(tv["valid"], pv, 0.0)
                    yv = np.log(label_evals(ev, tot_v, c))
                    mask = tv["valid"] & (tot_v > c)
                    r2 = 1 - ((pv[mask] - yv[mask]) ** 2).mean() / yv[mask].var()
                    for r in TARGETS:
                        mult = fit_multiplier(ev, tot_v, pv, tv["valid"], c, r)
                        if mult is None:
                            continue
                        rec, e = simulate(ev, tot_v, learned_budget(pv, tv["valid"], tot_v, c, mult))
                        sim.append({"method": "learned", "model": tag, "kind": kind,
                                    "use_query": use_q, "ef_max": ef, "checkpoint": c,
                                    "target": r, "multiplier": mult, "val_r2_log": r2,
                                    "recall": rec.mean(), "evals": e.mean()})
                    write_model(m, models_dir / f"{tag}.txt", c, Ql.shape[1] if use_q else 0)
                    with open(log, "a") as f:
                        f.write(f"{tag}: val R^2(log T) {r2:.4f}\n")
    write_json(run_dir / "models.sha256.json", model_hashes(run_dir))

    # Learned and cap: the setting with the fewest simulated val evaluations at each target.
    for r in TARGETS:
        for method in ["learned", "cap"]:
            cands = [s for s in sim if s["method"] == method and s.get("target") == r]
            if cands:
                chosen[f"{method}@{r}"] = min(cands, key=lambda s: s["evals"])

    # Fixed ef and patience: real single-thread runs on val (recall only, no timing).
    base = read_fbin(fbin("base"))
    configs = [(f"fixed_ef{ef}", "fixed", ef, 0, 0, "-", 1) for ef in FIXED_EF]
    configs += [(f"pat_ef{ef}_n{n}", "adaptive", ef, 0, n, "-", 1)
                for ef in PATIENCE_EF for n in PATIENCE_N]
    runs = run_configs(run_dir, "val_sweep", "val", configs, 0, log)
    val_runs = []
    for name, res in runs.items():
        rec, tie = recall_rows(res["ids"], val_gt, val_gt_d, base, Qv)
        val_runs.append({"name": name, "recall": rec.mean(), "recall_tie": tie.mean(),
                         "evals": float(res["stats"]["evals"].mean())})
    # Fixed-ef runs record no stats; count their evaluations with the equivalent adaptive run.
    fixed_evals = run_configs(run_dir, "val_fixed_evals", "val",
                              [(f"fixed_ef{ef}", "adaptive", ef, 0, 0, "-", 1) for ef in FIXED_EF], 0, log)
    for v in val_runs:
        if v["name"].startswith("fixed_"):
            v["evals"] = float(fixed_evals[v["name"]]["stats"]["evals"].mean())
    for r in TARGETS:
        ok = [v for v in val_runs if v["name"].startswith("fixed_") and v["recall"] >= r]
        if ok:
            chosen[f"fixed@{r}"] = min(ok, key=lambda v: v["evals"])
        ok = [v for v in val_runs if v["name"].startswith("pat_") and v["recall"] >= r]
        if ok:
            chosen[f"patience@{r}"] = min(ok, key=lambda v: v["evals"])

    # Confirm the simulated learned and cap settings with real val runs.
    confirm = []
    for key, s in chosen.items():
        if key.startswith("learned"):
            confirm.append((key.replace("@", "_"), "adaptive", s["ef_max"], 0, 0,
                            str(models_dir / f"{s['model']}.txt"), repr(float(s["multiplier"]))))
        elif key.startswith("cap"):
            confirm.append((key.replace("@", "_"), "adaptive", s["ef_max"], s["max_evals"], 0, "-", 1))
    real = run_configs(run_dir, "val_confirm", "val", confirm, 0, log)
    for key, s in chosen.items():
        n = key.replace("@", "_")
        if n in real:
            rec, _ = recall_rows(real[n]["ids"], val_gt, val_gt_d, base, Qv)
            s["val_recall_real"] = rec.mean()
            s["val_evals_real"] = float(real[n]["stats"]["evals"].mean())

    write_json(run_dir / "selection.json", {"chosen": chosen, "simulated": sim, "val_runs": val_runs})
    print(json.dumps(chosen, indent=1, default=float))


def test_configs(run_dir, sel):
    configs = [(f"fixed_ef{ef}", "fixed", ef, 0, 0, "-", 1) for ef in FIXED_EF]
    for key, s in sel.items():
        method, r = key.split("@")
        name = f"{method}_{r}"
        if method == "fixed":
            continue  # already in the sweep
        if method == "patience":
            ef, n = s["name"].split("_")[1:]
            configs.append((name, "adaptive", int(ef[2:]), 0, int(n[1:]), "-", 1))
        elif method == "cap":
            configs.append((name, "adaptive", s["ef_max"], s["max_evals"], 0, "-", 1))
        else:
            configs.append((name, "adaptive", s["ef_max"], 0, 0,
                            str(run_dir / "models" / f"{s['model']}.txt"), repr(float(s["multiplier"]))))
    assert len({c[0] for c in configs}) == len(configs)
    return configs


def measure_overhead(run_dir, sel, log):
    """Model inference cost (C++ microbenchmark) for each chosen learned model."""
    overhead = {}
    for r in TARGETS:
        s = sel.get(f"learned@{r}")
        if s:
            t = load_trace(run_dir, tname("val", s["ef_max"], s["checkpoint"]))
            fpath = runs_dir(run_dir) / "overhead_inputs.f32"
            X = t["features"][t["valid"]]
            if s["use_query"]:
                X = np.hstack([X, model_queries("val")[t["valid"]]])
            X.astype(np.float32).tofile(fpath)
            out = sh([TOOL, "overhead", str(run_dir / "models" / f"{s['model']}.txt"), str(fpath), "20"], log)
            overhead[s["model"]] = json.loads(out.strip().splitlines()[-1])
    return overhead


def stage_test(run_dir):
    pin_inputs(run_dir)
    refuse_overwrite(run_dir / "test_run.json")
    sel = read_json(run_dir / "selection.json")["chosen"]
    log = run_dir / "test.log"
    configs = test_configs(run_dir, sel)
    verify_models(run_dir, [pathlib.Path(c[5]).name for c in configs if c[5] != "-"])
    env = environment()  # measurement provenance, captured when the measurement runs
    # Distance-evaluation counts for fixed ef come from the equivalent adaptive run (untimed).
    run_configs(run_dir, "test_evals", "test",
                [(f"evals_ef{ef}", "adaptive", ef, 0, 0, "-", 1) for ef in FIXED_EF], 0, log)
    t0 = time.time()
    run_configs(run_dir, "test", "test", configs, PASSES, log)
    rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
           "seconds": time.time() - t0, "passes": PASSES, "warmup_passes": 1,
           "pinned_cpu": 3, "threads": 1, "order": "config order rotated by one each pass",
           "configs": configs, "environment": env,
           "model_overhead": measure_overhead(run_dir, sel, log)}
    write_json(run_dir / "test_run.json", rec)


# ---------------------------------------------------------------- audit bundle and report

def bundle_dir(run_dir):
    return DATA / "bundles" / pathlib.Path(run_dir).name


def stage_bundle(run_dir):
    """Seal everything the report needs (full per-pass timings, returned ids and distances, stats,
    test ground truth, oracle inputs, selection, provenance, hashes) into one directory and a .tar.
    The report stage reads only this bundle."""
    pin_inputs(run_dir)
    out = bundle_dir(run_dir)
    refuse_overwrite(out)
    tr = read_json(run_dir / "test_run.json")
    sel = read_json(run_dir / "selection.json")["chosen"]
    raw = runs_dir(run_dir)
    names = [c[0] for c in tr["configs"] if not c[0].startswith("evals_")]
    res = load_runs(raw / "test", names, tr["passes"])
    ev_dir = raw / "test_evals"
    if (ev_dir / f"evals_ef{FIXED_EF[0]}.stats").exists():
        ev = load_runs(ev_dir, [f"evals_ef{ef}" for ef in FIXED_EF], 0)
    else:  # the first SIFT run timed the evaluation-count configs in the same pass as the rest
        ev = load_runs(raw / "test", [f"evals_ef{ef}" for ef in FIXED_EF], tr["passes"])
    gt_ids, gt_d = read_gt("test")
    base, Qt = read_fbin(fbin("base")), read_fbin(fbin("test"))

    arrays = {"gt_ids": gt_ids, "gt_dists": gt_d}
    for n, r in res.items():
        arrays[f"{n}__ids"] = r["ids"]
        arrays[f"{n}__dists"] = returned_dists(r["ids"], base, Qt)
        arrays[f"{n}__lat_ns"] = r["lat"]
        arrays[f"{n}__stats"] = r["stats"]
    for n, r in ev.items():
        arrays[f"{n}__stats"] = r["stats"]
    oracle_inputs = {}
    for r in TARGETS:
        s = sel.get(f"learned@{r}")
        if s:
            nm = tname("test", s["ef_max"], s["checkpoint"])
            arrays[f"oracle_{r}__entry"] = entry_evals(load_trace(run_dir, nm), gt_ids)
            oracle_inputs[str(r)] = {"ef_max": s["ef_max"], "trace": nm}
    out.mkdir(parents=True)
    np.savez_compressed(out / "arrays.npz", **arrays)

    legacy = {}
    if "environment" not in tr:  # runs made before the test stage recorded provenance itself
        old = read_json(run_dir / "summary.json")
        legacy = {"environment": {**old["environment"], "note": (
            "Legacy run: the test stage did not record its environment. Captured by the first "
            "report generation right after the timing run, on the same machine; git_commit is "
            "the commit checked out at that moment.")},
            "model_overhead": {"values": old["model_overhead"], "note": (
                "Measured by the first report generation on the same machine, not by the test stage.")}}
    meta = {
        "format": BUNDLE_FORMAT, "dataset": DS, "metric": METRIC, "run_id": pathlib.Path(run_dir).name,
        "k": K, "targets": TARGETS, "fixed_ef": FIXED_EF, "configs": names,
        "selection": sel, "test_run": tr, "oracle": oracle_inputs,
        "measurement_environment": tr.get("environment", legacy.get("environment")),
        "model_overhead": tr.get("model_overhead", legacy.get("model_overhead")),
        "inputs": read_json(run_dir / "inputs.json"), "traces": read_json(run_dir / "traces.json"),
        "models_sha256": read_json(run_dir / "models.sha256.json"),
        "raw_files_sha256": {str(p.relative_to(raw)): sha256(p) for p in sorted(raw.rglob("*")) if p.is_file()},
        "bundled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for extra in ["multithread.json", "stress.json", "split_manifest.json"]:
        if (run_dir / extra).exists():
            meta[f"file_sha256:{extra}"] = sha256(run_dir / extra)
    write_json(out / "meta.json", meta)
    shutil.copy(run_dir / "selection.json", out / "selection.json")
    tar = out.parent / f"{meta['run_id']}_audit_bundle.tar"
    with tarfile.open(tar, "w") as t:
        for f in ["arrays.npz", "meta.json", "selection.json"]:
            t.add(out / f, arcname=f"{meta['run_id']}/{f}")
    write_json(run_dir / "bundle.json", {
        "file": tar.name, "sha256": sha256(tar), "bytes": tar.stat().st_size,
        "arrays_sha256": sha256(out / "arrays.npz"), "meta_sha256": sha256(out / "meta.json"),
        "release": "https://github.com/jaideepgorijavolu-oss/vecsearch/releases/tag/adaptive-search-data-v1"})
    print(f"wrote {tar} ({tar.stat().st_size / 2**20:.1f} MiB)")


def render_report(bundle, out_dir):
    """Tables, plot and summary.json from an audit bundle directory only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle, out_dir = pathlib.Path(bundle), pathlib.Path(out_dir)
    meta = read_json(bundle / "meta.json")
    if meta["format"] != BUNDLE_FORMAT:
        raise StudyError(f"unsupported bundle format {meta['format']}")
    a = np.load(bundle / "arrays.npz")
    sel, targets = meta["selection"], meta["targets"]
    gt_ids, gt_d = a["gt_ids"], a["gt_dists"]

    rows, per_query = {}, {}
    for n in meta["configs"]:
        rec, tie = recall_from(a[f"{n}__ids"], a[f"{n}__dists"], gt_ids, gt_d)
        lat_us = a[f"{n}__lat_ns"] / 1e3
        st = a[f"{n.replace('fixed_', 'evals_')}__stats"] if n.startswith("fixed_") else a[f"{n}__stats"]
        per_pass_mean = lat_us.mean(1)
        rows[n] = {
            "recall": rec.mean(), "recall_tie": tie.mean(),
            "recall_p10": float(np.quantile(rec, 0.10)), "frac_recall_below_0.8": float((rec < 0.8).mean()),
            "mean_us": float(np.median(per_pass_mean)),
            "mean_us_min": float(per_pass_mean.min()), "mean_us_max": float(per_pass_mean.max()),
            "p50_us": float(np.median(np.percentile(lat_us, 50, axis=1))),
            "p95_us": float(np.median(np.percentile(lat_us, 95, axis=1))),
            "p99_us": float(np.median(np.percentile(lat_us, 99, axis=1))),
            "qps_1thread": float(1e6 / np.median(per_pass_mean)),
            "evals": float(st["evals"].mean()), "expansions": float(st["exp"].mean()),
            "model_used_frac": float(st["model_used"].mean()),
        }
        per_query[n] = {"recall": rec.astype(np.float32),
                        "lat_us_median": np.median(lat_us, 0).astype(np.float32), "evals": st["evals"]}
    np.savez_compressed(out_dir / "per_query_test.npz",
                        **{f"{n}__{k}": v for n, d in per_query.items() for k, v in d.items()})

    oracle = {}
    for r, o in meta["oracle"].items():
        rec, ev = greedy_oracle(a[f"oracle_{r}__entry"], float(r))
        oracle[r] = {"ef_max": o["ef_max"], "recall": rec, "evals": ev}

    table = []
    for r in targets:
        for method in ["fixed", "cap", "patience", "learned"]:
            s = sel.get(f"{method}@{r}")
            if s:
                n = s["name"] if method == "fixed" else f"{method}_{r}"
                table.append({"target": r, "method": method, "config": n, **rows[n]})

    summary = {"rows": rows, "matched": table, "oracle": oracle, "model_overhead": meta["model_overhead"],
               "selection": sel, "measurement_environment": meta["measurement_environment"],
               "bundle": {"run_id": meta["run_id"], "meta_sha256": sha256(bundle / "meta.json"),
                          "arrays_sha256": sha256(bundle / "arrays.npz")},
               "report_environment": {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                      "python": platform.python_version(), "numpy": np.__version__}}
    write_json(out_dir / "summary.json", summary)

    lines = ["| target (val) | method | config | test R@10 | tie-aware | mean µs | p50 | p95 | p99 | QPS 1-thr | layer-0 evals | R@10<0.8 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        lines.append(f"| {t['target']} | {t['method']} | {t['config']} | {t['recall']:.4f} | {t['recall_tie']:.4f} | "
                     f"{t['mean_us']:.1f} ({t['mean_us_min']:.1f}–{t['mean_us_max']:.1f}) | {t['p50_us']:.1f} | "
                     f"{t['p95_us']:.1f} | {t['p99_us']:.1f} | {t['qps_1thread']:.0f} | {t['evals']:.0f} | "
                     f"{t['frac_recall_below_0.8']:.3f} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fx = sorted([(rows[n]["recall"], rows[n]["mean_us"], rows[n]["evals"]) for n in rows if n.startswith("fixed_")])
    for ax, key, lab in [(axes[0], 1, "mean latency per query (µs), 1 thread"),
                         (axes[1], 2, "layer-0 distance evaluations per query")]:
        ax.plot([f[0] for f in fx], [f[key] for f in fx], "-o", ms=3, color="#555", label="fixed ef")
        for method, mk, col in [("cap", "s", "#1f77b4"), ("patience", "^", "#2ca02c"), ("learned", "D", "#d62728")]:
            pts = [t for t in table if t["method"] == method]
            ax.scatter([p["recall"] for p in pts], [p["mean_us" if key == 1 else "evals"] for p in pts],
                       marker=mk, color=col, s=40, zorder=3, label=method)
        if key == 2 and oracle:
            ax.scatter([o["recall"] for o in oracle.values()], [o["evals"] for o in oracle.values()],
                       marker="x", color="k", label="oracle (not deployable)")
            ax.legend(fontsize=8)
        ax.set_xlabel("recall@10 (test)"); ax.set_ylabel(lab); ax.set_yscale("log")
        ax.set_xlim(min(targets) - 0.05, 1.0); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{'SIFT1M' if meta['dataset'] == 'sift' else 'GloVe-100'}, M=16, efc=200: test queries, "
                 "settings chosen on validation")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_latency.png", dpi=130)
    print("\n".join(lines))


def stage_report(run_dir):
    """Regenerates the report from the audit bundle: $BUNDLE if set (e.g. a downloaded copy),
    else this run's bundle under $VECSEARCH_DATA. Needs no index, traces or raw run outputs."""
    bundle = pathlib.Path(os.environ.get("BUNDLE", bundle_dir(run_dir)))
    render_report(bundle, run_dir)


def stage_stress(run_dir):
    """Predefined stress condition: the same policies on a rebuilt index (graph seed 101), no
    retraining or recalibration. Recall and distance evaluations from traces (exact, as checked
    on val); no timing."""
    pin_inputs(run_dir)
    refuse_overwrite(run_dir / "stress.json")
    log = run_dir / "stress.log"
    idx = DATA / f"{DS}_m16_efc200_s101.hnsw"
    if not idx.exists():
        sh([TOOL, "build", str(fbin("base")), str(idx), "16", "200", "101", METRIC], log)
    sel = read_json(run_dir / "selection.json")["chosen"]
    gt_ids, _ = read_gt("test")
    Qt = model_queries("test")
    out = {"index_sha256": sha256(idx)}
    for r in TARGETS:
        for method in ["learned", "cap"]:
            s = sel.get(f"{method}@{r}")
            c = s.get("checkpoint", CHECKPOINTS[0])
            name = f"stress_s101/{tname('test', s['ef_max'], c)}"
            record_trace(run_dir, name, ensure_trace(idx, fbin("test"), s["ef_max"], c, log))
            t = load_trace(run_dir, name)
            e, tot = entry_evals(t, gt_ids), t["evals"].astype(float)
            if method == "cap":
                budget = np.full(len(tot), s["max_evals"])
            else:
                m = parse_model((run_dir / "models" / f"{s['model']}.txt").read_text().split())
                pv = np.where(t["valid"], predict(m, t["features"], Qt if s["use_query"] else None), 0)
                budget = learned_budget(pv, t["valid"], tot, c, s["multiplier"])
            rec, ev = simulate(e, tot, budget)
            out[f"{method}@{r}"] = {"recall": rec.mean(), "evals": ev.mean()}
    write_json(run_dir / "stress.json", out)
    print(json.dumps(out, indent=1, default=float))


def stage_mt(run_dir):
    """Representative multithreaded throughput for the matched configs: all test queries in one
    batch call, 16 threads (all logical CPUs), best of 5."""
    pin_inputs(run_dir)
    refuse_overwrite(run_dir / "multithread.json")
    sel = read_json(run_dir / "selection.json")["chosen"]
    tr = read_json(run_dir / "test_run.json")
    want = {sel[k]["name"] if k.startswith("fixed") else k.replace("@", "_") for k in sel}
    configs = [c for c in tr["configs"] if c[0] in want]
    verify_models(run_dir, [pathlib.Path(c[5]).name for c in configs if c[5] != "-"])
    raw = runs_dir(run_dir) / "mt"
    raw.mkdir(parents=True)
    cfg = raw / "configs.txt"
    cfg.write_text("".join(" ".join(str(x) for x in c) + "\n" for c in configs))
    env = environment()
    out = sh([TOOL, "batch", str(INDEX), str(fbin("test")), str(cfg), "5", str(os.cpu_count())],
             run_dir / "mt.log")
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    write_json(run_dir / "multithread.json", rows)
    write_json(run_dir / "multithread_env.json", env)
    for r in rows:
        print(f"{r['name']:>16}  {r['qps']:>10.0f} QPS ({r['threads']} threads)")


def stage_adopt(run_dir):
    """One-off for runs made before run isolation (sift1m_v1, glove_v1). Nothing is retrained or
    re-measured. It (1) pins the input hashes after checking them against the ones recorded at
    preparation; (2) regenerates every trace the run used through the fingerprinted cache and
    requires it to be byte-identical to the original file, then records it; (3) records model
    hashes; (4) moves the run's raw outputs into its own runs_by_id directory."""
    name = pathlib.Path(run_dir).name
    legacy_traces = DATA / ("traces" if DS == "sift" else f"traces_{DS}")
    legacy_runs = DATA / ("runs" if DS == "sift" else f"runs_{DS}")
    prep = [json.loads(l) for l in (run_dir / "prepare.log").read_text().splitlines() if l.startswith('{"snapshot"')]
    manifest = read_json(run_dir / "split_manifest.json")["outputs_sha256"]
    cur = current_inputs()
    if prep[-1]["sha256"] != cur["index"][INDEX.name]:
        raise StudyError("index differs from the snapshot recorded at preparation")
    for s, h in cur["fbin"].items():
        if manifest[f"{DS}_{s}.fbin"] != h:
            raise StudyError(f"{s}.fbin differs from the split manifest")
    pin_inputs(run_dir, note="Pinned after the run by `adopt`; index and fbin hashes verified "
                             "against prepare.log and split_manifest.json. GT hashes recorded here.")
    log = run_dir / "adopt.log"
    report = {"identical": [], "missing_legacy": []}
    jobs = [(INDEX, s, ef, c, tname(s, ef, c), legacy_traces / f"{s}_ef{ef}_c{c}.trace")
            for s in ["learn", "val", "test"] for ef in EF_MAX for c in CHECKPOINTS]
    if DS == "sift" and (run_dir / "stress.json").exists():
        idx2 = DATA / "sift_m16_efc200_s101.hnsw"
        if sha256(idx2) != read_json(run_dir / "stress.json")["index_sha256"]:
            raise StudyError("stress index differs from the one stress.json recorded")
        sel = read_json(run_dir / "selection.json")["chosen"]
        for r in TARGETS:
            for method in ["learned", "cap"]:
                s = sel[f"{method}@{r}"]
                c = s.get("checkpoint", CHECKPOINTS[0])
                jobs.append((idx2, "test", s["ef_max"], c, f"stress_s101/{tname('test', s['ef_max'], c)}",
                             legacy_traces / f"stress_test_ef{s['ef_max']}_c{c}.trace"))
    for index, split, ef, c, nm, legacy in jobs:
        key = ensure_trace(index, fbin(split), ef, c, log)
        if not legacy.exists():
            report["missing_legacy"].append(nm)
            continue
        if sha256(cache_paths(key)[0]) != sha256(legacy):
            raise StudyError(f"regenerated trace {nm} differs from the original {legacy}")
        record_trace(run_dir, nm, key)
        report["identical"].append(nm)
    if report["missing_legacy"]:
        raise StudyError(f"original traces missing: {report['missing_legacy']}")
    if not (run_dir / "models.sha256.json").exists():
        write_json(run_dir / "models.sha256.json", model_hashes(run_dir))
    dest = runs_dir(run_dir)
    if not dest.exists():
        dest.mkdir(parents=True)
        for stage in sorted(p for p in legacy_runs.iterdir() if p.is_dir()):
            shutil.move(str(stage), str(dest / stage.name))
    report["raw_dir"] = str(dest)
    write_json(run_dir / "adopt.json", report)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}))


STAGES = {"traces": stage_traces, "select": stage_select, "test": stage_test, "bundle": stage_bundle,
          "report": stage_report, "stress": stage_stress, "mt": stage_mt, "adopt": stage_adopt}

if __name__ == "__main__":
    stage, run_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        STAGES[stage](run_dir)
    except StudyError as e:
        sys.exit(f"error: {e}")
