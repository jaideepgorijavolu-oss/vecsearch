"""Adaptive-search study on SIFT1M (docs/adaptive_search/PROTOCOL.md). Run in the dev container.

  python3 bench/ann/adaptive/study.py <stage> <run_dir>

Stages, in order:
  traces   unbudgeted search traces (all threads) for learn / val / test at every ef_max and checkpoint
  select   train models on learn; pick every method's setting on val (simulation + real val runs)
  test     single-thread timing runs on the final test queries (needs a quiet machine)
  report   tables, plots and summary.json from the raw outputs

Large intermediate files live in $VECSEARCH_DATA/adaptive; small outputs in <run_dir>.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

K = 10
EF_MAX = [128, 256, 512]
CHECKPOINTS = [100, 200, 400]
TARGETS = [0.90, 0.95, 0.99]
FIXED_EF = [10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256,
            320, 384, 512]
PATIENCE_EF = [64, 128, 256, 512]
PATIENCE_N = [2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 65, 80]
PASSES = 5
F = 9  # kNumTerminationFeatures
FEATURE_NAMES = ["log_entry", "log_d1", "log_dk", "d1/entry", "dk/entry", "dk/d1", "cand/dk",
                 "stale_frac", "topk_changes"]

DATA = pathlib.Path(os.environ.get("VECSEARCH_DATA", "/data")) / "adaptive"
TOOL = "build/bench/bench/ann/adaptive_eval"
INDEX = DATA / "sift_m16_efc200_s100.hnsw"


# ---------------------------------------------------------------- io

def fbin(name):
    return DATA / f"sift_{name}.fbin"


def read_fbin(path):
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(n, d)


def read_gt(split):
    path = DATA / f"sift_{split}.gt"
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


def trace_path(split, ef, c):
    return DATA / "traces" / f"{split}_ef{ef}_c{c}.trace"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd, log):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(json.dumps({"cmd": cmd, "rc": r.returncode, "seconds": time.time() - t0}) + "\n")
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd} failed: {r.stderr[-2000:]}")
    return r.stdout


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
    nq = len(entry)
    steps = np.sort(np.where(np.isfinite(entry), entry, np.inf), axis=1)
    pos = np.zeros(nq, dtype=int)
    cur = np.zeros(nq)
    need = int(np.ceil(target * nq * K))
    have = 0
    import heapq
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


# ---------------------------------------------------------------- runs

def run_configs(run_dir, name, split, configs, passes, log):
    """configs: list of (name, kind, ef, max_evals, patience, model_path, mult)."""
    out = DATA / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
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
        st = np.fromfile(out / f"{n}.stats", dtype=np.dtype([("evals", "<u4"), ("exp", "<u4"),
                                                             ("budget", "<u4"), ("stop", "u1"),
                                                             ("model_used", "u1")]))
        lat = np.fromfile(out / f"{n}.lat", dtype=np.int64)
        res[n] = {"ids": ids, "stats": st,
                  "lat": lat.reshape(passes, -1) if passes > 0 else None}
    return res


def recall_rows(ids, gt_ids, gt_d, base, queries):
    by_id = np.array([len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt_ids.tolist())]) / K
    # Tie-aware: a returned point counts if it is no further than the true k-th neighbor.
    d = ((base[ids] - queries[:, None, :]) ** 2).sum(-1)
    tie = (d <= gt_d[:, -1:] * (1 + 1e-6) + 1e-6).sum(1) / K
    return by_id, tie


# ---------------------------------------------------------------- stages

def stage_traces(run_dir):
    (DATA / "traces").mkdir(exist_ok=True)
    log = run_dir / "traces.log"
    for split in ["learn", "val", "test"]:
        for ef in EF_MAX:
            for c in CHECKPOINTS:
                p = trace_path(split, ef, c)
                if not p.exists():
                    sh([TOOL, "trace", str(INDEX), str(fbin(split)), str(ef), str(c), str(p)], log)


def stage_select(run_dir):
    log = run_dir / "select.log"
    learn_gt, _ = read_gt("learn")
    val_gt, val_gt_d = read_gt("val")
    Ql, Qv = read_fbin(fbin("learn")), read_fbin(fbin("val"))
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    sim, chosen = [], {}

    for ef in EF_MAX:
        tl, tv = read_trace(trace_path("learn", ef, CHECKPOINTS[0])), read_trace(trace_path("val", ef, CHECKPOINTS[0]))
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
            tl, tv = read_trace(trace_path("learn", ef, c)), read_trace(trace_path("val", ef, c))
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
        # Fixed-ef runs record no stats; count their evaluations with the equivalent adaptive run.
        val_runs.append({"name": name, "recall": rec.mean(), "recall_tie": tie.mean(),
                         "evals": float(res["stats"]["evals"].mean())})
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
                            str(models_dir / f"{s['model']}.txt"), repr(s["multiplier"])))
        elif key.startswith("cap"):
            confirm.append((key.replace("@", "_"), "adaptive", s["ef_max"], s["max_evals"], 0, "-", 1))
    real = run_configs(run_dir, "val_confirm", "val", confirm, 0, log)
    for key, s in chosen.items():
        n = key.replace("@", "_")
        if n in real:
            rec, _ = recall_rows(real[n]["ids"], val_gt, val_gt_d, base, Qv)
            s["val_recall_real"] = rec.mean()
            s["val_evals_real"] = float(real[n]["stats"]["evals"].mean())

    (run_dir / "selection.json").write_text(json.dumps(
        {"chosen": chosen, "simulated": sim, "val_runs": val_runs}, indent=1, default=float))
    print(json.dumps(chosen, indent=1, default=float))


def stage_test(run_dir):
    sel = json.loads((run_dir / "selection.json").read_text())["chosen"]
    log = run_dir / "test.log"
    configs = [(f"fixed_ef{ef}", "fixed", ef, 0, 0, "-", 1) for ef in FIXED_EF]
    seen = {c[0] for c in configs}
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
                            str(run_dir / "models" / f"{s['model']}.txt"), repr(s["multiplier"])))
    configs += [(f"evals_ef{ef}", "adaptive", ef, 0, 0, "-", 1) for ef in FIXED_EF]  # eval counts
    assert len({c[0] for c in configs}) == len(configs) and seen
    t0 = time.time()
    run_configs(run_dir, "test", "test", configs, PASSES, log)
    env = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
           "seconds": time.time() - t0, "passes": PASSES, "warmup_passes": 1,
           "pinned_cpu": 3, "threads": 1, "order": "config order rotated by one each pass",
           "configs": configs}
    (run_dir / "test_run.json").write_text(json.dumps(env, indent=1))


def stage_report(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = json.loads((run_dir / "selection.json").read_text())["chosen"]
    tr = json.loads((run_dir / "test_run.json").read_text())
    names = [c[0] for c in tr["configs"]]
    res = load_runs(DATA / "runs" / "test", names, PASSES)
    gt_ids, gt_d = read_gt("test")
    base, Qt = read_fbin(fbin("base")), read_fbin(fbin("test"))

    rows, per_query = {}, {}
    for n in names:
        if n.startswith("evals_"):
            continue
        r = res[n]
        rec, tie = recall_rows(r["ids"], gt_ids, gt_d, base, Qt)
        lat_us = r["lat"] / 1e3
        st = res[n.replace("fixed_", "evals_")]["stats"] if n.startswith("fixed_") else r["stats"]
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
        per_query[n] = {"recall": rec.astype(np.float32), "lat_us_median": np.median(lat_us, 0).astype(np.float32),
                        "evals": st["evals"]}
    np.savez_compressed(run_dir / "per_query_test.npz",
                        **{f"{n}__{k}": v for n, d in per_query.items() for k, v in d.items()})

    # Oracle (diagnostic): greedy per-query stopping on the chosen learned ef_max trace.
    oracle = {}
    for r in TARGETS:
        s = sel.get(f"learned@{r}")
        if s:
            t = read_trace(trace_path("test", s["ef_max"], s["checkpoint"]))
            rec, ev = greedy_oracle(entry_evals(t, gt_ids), r)
            oracle[str(r)] = {"ef_max": s["ef_max"], "recall": rec, "evals": ev}

    # Matched-quality table: each method's setting was chosen on val for the target.
    table = []
    for r in TARGETS:
        for method in ["fixed", "cap", "patience", "learned"]:
            s = sel.get(f"{method}@{r}")
            if not s:
                continue
            n = s["name"] if method == "fixed" else f"{method}_{r}"
            table.append({"target": r, "method": method, "config": n, **rows[n]})

    # Model inference cost (C++ microbenchmark) for each chosen learned model.
    overhead = {}
    for r in TARGETS:
        s = sel.get(f"learned@{r}")
        if s and not s["use_query"]:
            t = read_trace(trace_path("val", s["ef_max"], s["checkpoint"]))
            fpath = DATA / "val_features.f32"
            t["features"][t["valid"]].astype(np.float32).tofile(fpath)
            out = subprocess.run([TOOL, "overhead", str(run_dir / "models" / f"{s['model']}.txt"),
                                  str(fpath), "20"], capture_output=True, text=True, check=True)
            overhead[s["model"]] = json.loads(out.stdout)

    summary = {"rows": rows, "matched": table, "oracle": oracle, "model_overhead": overhead,
               "selection": sel, "environment": environment(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    lines = ["| target (val) | method | config | test R@10 | tie-aware | mean µs | p50 | p95 | p99 | QPS 1-thr | layer-0 evals | R@10<0.8 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        lines.append(f"| {t['target']} | {t['method']} | {t['config']} | {t['recall']:.4f} | {t['recall_tie']:.4f} | "
                     f"{t['mean_us']:.1f} ({t['mean_us_min']:.1f}–{t['mean_us_max']:.1f}) | {t['p50_us']:.1f} | "
                     f"{t['p95_us']:.1f} | {t['p99_us']:.1f} | {t['qps_1thread']:.0f} | {t['evals']:.0f} | "
                     f"{t['frac_recall_below_0.8']:.3f} |")
    (run_dir / "table.md").write_text("\n".join(lines) + "\n")

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
        ax.set_xlabel("recall@10 (test)"); ax.set_ylabel(lab); ax.set_yscale("log")
        ax.set_xlim(0.85, 1.0); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("SIFT1M, M=16, efc=200: test queries, settings chosen on validation")
    fig.tight_layout()
    fig.savefig(run_dir / "recall_latency.png", dpi=130)
    print("\n".join(lines))


def environment(run_dir):
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, shell=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return str(e)
    import sklearn
    return {"cpu": cmd("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
            "nproc": os.cpu_count(), "kernel": platform.platform(),
            "compiler": cmd("g++ --version | head -1"), "build": "CMake preset 'bench' (Release, -O3), AVX2 kernels",
            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "git_commit": cmd("git rev-parse HEAD"), "git_dirty": cmd("git status --porcelain | wc -l"),
            "index_sha256": sha256(INDEX)}


if __name__ == "__main__":
    stage, run_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    {"traces": stage_traces, "select": stage_select, "test": stage_test, "report": stage_report}[stage](run_dir)
