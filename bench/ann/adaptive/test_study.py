"""Regression tests for run isolation in study.py (review item R1). They drive the real C++
adaptive_eval tool on a tiny synthetic dataset in a temporary data root.

  cmake --preset bench && cmake --build --preset bench --target adaptive_eval
  python3 -m pytest -q bench/ann/adaptive/test_study.py
"""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def write_fbin(path, x):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.astype(np.float32).tofile(f)


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("VECSEARCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DATASET", "sift")
    import study as mod
    mod = importlib.reload(mod)
    mod.DATA.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, n in [("base", 2000), ("learn", 60), ("val", 40), ("test", 40)]:
        write_fbin(mod.fbin(name), rng.standard_normal((n, 16)))
    for s in ["learn", "val", "test"]:
        subprocess.run([mod.TOOL, "gt", str(mod.fbin("base")), str(mod.fbin(s)), "10",
                        str(mod.gt_path(s)), "l2"], check=True, capture_output=True)
    build_index(mod, seed=100)
    return mod


def build_index(mod, seed):
    subprocess.run([mod.TOOL, "build", str(mod.fbin("base")), str(mod.INDEX), "8", "50", str(seed),
                    "l2"], check=True, capture_output=True)
    mod._SHA_MEMO.clear()


def hashes(d):
    return {p.name: p.read_bytes() for p in sorted(pathlib.Path(d).rglob("*")) if p.is_file()}


def test_rebuilt_index_is_neither_reused_nor_accepted(study, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    log = tmp_path / "log"
    study.pin_inputs(run_a)
    key_a = study.ensure_trace(study.INDEX, study.fbin("test"), 64, 20, log)
    study.record_trace(run_a, "test/ef64/c20", key_a)

    build_index(study, seed=101)  # same path, different graph
    key_b = study.ensure_trace(study.INDEX, study.fbin("test"), 64, 20, log)
    assert key_b != key_a  # a new trace, not the cached one
    assert study.cache_paths(key_a)[0].exists() and study.cache_paths(key_b)[0].exists()
    with pytest.raises(study.StudyError, match="index differs"):
        study.pin_inputs(run_a)  # every later stage of run A stops
    with pytest.raises(study.StudyError, match="different inputs or code"):
        study.record_trace(run_a, "test/ef64/c20", key_b)


def test_changed_queries_or_code_change_the_key(study, monkeypatch):
    k1, _ = study.trace_key(study.INDEX, study.fbin("test"), 64, 20)
    monkeypatch.setattr(study, "code_fingerprint", lambda: "different code")
    k2, _ = study.trace_key(study.INDEX, study.fbin("test"), 64, 20)
    monkeypatch.undo()
    write_fbin(study.fbin("test"), np.ones((40, 16)))
    study._SHA_MEMO.clear()
    k3, _ = study.trace_key(study.INDEX, study.fbin("test"), 64, 20)
    assert len({k1, k2, k3}) == 3


def test_damaged_cache_entry_is_rejected(study, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    key = study.ensure_trace(study.INDEX, study.fbin("val"), 64, 20, tmp_path / "log")
    study.record_trace(run, "val/ef64/c20", key)
    path = study.cache_paths(key)[0]
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    study._SHA_MEMO.clear()
    with pytest.raises(study.StudyError, match="damaged"):
        study.ensure_trace(study.INDEX, study.fbin("val"), 64, 20, tmp_path / "log")
    with pytest.raises(study.StudyError, match="damaged"):
        study.load_trace(run, "val/ef64/c20")


def test_two_runs_on_one_data_root_do_not_share_raw_outputs(study, tmp_path):
    cfg = [("fixed_ef32", "fixed", 32, 0, 0, "-", 1)]
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    study.run_configs(run_a, "test", "test", cfg, 0, tmp_path / "log")
    before = hashes(study.runs_dir(run_a))
    study.run_configs(run_b, "test", "test", [("fixed_ef64", "fixed", 64, 0, 0, "-", 1)], 0,
                      tmp_path / "log")
    assert hashes(study.runs_dir(run_a)) == before
    assert study.runs_dir(run_a) != study.runs_dir(run_b)
    with pytest.raises(study.StudyError, match="never overwritten"):
        study.run_configs(run_a, "test", "test", cfg, 0, tmp_path / "log")


def test_changed_model_is_detected(study, tmp_path):
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    (run / "models" / "m.txt").write_text("model one\n")
    study.write_json(run / "models.sha256.json", study.model_hashes(run))
    study.verify_models(run, ["m.txt"])
    (run / "models" / "m.txt").write_text("model two\n")
    study._SHA_MEMO.clear()
    with pytest.raises(study.StudyError, match="differs"):
        study.verify_models(run, ["m.txt"])


def make_run(study, run, fixed_choice):
    run.mkdir()
    study.pin_inputs(run)
    (run / "models").mkdir()
    study.write_json(run / "models.sha256.json", {})
    study.write_json(run / "traces.json", {})
    study.write_json(run / "selection.json", {"chosen": {"fixed@0.9": {"name": fixed_choice}}})
    study.stage_test(run)
    study.stage_bundle(run)
    study.stage_report(run)


def test_reports_read_only_their_own_bundle_and_keep_measurement_provenance(study, tmp_path):
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    make_run(study, run_a, "fixed_ef16")
    table_a, summary_a = (run_a / "table.md").read_text(), json.loads((run_a / "summary.json").read_text())
    measured = json.loads((run_a / "test_run.json").read_text())["environment"]
    assert summary_a["measurement_environment"] == measured
    assert "report_environment" in summary_a and "git_commit" not in summary_a["report_environment"]

    make_run(study, run_b, "fixed_ef32")  # another run on the same data root
    study.stage_report(run_a)  # regenerate A after B exists
    assert (run_a / "table.md").read_text() == table_a
    assert "fixed_ef16" in table_a and "fixed_ef32" in (run_b / "table.md").read_text()
    summary_a2 = json.loads((run_a / "summary.json").read_text())
    assert summary_a2["measurement_environment"] == measured  # not relabeled by the new report

    for stage in [study.stage_test, study.stage_bundle]:
        with pytest.raises(study.StudyError, match="exists"):
            stage(run_a)


def test_changed_code_stops_every_resuming_stage(study, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    study.pin_inputs(run)
    key = study.ensure_trace(study.INDEX, study.fbin("val"), 64, 20, tmp_path / "log")
    study.record_trace(run, "val/ef64/c20", key)
    study.write_json(run / "selection.json", {"chosen": {"fixed@0.9": {"name": "fixed_ef16"}}})
    study.write_json(run / "models.sha256.json", {})

    monkeypatch.setattr(study, "code_fingerprint", lambda: "a different search implementation")
    for stage in [study.stage_traces, study.stage_select, study.stage_test, study.stage_mt,
                  study.stage_stress]:
        with pytest.raises(study.StudyError, match="search code changed"):
            stage(run)
    with pytest.raises(study.StudyError, match="different version of the search code"):
        study.load_trace(run, "val/ef64/c20")
    assert not study.runs_dir(run).exists()  # nothing was measured


def test_line_endings_do_not_change_the_code_fingerprint(study, tmp_path, monkeypatch):
    src = tmp_path / "src"
    for rel in study.CODE_FILES:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes((study.ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(study, "ROOT", src)
    lf = study.code_fingerprint()
    for rel in study.CODE_FILES:
        (src / rel).write_bytes((src / rel).read_bytes().replace(b"\n", b"\r\n"))
    assert study.code_fingerprint() == lf
