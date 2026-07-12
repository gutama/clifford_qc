"""Backlog 16 acceptance: one-command JSONL generation and summarization."""

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
try:
    import run_benchmark  # noqa: E402
    import summarize  # noqa: E402
finally:
    sys.path.remove(str(BENCH))


TINY_CONFIG = {
    "seeds": 2,
    "models": [
        {"model": {"type": "tfim", "n": 3, "J": 1.0, "h": 1.0},
         "pool": {"type": "local", "periodic_context": False}},
    ],
    "methods": {
        "exact": {"kind": "exact", "max_operators": 4},
        "doubling": {"kind": "confidence", "max_operators": 3,
                     "selector": {"delta": 0.1, "near_tol": 0.1},
                     "allocator": {"type": "uniform_doubling",
                                   "base": 64, "max_factor": 8}},
    },
}


def test_benchmark_matrix_generates_jsonl_and_summary(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(TINY_CONFIG))
    out = tmp_path / "results.jsonl"
    run_benchmark.main(["--config", str(config), "--out", str(out)])

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 * 2  # methods x seeds
    for row in rows:
        assert {"model", "method", "seed", "relative_error", "total_shots",
                "total_circuits", "operators", "status_counts",
                "selection_steps"} <= set(row)
    exact_rows = [r for r in rows if r["method"] == "exact"]
    assert all(r["total_shots"] == 0 for r in exact_rows)
    assert all(r["relative_error"] < 1e-6 for r in exact_rows)

    csv_path = tmp_path / "summary.csv"
    summarize.main([str(out), "--csv", str(csv_path)])
    captured = capsys.readouterr().out
    assert "| model | method |" in captured
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 (model, method) groups


def test_unknown_method_kind_rejected():
    with pytest.raises(ValueError, match="unknown method kind"):
        run_benchmark.build_run_kwargs({"kind": "bogus"}, seed=0)
    with pytest.raises(ValueError, match="unused method keys"):
        run_benchmark.build_run_kwargs({"kind": "exact", "typo_key": 1}, seed=0)
    with pytest.raises(ValueError, match="need an 'allocator'"):
        run_benchmark.build_run_kwargs({"kind": "confidence"}, seed=0)
    with pytest.raises(ValueError, match="unknown allocator type"):
        run_benchmark.build_run_kwargs(
            {"kind": "confidence", "allocator": {"type": "bogus"}}, seed=0)


def test_summarize_rejects_empty_results(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no benchmark rows"):
        summarize.main([str(empty)])
