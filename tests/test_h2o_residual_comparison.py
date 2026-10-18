"""Independent synthetic replay and adversarial scientific-record checks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from benchmarks import check_h2o_residual_comparison as checker
from benchmarks import run_h2o_residual_comparison as producer
from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.spin_packets import spin_squared_operator


@pytest.fixture(scope="module")
def synthetic():
    model = hubbard(4, t=1.0, U=3.0)
    backend = SectorStatevectorBackend(8, 4, 0)
    operator = backend.operator(model.hamiltonian)
    h = operator.restrict(np.arange(backend.dimension))
    s2 = backend.operator(spin_squared_operator(8)).restrict(np.arange(backend.dimension))
    frame, _ = producer.singlet_frame(backend.basis, 8, s2, 1e-9)
    config = json.loads((producer.ROOT / producer.CONFIG).read_text())
    config["input"]["active_spatial_orbitals"] = [1, 2]
    config.update(maximum_columns=frame.shape[1], maximum_growth_steps=40,
                  test_n_qubits=8, test_n_electrons=4)
    results = {arm: producer.trajectory(operator.matvec, h.diagonal().real, backend.basis,
                                       8, 240, frame, config, arm, s2) for arm in producer.ARMS}
    return config, h, backend.basis.tolist(), s2, results


@pytest.mark.parametrize("arm", producer.ARMS)
def test_three_arms_replay_with_independent_spin_and_hamiltonian(synthetic, arm):
    config, h, words, s2, results = synthetic
    np.testing.assert_allclose(s2, checker.independent_spin_squared(words, 8), atol=1e-12)
    assert checker.verify_worker(results[arm], config, h, words, s2) == []
    final = results[arm]["trajectory"][-1]
    assert abs(final["energy"] - np.linalg.eigvalsh(h)[0]) < 1e-8
    assert final["spin_action_norm"] < 1e-9


@pytest.mark.parametrize("mutation", ["energy", "residual", "selection", "actions", "support", "spin", "stop", "column", "payload"])
def test_independent_replay_rejects_tampered_records(synthetic, mutation):
    config, h, words, s2, results = synthetic
    result = deepcopy(results["residual_packets"])
    row = result["trajectory"][1]
    if mutation == "energy": row["energy"] += .01
    elif mutation == "residual": row["residual_norm"] = 0
    elif mutation == "selection": row["selection"]["candidate"] = -1
    elif mutation == "actions": row["h_action_calls"] -= 1
    elif mutation == "support": row["amplitude_nonzeros"] += 1
    elif mutation == "spin": row["spin_action_norm"] += .1
    elif mutation == "stop": result["stop_reason"] = "residual_tolerance_invalid"
    elif mutation == "column": row["added_columns"][0]["real"][0] += .1
    elif mutation == "payload": row["retained_payload_bytes"] -= 16
    assert checker.verify_worker(result, config, h, words, s2)


def test_default_is_result_free_and_execution_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.setattr(producer, "execute", lambda c: pytest.fail("science reached"))
    assert producer.main([]) == 0
    output = tmp_path / "existing.json"
    output.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        producer.main(["--execute", "--output", str(output)])
    assert output.read_text() == "previous evidence"


def test_committed_campaign_replays():
    path = producer.ROOT / producer.RECORD
    if not path.exists() or not (producer.ROOT / producer.RECEIPT).exists():
        pytest.skip("observations are published only after committed implementation")
    assert checker.verify(json.loads(path.read_text())) == []


def test_timeout_campaign_retains_all_failures_and_report_denominators(monkeypatch):
    from benchmarks import phase17_ceo_oracle as oracle
    from benchmarks import summarize_h2o_residual_comparison as report
    from clifford_qc.backends import sector_statevector

    config = producer.load_design()
    monkeypatch.setattr(producer, "preflight", lambda c: ("1"*40, "2"*40, {}))
    monkeypatch.setattr(producer, "commit_binding", lambda root, commit, path:
                        {"sha256": producer.sha256(root / path)})
    monkeypatch.setattr(producer, "stamp_record", lambda record, **kwargs: record)
    monkeypatch.setattr(oracle, "hamiltonian", lambda path, words: (np.diag([0., 1.]), {}))
    monkeypatch.setattr(sector_statevector, "sector_basis", lambda *args: np.array([0, 1]))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 120)

    monkeypatch.setattr(producer.subprocess, "run", timeout)
    record = producer.execute(config)
    assert len(record["workers"]) == 9
    assert all(w["status"] == "timeout" and "result" not in w for w in record["workers"])
    text = report.render(record)
    assert "failed/timed out: 9" in text
    assert text.count("| 0/3 |") == 6


@pytest.mark.parametrize("field", ["timing", "rss", "provenance", "source"])
def test_preserved_observation_binding_rejects_metadata_changes(field):
    path = producer.ROOT / producer.RECORD
    if not (producer.ROOT / producer.RECEIPT).exists():
        pytest.skip("observation publication not yet committed")
    record = json.loads(path.read_text())
    if field == "timing": record["workers"][0]["process_wall_seconds"] += 1
    elif field == "rss": record["workers"][0]["result"]["final_peak_rss_bytes"] += 1
    elif field == "provenance": record["provenance"]["utc"] = "changed"
    elif field == "source": record["source_commit"] = "0"*40
    with pytest.raises(ValueError, match="inspected object"):
        checker.verify(record)


def test_bundle_restores_history_from_the_baseline_without_creating_refs(monkeypatch, tmp_path):
    if not (producer.ROOT / producer.RECEIPT).exists():
        pytest.skip("observation publication not yet committed")
    root = tmp_path / "fresh"
    receipt = json.loads((producer.ROOT / producer.RECEIPT).read_text())
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "fetch", "--quiet", str(producer.ROOT), receipt["base_commit"]], cwd=root, check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=root, check=True)
    assert subprocess.run(["git", "cat-file", "-e", receipt["observation_commit"]],
                          cwd=root, capture_output=True).returncode != 0
    paths = set(producer.SOURCES) | {producer.CONFIG, producer.DECLARATION,
                                   producer.RECORD, producer.RECEIPT, producer.BUNDLE}
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(producer.ROOT / relative, target)
    refs_before = subprocess.run(["git", "show-ref"], cwd=root, capture_output=True).stdout
    monkeypatch.setattr(checker, "ROOT", Path(root))
    record = json.loads((root / producer.RECORD).read_text())
    checker.restore_and_bind(record)
    subprocess.run(["git", "cat-file", "-e", receipt["observation_commit"]], cwd=root, check=True)
    assert subprocess.run(["git", "show-ref"], cwd=root, capture_output=True).stdout == refs_before


def test_report_changes_when_target_outcomes_change():
    from benchmarks import summarize_h2o_residual_comparison as report
    path = producer.ROOT / producer.RECORD
    if not path.exists():
        pytest.skip("campaign observations not yet produced")
    record = json.loads(path.read_text())
    original = report.render(record)
    for worker in record["workers"]:
        for row in worker["result"]["trajectory"]:
            row["target_reached"] = False
    assert report.render(record) != original
    assert report.render(record).count("| 0/3 |") == 3
