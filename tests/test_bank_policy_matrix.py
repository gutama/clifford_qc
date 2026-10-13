"""Actual isolated toy workers and corruption/failure gates, no molecular laws."""
from copy import deepcopy
from dataclasses import asdict
import inspect
import os
import sys

import numpy as np
import pytest

from benchmarks import bank_policy_contract as contract
from benchmarks import bank_policy_worker as worker
from benchmarks import check_bank_policy_matrix as checker
from benchmarks import run_bank_policy_matrix as producer
from clifford_qc.subspace.adaptive import ACASEConfig


def toy_config():
    return {"systems": [{"id": "undeclared_tfim3", "kind": "toy", "n_qubits": 3}],
            "options": asdict(ACASEConfig(max_size=3, gamma=0.3)),
            "execution": {"rss_ceiling_bytes": 2 * 2**30, "host_reserve_bytes": 2**29,
                          "timeout_seconds": 60, "sample_seconds": 0.01, "frontier_pairs": 1}}


@pytest.fixture(scope="module")
def toy():
    config = toy_config()
    cells = producer.run_campaign(config, lambda cells: None)
    record = {"schema": contract.SCHEMA, "phase": "2M-C/D",
              "configuration_sha256": contract.EXPECTED_CONFIG_SHA256,
              "claim_boundary": contract.CLAIM_BOUNDARY, "cells": cells, "verdict": "VALID",
              "execution": {"parent_pid": os.getpid(), "serial_workers": True}, "provenance": {}}
    return config, record


def test_fresh_workers_preserve_full_scientific_and_logical_trace(toy):
    config, record = toy
    assert checker.check_record(record, config, provenance=False) == []
    assert len({(c["worker"]["pid"], c["worker"]["start_ticks"]) for c in record["cells"]}) == 4
    assert all(c["worker"]["pid"] != os.getpid() for c in record["cells"])
    stream = record["cells"][3]["costs"]
    assert stream["evicted_rows"] > 0
    assert stream["recomputed_rows"] > 0
    assert stream["resident_coefficients"] < record["cells"][3]["scientific"]["logical"]["coefficient_occurrences"]
    assert all(rows <= bound for rows, bound in stream["row_observations"])


def test_direct_product_reference_matches_without_a_storage_bank(toy):
    config, record = toy
    checked = checker.reference_check({"system": config["systems"][0],
                                       "scientific": record["cells"][0]["scientific"]})
    assert all(value is True for value in checked.values())
    isolated = checker.rebuild_reference(config["systems"][0], record["cells"][0]["scientific"])
    assert isolated == checked


@pytest.mark.parametrize("field", [
    "energy", "labels", "scores", "matrix", "logical_pairs", "logical_coefficients",
    "ownership", "row_limit", "payload", "recompute", "worker_reuse", "threads",
    "ceiling", "verdict", "missing_cell", "bool_count", "claim_boundary", "nan_timing",
])
def test_checker_rejects_corrupted_observation(toy, field):
    config, original = toy
    record = deepcopy(original)
    cell = record["cells"][3]
    science, costs = cell["scientific"], cell["costs"]
    if field == "energy": science["energy"] = (-10.0).hex()
    elif field == "labels": science["labels"][-1] = "forged"
    elif field == "scores": science["scores"][0]["candidate"]["rejected"] = "forged"
    elif field == "matrix":
        matrix = checker.decode_matrix(science["solves"][-1]["H"], len(science["indices"])).copy()
        matrix[0, 0] += 0.1
        science["solves"][-1]["H"] = contract.matrix_record(matrix)
    elif field == "logical_pairs": science["logical"]["pairs_built"] += 1
    elif field == "logical_coefficients": science["logical"]["coefficient_occurrences"] += 1
    elif field == "ownership": science["logical"]["pair_ownership"][0][1] += 1
    elif field == "row_limit": costs["row_observations"][0][0] = costs["row_observations"][0][1] + 1
    elif field == "payload": costs["coefficient_payload_estimate_bytes"] += 24
    elif field == "recompute": costs["recomputed_rows"] += 2
    elif field == "worker_reuse": cell["worker"] = deepcopy(record["cells"][0]["worker"])
    elif field == "threads": cell["worker"]["threads"]["OMP_NUM_THREADS"] = "4"
    elif field == "ceiling": cell["parent"]["rss_ceiling_bytes"] *= 2
    elif field == "verdict": record["verdict"] = "INVALID"
    elif field == "missing_cell": record["cells"].pop()
    elif field == "bool_count": costs["resident_coefficients"] = True
    elif field == "claim_boundary": record["claim_boundary"] = "quantum advantage"
    elif field == "nan_timing": costs["solve_seconds"] = float("nan")
    assert checker.check_record(record, config, provenance=False)


def test_coherent_forgery_is_caught_by_direct_rebuild(toy):
    config, record = toy
    science = deepcopy(record["cells"][0]["scientific"])
    size = len(science["indices"])
    matrix = checker.decode_matrix(science["solves"][-1]["H"], size).copy()
    matrix[0, 0] += 0.1
    science["solves"][-1]["H"] = contract.matrix_record(matrix)
    assert checker.reference_check({"system": config["systems"][0],
                                    "scientific": science})["matrices_bitwise_equal"] is False


@pytest.mark.parametrize("limit,status", [("timeout_seconds", "timeout"), ("rss_ceiling_bytes", "memory_limit")])
def test_parent_stops_and_records_failed_worker(limit, status, monkeypatch):
    original = producer.subprocess.Popen
    def slow_worker(command, **kwargs):
        return original([sys.executable, "-c", "import time; time.sleep(2)"], **kwargs)
    if limit == "timeout_seconds":
        monkeypatch.setattr(producer.subprocess, "Popen", slow_worker)
    config = toy_config()
    config["execution"][limit] = 0.001 if limit == "timeout_seconds" else 1
    cell = producer.run_cell(config["systems"][0], contract.ARMS[0], config)
    assert cell["status"] == status
    if limit == "timeout_seconds":
        assert cell["parent"]["returncode"] != 0
        assert "scientific" not in cell
    elif cell["parent"]["returncode"] == 0:
        # A completed worker's kernel high-water mark independently catches
        # a crossing that a polling parent missed; it still cannot be VALID.
        assert cell["costs"]["worker_high_water_rss_bytes"] > 1


def test_incomplete_matrix_cannot_claim_completion(toy):
    config, original = toy
    record = deepcopy(original)
    record["cells"][0]["status"] = "worker_failed"
    record["verdict"] = "INCOMPLETE"
    assert any("INCOMPLETE" in problem for problem in checker.check_record(record, config, provenance=False))


def test_insufficient_capacity_refuses_without_sampling_or_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(producer, "host_capacity", lambda: {"available_bytes": 8 * 2**30})
    with pytest.raises(ValueError, match="insufficient host capacity"):
        producer.require_host_capacity(contract.load_config()["execution"])
    monkeypatch.setattr(contract, "RECORD", tmp_path / "absent.json")
    def forbidden(*args, **kwargs): raise AssertionError("declared molecular laws were evaluated")
    monkeypatch.setattr(worker, "run_arm", forbidden)
    assert producer.main([]) == 0
    assert checker.main([]) == 0


def test_atomic_publication_and_failed_serialization(tmp_path):
    path = tmp_path / "record.json"
    with pytest.raises(ValueError): contract.atomic_json(path, {"bad": float("nan")}, exclusive=True)
    assert not path.exists()
    contract.atomic_json(path, {"ok": 1}, exclusive=True)
    with pytest.raises(FileExistsError): contract.atomic_json(path, {"ok": 2}, exclusive=True)
    assert contract.read_json(path) == {"ok": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]


def test_json_duplicates_and_declaration_drift(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"value": 1, "value": 2}')
    with pytest.raises(ValueError, match="duplicate"): contract.read_json(path)
    path.write_text(contract.CONFIG.read_text() + " ")
    with pytest.raises(ValueError, match="configuration differs"): contract.load_config(path)


def test_matrix_encoding_rejects_nonfinite_and_bool_shape():
    with pytest.raises(ValueError): contract.matrix_record(np.array([[float("nan")]]))
    value = contract.matrix_record(np.eye(1))
    value["shape"] = [True, True]
    with pytest.raises(ValueError): checker.decode_matrix(value, 1)


def test_checker_does_not_import_worker_or_producer():
    source = inspect.getsource(checker)
    assert "import bank_policy_worker" not in source
    assert "import run_bank_policy_matrix" not in source


def test_failed_campaign_retains_claim_when_error_persistence_fails(monkeypatch, tmp_path):
    claim = tmp_path / "claim.json"
    monkeypatch.setattr(producer, "execution_preflight", lambda: (toy_config(), {}, {}, claim))
    def failed(*args): raise RuntimeError("worker transport failed")
    monkeypatch.setattr(producer, "run_campaign", failed)
    original = contract.atomic_json
    def write(path, value, **kwargs):
        if not kwargs.get("exclusive"):
            raise OSError("disk full")
        return original(path, value, **kwargs)
    monkeypatch.setattr(contract, "atomic_json", write)
    assert producer.main(["--execute"]) == 1
    assert contract.read_json(claim)["status"] == "claimed"


def environment_fixture():
    dependencies = {"numpy": "2.5.2", "scipy": "1.18.0"}
    reference = {"python": "3.12.13", "dependencies": dict(dependencies)}
    record = {"provenance": {"python": "3.12.14", "dependencies": dict(dependencies)},
              "cells": [{"status": "completed", "worker": {
                  "python": "3.12.14", **dependencies}}]}
    return record, reference


def test_record_environment_allows_reference_python_patch_difference():
    record, reference = environment_fixture()
    assert checker.environment_problems(record, reference) == []


@pytest.mark.parametrize("field", [
    "producer_python_minor", "producer_python_type", "worker_python_minor",
    "worker_python_patch", "producer_numpy", "producer_scipy",
    "worker_numpy", "worker_scipy",
])
def test_record_environment_rejects_interpreter_and_library_drift(field):
    record, reference = environment_fixture()
    if field == "producer_python_minor":
        record["provenance"]["python"] = "3.11.14"
    elif field == "producer_python_type":
        record["provenance"]["python"] = False
    elif field == "worker_python_minor":
        record["cells"][0]["worker"]["python"] = "3.11.14"
    elif field == "worker_python_patch":
        record["cells"][0]["worker"]["python"] = "3.12.15"
    else:
        owner, library = field.split("_")
        if owner == "producer":
            record["provenance"]["dependencies"][library] = {"numpy": "2.5.3", "scipy": "1.18.1"}[library]
        else:
            record["cells"][0]["worker"][library] = {"numpy": "2.5.3", "scipy": "1.18.1"}[library]
    expected = "python" if "python" in field else field.split("_")[1]
    problems = checker.environment_problems(record, reference)
    assert any(problem.endswith(f": {expected}") for problem in problems)


@pytest.mark.parametrize("kind", ["S", "H"])
def test_direct_rebuild_rejects_changed_ieee_signed_zero(toy, kind):
    config, record = toy
    science = deepcopy(record["cells"][0]["scientific"])
    size = len(science["indices"])
    original = checker.decode_matrix(science["solves"][-1][kind], size)
    matrix = original.copy()
    assert matrix[0, 0].imag.hex() == "0x0.0p+0"
    matrix[0, 0] = complex(matrix[0, 0].real, -0.0)
    assert np.array_equal(matrix, original)
    assert matrix.tobytes() != original.tobytes()
    science["solves"][-1][kind] = contract.matrix_record(matrix)
    checked = checker.reference_check({"system": config["systems"][0],
                                       "scientific": science})
    assert checked["matrices_bitwise_equal"] is False
