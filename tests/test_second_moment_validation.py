"""The SecondMomentBank validation producer and checker, on the Hubbard dimer.

A toy preflight record is produced for two banks on the undeclared dimer, and
the validation must accept exactly the banks it licenses, pass every check, and
re-derive under its checker. Each tampered field must be caught.
"""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks import check_phase15_preregistration as gate
from benchmarks import check_second_moment_validation as checker
from benchmarks import run_phase15_h2_preflight as preflight_producer
from benchmarks import run_second_moment_validation as producer
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.generators import identity_generator


@pytest.fixture(scope="module")
def dimer():
    model = hubbard(2)
    raw = _raw_pool(model)
    identity = identity_generator(model.n)
    return {"toy": (model, [identity, *raw[:2]]), "toy_small": (model, [identity, raw[0]])}


@pytest.fixture(scope="module")
def toy(dimer):
    config = copy.deepcopy(gate.load_config())
    config["banks"]["systems"] = sorted(dimer)
    config["measured_before_freezing"] = {
        name: gate.quantities_for(config, *inputs) for name, inputs in dimer.items()}
    preflight = preflight_producer.run_preflight(config, config_bytes=b"{}",
                                                 inputs=dimer.__getitem__)
    raw = json.dumps(preflight).encode()
    record = producer.run_validation(preflight, inputs=dimer.__getitem__,
                                     preflight_bytes=raw)
    return preflight, raw, record


def test_the_toy_validation_passes_and_re_derives(toy, dimer):
    preflight, raw, record = toy
    assert record["summary"]["all_validated"] is True
    for entry in record["banks"].values():
        assert all(entry["checks"].values())
        assert any(row["resolved"] for row in entry["roots"])
    assert checker.declaration_problems(record, preflight, raw) == []
    assert checker.derived_problems(record, preflight) == []
    assert checker.recompute_problems(record, preflight, inputs=dimer.__getitem__) == []


def test_only_licensed_banks_are_validated(toy, dimer):
    preflight, raw, _ = toy
    with pytest.raises(ValueError, match="does not license"):
        producer.run_validation(preflight, banks=["h2o_cas8e6o"], inputs=dimer.__getitem__)
    refused = copy.deepcopy(preflight)
    refused["decision"]["verdict"] = "NONE"
    with pytest.raises(ValueError, match="licenses no SecondMomentBank"):
        producer.run_validation(refused, inputs=dimer.__getitem__)


def _bank(record):
    return record["banks"]["toy"]


TAMPERS = {
    "a variance": (lambda r: _bank(r)["roots"][0].__setitem__("variance", 3.0),
                   "variance_minus_dense drifted"),
    "a residual norm": (lambda r: _bank(r)["roots"][0].__setitem__("residual_norm", 9.0),
                        "clipped root"),
    "a resolved flag": (lambda r: _bank(r)["roots"][0].__setitem__("resolved", False),
                        "resolved flag"),
    "a check flag": (lambda r: _bank(r)["checks"].__setitem__("block_psd", False),
                     "recorded checks"),
    "a status": (lambda r: _bank(r).__setitem__("status", "FAILED"), "recorded status"),
    "the summary": (lambda r: r["summary"].__setitem__("all_validated", False),
                    "all_validated contradicts"),
    "a count": (lambda r: _bank(r)["resources"].__setitem__(
        "second_moment_word_universe", 1), "recorded checks"),
    "a ground residual": (lambda r: r["summary"]["ground_residual_norms"].__setitem__(
        "toy", 0.0), "ground residual norms"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_tamper_is_caught(toy, dimer, name):
    preflight, raw, record = toy
    broken = copy.deepcopy(record)
    edit, expected = TAMPERS[name]
    edit(broken)
    problems = (checker.declaration_problems(broken, preflight, raw)
                + checker.derived_problems(broken, preflight)
                + checker.recompute_problems(broken, preflight, inputs=dimer.__getitem__))
    assert any(expected in problem for problem in problems), problems


def test_a_changed_preflight_breaks_the_digest(toy):
    preflight, raw, record = toy
    assert any("preflight record changed" in problem for problem in
               checker.declaration_problems(record, preflight, raw + b" "))


def test_the_committed_record_re_derives_its_checks():
    """The five-bank record, re-read without rebuilding a bank."""
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    preflight_bytes = producer.PREFLIGHT.read_bytes()
    preflight = json.loads(preflight_bytes)
    assert checker.declaration_problems(record, preflight, preflight_bytes) == []
    assert checker.derived_problems(record, preflight) == []
    assert record["summary"]["all_validated"] is True
    assert record["provenance"]["git_dirty"] is False


def test_the_producer_refuses_a_subset_on_the_committed_path(capsys):
    assert producer.main(["--banks", "h4"]) == 1
    assert "bank subset" in capsys.readouterr().out


def test_the_checker_fails_without_a_record(tmp_path, capsys):
    assert checker.main(["--record", str(tmp_path / "absent.json")]) == 1
    assert "missing record" in capsys.readouterr().out
