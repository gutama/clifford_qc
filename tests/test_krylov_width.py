"""Regression tests for the Krylov word-universe identity and certificate."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.lattice import hubbard, kitaev_honeycomb
from clifford_qc.models.spin import tfim
from clifford_qc.multivector import MV
from clifford_qc.subspace import (MatrixElementBank, identity_generator,
                                  krylov_response)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run_krylov_width import (  # noqa: E402
    REPORTED_THRESHOLD,
    krylov_word_diagnostics,
    krylov_word_universe,
)


def _direct_universe(model, order: int) -> int:
    rho = ExactMVBackend().state(model.reference, ())
    generators = [identity_generator(model.n),
                  *krylov_response(model.hamiltonian, order)]
    bank = MatrixElementBank(rho, model.hamiltonian, generators)
    bank.solve()
    return len(bank._universe)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_identity_matches_direct_enumeration_on_a_hubbard_plaquette(order):
    model = hubbard((2, 2), t=1.0, U=4.0, periodic=False)
    counts = krylov_word_universe(model.hamiltonian, order, thresholds=(0.0,))
    assert counts[repr(0.0)] == _direct_universe(model, order)


@pytest.mark.parametrize("model_factory", [
    lambda: kitaev_honeycomb(2, 2, periodic=False),
    lambda: tfim(4, J=1.0, h=0.5),
])
def test_identity_matches_direct_enumeration_on_spin_models(model_factory):
    model = model_factory()
    counts = krylov_word_universe(model.hamiltonian, 3, thresholds=(0.0,))
    assert counts[repr(0.0)] == _direct_universe(model, 3)


def test_exponent_range_is_not_off_by_one():
    model = hubbard((2, 2), t=1.0, U=4.0, periodic=False)
    H = model.hamiltonian.to_mv()
    correct = _direct_universe(model, 3)

    def union_to(highest: int) -> int:
        words: set[int] = set()
        power = MV.scalar(H.n, 1.0)
        words |= set(power.terms)
        for _ in range(1, highest + 1):
            power = power * H
            words |= set(power.terms)
        return len(words)

    assert union_to(2 * 3 + 1) == correct
    assert union_to(2 * 3) < correct
    assert union_to(2 * 3 + 2) >= correct


def test_support_sweep_is_nested_and_reported_pencil_is_certified():
    model = hubbard((2, 2), t=1.0, U=4.0, periodic=False)
    rho = ExactMVBackend().state(model.reference, ())
    thresholds = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
    diagnostics = krylov_word_diagnostics(
        model.hamiltonian, rho, 4, thresholds=thresholds)
    rows = diagnostics["by_threshold"]
    counts = [rows[repr(tol)]["word_universe"] for tol in thresholds]
    assert counts == sorted(counts, reverse=True)
    assert rows[repr(REPORTED_THRESHOLD)]["certificate_passed"]


@pytest.fixture()
def committed_record():
    path = (Path(__file__).resolve().parents[1] / "benchmarks"
            / "reference_results" / "krylov_width.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_with(monkeypatch, fresh) -> int:
    import check_krylov_width
    import run_krylov_width

    monkeypatch.setattr(run_krylov_width, "build_record", lambda: fresh)
    return check_krylov_width.main()


def test_gate_accepts_an_unchanged_record(monkeypatch, committed_record):
    assert _gate_with(monkeypatch, copy.deepcopy(committed_record)) == 0


def test_gate_tolerates_round_off_churn_in_raw_counts(
        monkeypatch, committed_record):
    fresh = copy.deepcopy(committed_record)
    for row in fresh["rows"]:
        raw = row["pencil_diagnostics_by_threshold"]["0.0"]
        raw["word_universe"] += 7
        row["word_universe_by_threshold"]["0.0"] += 7
    assert _gate_with(monkeypatch, fresh) == 0


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda r: r["rows"][0].__setitem__(
        "word_universe", r["rows"][0]["word_universe"] + 1),
        id="reported-width-moved"),
    pytest.param(lambda r: r["rows"][0].__setitem__(
        "word_universe_certificate_passed", False),
        id="certificate-failed"),
    pytest.param(lambda r: r.__setitem__("rows", r["rows"][:-1]),
                 id="rung-disappeared"),
    pytest.param(lambda r: r.__setitem__("reported_threshold", 1e-6),
                 id="threshold-moved"),
])
def test_gate_rejects_a_real_regression(
        monkeypatch, committed_record, mutate):
    fresh = copy.deepcopy(committed_record)
    mutate(fresh)
    assert _gate_with(monkeypatch, fresh) == 1
