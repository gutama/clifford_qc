"""The Krylov word-universe identity, checked against direct enumeration.

``benchmarks/run_krylov_width.py`` fills in the ladder's missing measurement
width for the Krylov arm by replacing the O(M^2) enumeration of element
operators with the 2m+2 powers of H. That shortcut is the only reason those
numbers exist, and a wrong exponent range would silently produce a plausible
count -- so it is checked here against the bank that the ladder itself uses,
on cases small enough for the quadratic route to finish.
"""

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
from run_krylov_width import krylov_word_universe  # noqa: E402


def _direct_universe(model, order: int) -> int:
    """|W| the way the ladder computes it: every A_i^dag A_j and A_i^dag H A_j."""
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
    """A short range under-counts and a long one over-counts.

    The union runs to H^(2m+1) because the widest element is A_m^dag H A_m.
    Both neighbours are checked so the test fails if the bound drifts either
    way rather than only in the direction a typo happens to take.
    """
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


@pytest.fixture()
def committed_record():
    path = (Path(__file__).resolve().parents[1] / "benchmarks"
            / "reference_results" / "krylov_width.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_with(monkeypatch, fresh) -> int:
    """Run the CI gate against a supplied 'fresh' record."""
    import check_krylov_width
    import run_krylov_width

    monkeypatch.setattr(run_krylov_width, "build_record", lambda: fresh)
    return check_krylov_width.main()


def test_gate_accepts_an_unchanged_record(monkeypatch, committed_record):
    assert _gate_with(monkeypatch, copy.deepcopy(committed_record)) == 0


def test_gate_tolerates_round_off_churn_in_the_raw_counts(
        monkeypatch, committed_record):
    """The reason this gate exists instead of a bit-for-bit diff.

    Two runs disagree about how many round-off-level words H^17 carries, so a
    byte comparison fails for a reason that means nothing. Only the reported
    count is a claim, and only it is gated.
    """
    fresh = copy.deepcopy(committed_record)
    for row in fresh["rows"]:
        row["word_universe_by_threshold"]["0.0"] += 7
        row["word_universe_by_threshold"]["1e-12"] -= 3
    assert _gate_with(monkeypatch, fresh) == 0


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda r: r["rows"][0].__setitem__(
        "word_universe", r["rows"][0]["word_universe"] + 1),
        id="reported-width-moved"),
    pytest.param(lambda r: r.__setitem__("rows", r["rows"][:-1]),
                 id="rung-disappeared"),
    pytest.param(lambda r: r.__setitem__("reported_threshold", 1e-6),
                 id="threshold-moved"),
])
def test_gate_rejects_a_real_regression(monkeypatch, committed_record, mutate):
    fresh = copy.deepcopy(committed_record)
    mutate(fresh)
    assert _gate_with(monkeypatch, fresh) == 1


def test_thresholding_never_grows_the_universe():
    """Pruning can only remove words, and the sweep must be monotone.

    The Krylov count depends on the threshold precisely because round-off
    inflates it; a sweep that was not monotone would mean the pruning is
    interacting with the accumulation in a way the record does not describe.
    """
    model = hubbard((2, 2), t=1.0, U=4.0, periodic=False)
    thresholds = (0.0, 1e-14, 1e-12, 1e-10, 1e-8)
    counts = krylov_word_universe(model.hamiltonian, 4, thresholds=thresholds)
    values = [counts[repr(tol)] for tol in thresholds]
    assert values == sorted(values, reverse=True)
