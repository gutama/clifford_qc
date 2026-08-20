"""Contracts for the R3 protocol-axis (``mapping x k``) record."""

import copy
import json

import pytest

pytest.importorskip("stim")

from benchmarks.check_protocol_axis import contract_problems  # noqa: E402
from benchmarks.run_protocol_axis import REFERENCE  # noqa: E402


@pytest.fixture(scope="module")
def record():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_committed_record_passes_its_contract(record):
    assert contract_problems(record) == []


def test_k1_column_reproduces_the_frozen_mapping_axis(record):
    """The condition that makes this an extension of R2b, not a new experiment."""
    for system in record["systems"]:
        for arm in system["arms"]:
            k1 = _rung(arm, 1)
            assert k1["settings"] == arm["frozen_qwc_settings"], (
                system["system"], arm["mapping"])


def _rung(arm, block_size):
    """Select a rung by its ``block_size``, never by position.

    Rung order is an ordering choice of the producer, not part of the contract
    these tests exercise. Indexing by position would make them fail for the
    wrong reason -- or silently stop testing anything -- if that order changed.
    """
    return next(row for row in arm["rungs"] if row["block_size"] == block_size)


def _widest_rung(arm):
    return max(arm["rungs"], key=lambda row: row["block_size"])


def test_contract_catches_a_drifted_k1_column(record):
    broken = copy.deepcopy(record)
    _rung(broken["systems"][0]["arms"][0], 1)["settings"] += 1
    assert any("R2b froze" in problem for problem in contract_problems(broken))


def test_contract_catches_a_non_monotone_k_sweep(record):
    broken = copy.deepcopy(record)
    arm = broken["systems"][0]["arms"][0]
    _widest_rung(arm)["settings"] = _rung(arm, 1)["settings"] + 1
    assert any("non-increasing" in problem for problem in contract_problems(broken))


def test_contract_catches_entangling_gates_at_qwc(record):
    broken = copy.deepcopy(record)
    k1 = _rung(broken["systems"][0]["arms"][0], 1)
    k1["resource_metrics"]["N_2q"]["sum"] = 7
    assert any("two-qubit gates" in problem for problem in contract_problems(broken))


def test_contract_catches_an_overstated_evidence_tier(record):
    broken = copy.deepcopy(record)
    broken["protocol"]["evidence_tier"] = "exact"
    assert any("evidence tier" in problem for problem in contract_problems(broken))


def test_contract_catches_a_mixed_width_spread_group(record):
    broken = copy.deepcopy(record)
    system = broken["systems"][0]["system"]
    group = next(iter(broken["mapping_spread"][system].values()))
    group["arms"] = [arm["mapping"] for arm in broken["systems"][0]["arms"]]
    assert any("different" in problem for problem in contract_problems(broken))


def test_reduced_arms_are_not_compared_against_full_width_arms(record):
    """A ``+2q`` arm measures two fewer qubits, so its ``k = n`` is narrower."""
    for system in record["systems"]:
        widths = {arm["mapping"]: arm["measured_qubits"] for arm in system["arms"]}
        for width, payload in record["mapping_spread"][system["system"]].items():
            assert {widths[name] for name in payload["arms"]} == {int(width)}


# P5's first clause -- "G(k = n) agrees between mappings within tie-break
# noise" -- is a claim about a bank, not about a Hamiltonian, and the two H4
# banks in this record disagree about it. Both are listed so that a change to
# either is a visible edit rather than a loosened threshold.
FULL_WIDTH_CLOSURE = {
    "h4": {"k1": 1.713, "kn": 1.048, "closes_to_noise": True},
    "h4_converged": {"k1": 1.713, "kn": 1.200, "closes_to_noise": False},
    "beh2": {"k1": 8.610, "kn": 1.071, "closes_to_noise": True},
}
TIE_BREAK_NOISE = 1.1


def test_unreduced_arms_narrow_toward_full_commutation(record):
    """Every bank's full-width spread is narrower at ``k = n`` than at ``k = 1``.

    This much of P5's first clause survives everywhere in the record. Whether
    the endpoint reaches tie-break noise is a separate question, and the test
    below is where the two H4 banks part company.
    """
    for system in record["systems"]:
        key = system["system"]
        by_k = record["mapping_spread"][key][str(system["n_qubits"])]["by_block_size"]
        widest = max(by_k, key=lambda name: int(name))
        assert by_k[widest]["spread"] < by_k["1"]["spread"], key
        assert by_k["1"]["spread"] == pytest.approx(
            FULL_WIDTH_CLOSURE[key]["k1"], abs=1e-3
        )
        assert by_k[widest]["spread"] == pytest.approx(
            FULL_WIDTH_CLOSURE[key]["kn"], abs=1e-3
        )


def test_the_converged_h4_bank_does_not_close_to_tie_break_noise(record):
    """The second counter-example, and the one that costs P5 its first clause.

    On the budget-8 H4 bank the full-width arms reach 1.048 at ``k = 8``, which
    is what "agrees within tie-break noise" means. Carrying the same greedy on
    the same Hamiltonian to its own stopping threshold does not reproduce that:
    the converged bank narrows to 1.117 at ``k = 4`` and then *widens* to 1.200
    at ``k = 8``, missing the endpoint the clause predicts and missing it in the
    same non-monotone way the reduced arms already did.

    So the clause is not a property of H4. It is a property of one subspace on
    H4, and this record contains a second subspace on the same Hamiltonian that
    falsifies it.
    """
    for key, expected in FULL_WIDTH_CLOSURE.items():
        by_k = record["mapping_spread"][key]["8"]["by_block_size"]
        widest = max(by_k, key=lambda name: int(name))
        assert (by_k[widest]["spread"] < TIE_BREAK_NOISE) is expected[
            "closes_to_noise"
        ], key

    converged = record["mapping_spread"]["h4_converged"]["8"]["by_block_size"]
    assert converged["4"]["spread"] < converged["2"]["spread"]
    assert converged["8"]["spread"] > converged["4"]["spread"]


def test_reduced_arms_do_not_close_and_h4_widens(record):
    """The counter-example, pinned so it cannot vanish silently.

    P5 also predicts the gap closes *monotonically*. It does not. On H4 the two
    ``+2q`` arms narrow from 1.148 at ``k = 1`` to 1.054 at ``k = 4`` and then
    *widen* to 1.181 at ``k = 6`` -- the second-largest spread of any rung. BeH2
    at ``n = 8`` is non-monotone too, touching 1.0 at ``k = 2`` and ``k = 4``
    before rising to 1.071.

    So the prediction's first clause survives on the full-width arms and its
    monotonicity clause does not survive at all. Pinning the shape here keeps a
    later refactor from quietly turning the negative into a positive.
    """
    h4_reduced = record["mapping_spread"]["h4"]["6"]["by_block_size"]
    assert h4_reduced["4"]["spread"] < h4_reduced["1"]["spread"]
    assert h4_reduced["6"]["spread"] > h4_reduced["4"]["spread"]
    assert h4_reduced["6"]["spread"] > h4_reduced["1"]["spread"]

    beh2_full = record["mapping_spread"]["beh2"]["8"]["by_block_size"]
    assert beh2_full["2"]["spread"] == 1.0
    assert beh2_full["8"]["spread"] > beh2_full["4"]["spread"]
