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


def test_unreduced_arms_close_toward_full_commutation(record):
    """P5's first clause holds for the arms that keep the full register.

    H4 goes 1.713 -> 1.048 and BeH2 8.610 -> 1.071 between ``k = 1`` and
    ``k = n``, which is what "``G(k = n)`` agrees between mappings within
    tie-break noise" asserts.
    """
    for system in record["systems"]:
        groups = record["mapping_spread"][system["system"]]
        payload = groups[str(system["n_qubits"])]
        by_k = payload["by_block_size"]
        widest = max(by_k, key=lambda key: int(key))
        assert by_k[widest]["spread"] < by_k["1"]["spread"]
        assert by_k[widest]["spread"] < 1.1


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
