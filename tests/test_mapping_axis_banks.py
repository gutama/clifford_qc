"""The declared banks are re-derivable, and the second one is priceable.

``mapping_axis.json`` freezes each system's subspace as a list of labels. For
most systems those labels are copied from an earlier frozen record, which is
its own provenance. ``h4_converged`` instead cites a *rule* -- grow A-CASE
until its own lowering threshold stops it -- so the freeze is only as good as
a test that re-runs that rule and gets the same labels back. That is what this
module is.

It also pins the two facts that make the bank worth declaring at all: it clears
the 1.6 mHa target that the frozen budget-8 bank misses, and lifting the budget
cap that produced it leaves the BeH2 bank untouched, so BeH2's existing prices
still describe the bank they were measured on.
"""

import json

import pytest

from benchmarks.run_clifford_hierarchy import ACCURACY_TARGET_MILLIHARTREE
from benchmarks.run_mapping_axis import (
    _build_model,
    _raw_pool,
    _selected_generators,
    load_config,
)
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.subspace import run_acase
from clifford_qc.subspace.elements import MatrixElementBank


# Larger than any pool this benchmark declares, so no run stops on the budget.
UNBOUND_BUDGET = 64


def _spec(key: str) -> dict:
    config = load_config()
    return next(spec for spec in config["systems"] if spec["key"] == key)


def _grow(key: str, budget: int):
    """Run the declared growth rule on one system and return its outcome."""
    model, _ = _build_model(_spec(key))
    sector = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact = float(sector.ground_state(model.hamiltonian, k=1, method="dense")[0][0])
    reference = ExactMVBackend().state(model.reference, ())
    run = run_acase(
        reference,
        model.hamiltonian,
        _raw_pool(model),
        max_size=budget,
        exact_ground_energy=exact,
    )
    return run, exact


def test_the_converged_h4_labels_are_what_the_declared_rule_produces():
    spec = _spec("h4_converged")
    run, exact = _grow("h4_converged", UNBOUND_BUDGET)

    # The rule is "stopped by its own threshold", so a run that merely ran out
    # of budget would reproduce the labels while falsifying the declaration.
    assert run.stopped_reason == "predicted lowering below threshold"
    assert list(run.result.basis_labels) == list(spec["selected_labels"])
    assert float(run.result.ground_energy) == pytest.approx(
        spec["expected_selected_energy"], abs=spec["selected_energy_tolerance"]
    )
    assert (float(run.result.ground_energy) - exact) * 1e3 == pytest.approx(
        0.765862, abs=1e-6
    )


def test_the_converged_h4_bank_extends_the_frozen_one_rather_than_replacing_it():
    frozen = _spec("h4")["selected_labels"]
    converged = _spec("h4_converged")["selected_labels"]
    # Same instance, same greedy, same order of additions -- the budget-8 bank
    # is a prefix of the converged one. That is what lets the record's two H4
    # rows be read against each other instead of as unrelated systems.
    assert converged[: len(frozen)] == frozen
    assert len(converged) == 15


def test_the_converged_bank_clears_the_target_and_the_frozen_bank_does_not():
    model, _ = _build_model(_spec("h4"))
    sector = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact = float(sector.ground_state(model.hamiltonian, k=1, method="dense")[0][0])
    reference = ExactMVBackend().state(model.reference, ())

    biases = {}
    for key in ("h4", "h4_converged"):
        _, selected = _selected_generators(model, _spec(key))
        bank = MatrixElementBank(reference, model.hamiltonian, selected)
        biases[key] = abs(float(bank.solve().ground_energy) - exact) * 1e3

    assert biases["h4"] > ACCURACY_TARGET_MILLIHARTREE
    assert biases["h4_converged"] < ACCURACY_TARGET_MILLIHARTREE
    # Not a marginal pass: the whole point of choosing the converged bank over
    # the smallest one that happens to clear is that it clears with room.
    assert biases["h4_converged"] < 0.5 * ACCURACY_TARGET_MILLIHARTREE


def test_lifting_the_budget_leaves_the_beh2_bank_untouched():
    spec = _spec("beh2")
    capped, _ = _grow("beh2", 8)
    lifted, _ = _grow("beh2", UNBOUND_BUDGET)

    # BeH2 already stopped on its own threshold under the budget-8 cap, so the
    # lifted rule is a no-op for it. If it were not, BeH2's frozen prices would
    # no longer describe the bank the config declares.
    assert capped.stopped_reason == lifted.stopped_reason
    assert list(capped.result.basis_labels) == list(lifted.result.basis_labels)
    assert list(lifted.result.basis_labels) == list(spec["selected_labels"])
    assert float(lifted.result.ground_energy) == pytest.approx(
        float(capped.result.ground_energy), abs=1e-12
    )


def test_every_declared_system_cites_a_selection_source_or_a_rule():
    for spec in load_config()["systems"]:
        assert spec.get("selection_source") or spec.get("selection_rule"), spec["key"]


def test_the_protocol_axis_grid_prices_the_systems_the_mapping_axis_declares():
    from benchmarks.run_protocol_axis import CONFIG

    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    declared = {spec["key"] for spec in load_config()["systems"]}
    assert set(protocol["systems"]) <= declared
    assert "h4_converged" in protocol["systems"]
