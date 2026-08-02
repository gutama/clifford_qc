"""Smoke and contract tests for the experimental configuration-packet benchmark."""

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))
try:
    import run_configuration_packets  # noqa: E402
finally:
    sys.path.remove(str(BENCH))


def test_sector_configuration_order_is_complete_deterministic_and_physics_keyed():
    from clifford_qc.models.lattice import hubbard
    from clifford_qc.subspace import occupied_spin_orbitals

    model = hubbard((2, 2))
    configurations = run_configuration_packets.ordered_sector_configurations(model)
    assert len(configurations) == 35      # C(4,2)^2 minus the reference
    assert tuple(occupied_spin_orbitals(model)) not in configurations
    keys = [run_configuration_packets.physics_order_key(
        occupied, occupied_spin_orbitals(model), 4)
        for occupied in configurations]
    assert keys == sorted(keys)


def test_small_staged_benchmark_reports_invariants_and_resource_ledger():
    pytest.importorskip("scipy")
    record = run_configuration_packets.run_experiment(
        shape=2, coarse_size=3, final_size=4, max_packet_support=2)
    assert record["orthogonality_residual"] < 1e-12
    assert record["configuration_leaves"] == 3
    assert len(record["results"]) == 4
    for row in record["results"]:
        assert {"energy", "error", "basis_size", "word_universe",
                "max_generator_support", "max_element_support"} <= set(row)
        assert row["energy"] >= record["exact_ground_energy"] - 1e-9
