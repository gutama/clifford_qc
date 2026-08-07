"""Smoke tests for the Phase 10 primary-system benchmark driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
DRIVER = ROOT / "benchmarks" / "run_phase10_hybrid.py"


def _driver():
    spec = importlib.util.spec_from_file_location("run_phase10_hybrid", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_primary_system_set_matches_phase10d():
    module = _driver()
    assert module.PRIMARY_SYSTEMS == (
        "hubbard_2x2",
        "hubbard_2x3",
        "h4_equilibrium",
        "h4_stretched",
        "fcidump_h4_equilibrium",
    )


def test_hubbard_smoke_has_honest_controls_and_matched_budget():
    module = _driver()
    record = module.run_system(
        "hubbard_2x2", shots=16, seed=3, max_size=2,
        max_generators=12, max_support=16, max_packet_support=8)

    assert record["settings"]["sampling_evidence"] == "oracle"
    assert record["full_qsci"]["energy"] >= record["exact_energy"] - 1e-9
    assert record["bare_acase"]["energy"] >= record["exact_energy"] - 1e-9
    assert set(record["phase9_controls"]) == {
        "family_closure", "budget_selected_ci"}
    assert record["phase9_controls"]["budget_selected_ci"][
        "determinant_count"] <= 3
    assert record["family_projection"]["incremental_word_universe"] >= 0
    for arm in record["hybrid_arms"]:
        assert arm["basis_size"] <= 3
        assert arm["energy"] >= record["exact_energy"] - 1e-9
