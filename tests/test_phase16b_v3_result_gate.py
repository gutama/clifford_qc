"""What the Phase 16B v3 result gate refuses.

Two holes found in review, both of which let a record pass that should not
have.

The first: the gate audited whatever cells the record happened to carry, which
establishes that the present ones are well formed and nothing at all about the
absent ones. Deleting one budget cell from the committed record left it passing
as "complete". The declared sweep is a Cartesian product written down in the
frozen config, so the comparison is against that product.

The second, and the more serious one: the config freezes a continuity rule
requiring the ungrouped scheme to reproduce v2's qualifying ordering, the run
did not satisfy it, and the gate stamped the record OK anyway. An explanation of
why a preregistered check failed does not convert it into a check that passed.
The gate now derives conformance from the comparisons the record carries --
never from its own summary flag, which a bad record could simply set -- requires
a run that failed the rule to say so, and withholds the conforming stamp while
the deviation stands.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks import check_phase16b_v3_feasibility as gate

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "benchmarks" / "configs" / "phase16b_v3_feasibility.json"
RECORD = ROOT / "benchmarks" / "reference_results" / "phase16b_v3_feasibility.json"

pytestmark = pytest.mark.skipif(not RECORD.exists(), reason="no committed v3 record")


@pytest.fixture(scope="module")
def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def first_scheme(record):
    for name, entry in record["instances"].items():
        for scheme, scheme_entry in entry.get("schemes", {}).items():
            return name, scheme, scheme_entry
    raise AssertionError("the committed record carries no scheme results")


# ------------------------------------------------------------------ coverage

def test_committed_record_carries_the_whole_declared_sweep(config, record):
    declared = gate.expected_cells(config)
    for name, entry in record["instances"].items():
        for scheme, scheme_entry in entry.get("schemes", {}).items():
            assert gate.recorded_cells(scheme_entry) == declared, f"{name}/{scheme}"


def test_expected_cells_is_the_declared_cartesian_product(config):
    cells = gate.expected_cells(config)
    budgets = len(config["cost_model"]["budget_grid_decades"])
    regularizers = len(config["regularizers"])
    sizes = sum(len(config["basis_sizes_by_arm"][arm]) for arm in gate.priceable_arms(config))
    assert len(cells) == sizes * regularizers * budgets
    assert "exact_diag" not in {arm for arm, _, _, _ in cells}


def test_a_deleted_budget_cell_is_caught(config, record):
    """The reviewer's exact reproduction: one cell removed, record still 'complete'."""
    mutated = copy.deepcopy(record)
    cells = mutated["instances"]["h4_chain_075"]["schemes"]["qwc"]["arms"]
    del cells["acase_control"]["budget"]["ridge"]["4"]["1e20"]
    problems: list[str] = []
    gate.check_coverage(config, "h4_chain_075", "qwc",
                        mutated["instances"]["h4_chain_075"]["schemes"]["qwc"], problems)
    assert any("declared budget cells are absent" in p for p in problems)


def test_a_dropped_basis_size_is_caught(config, record):
    mutated = copy.deepcopy(record)
    scheme_entry = mutated["instances"]["h4_chain_075"]["schemes"]["qwc"]
    del scheme_entry["arms"]["rt_unitary"]["sizes"]["12"]
    del scheme_entry["arms"]["rt_unitary"]["budget"]["hard_truncation"]["12"]
    problems: list[str] = []
    gate.check_coverage(config, "h4_chain_075", "qwc", scheme_entry, problems)
    assert any("no per-size record" in p for p in problems)


def test_a_cell_outside_the_declared_sweep_is_caught(config, record):
    mutated = copy.deepcopy(record)
    scheme_entry = mutated["instances"]["h4_chain_075"]["schemes"]["qwc"]
    scheme_entry["arms"]["rt_unitary"]["budget"]["hard_truncation"]["12"]["1e21"] = {}
    problems: list[str] = []
    gate.check_coverage(config, "h4_chain_075", "qwc", scheme_entry, problems)
    assert any("outside the declared sweep" in p for p in problems)


# --------------------------------------------------------------- conformance

def test_committed_record_declares_its_own_deviation(config, record):
    problems: list[str] = []
    status = gate.check_conformance(record, config, problems)
    assert problems == []
    assert status == "deviating"
    assert record["protocol_conformance"]["continuity_rule_satisfied"] is False
    assert record["protocol_conformance"]["deviation"]["cause"]


def test_a_record_claiming_conformance_it_does_not_have_is_caught(config, record):
    """The failure mode that matters: an explanation dressed up as a pass."""
    mutated = copy.deepcopy(record)
    mutated["protocol_conformance"]["status"] = "conforming"
    mutated["protocol_conformance"]["continuity_rule_satisfied"] = True
    problems: list[str] = []
    gate.check_conformance(mutated, config, problems)
    assert any("continuity rule was satisfied" in p for p in problems)
    assert any("protocol status is 'conforming'" in p for p in problems)


def test_conformance_is_derived_from_the_comparisons_not_the_summary_flag(config, record):
    """Flipping only the summary flag must not buy a pass."""
    mutated = copy.deepcopy(record)
    mutated["continuity_check"]["orderings_agree"] = True
    problems: list[str] = []
    status = gate.check_conformance(mutated, config, problems)
    assert status == "deviating"
    assert any("orderings_agree disagrees" in p for p in problems)


def test_a_flipped_per_instance_agree_flag_is_caught(config, record):
    mutated = copy.deepcopy(record)
    name = next(iter(mutated["continuity_check"]["instances"]))
    mutated["continuity_check"]["instances"][name]["agree"] = True
    problems: list[str] = []
    gate.check_conformance(mutated, config, problems)
    assert any("agree flag disagrees" in p for p in problems)


def test_a_missing_conformance_block_is_caught(config, record):
    mutated = copy.deepcopy(record)
    del mutated["protocol_conformance"]
    problems: list[str] = []
    assert gate.check_conformance(mutated, config, problems) == "unstated"
    assert any("no protocol_conformance block" in p for p in problems)


def test_a_deviation_without_a_cause_is_caught(config, record):
    mutated = copy.deepcopy(record)
    mutated["protocol_conformance"]["deviation"] = {}
    problems: list[str] = []
    gate.check_conformance(mutated, config, problems)
    assert any("must carry a deviation" in p for p in problems)


def test_a_restated_continuity_rule_is_caught(config, record):
    """The frozen rule is the authority, not the record's copy of it."""
    mutated = copy.deepcopy(record)
    mutated["protocol_conformance"]["continuity_rule"] = "ungrouped may differ from v2"
    problems: list[str] = []
    gate.check_conformance(mutated, config, problems)
    assert any("stated continuity rule is not the frozen one" in p for p in problems)
