"""Contracts for the Phase 2M-A bank storage ledger.

2M-A's failure modes are specific and all quiet. A packed-byte identity that
stops holding silently invalidates the committed back-fill, because that
recovery is a division by the same constant. A ledger whose frontier fields are
zero on every row looks complete and has measured nothing, because a bank built
from a fixed label list never rejects a candidate. A measured byte rate taken
without deduplicating shared word-code boxes overstates what packing could
recover, in the direction that flatters the phase. And a baseline that grades
its own go/no-go turns "here is what 2M-B must beat" into "2M-B passes".

These tests pin the ledger's arithmetic on a live bank, the policy's
consequences, the identity the back-fill depends on, and the checker's ability
to catch each of those going wrong.
"""

from __future__ import annotations

import copy
import json
import sys

import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.subspace.adaptive import run_acase
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.projection import STORAGE_POLICY

from benchmarks.check_bank_storage_ledger import (
    INTERPRETER_DEPENDENT_FIELDS,
    INTERPRETER_RTOL,
    attribution_problems,
    baseline_problems,
    committed_row_problems,
    contract_problems,
    interpreter_field_problems,
    measured_bank_problems,
    rate_problems,
)
from benchmarks.run_bank_storage_ledger import (
    BANK_DOMINANCE_FRACTION,
    PACKED_BYTES_PER_COEFFICIENT,
    REFERENCE,
    REQUIRED_LEDGER_FIELDS,
    ROOT,
    load_config,
)

H4 = ROOT / "benchmarks" / "data" / "h4_sto3g_r0.9.FCIDUMP"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixed_bank() -> MatrixElementBank:
    model = fcidump_model(H4, name="h4")
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    generators = [identity_generator(model.n), *pool[:5]]
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian, generators)
    bank.matrices()
    return bank


@pytest.fixture(scope="module")
def adaptive_bank():
    model = fcidump_model(H4, name="h4")
    rho = ExactMVBackend().state(model.reference, ())
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    result = run_acase(rho, model.hamiltonian, pool,
                       initial=[identity_generator(model.n)], max_size=4)
    return result.bank, result.indices


# --------------------------------------------------------------------------
# The ledger on a live bank
# --------------------------------------------------------------------------

def test_resources_carries_every_declared_ledger_field(fixed_bank):
    resources = fixed_bank.resources()
    missing = [field for field in REQUIRED_LEDGER_FIELDS if field not in resources]
    assert missing == []


def test_packed_identity_is_exact(fixed_bank):
    """The identity the committed back-fill recovers T_coeff through."""
    resources = fixed_bank.resources()
    assert resources["cached_operator_bytes"] == (
        PACKED_BYTES_PER_COEFFICIENT * resources["coefficient_occurrences"])


def test_coefficient_occurrences_is_the_summed_support(fixed_bank):
    rows = fixed_bank._resident_operators()
    assert fixed_bank.resources()["coefficient_occurrences"] == sum(
        row.nnz() for row in rows)


def test_a_fixed_label_bank_has_no_frontier(fixed_bank):
    """Every pair it builds is a retained-block pair, so the fraction is one.

    This is why the record cannot measure eviction headroom on these banks and
    carries adaptive arms beside them.
    """
    resources = fixed_bank.resources()
    assert resources["selection_pairs"] == 0
    assert resources["retained_pair_fraction"] == 1.0
    assert resources["retained_block_pairs"] == resources["complete_block_pairs"]


def test_an_adaptive_bank_retains_a_minority_of_what_it_builds(adaptive_bank):
    """The frontier the ledger exists to baseline, on a live selection run."""
    bank, indices = adaptive_bank
    resources = bank.resources(indices)
    assert resources["selection_pairs"] > 0
    assert resources["retained_pair_fraction"] < 1.0
    assert (resources["retained_block_pairs"] + resources["selection_pairs"]
            == resources["pairs_built"])


def test_retain_all_never_frees_a_row(fixed_bank):
    """The policy's defining consequence, not an assumption about it."""
    resources = fixed_bank.resources()
    assert resources["storage_policy"] == STORAGE_POLICY == "retain_all"
    assert resources["evicted_rows"] == 0
    assert resources["recomputed_rows"] == 0
    assert resources["spill_bytes"] == 0
    assert (resources["peak_resident_operator_rows"]
            == resources["resident_operator_rows"])


def test_peak_resident_rows_is_counted_not_derived():
    """It tracks growth as it happens, so 2M-C's policies inherit a live field."""
    model = fcidump_model(H4, name="h4")
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian,
                             [identity_generator(model.n), *pool[:3]])
    assert bank.resources()["peak_resident_operator_rows"] == 0
    bank.matrices()
    grown = bank.resources()["peak_resident_operator_rows"]
    assert grown == bank.resources()["resident_operator_rows"] > 0


def test_measured_bytes_exceed_the_packed_model(fixed_bank):
    """The packing hypothesis, measured on a real bank rather than assumed."""
    measured = fixed_bank.measured_storage_bytes()
    assert measured["measured_operator_bytes"] > measured["packed_operator_bytes"]
    assert measured["packing_headroom"] > 1.0
    assert (measured["measured_container_bytes"] + measured["measured_boxed_bytes"]
            == measured["measured_operator_bytes"])


def test_boxed_slots_account_for_two_per_coefficient(fixed_bank):
    """Deduplicated plus shared covers every key and value, once each."""
    measured = fixed_bank.measured_storage_bytes()
    assert (measured["distinct_boxed_objects"] + measured["shared_boxed_slots"]
            == 2 * measured["coefficient_occurrences"])


def test_word_code_boxes_are_shared_across_rows(fixed_bank):
    """The reason the walk deduplicates: a per-row sum counts one box many times.

    The memoized word product hands every row carrying a word the same integer
    object, so a bank of any size has shared slots. Without the deduplication
    the measured rate -- and with it the packing headroom 2M-B is gated on --
    comes out too high.
    """
    measured = fixed_bank.measured_storage_bytes()
    assert measured["shared_boxed_slots"] > 0
    naive = 0
    for row in fixed_bank._resident_operators():
        container, boxes = row.boxed_storage()
        naive += container + sum(sys.getsizeof(box) for box in boxes)
    assert naive > measured["measured_operator_bytes"]


def test_boxed_storage_reports_two_boxes_per_term():
    mv = MV.from_terms(2, {"XY": 1.0, "ZZ": 2.0j})
    size, boxes = mv.boxed_storage()
    assert size > 0
    assert len(boxes) == 2 * mv.nnz()


# --------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------

def test_config_declares_every_gate_with_its_statement():
    gates = load_config()["gates"]
    for gate, statement in (
            ("packed_identity_is_exact", "packed_identity_statement"),
            ("measured_rate_is_deduplicated", "measured_rate_statement"),
            ("frontier_fields_are_exercised", "frontier_statement"),
            ("no_cost_fields", "no_cost_fields_statement"),
            ("no_go_no_go_verdict", "no_go_no_go_statement")):
        assert gates[gate] is True
        assert gates[statement].strip()


def test_config_and_producer_agree_on_the_packed_model():
    config = load_config()
    assert config["packed_bytes_per_coefficient"] == PACKED_BYTES_PER_COEFFICIENT
    assert config["storage_policy"] == STORAGE_POLICY


def test_config_rejects_a_divergent_packed_model(tmp_path):
    config = load_config()
    config["packed_bytes_per_coefficient"] = 32
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="packed model"):
        load_config(path)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def test_committed_record_passes_every_contract(record):
    assert contract_problems(record) == []


def test_record_prices_a_live_frontier(record):
    assert any(row["selection_pairs"] > 0 for row in record["measured_banks"])


def test_record_withholds_the_go_no_go(record):
    assert record["baseline"]["verdict"] == "baseline_only_no_go_no_go_evaluated"
    assert record["baseline"]["why_no_verdict"].strip()


def test_committed_rows_retain_a_small_minority_of_their_pairs(record):
    """PLAN.md's 3.0--8.8%, re-derived per row rather than compared to the text."""
    for row in record["committed_records"]:
        basis = row["subspace_size_m"]
        assert row["retained_block_pairs"] == basis * (basis + 1) // 2
        assert 0.0 < row["retained_pair_fraction"] < 0.1


def test_the_bank_dominates_peak_on_the_large_committed_rows(record):
    """Phase 2M's premise: the bank is the allocation that failed."""
    dominated = record["peak_rss_attribution"]["records_where_the_bank_dominates_peak"]
    assert "beh2_stretched_results.json" in dominated
    assert "h2o_results.json" in dominated
    for row in record["peak_rss_attribution"]["rows"]:
        if row["record"] in dominated:
            assert row["attributed_fraction_of_peak"] >= BANK_DOMINANCE_FRACTION


def test_attribution_uses_the_conservative_rate(record):
    """The minimum, because the committed banks are larger than any measured here."""
    rate = record["measured_rate"]
    assert (record["peak_rss_attribution"]["bytes_per_coefficient_used"]
            == rate["minimum_bytes_per_coefficient"])
    assert (rate["minimum_bytes_per_coefficient"]
            <= rate["pooled_bytes_per_coefficient"]
            <= rate["maximum_bytes_per_coefficient"])


def test_word_product_cache_prices_its_ceiling_not_its_fill(record):
    """Hits, misses and live entries are process history and must not be published."""
    cache = record["word_product_cache"]
    assert cache["maxsize"] > 0
    assert cache["lower_bound_ceiling_bytes"] == (
        cache["lower_bound_bytes_per_entry"] * cache["maxsize"])
    assert not {"hits", "misses", "live_entries"} & set(cache)


# --------------------------------------------------------------------------
# The checker catches each failure
# --------------------------------------------------------------------------

def _broken(record: dict, mutate) -> dict:
    copied = copy.deepcopy(record)
    mutate(copied)
    return copied


@pytest.mark.parametrize("name,mutate,gate", [
    ("packed identity",
     lambda r: r["measured_banks"][0].update(
         coefficient_occurrences=r["measured_banks"][0]["coefficient_occurrences"] + 1),
     measured_bank_problems),
    ("pair arithmetic",
     lambda r: r["measured_banks"][0].update(selection_pairs=7),
     measured_bank_problems),
    ("eviction under retain_all",
     lambda r: r["measured_banks"][0].update(evicted_rows=1),
     measured_bank_problems),
    ("a freed row under retain_all",
     lambda r: r["measured_banks"][0].update(peak_resident_operator_rows=1),
     measured_bank_problems),
    ("boxed slots that do not account",
     lambda r: r["measured_banks"][0].update(shared_boxed_slots=0),
     measured_bank_problems),
    ("a back-filled row that was transcribed",
     lambda r: r["committed_records"][1].update(pairs_built=999),
     committed_row_problems),
    ("a hand-fitted retained fraction",
     lambda r: r["committed_records"][1].update(retained_pair_fraction=0.5),
     committed_row_problems),
    ("a pooled rate that is not the quotient",
     lambda r: r["measured_rate"].update(pooled_coefficient_occurrences=1),
     rate_problems),
    ("an attribution at a flattering rate",
     lambda r: r["peak_rss_attribution"].update(bytes_per_coefficient_used=1e4),
     attribution_problems),
    ("a dominance verdict written by hand",
     lambda r: r["peak_rss_attribution"]["rows"][-1].update(bank_dominates_peak=True),
     attribution_problems),
    ("a baseline that grades itself",
     lambda r: r["baseline"].update(verdict="go"),
     baseline_problems),
    ("a retained-pair range that is not the range",
     lambda r: r["baseline"].update(committed_retained_pair_fraction_range=[0.0, 1.0]),
     baseline_problems),
])
def test_checker_catches(record, name, mutate, gate):
    assert gate(_broken(record, mutate)) != [], name


def test_checker_catches_a_no_frontier_record(record):
    """Consistently zeroed, so only the frontier gate itself can catch it."""
    broken = copy.deepcopy(record)
    for row in broken["measured_banks"]:
        row["selection_pairs"] = 0
        row["pairs_built"] = row["retained_block_pairs"]
        row["retained_pair_fraction"] = 1.0
    problems = measured_bank_problems(broken)
    assert any("rejected a candidate" in problem for problem in problems)


def test_checker_catches_a_cost_field(record):
    assert contract_problems(_broken(record, lambda r: r.update(k_star=4))) != []


def test_checker_catches_an_upgraded_evidence_tier(record):
    assert contract_problems(_broken(record, lambda r: r.update(evidence_tier="exact"))) != []


def test_checker_survives_a_malformed_record(record):
    """A guarded path reports the record rather than raising a traceback."""
    problems = contract_problems(_broken(record, lambda r: r.update(measured_banks="x")))
    assert any("malformed record" in problem for problem in problems)


# --------------------------------------------------------------------------
# The interpreter tolerance
# --------------------------------------------------------------------------

def test_measured_bytes_are_excluded_from_the_exact_comparison():
    """Every published measured-byte field is declared interpreter-dependent."""
    for field in ("measured_operator_bytes", "measured_bytes_per_coefficient",
                  "packing_headroom", "attributed_row_bytes",
                  "pooled_bytes_per_coefficient"):
        assert field in INTERPRETER_DEPENDENT_FIELDS


def test_interpreter_tolerance_admits_layout_drift_and_rejects_a_regression(record):
    inside = _broken(record, lambda r: r["measured_banks"][0].update(
        measured_operator_bytes=int(
            r["measured_banks"][0]["measured_operator_bytes"]
            * (1 + INTERPRETER_RTOL / 4))))
    assert interpreter_field_problems(record, inside) == []
    outside = _broken(record, lambda r: r["measured_banks"][0].update(
        measured_operator_bytes=int(
            r["measured_banks"][0]["measured_operator_bytes"]
            * (1 + 10 * INTERPRETER_RTOL))))
    assert interpreter_field_problems(record, outside) != []
