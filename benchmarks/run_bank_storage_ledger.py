"""Phase 2M-A -- the matrix-element bank's storage ledger and frozen baseline.

Phase 2 made repeated adaptive solves credible by retaining every built pair,
and the larger molecular records then exposed the other side of that decision:
two OOM failures whose cause PLAN.md section 5 records as a calibrated
extrapolation rather than a measurement.  2M-B proposes packed storage and 2M-C
proposes eviction.  Neither can be gated without the baseline they are supposed
to beat, and the specific number the plan carries -- the byte ratio between the
packed representation and the ``dict[int, complex]`` in front of it -- is
written there as a hypothesis to measure.

**Three populations, because they answer different questions.**  The frozen
mapping-axis banks give the bytes-per-coefficient rate over two decades of
coefficient count, and they are rebuilt from scratch by the checker.  Two small
adaptive arms give the retained-against-frontier ratio, which a fixed label list
cannot produce because it never rejects a candidate.  The seven committed
molecular records supply the scale that actually failed, and they are read
rather than rerun: ``T_coeff`` comes back from ``cached_operator_bytes / 24``,
exactly as the plan specifies, and nothing else about them is touched.

**What the record is for.**  The measured rate applied to a committed row's
recovered ``T_coeff`` calibrates how much of that row's already-recorded peak
RSS may be attributed to resident coefficient payload. It is an extrapolation
from smaller live banks, interpreted beside the independent OOM witnesses, not
a direct object-graph measurement of the large run.

    python benchmarks/run_bank_storage_ledger.py
    python benchmarks/check_bank_storage_ledger.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clifford_qc.backends import ExactMVBackend
from clifford_qc.pauli_kernel import _word_mul_unchecked
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.adaptive import run_acase
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.projection import STORAGE_POLICY

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_mapping_axis import (
        _build_model,
        _raw_pool,
        _selected_generators,
        load_config as load_mapping_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_mapping_axis import (
        _build_model,
        _raw_pool,
        _selected_generators,
        load_config as load_mapping_config,
    )

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "bank_storage_ledger.json"
REFERENCE = HERE / "reference_results" / "bank_storage_ledger.json"
SCHEMA = "clifford_qc.bank_storage_ledger.v1"
CONFIG_SCHEMA = "clifford_qc.bank_storage_ledger_config.v1"

# The packed model ``MV.memory_estimate`` implements, and therefore what
# ``cached_operator_bytes`` has always meant. Declared here so a record built
# after that estimator changed fails loudly instead of back-filling the
# committed rows through a divisor that no longer describes them.
PACKED_BYTES_PER_COEFFICIENT = 24

# Ledger fields ``resources()`` must carry for this record to be assemblable.
# Named rather than read opportunistically: a missing field would otherwise
# surface as a null in the record and a confusing comparison later.
# The share of a committed row's recorded peak RSS above which the retained
# coefficient rows are called the dominant allocation. Declared here, with its
# first use, rather than chosen after reading the rows: a majority is the
# weakest threshold that supports the word "dominates", and the record carries
# every fraction so a reader can apply their own.
BANK_DOMINANCE_FRACTION = 0.5

REQUIRED_LEDGER_FIELDS = (
    "coefficient_occurrences",
    "resident_word_universe",
    "coefficient_reuse",
    "coefficient_occurrences_per_selected_element_word",
    "storage_policy",
    "resident_operator_rows",
    "peak_resident_operator_rows",
    "retained_block_pairs",
    "complete_block_pairs",
    "selection_pairs",
    "retained_pair_fraction",
    "evicted_rows",
    "recomputed_rows",
    "spill_bytes",
)

# Every field this record reads out of a committed molecular record. It reads
# these and writes none of them; see the config's ``preserved_results``.
COMMITTED_FIELDS = (
    "adaptive_cached_operator_bytes",
    "adaptive_pairs_built",
    "subspace_size_m",
    "element_word_universe",
    "adaptive_peak_rss_bytes",
    "adaptive_peak_rss_delta_bytes",
)


def load_config(path: Path | None = None) -> dict:
    """Read the frozen ledger configuration and check its own declarations."""
    config = json.loads((CONFIG if path is None else path).read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected config schema {config.get('schema')!r}")
    if config.get("packed_bytes_per_coefficient") != PACKED_BYTES_PER_COEFFICIENT:
        raise ValueError(
            "the config's packed model disagrees with this producer's: it "
            f"declares {config.get('packed_bytes_per_coefficient')!r} bytes per "
            f"coefficient against {PACKED_BYTES_PER_COEFFICIENT}. The committed "
            "back-fill divides by that constant, so the two cannot differ"
        )
    if config.get("storage_policy") != STORAGE_POLICY:
        raise ValueError(
            f"the config declares policy {config.get('storage_policy')!r} but the "
            f"bank implements {STORAGE_POLICY!r}")
    for gate, statement in (
            ("packed_identity_is_exact", "packed_identity_statement"),
            ("measured_rate_is_deduplicated", "measured_rate_statement"),
            ("reuse_populations_are_aligned", "reuse_population_statement"),
            ("frontier_fields_are_exercised", "frontier_statement"),
            ("no_cost_fields", "no_cost_fields_statement"),
            ("no_go_no_go_verdict", "no_go_no_go_statement")):
        if config["gates"].get(gate) is not True:
            raise ValueError(f"config gate {gate} is not asserted")
        if not config["gates"].get(statement):
            raise ValueError(f"config gate {gate} carries no statement in {statement}")
    if not config.get("claim_boundary"):
        raise ValueError("config declares no claim boundary")
    return config


def _ledger(resources: dict) -> dict:
    """The 2M-A subset of one ``resources()`` dump, with its identity checked."""
    missing = [field for field in REQUIRED_LEDGER_FIELDS if field not in resources]
    if missing:
        raise ValueError(
            f"resources() is missing the 2M-A ledger fields {missing}; the bank "
            "and this producer disagree about what a storage ledger contains")
    occurrences = resources["coefficient_occurrences"]
    packed = resources["cached_operator_bytes"]
    if packed != PACKED_BYTES_PER_COEFFICIENT * occurrences:
        raise ValueError(
            f"cached_operator_bytes {packed} is not "
            f"{PACKED_BYTES_PER_COEFFICIENT} * {occurrences}; the identity the "
            "committed back-fill depends on does not hold here, so neither the "
            "back-fill nor this row may be published")
    return {field: resources[field] for field in REQUIRED_LEDGER_FIELDS} | {
        "basis_size": resources["basis_size"],
        "word_universe": resources["word_universe"],
        "pairs_built": resources["pairs_built"],
        "packed_operator_bytes": packed,
    }


def _measured_row(key: str, kind: str, n_qubits: int, bank: MatrixElementBank,
                  indices) -> dict:
    """One priced bank: its ledger, its measured bytes, and their quotient."""
    resources = bank.resources(indices)
    measured = bank.measured_storage_bytes()
    row = {"system": key, "kind": kind, "n_qubits": n_qubits}
    row.update(_ledger(resources))
    # Both paths price the whole resident cache, rejected-candidate rows included.
    # ``word_universe`` alone remains subset-scoped because it is measurement cost.
    if measured["coefficient_occurrences"] != resources["coefficient_occurrences"]:
        raise ValueError("the measured walk and resources() price different rows")
    row.update({
        "measured_operator_bytes": measured["measured_operator_bytes"],
        "measured_container_bytes": measured["measured_container_bytes"],
        "measured_boxed_bytes": measured["measured_boxed_bytes"],
        "measured_coefficient_occurrences": measured["coefficient_occurrences"],
        "shared_boxed_slots": measured["shared_boxed_slots"],
        "distinct_boxed_objects": measured["distinct_boxed_objects"],
        "measured_bytes_per_coefficient": measured["measured_bytes_per_coefficient"],
        "packing_headroom": measured["packing_headroom"],
    })
    return row


def _measured_banks(config: dict) -> list[dict]:
    """The frozen mapping-axis banks at their declared labels."""
    mapping = load_mapping_config()
    by_key = {spec["key"]: spec for spec in mapping["systems"]}
    rows = []
    for key in config["measured_banks"]["systems"]:
        spec = by_key.get(key)
        if spec is None:
            raise ValueError(
                f"{key!r} is declared here but absent from "
                "benchmarks/configs/mapping_axis.json; this record prices the "
                "mapping instances and may not invent one")
        model, _ = _build_model(spec)
        _, selected = _selected_generators(model, spec)
        bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                                 model.hamiltonian, selected)
        bank.matrices()
        rows.append(_measured_row(key, "fixed_label_bank", model.n, bank, None))
    return rows


def _adaptive_arms(config: dict) -> list[dict]:
    """Small exact-selection arms, so the frontier fields have a live frontier."""
    mapping = load_mapping_config()
    by_key = {spec["key"]: spec for spec in mapping["systems"]}
    arms = config["adaptive_arms"]
    rows = []
    for key in arms["systems"]:
        model, _ = _build_model(by_key[key])
        rho = ExactMVBackend().state(model.reference, ())
        pool = _raw_pool(model)
        result = run_acase(rho, model.hamiltonian, pool,
                           initial=[identity_generator(model.n)],
                           max_size=int(arms["max_size"]))
        row = _measured_row(key, "adaptive_arm", model.n, result.bank, result.indices)
        row["candidate_pool_size"] = len(pool)
        rows.append(row)
    return rows


def _committed_rows(config: dict) -> list[dict]:
    """Back-fill the committed molecular records without rerunning them."""
    declared = config["committed_records"]
    directory = ROOT / declared["directory"]
    rows = []
    for name in declared["records"]:
        payload = json.loads((directory / name).read_text(encoding="utf-8"))
        missing = [field for field in COMMITTED_FIELDS if field not in payload]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        packed = payload["adaptive_cached_operator_bytes"]
        occurrences, remainder = divmod(packed, PACKED_BYTES_PER_COEFFICIENT)
        if remainder:
            raise ValueError(
                f"{name}: adaptive_cached_operator_bytes {packed} is not a "
                f"multiple of {PACKED_BYTES_PER_COEFFICIENT}, so it was not "
                "produced under the packed model this recovery assumes and "
                "T_coeff cannot be recovered from it")
        basis_size = payload["subspace_size_m"]
        pairs_built = payload["adaptive_pairs_built"]
        retained = basis_size * (basis_size + 1) // 2
        universe = payload["element_word_universe"]
        rows.append({
            "record": name,
            "subspace_size_m": basis_size,
            "pairs_built": pairs_built,
            "retained_block_pairs": retained,
            "selection_pairs": pairs_built - retained,
            "retained_pair_fraction": retained / pairs_built,
            "word_universe": universe,
            "resident_word_universe": None,
            "packed_operator_bytes": packed,
            "coefficient_occurrences": occurrences,
            "coefficient_reuse": None,
            "coefficient_occurrences_per_selected_element_word": occurrences / universe,
            "peak_rss_bytes": payload["adaptive_peak_rss_bytes"],
            "peak_rss_delta_bytes": payload["adaptive_peak_rss_delta_bytes"],
            "recovery": "cached_operator_bytes / 24, exact",
            "reuse_recovery_boundary": (
                "The record preserves selected-subspace W but not the word union "
                "of every rejected row still resident in the cache. T_coeff/W is "
                "therefore reported as resident coefficients per selected word, "
                "not as cross-row coefficient reuse; true reuse is unavailable "
                "without rerunning the bank."),
        })
    return rows


def _measured_rate(rows: list[dict]) -> dict:
    """The bytes-per-coefficient rate, pooled and as a range.

    Pooled rather than averaged over rows: the question is what a coefficient
    costs, and a mean over banks spanning two decades of coefficient count would
    weight the smallest bank as heavily as the largest. The range is carried
    beside it because the rate is not constant -- a larger bank shares more
    word-code integers, so its per-coefficient cost is lower -- and the
    attribution below has to be read against the end of the range it uses.
    """
    measured = sum(row["measured_operator_bytes"] for row in rows)
    occurrences = sum(row["measured_coefficient_occurrences"] for row in rows)
    per_row = {row["system"] + "/" + row["kind"]:
               row["measured_bytes_per_coefficient"] for row in rows}
    low = min(per_row.values())
    high = max(per_row.values())
    return {
        "pooled_measured_bytes": measured,
        "pooled_coefficient_occurrences": occurrences,
        "pooled_bytes_per_coefficient": measured / occurrences,
        "minimum_bytes_per_coefficient": low,
        "maximum_bytes_per_coefficient": high,
        "per_bank_bytes_per_coefficient": per_row,
        "pooled_packing_headroom": (measured
                                    / (PACKED_BYTES_PER_COEFFICIENT * occurrences)),
        "why_the_rate_varies": (
            "A word-code integer is one object shared by every row carrying that "
            "word, because the word product is memoized. A bank with more "
            "coefficients over a comparable word universe therefore amortizes "
            "those boxes further and costs less per coefficient. The committed "
            "rows below have coefficient counts one to three decades above every "
            "bank measured here. The low end of the observed range is used as a "
            "deliberately low-rate calibration point, not as a proven lower bound "
            "on a larger CPython dictionary."),
    }


def _attribution(rate: dict, committed: list[dict]) -> dict:
    """How much of each committed row's recorded peak RSS the rows explain.

    This is an attribution and not a measurement of those runs. It multiplies a
    rate measured on small banks by a coefficient count recovered from the
    record, and compares the product against a peak the record already states.
    It cannot separate the remainder into allocator slack, transient product
    storage or the run's own non-bank allocations, and it says nothing about any
    run outside the committed set.

    Peak is the denominator that carries the claim and delta is reported beside
    it, because the two behave differently and only one of them is a property of
    the bank. ``adaptive_peak_rss_delta_bytes`` subtracts the resident size at
    the *start* of the adaptive block, so a process that had already allocated
    and freed memory hands the bank pages the allocator still holds; the bank
    then grows into them and the delta understates it. That is not hypothetical
    here -- PLAN.md records a sequential run that carried BeH2's bank into H2O's
    baseline, and the producer was changed to free it explicitly -- so the delta
    fraction is a lower bound whose denominator depends on run history, while
    the peak fraction does not.
    """
    per_coefficient = rate["minimum_bytes_per_coefficient"]
    rows = []
    for row in committed:
        attributed = per_coefficient * row["coefficient_occurrences"]
        delta = row["peak_rss_delta_bytes"]
        peak = row["peak_rss_bytes"]
        rows.append({
            "record": row["record"],
            "coefficient_occurrences": row["coefficient_occurrences"],
            "attributed_row_bytes": attributed,
            "peak_rss_bytes": peak,
            "attributed_fraction_of_peak": attributed / peak,
            "peak_rss_delta_bytes": delta,
            "attributed_fraction_of_delta": (attributed / delta) if delta else None,
            "attributed_majority_of_peak": (
                attributed / peak) >= BANK_DOMINANCE_FRACTION,
        })
    dominated = [row["record"] for row in rows if row["attributed_majority_of_peak"]]
    fractions = [row["attributed_fraction_of_peak"] for row in rows
                 if row["attributed_majority_of_peak"]]
    return {
        "bytes_per_coefficient_used": per_coefficient,
        "basis": (
            "the minimum observed rate, used as the low-rate sensitivity case: the "
            "committed banks carry one to three decades more coefficients than "
            "any bank measured here, but CPython dict capacity is discontinuous, "
            "so this extrapolated rate is not claimed as a mathematical bound"),
        "attributed_majority_fraction": BANK_DOMINANCE_FRACTION,
        "rows": rows,
        "records_with_attributed_majority_of_peak": dominated,
        "majority_attribution_fraction_range": ([min(fractions), max(fractions)]
                                                if fractions else None),
        "reading": (
            "On every committed row whose bank is large enough to matter, the "
            "resident coefficient payload is attributed a majority of the recorded "
            "peak, at a strikingly stable fraction. This is a measurement-calibrated "
            "attribution, consistent with the two independent OOM witnesses that "
            "identify the bank as dominant; it is not a direct measurement of the "
            "large banks' object graphs. The rows where it does not dominate are "
            "the small ones, where a per-run baseline "
            "unrelated to the bank sets the peak -- hf carries a 2.9 GiB peak "
            "against 0.06 GiB of attributed rows -- so they are reported "
            "separately rather than averaged in, and they are not evidence "
            "against the premise."),
        "what_this_does_not_establish": (
            "Not that packing the rows reduces the peak by the attributed "
            "fraction. The remainder is unattributed rather than accounted for, "
            "and a packed representation that halves retained bytes may leave "
            "transient product storage, allocator slack and the memoized word "
            "product untouched. The primary end-to-end test in 2M's go/no-go is "
            "a configuration completing under a memory ceiling, and it stays the "
            "test precisely because no attribution can stand in for it."),
    }


def _word_product_cache() -> dict:
    """The memoized word product's ceiling, which neither 2M-B nor 2M-C reaches.

    Reported because a storage ledger that omitted it would let the packed
    representation hit its target while the process kept a fixed allocation of
    the same order. The bytes are a *lower* bound: they price the boxed key and
    value objects behind one live entry and exclude the cache's own hash table
    and its one LRU list node per entry, which are real and not portable to
    measure from outside ``functools``.

    The **ceiling** is priced rather than the current fill, and that is a
    correctness requirement rather than a simplification. Hits, misses and live
    entries are cumulative over the process, so a record carrying them would
    report different numbers depending on what ran before it -- measured here at
    1.50M against 1.54M hits for the same banks, with and without an unrelated
    warm-up. The ceiling is a declared constant of the implementation, it is
    what the process will reach on any workload large enough to matter, and
    these banks already saturate it.
    """
    maxsize = _word_mul_unchecked.cache_info().maxsize
    # One entry owns a 3-item key tuple and a 2-item value tuple. Count only those
    # containers: their referents may be shared with bank rows or other entries,
    # so multiplying their boxed sizes would not be a defensible lower bound.
    key = (8, 1 << 15, 1 << 15)
    value = (1 + 0j, 1 << 15)
    per_entry = sys.getsizeof(key) + sys.getsizeof(value)
    return {
        "function": "clifford_qc.pauli_kernel._word_mul_unchecked",
        "maxsize": maxsize,
        "lower_bound_bytes_per_entry": per_entry,
        "lower_bound_ceiling_bytes": per_entry * maxsize,
        "excluded_from_the_bound": (
            "the cache's own hash table, its one LRU list node per entry, and all "
            "boxed key/value referents because those objects may be shared"),
        "fill_is_not_recorded": (
            "Hits, misses and live entries are cumulative over the process and "
            "depend on what ran before this record was built, so they are "
            "measured but not published; the ceiling is the implementation "
            "constant and the banks priced here already reach it."),
        "why_it_is_here": (
            "2M-B packs operator rows and 2M-C evicts them; neither lever "
            "touches this cache, whose ceiling is a fixed allocation of roughly "
            "0.11 GiB before hash/LRU metadata. On the small banks measured here "
            "that floor is larger than the resident coefficient payload itself; "
            "on the committed "
            "molecular banks it is a constant beside rows one to three decades "
            "bigger. So it is not a competing explanation for the two OOM "
            "failures, and it is also not something the two proposed levers "
            "remove -- a packed bank that meets its 3x target still carries it."),
    }


def _baseline(rate: dict, measured: list[dict], committed: list[dict]) -> dict:
    """What 2M-B and 2M-C will be read against, with no verdict attached."""
    frontier = [row for row in measured if row["selection_pairs"] > 0]
    fractions = [row["retained_pair_fraction"] for row in committed]
    return {
        "storage_policy": STORAGE_POLICY,
        "packing_headroom_available": rate["pooled_packing_headroom"],
        # Ascending, so it reads as a range. The low end comes from the bank
        # with the *highest* per-coefficient cost, which is the smallest one.
        "packing_headroom_range": [rate["minimum_bytes_per_coefficient"]
                                   / PACKED_BYTES_PER_COEFFICIENT,
                                   rate["maximum_bytes_per_coefficient"]
                                   / PACKED_BYTES_PER_COEFFICIENT],
        "committed_retained_pair_fraction_range": [min(fractions), max(fractions)],
        "eviction_headroom_statement": (
            "The committed rows retain between "
            f"{min(fractions):.1%} and {max(fractions):.1%} of the pairs they "
            "built, so the rest is frontier or rejected-candidate storage. That "
            "ratio is eviction headroom and not an achieved reduction: a policy "
            "that discards those rows pays recomputation or I/O for them, and "
            "2M-C is required to report that cost rather than present the "
            "freed bytes alone."),
        "frontier_exercised_on": [row["system"] for row in frontier],
        "verdict": "baseline_only_no_go_no_go_evaluated",
        "why_no_verdict": (
            "2M's go/no-go asks for a 3x reduction in resident coefficient "
            "storage at unchanged T_coeff, a streaming policy that bounds live "
            "rows by the retained block plus its batch, and the stretched-H2O "
            "M=31 configuration completing below 15 GiB. None of the three can "
            "be evaluated before the packed representation and the policies "
            "exist. This record fixes the numbers they will be compared against "
            "and reports the headroom available to them; a headroom is not a "
            "reduction, and no combined packing-and-eviction figure follows "
            "from it."),
    }


def build_record() -> dict:
    config = load_config()
    measured = _measured_banks(config) + _adaptive_arms(config)
    committed = _committed_rows(config)
    rate = _measured_rate(measured)
    record = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_tier": "structural",
        "estimand": config["estimand"],
        "storage_policy": STORAGE_POLICY,
        "packed_bytes_per_coefficient": PACKED_BYTES_PER_COEFFICIENT,
        "packed_model_statement": config["packed_model_statement"],
        "interpreter_dependence": (
            "Every measured byte count is a property of the CPython build that "
            "produced it: dict capacity, integer boxing and object headers are "
            "implementation details rather than portable constants. The record "
            "declares its interpreter in provenance and the checker compares "
            "these fields under a declared relative tolerance, while the packed "
            "bytes, coefficient counts and pair counts stay exact."),
        "measured_banks": measured,
        "measured_rate": rate,
        "committed_records": committed,
        "peak_rss_attribution": _attribution(rate, committed),
        "word_product_cache": _word_product_cache(),
        "baseline": _baseline(rate, measured, committed),
        "claim_boundary": config["claim_boundary"],
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record()
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"policy {record['storage_policy']}  packed model "
          f"{record['packed_bytes_per_coefficient']} B/coefficient\n")
    print(f"{'bank':28s} {'n':>2s} {'M':>3s} {'T_coeff':>10s} {'B/coef':>7s} "
          f"{'headroom':>8s} {'retained':>9s}")
    for row in record["measured_banks"]:
        print(f"{row['system'] + '/' + row['kind']:28s} {row['n_qubits']:2d} "
              f"{row['basis_size']:3d} {row['coefficient_occurrences']:10d} "
              f"{row['measured_bytes_per_coefficient']:7.2f} "
              f"{row['packing_headroom']:8.3f} "
              f"{row['retained_pair_fraction']:8.1%}")
    rate = record["measured_rate"]
    print(f"\npooled {rate['pooled_bytes_per_coefficient']:.2f} B/coefficient "
          f"({rate['minimum_bytes_per_coefficient']:.2f}-"
          f"{rate['maximum_bytes_per_coefficient']:.2f}), "
          f"headroom {rate['pooled_packing_headroom']:.3f}x\n")
    print(f"{'committed record':26s} {'T_coeff':>12s} {'retained':>9s} "
          f"{'attributed':>11s} {'of peak RSS':>12s}")
    attribution = record["peak_rss_attribution"]
    by_record = {row["record"]: row for row in attribution["rows"]}
    for row in record["committed_records"]:
        priced = by_record[row["record"]]
        print(f"{row['record']:26s} {row['coefficient_occurrences']:12d} "
              f"{row['retained_pair_fraction']:8.1%} "
              f"{priced['attributed_row_bytes'] / 2**30:10.2f}G "
              f"{priced['attributed_fraction_of_peak']:12.2f}"
              f"{'  *' if priced['attributed_majority_of_peak'] else ''}")
    print(f"\n  * calibrated resident-row attribution exceeds half of peak RSS, on "
          f"{len(attribution['records_with_attributed_majority_of_peak'])} of "
          f"{len(attribution['rows'])} committed rows")
    print(f"\n{record['baseline']['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
