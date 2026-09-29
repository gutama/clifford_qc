"""Verify the Phase 15 preflight record against its preregistration.

``run_phase15_h2_preflight.py`` writes
``benchmarks/reference_results/phase15_h2_preflight.json``. This gate trusts
none of the record's summary fields.

1. **Declaration.** The record names the config by digest, and the config is
   unchanged since the run. It carries its own claim boundary, quotes the
   config's rather than inheriting it, and makes no advantage claim.
2. **Completeness.** Every declared bank is present, with one row per block
   pair and every pair ``i <= j`` exactly once.
3. **The freeze.** Each bank's ``measured_at_execution`` equals the config's
   ``measured_before_freezing``.
4. **Arithmetic.** Every total is re-derived from the per-row counts, and the
   word universes are checked against the bounds their union implies.
5. **Rule.** Each clause, status, the eligible set, the verdict and its quoted
   consequence are re-derived under the frozen rule. This implementation is
   separate from the producer's.
6. **Second moments.** The recorded row pairings are recompared against
   ``<psi|A_i† H² A_j|psi>`` from dense statevectors, with no multivector
   product. The block's smallest eigenvalue is recomputed, and every
   deterministic check is re-derived from its values.
7. **Recount.** Every row of every bank is recomputed through the declared
   products, and each count and word-set digest must match exactly. This costs
   as much as the run. ``--no-recount`` skips it, and ``--banks`` restricts it.
8. **Order.** Through the preregistration gate, the config's last change must
   strictly precede the record's commit.

    python benchmarks/check_phase15_h2_preflight.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_preregistration as gate
    from benchmarks import run_phase15_h2_preflight as producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as gate
    import run_phase15_h2_preflight as producer

STATUSES = {"ELIGIBLE", "WORD_PROHIBITIVE", "STORAGE_PROHIBITIVE",
            "BOTH_PROHIBITIVE", "CONVENTION_SENSITIVE", "INVALID"}
DENIALS = ("holds no count", "no second-moment row")
PAIRED = ("h2_terms", "k_word_universe", "combined_word_universe",
          "k_coefficient_occurrences", "total_coefficient_occurrences")


def _close(a, b, tol=1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def declaration_problems(config: dict, record: dict, config_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != producer.SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("config_digest") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("the config changed after the run: its digest no longer matches")
    boundary = str(record.get("claim_boundary", ""))
    if boundary == config["claim_boundary"]:
        problems.append("the record inherits the config's claim boundary instead of "
                        "stating its own")
    if any(phrase in boundary for phrase in DENIALS):
        problems.append("the record's claim boundary denies the counts it reports")
    quoted = record.get("preregistration", {}).get("config_claim_boundary_at_landing")
    if quoted != config["claim_boundary"]:
        problems.append("the record does not quote the config's boundary at landing")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be false")
    for key in ("clauses", "pruning", "row_definition"):
        if record.get(key) != config[key]:
            problems.append(f"the record's {key} is not the config's")
    return problems


def completeness_problems(config: dict, record: dict) -> list[str]:
    declared = config["banks"]["systems"]
    banks = record.get("banks", {})
    if sorted(banks) != sorted(declared):
        return [f"banks {sorted(banks)} are not the declared {sorted(declared)}"]
    problems = []
    for name in declared:
        entry = banks[name]
        size = config["measured_before_freezing"][name]["basis_size"]
        wanted = {(i, j) for j in range(size) for i in range(j + 1)}
        seen = [(int(row[0]), int(row[1])) for row in entry["rows"]]
        if len(seen) != len(set(seen)) or set(seen) != wanted:
            problems.append(f"{name}: the rows are not every pair i <= j exactly once")
        for key in PAIRED:
            if key not in entry["counts"] or f"{key}_strict" not in entry["counts"]:
                problems.append(f"{name}: {key} is missing at one of the thresholds")
    return problems


def freeze_problems(config: dict, record: dict) -> list[str]:
    problems = []
    for name, entry in record["banks"].items():
        frozen = config["measured_before_freezing"][name]
        measured = entry["measured_at_execution"]
        for key in gate._INT_FIELDS + gate._BOOL_FIELDS:
            if measured.get(key) != frozen[key]:
                problems.append(f"{name}: {key} at execution {measured.get(key)!r} is "
                                f"not the frozen {frozen[key]!r}")
        for key in gate._FLOAT_FIELDS:
            if not _close(measured.get(key, float("nan")), frozen[key], 1e-9):
                problems.append(f"{name}: {key} at execution differs from the freeze")
    return problems


def arithmetic_problems(record: dict) -> list[str]:
    """Totals from the rows, and the bounds a union of two sets must respect."""
    problems = []
    for name, entry in record["banks"].items():
        counts, rows = entry["counts"], entry["rows"]
        expected = {
            "block_pairs": len(rows),
            "k_coefficient_occurrences": sum(int(row[2]) for row in rows),
            "k_coefficient_occurrences_strict": sum(int(row[3]) for row in rows),
            "largest_row_terms": max((int(row[2]) for row in rows), default=0),
        }
        sh = counts["sh_coefficient_occurrences"]
        expected["total_coefficient_occurrences"] = sh + expected["k_coefficient_occurrences"]
        expected["total_coefficient_occurrences_strict"] = (
            sh + expected["k_coefficient_occurrences_strict"])
        expected["k_near_threshold_coefficients"] = (
            expected["k_coefficient_occurrences"] - expected["k_coefficient_occurrences_strict"])
        expected["h2_near_threshold_coefficients"] = (
            counts["h2_terms"] - counts["h2_terms_strict"])
        expected["packed_bytes"] = 24 * expected["total_coefficient_occurrences"]
        expected["additional_words"] = (
            counts["combined_word_universe"] - counts["sh_word_universe"])
        for key, value in expected.items():
            if counts.get(key) != value:
                problems.append(f"{name}: {key} is {counts.get(key)}, the rows give {value}")
        words = counts["sh_word_universe"]
        for suffix in ("", "_strict"):
            k = counts[f"k_word_universe{suffix}"]
            union = counts[f"combined_word_universe{suffix}"]
            if not max(words, k) <= union <= words + k:
                problems.append(f"{name}: combined_word_universe{suffix} {union} is not "
                                f"a union of {words} and {k} words")
            if not _close(counts[f"word_ratio{suffix}"], union / words):
                problems.append(f"{name}: word_ratio{suffix} drifted")
        for key in PAIRED:
            if counts[f"{key}_strict"] > counts[key]:
                problems.append(f"{name}: strict {key} exceeds the declared count")
        ceiling = entry["measured_at_execution"]["sector_word_ceiling"]
        if counts["combined_word_universe"] > ceiling:
            problems.append(f"{name}: more words than the sector holds")
    return problems


def derive_status(config: dict, counts: dict, checks: dict) -> str:
    """The frozen bank rule, restated independently of the producer and gate."""
    if not all(checks.values()):
        return "INVALID"
    ratio = config["clauses"]["word"]["max_ratio"]
    anchor = config["clauses"]["storage"]["max_coefficient_occurrences"]
    outcome = {}
    for suffix in ("", "_strict"):
        word = counts[f"combined_word_universe{suffix}"] <= ratio * counts["sh_word_universe"]
        storage = counts[f"total_coefficient_occurrences{suffix}"] <= anchor
        outcome[suffix] = (word, storage)
    if outcome[""] != outcome["_strict"]:
        return "CONVENTION_SENSITIVE"
    return {(True, True): "ELIGIBLE", (False, True): "WORD_PROHIBITIVE",
            (True, False): "STORAGE_PROHIBITIVE",
            (False, False): "BOTH_PROHIBITIVE"}[outcome[""]]


def derive_verdict(statuses: list[str]) -> str:
    if "INVALID" in statuses:
        return "INVALID"
    eligible = [s for s in statuses if s == "ELIGIBLE"]
    if len(eligible) == len(statuses):
        return "FULL"
    return "NONE" if not eligible else "RESTRICTED"


def check_values(entry: dict) -> dict:
    """The deterministic checks, re-derived from the values the record holds."""
    counts, dense, frozen = entry["counts"], entry["dense_check"], entry["measured_at_execution"]
    tol = producer.DENSE_TOLERANCE
    return {
        "baseline_matches_freeze": (
            counts["sh_word_universe"] == frozen["sh_word_universe"]
            and counts["sh_coefficient_occurrences"] == frozen["sh_coefficient_occurrences"]
            and counts["hamiltonian_terms"] == frozen["hamiltonian_terms"]),
        "rows_complete": counts["block_pairs"] == frozen["block_pairs"],
        "sector_closed": entry["sector_violations"] == 0,
        "rows_reproduce_dense_second_moments": dense["max_relative_deviation"] <= tol,
        "gram_psd": dense["min_eigenvalue"] >= -tol * max(dense["max_abs_entry"], 1.0),
        "strict_not_above_declared": all(
            counts[f"{key}_strict"] <= counts[key] for key in PAIRED),
    }


def rule_problems(config: dict, record: dict) -> list[str]:
    problems = []
    statuses = {}
    for name, entry in record["banks"].items():
        checks = check_values(entry)
        if entry["deterministic_checks"] != checks:
            problems.append(f"{name}: recorded deterministic checks are not what "
                            "their values give")
        status = derive_status(config, entry["counts"], checks)
        statuses[name] = status
        decision = entry["decision"]
        if decision.get("status") != status:
            problems.append(f"{name}: recorded status {decision.get('status')} is not "
                            f"the rule's {status}")
        if status != "INVALID":
            ratio = config["clauses"]["word"]["max_ratio"]
            anchor = config["clauses"]["storage"]["max_coefficient_occurrences"]
            counts = entry["counts"]
            for label, suffix in (("declared", ""), ("strict", "_strict")):
                want = {"word": counts[f"combined_word_universe{suffix}"]
                        <= ratio * counts["sh_word_universe"],
                        "storage": counts[f"total_coefficient_occurrences{suffix}"] <= anchor}
                if decision.get("clauses", {}).get(label) != want:
                    problems.append(f"{name}: recorded {label} clauses are not the rule's")
    declared = config["banks"]["systems"]
    verdict = derive_verdict([statuses[name] for name in declared])
    summary = record.get("decision", {})
    if summary.get("bank_statuses") != {name: statuses[name] for name in declared}:
        problems.append("the summary's bank statuses are not the banks' own")
    if summary.get("eligible_banks") != [n for n in declared if statuses[n] == "ELIGIBLE"]:
        problems.append("the summary's eligible banks are not the ELIGIBLE ones")
    if summary.get("verdict") != verdict:
        problems.append(f"recorded verdict {summary.get('verdict')} is not the rule's {verdict}")
    if summary.get("consequence") != config["consequences"].get(verdict):
        problems.append("the recorded consequence does not quote the config's for the verdict")
    if verdict not in set(config["reachable_verdicts"]) | {"INVALID"}:
        problems.append(f"verdict {verdict} is one the declaration showed unreachable")
    return problems


def dense_problems(config: dict, record: dict, *, inputs=None) -> list[str]:
    """Recorded pairings against dense statevectors: no multivector product."""
    from clifford_qc.subspace.contracts import as_multivector

    inputs = gate.bank_inputs if inputs is None else inputs
    problems = []
    tol = producer.DENSE_TOLERANCE
    for name, entry in record["banks"].items():
        model, selected = inputs(name)
        dense = producer.dense_second_moments(model, selected,
                                              as_multivector(model.hamiltonian))
        size = len(selected)
        block = np.zeros((size, size), dtype=complex)
        worst = 0.0
        for i, j, _, _, real, imag in entry["rows"]:
            value = complex(real, imag)
            block[int(i), int(j)] = value
            deviation = abs(value - dense[int(i), int(j)]) / (1.0 + abs(dense[int(i), int(j)]))
            worst = max(worst, deviation)
        if worst > tol:
            problems.append(f"{name}: a recorded pairing is {worst:.2e} from the dense "
                            "second moment")
        full = np.triu(block) + np.triu(block, 1).conj().T
        lowest = float(np.linalg.eigvalsh(0.5 * (full + full.conj().T))[0])
        recorded = entry["dense_check"]
        if abs(recorded["min_eigenvalue"] - lowest) > tol * max(recorded["max_abs_entry"], 1.0):
            problems.append(f"{name}: the block's smallest eigenvalue recomputes to "
                            f"{lowest:.3e}, recorded {recorded['min_eigenvalue']:.3e}")
        if not _close(recorded["max_abs_entry"], float(np.max(np.abs(full))), 1e-9):
            problems.append(f"{name}: max_abs_entry drifted")
    return problems


def recount_problems(config: dict, record: dict, *, banks=None, inputs=None) -> list[str]:
    """Every row recomputed through the declared products; counts must match exactly."""
    inputs = gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        entry = record["banks"][name]
        model, selected = inputs(name)
        counted = producer.count_bank(config, model, selected)
        for key, value in counted["counts"].items():
            if isinstance(value, float):
                if not _close(entry["counts"].get(key), value):
                    problems.append(f"{name}: {key} recounts to {value}, recorded "
                                    f"{entry['counts'].get(key)}")
            elif entry["counts"].get(key) != value:
                problems.append(f"{name}: {key} recounts to {value}, recorded "
                                f"{entry['counts'].get(key)}")
        if entry["digests"] != counted["digests"]:
            problems.append(f"{name}: a word-set digest recounts differently")
        if entry["sector_violations"] != counted["sector_violations"]:
            problems.append(f"{name}: sector_violations recounts differently")
        for got, want in zip(counted["rows"], entry["rows"]):
            if got[:4] != [int(v) for v in want[:4]] or not (
                    _close(got[4], want[4], 1e-10) and _close(got[5], want[5], 1e-10)):
                problems.append(f"{name}: row ({got[0]}, {got[1]}) recounts to {got}, "
                                f"recorded {want}")
                break
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--no-recount", action="store_true",
                        help="skip recomputing every row; rules, arithmetic and the "
                             "dense comparison still run")
    parser.add_argument("--banks", nargs="*", default=None,
                        help="recount only these banks")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    problems = declaration_problems(config, record, config_bytes)
    problems += completeness_problems(config, record)
    if not problems:
        problems += freeze_problems(config, record)
        problems += arithmetic_problems(record)
        problems += rule_problems(config, record)
        problems += dense_problems(config, record)
        if not args.no_recount:
            problems += recount_problems(config, record, banks=args.banks)
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("the record's provenance does not show a clean tree")
    notes: list[str] = []
    if args.record.resolve() == producer.RECORD.resolve():
        problems += gate.commit_order_problems(notes)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    decision = record["decision"]
    print(f"OK Phase 15 preflight record re-derives under the frozen rule: verdict "
          f"{decision['verdict']}, statuses {decision['bank_statuses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
