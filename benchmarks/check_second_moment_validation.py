"""Verify the SecondMomentBank validation record.

``run_second_moment_validation.py`` writes
``benchmarks/reference_results/second_moment_validation.json``. This gate:

1. requires the record's preflight digest to match the committed preflight
   record, and its banks to be exactly the ones that record licenses;
2. re-derives every check and the summary from the recorded values, so a
   flag cannot disagree with the numbers under it;
3. recomputes every bank and requires the counts and digest to match exactly,
   and every energy, variance and residual norm to match to rounding
   (``--banks`` restricts this, ``--no-recompute`` skips it).

    python benchmarks/check_second_moment_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_preregistration as gate
    from benchmarks import run_second_moment_validation as producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as gate
    import run_second_moment_validation as producer


def _close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def declaration_problems(record: dict, preflight: dict, preflight_bytes: bytes) -> list[str]:
    problems = []
    if record.get("schema") != producer.SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("preflight", {}).get("sha256") != hashlib.sha256(preflight_bytes).hexdigest():
        problems.append("the preflight record changed after this validation")
    licensed = producer.licensed_banks(preflight)
    if sorted(record.get("banks", {})) != sorted(licensed):
        problems.append(f"banks {sorted(record.get('banks', {}))} are not the licensed "
                        f"{sorted(licensed)}")
    if record.get("quantum_advantage_claim") is not False:
        problems.append("quantum_advantage_claim must be false")
    return problems


def derived_problems(record: dict, preflight: dict) -> list[str]:
    """Checks and summary re-derived from the recorded values."""
    resolution = record["tolerances"]["variance_resolution"]
    tolerance = record["tolerances"]["block_relative"]
    problems = []
    for name, entry in record["banks"].items():
        counts = preflight["banks"][name]["counts"]
        block = entry["block_check"]
        roots = entry["roots"]
        for row in roots:
            residual = row["dense_residual_norm"]
            if not _close(row["variance_minus_dense"], row["variance"] - residual ** 2,
                          rel=1e-12, abs_=1e-15):
                problems.append(f"{name} root {row['root']}: variance_minus_dense drifted")
            if not _close(row["residual_norm"], math.sqrt(max(row["variance"], 0.0)),
                          rel=1e-12, abs_=1e-15):
                problems.append(f"{name} root {row['root']}: residual_norm is not the "
                                "clipped root of the variance")
            resolved = row["variance"] > resolution * row["cancellation_scale"]
            if row["resolved"] != resolved:
                problems.append(f"{name} root {row['root']}: resolved flag is wrong")
        checks = {
            "block_matches_dense": block["max_relative_deviation"] <= tolerance,
            "block_psd": block["min_eigenvalue"] >= -tolerance * max(block["max_abs_entry"], 1.0),
            "variances_match_dense": all(
                abs(row["variance_minus_dense"]) <= resolution * row["cancellation_scale"]
                for row in roots),
            "counts_match_preflight": all(
                entry["resources"][mine] == counts[theirs]
                for mine, theirs in producer.COUNT_FIELDS),
            "digest_matches_preflight": (
                entry["k_words_sha256"]
                == preflight["banks"][name]["digests"]["k_words_sha256"]),
        }
        if entry["checks"] != checks:
            problems.append(f"{name}: recorded checks are not what their values give")
        status = "VALIDATED" if all(checks.values()) else "FAILED"
        if entry["status"] != status:
            problems.append(f"{name}: recorded status {entry['status']} is not {status}")
    statuses = {name: entry["status"] for name, entry in record["banks"].items()}
    summary = record["summary"]
    if summary["bank_statuses"] != statuses:
        problems.append("the summary's statuses are not the banks' own")
    everything = all(s == "VALIDATED" for s in statuses.values()) and bool(statuses)
    if summary["all_validated"] is not everything:
        problems.append("all_validated contradicts the bank statuses")
    grounds = {name: entry["roots"][0]["residual_norm"]
               for name, entry in record["banks"].items()}
    if summary["ground_residual_norms"] != grounds:
        problems.append("the summary's ground residual norms are not the roots' own")
    return problems


def recompute_problems(record: dict, preflight: dict, *, banks=None, inputs=None) -> list[str]:
    """Every bank rebuilt; counts exact, values to rounding."""
    inputs = gate.bank_inputs if inputs is None else inputs
    problems = []
    for name in (record["banks"] if banks is None else banks):
        entry = record["banks"][name]
        model, selected = inputs(name)
        frozen = preflight["banks"][name]
        got = producer.validate_bank(model, selected, frozen["counts"],
                                     frozen["digests"]["k_words_sha256"])
        if got["resources"] != entry["resources"] or got["k_words_sha256"] != entry["k_words_sha256"]:
            problems.append(f"{name}: counts or digest recompute differently")
        if got["checks"] != entry["checks"] or got["status"] != entry["status"]:
            problems.append(f"{name}: checks recompute to {got['checks']}")
        if len(got["roots"]) != len(entry["roots"]):
            problems.append(f"{name}: {len(got['roots'])} roots recomputed, "
                            f"{len(entry['roots'])} recorded")
            continue
        for mine, theirs in zip(got["roots"], entry["roots"]):
            scale = theirs["cancellation_scale"]
            if not (_close(mine["energy"], theirs["energy"], rel=1e-10)
                    and abs(mine["variance"] - theirs["variance"]) <= 1e-12 * scale
                    and _close(mine["dense_residual_norm"], theirs["dense_residual_norm"],
                               rel=1e-8, abs_=1e-12)):
                problems.append(f"{name} root {theirs['root']}: recomputes to "
                                f"E={mine['energy']}, var={mine['variance']}")
                break
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=producer.RECORD)
    parser.add_argument("--banks", nargs="*", default=None, help="recompute only these banks")
    parser.add_argument("--no-recompute", action="store_true",
                        help="re-derive checks and summary without rebuilding any bank")
    args = parser.parse_args(argv)
    if not args.record.exists():
        print(f"FAIL missing record {args.record}")
        return 1
    record = json.loads(args.record.read_text(encoding="utf-8"))
    preflight_bytes = producer.PREFLIGHT.read_bytes()
    preflight = json.loads(preflight_bytes)
    problems = declaration_problems(record, preflight, preflight_bytes)
    if not problems:
        problems += derived_problems(record, preflight)
        if not args.no_recompute:
            problems += recompute_problems(record, preflight, banks=args.banks)
    if not record["summary"]["all_validated"]:
        problems.append("the record does not validate every licensed bank")
    if record.get("provenance", {}).get("git_dirty") is not False:
        problems.append("the record's provenance does not show a clean tree")
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    grounds = ", ".join(f"{name} {value:.3e}" for name, value in
                        record["summary"]["ground_residual_norms"].items())
    print(f"OK SecondMomentBank validates on every licensed bank; ground residuals: {grounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
