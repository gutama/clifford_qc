"""Check committed response records against a fresh deterministic run.

The grouped histograms, replica accounting, seeds, and all discrete metadata
must agree exactly.  Floating-point outputs are compared numerically because
LAPACK and BLAS implementations can move the last few serialized bits without
changing the measured pipeline or any reported digit.  A byte comparison would
therefore reject an independently reproduced record for platform noise while
still saying nothing stronger about its scientific content.

A flat diff is the wrong report for these records.  Two fewer surviving
bootstrap replicas move every percentile band, so one upstream disagreement
prints as eight hundred downstream ones and the reader cannot tell a changed
pipeline from a changed replica census.  The comparison is therefore run in
four tiers, reported separately, and the exit status names the earliest tier
that failed:

``contract``
    Seeds, shot budget, word and group counts, basis size and family, and the
    bootstrap's declared method.  These are inputs.  A mismatch means the two
    runs did not attempt the same experiment, and nothing below is meaningful.

``point``
    The exact spectrum, the overlap conditioning, and every point estimate.
    These are computed from the raw cache and never touch the bootstrap RNG.
    A mismatch means the measured pipeline itself changed.

``census``
    How many replicas survived the thresholded-rank and root-identity gates,
    and why the rest were rejected.  Acceptance is decided by the *sign* of the
    resampled overlap's smallest eigenvalue, so this tier is where a genuinely
    portable run and a merely similar one separate.

``intervals``
    The percentile bands.  Every one of them is conditional on the census
    above, so this tier is reported as a summary -- how many bands moved and by
    how much -- rather than as one line per frequency point.

Run from the repository root:

    python paper_acase/check_response_records.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

from clifford_qc.reproducibility import compare_json_records
from run_response_record import build_record

HERE = Path(__file__).resolve().parent
ATOL = 1e-11
RTOL = 1e-11
MAX_REPORTED_PER_TIER = 6

RECORDS = (
    (HERE / "data" / "response_bootstrap.json", "determinant"),
    (HERE / "data" / "response_bootstrap_illconditioned.json", "krylov"),
)

# Tier membership by top-level key, then by the leaf names that split a shared
# object.  ``measured`` and ``spectrum`` each carry both a point estimate and
# replica-conditional bands, so they are split rather than assigned whole.
_CONTRACT_KEYS = ("schema", "source_main_commit", "system", "measurement")
_CENSUS_KEYS = ("replicates_succeeded", "replicates_requested",
                "acceptance_rate", "failures", "replica_census")
_INTERVAL_LEAVES = ("lower", "upper")


def _split_intervals(node):
    """Return ``(point_part, interval_part)`` for a nested estimate/band tree.

    ``lower``/``upper`` leaves are replica-conditional; everything else is a
    point quantity computed from the unresampled cache.
    """
    if isinstance(node, dict):
        point, bands = {}, {}
        for key, value in node.items():
            if key in _INTERVAL_LEAVES:
                bands[key] = value
                continue
            sub_point, sub_bands = _split_intervals(value)
            if sub_point is not None:
                point[key] = sub_point
            if sub_bands is not None:
                bands[key] = sub_bands
        return (point or None), (bands or None)
    if isinstance(node, list):
        points, bands = [], []
        for item in node:
            sub_point, sub_bands = _split_intervals(item)
            points.append(sub_point)
            bands.append(sub_bands)
        if all(item is None for item in points):
            points = None
        if all(item is None for item in bands):
            bands = None
        return points, bands
    return node, None


def _tiers(record: dict) -> dict[str, dict]:
    """Partition one record into the four comparison tiers."""
    basis = dict(record["basis"])
    condition = basis.pop("condition_number", None)
    bootstrap = dict(record["bootstrap"])
    census = {key: bootstrap.pop(key) for key in _CENSUS_KEYS if key in bootstrap}

    measured_point, measured_bands = _split_intervals(record["measured"])
    spectrum = dict(record["spectrum"])
    spectrum_bands = {key: spectrum.pop(key)
                      for key in _INTERVAL_LEAVES if key in spectrum}

    return {
        "contract": {key: record[key] for key in _CONTRACT_KEYS if key in record}
                    | {"basis": basis, "bootstrap_declaration": bootstrap},
        "point": {
            "exact": record["exact"],
            "overlap_condition_number": condition,
            "measured": measured_point,
            "spectrum": spectrum,
        },
        "census": census,
        "intervals": {"measured": measured_bands, "spectrum": spectrum_bands},
    }


def _worst_relative(problems: list[str]) -> float:
    """Largest relative deviation named in a list of comparison problems."""
    worst = 0.0
    for problem in problems:
        head, _, tail = problem.partition("expected ")
        if not tail:
            continue
        expected_text, _, got_text = tail.partition(", got ")
        try:
            expected, got = float(expected_text), float(got_text)
        except ValueError:
            continue
        scale = max(abs(expected), abs(got))
        if scale > 0.0:
            worst = max(worst, abs(expected - got) / scale)
    return worst


def _census_summary(committed: dict, fresh: dict) -> str:
    """One line naming the replica-census difference, if there is one."""
    made = committed.get("replicates_succeeded")
    now = fresh.get("replicates_succeeded")
    total = committed.get("replicates_requested")
    if made == now:
        return f"{now}/{total} replicas accepted, as committed"
    return (f"{now}/{total} replicas accepted here against {made}/{total} "
            f"committed ({now - made:+d}); rejection reasons "
            f"committed={committed.get('failures')} fresh={fresh.get('failures')}")


def _movers(committed: dict, fresh: dict) -> list[str]:
    """Name the individual replicas whose outcome changed.

    This is what the per-replica fingerprint buys.  Each line reports the
    replica index, the outcome on both sides, and the smallest overlap
    eigenvalue that decided it -- so a marginal sign flip near zero is
    immediately distinguishable from a pipeline that moved the eigenvalue
    wholesale.
    """
    before = {row["index"]: row for row in committed.get("replica_census", [])}
    after = {row["index"]: row for row in fresh.get("replica_census", [])}
    lines: list[str] = []
    for index in sorted(before.keys() & after.keys()):
        was, now = before[index], after[index]
        if was["outcome"] == now["outcome"]:
            continue
        old_min, new_min = was["overlap_eigenvalue_min"], now["overlap_eigenvalue_min"]
        shift = ("n/a" if old_min is None or new_min is None
                 else f"{new_min - old_min:+.3e}")
        lines.append(
            f"replica {index}: {was['outcome']} -> {now['outcome']}, "
            f"lambda_min {old_min:+.6e} -> {new_min:+.6e} (shift {shift})"
            if old_min is not None and new_min is not None else
            f"replica {index}: {was['outcome']} -> {now['outcome']}, "
            f"lambda_min {old_min} -> {new_min}")
    return lines


def main() -> int:
    tier_order = ("contract", "point", "census", "intervals")
    failed_tiers: set[str] = set()

    for record_path, family in RECORDS:
        committed = json.loads(record_path.read_text(encoding="utf-8"))
        fresh = build_record(generators=family)
        committed_tiers, fresh_tiers = _tiers(committed), _tiers(fresh)

        print(f"== {record_path.name}")
        for tier in tier_order:
            problems = compare_json_records(
                committed_tiers[tier], fresh_tiers[tier],
                f"{record_path.name}.{tier}", atol=ATOL, rtol=RTOL)
            if not problems:
                if tier == "census":
                    print(f"   census    OK  "
                          f"{_census_summary(committed_tiers['census'], fresh_tiers['census'])}")
                else:
                    print(f"   {tier:9s} OK")
                continue

            failed_tiers.add(tier)
            if tier == "census":
                print(f"   census    FAIL  "
                      f"{_census_summary(committed_tiers['census'], fresh_tiers['census'])}")
                movers = _movers(committed_tiers["census"], fresh_tiers["census"])
                for line in movers[:MAX_REPORTED_PER_TIER]:
                    print(f"      {line}")
                if len(movers) > MAX_REPORTED_PER_TIER:
                    print(f"      ... {len(movers) - MAX_REPORTED_PER_TIER} more")
                if not movers:
                    print("      no per-replica fingerprint to diff; "
                          "regenerate the record to gain one")
                continue
            if tier == "intervals":
                print(f"   intervals FAIL  {len(problems)} band value(s) moved, "
                      f"worst relative shift {_worst_relative(problems):.2e}; "
                      f"every band is conditional on the census above")
                continue
            print(f"   {tier:9s} FAIL  {len(problems)} mismatch(es), "
                  f"worst relative shift {_worst_relative(problems):.2e}")
            for problem in problems[:MAX_REPORTED_PER_TIER]:
                print(f"      {problem}")
            if len(problems) > MAX_REPORTED_PER_TIER:
                print(f"      ... {len(problems) - MAX_REPORTED_PER_TIER} more")

    print()
    if not failed_tiers:
        print("OK (discrete fields exact; floating fields within "
              f"atol={ATOL:g}, rtol={RTOL:g})")
        return 0

    earliest = next(tier for tier in tier_order if tier in failed_tiers)
    diagnosis = {
        "contract": "the two runs did not attempt the same experiment",
        "point": "the measured pipeline itself changed",
        "census": ("the pipeline and every point estimate reproduce, but a "
                   "different set of bootstrap replicas survived the rank "
                   "gate, so the reported intervals are not portable"),
        "intervals": ("the pipeline, the point estimates, and the replica "
                      "census all reproduce; only band values moved"),
    }[earliest]
    print(f"FAIL earliest failing tier: {earliest} -- {diagnosis}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
