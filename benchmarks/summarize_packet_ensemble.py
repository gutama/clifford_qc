"""Seed-clustered inference for the Phase 11B packet ensemble.

Orderings and shot budgets are repeated conditions within a random seed.  They
must not be treated as independent observations.  This summarizer therefore:

* keeps the cell-level packet win rate as a descriptive statistic only;
* reduces each seed to its median paired log10 error ratio within each group;
* applies the exact sign test to those seed-level effects; and
* bootstraps whole seed effects to obtain the confidence interval.

Negative log10 ratios favour the packet arm.  Multiple JSONL inputs are
accepted so an immutable earlier sweep can be combined with a later molecular
extension without disguising their separate provenance records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CONTROL_ORDERING = "random"
MIN_EFFECT_LOG10 = 0.01


def _binomial_sign_test(wins: int, trials: int) -> float:
    """Exact two-sided sign-test p-value at p=0.5."""
    if trials == 0:
        return float("nan")
    tail = min(wins, trials - wins)
    cdf = sum(math.comb(trials, k) for k in range(tail + 1)) / (2.0 ** trials)
    return float(min(1.0, 2.0 * cdf))


def _seed_effects(cells: list[dict]) -> np.ndarray:
    """One equally weighted median paired effect for each random seed."""
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        by_seed[int(cell["seed"])].append(float(cell["log_ratio"]))
    return np.asarray([
        float(np.median(by_seed[seed])) for seed in sorted(by_seed)
    ], dtype=float)


def _cluster_bootstrap_ci(seed_effects: np.ndarray, *, replicates: int = 10000,
                          seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI obtained by resampling independent seed clusters."""
    values = np.asarray(seed_effects, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(values, size=(replicates, values.size), replace=True), axis=1)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _matched(cells: list[dict]) -> tuple[list[dict], int, int]:
    eligible = [cell for cell in cells if cell.get("packet_eligible")]
    matched = [cell for cell in eligible if cell.get("matched_M", True)]
    return matched, len(cells) - len(eligible), len(eligible) - len(matched)


def _summarize(cells: list[dict], label: str) -> dict:
    matched, ineligible, unmatched = _matched(cells)
    cell_wins = sum(
        1 for cell in matched if cell["packet_error"] < cell["dressed_error"])
    cell_losses = sum(
        1 for cell in matched if cell["packet_error"] > cell["dressed_error"])
    cell_decided = cell_wins + cell_losses

    seed_effects = _seed_effects(matched)
    seed_wins = int(np.sum(seed_effects < 0.0))
    seed_losses = int(np.sum(seed_effects > 0.0))
    seed_decided = seed_wins + seed_losses
    lo, hi = _cluster_bootstrap_ci(seed_effects)
    return {
        "group": label,
        "cells": len(cells),
        "ineligible": ineligible,
        "unmatched_M": unmatched,
        "compared": len(matched),
        "seed_clusters": int(seed_effects.size),
        "cell_win_rate": (cell_wins / cell_decided) if cell_decided else float("nan"),
        "seed_wins": seed_wins,
        "seed_losses": seed_losses,
        "seed_ties": int(seed_effects.size) - seed_decided,
        "seed_win_rate": (seed_wins / seed_decided) if seed_decided else float("nan"),
        "sign_test_p": _binomial_sign_test(seed_wins, seed_decided),
        "median_log_ratio": (
            float(np.median(seed_effects)) if seed_effects.size else float("nan")),
        "ci_low": lo,
        "ci_high": hi,
        "median_packet_directions": (
            float(np.median([cell.get("packet_directions", 0) for cell in matched]))
            if matched else float("nan")),
        "median_extra_selection_work": (
            float(np.median([
                cell["packet_selection_work"] - cell["dressed_selection_work"]
                for cell in matched
            ])) if matched else float("nan")),
    }


def _verdict(row: dict) -> str:
    """Apply the predeclared consistency plus material-effect rule."""
    if row["seed_clusters"] < 8 or math.isnan(row["ci_high"]):
        return "insufficient"
    median = row["median_log_ratio"]
    significant = row["sign_test_p"] < 0.05
    if significant and row["ci_high"] < 0.0:
        return "packets better" if -median >= MIN_EFFECT_LOG10 else "negligible"
    if significant and row["ci_low"] > 0.0:
        return "packets worse" if median >= MIN_EFFECT_LOG10 else "negligible"
    return "no effect"


def _load(paths: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    cells: list[dict] = []
    headers: list[dict] = []
    systems: list[dict] = []
    seen: set[tuple] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record") == "header":
                headers.append({"path": str(path), **row})
                continue
            if row.get("record") == "system":
                systems.append({"path": str(path), **row})
                continue
            key = (row["system"], row["ordering"], int(row["shots"]), int(row["seed"]))
            if key in seen:
                raise ValueError(f"duplicate treatment cell across inputs: {key}")
            seen.add(key)
            cells.append(row)
    return cells, headers, systems


def _verified_sources(paths: list[Path]) -> dict[str, str]:
    """Map result paths to externally verified source commits when available."""
    verified: dict[str, str] = {}
    candidates = {
        path.parent / "packet_seed_ensemble_molecular.provenance.json"
        for path in paths
    }
    for candidate in candidates:
        if not candidate.exists():
            continue
        record = json.loads(candidate.read_text(encoding="utf-8"))
        source_sha = record.get("source_code_git_sha")
        for result_path, expected in record.get("records", {}).items():
            path = Path(result_path)
            if path not in paths or not path.exists():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected.get("sha256"):
                raise ValueError(
                    f"source-verification digest mismatch for {result_path}")
            if source_sha:
                verified[str(path)] = str(source_sha)
    return verified


def summarize(paths: list[Path]) -> tuple[str, list[dict]]:
    cells, headers, system_records = _load(paths)
    if not cells:
        raise ValueError("no treatment cells in input")

    systems = sorted({cell["system"] for cell in cells})
    orderings = sorted({cell["ordering"] for cell in cells})
    shot_grid = sorted({int(cell["shots"]) for cell in cells})
    seed_grid = sorted({int(cell["seed"]) for cell in cells})
    verified_sources = _verified_sources(paths)

    policy = [cell for cell in cells if cell["ordering"] != CONTROL_ORDERING]
    control = [cell for cell in cells if cell["ordering"] == CONTROL_ORDERING]
    groups = [_summarize(policy, "POOLED (policy orderings)")]
    if control:
        groups.append(_summarize(control, f"CONTROL ({CONTROL_ORDERING})"))
    for ordering in orderings:
        tag = "control" if ordering == CONTROL_ORDERING else "order"
        groups.append(_summarize(
            [cell for cell in cells if cell["ordering"] == ordering],
            f"{tag}={ordering}"))
    for system in systems:
        groups.append(_summarize(
            [cell for cell in policy if cell["system"] == system],
            f"system={system} (policy only)"))
    for shots in shot_grid:
        groups.append(_summarize(
            [cell for cell in policy if int(cell["shots"]) == shots],
            f"shots={shots} (policy only)"))
    for system in systems:
        for ordering in orderings:
            subset = [
                cell for cell in cells
                if cell["system"] == system and cell["ordering"] == ordering
            ]
            if subset:
                groups.append(_summarize(subset, f"{system} / {ordering}"))

    source_text = ", ".join(f"`{path}`" for path in paths)
    factorial = len(systems) * len(orderings) * len(shot_grid) * len(seed_grid)
    design = (
        f"complete {len(systems)} systems x {len(orderings)} orderings x "
        f"{len(shot_grid)} shot settings x {len(seed_grid)} seed clusters"
        if len(cells) == factorial else
        f"unbalanced design spanning {len(systems)} systems, {len(orderings)} "
        f"orderings, {len(shot_grid)} shot settings, and {len(seed_grid)} seed clusters"
    )
    lines = [
        "# Packet seed-ensemble: seed-clustered dressed-vs-packet inference",
        "",
        f"Sources: {source_text} — {len(cells)} treatment cells ({design}).",
        "",
        "Orderings and shot settings are repeated conditions within seed. "
        "Inference therefore uses one median paired effect per seed and "
        "resamples whole seed clusters. Cell win rate is descriptive only.",
        "",
        "Negative median log10 ratio favours packets. A confidence interval "
        "straddling zero or a seed-level sign-test p >= 0.05 is a no-effect "
        "result. Reported p-values are unadjusted across secondary groups.",
        "",
        f"`{CONTROL_ORDERING}` is the predeclared ordering control and is kept "
        "separate from the candidate-policy pool by design.",
        "",
        "| Group | cells | seeds | cell win | seed win | seed sign p | "
        "median seed log10 ratio | cluster 95% CI | verdict |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for row in groups:
        if row["seed_clusters"] == 0:
            lines.append(
                f"| {row['group']} | {row['compared']} | 0 | — | — | — | — | — | insufficient |")
            continue
        lines.append(
            f"| {row['group']} | {row['compared']} | {row['seed_clusters']} | "
            f"{row['cell_win_rate']:.2f} | {row['seed_win_rate']:.2f} | "
            f"{row['sign_test_p']:.3f} | {row['median_log_ratio']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {_verdict(row)} |")

    policy_overall = groups[0]
    _, total_ineligible, total_unmatched = _matched(cells)
    lines += [
        "",
        "## Cost side",
        "",
        f"- median extra frontier scoring, policy orderings: "
        f"`{policy_overall['median_extra_selection_work']:+.0f}` candidates per run",
        f"- median packet directions retained, policy orderings: "
        f"`{policy_overall['median_packet_directions']:.1f}`",
        f"- packet-ineligible cells: `{total_ineligible}` total; "
        f"`{policy_overall['ineligible']}` in policy orderings",
        f"- cells dropped for unmatched M: `{total_unmatched}` total; "
        f"`{policy_overall['unmatched_M']}` in policy orderings",
    ]

    if system_records:
        lines += ["", "## Molecular/system records", ""]
        for record in system_records:
            lines.append(
                f"- `{record['system']}`: {record.get('model', 'unknown model')}; "
                f"{record.get('n_qubits', '?')} qubits; sector dimension "
                f"{record.get('sector_dimension', '?')} (`{record['path']}`)")

    if headers:
        lines += ["", "## Provenance", ""]
        for header in headers:
            prov = header.get("provenance", {})
            verified = verified_sources.get(header["path"])
            verification_text = (
                f"; source code verified at git `{verified}`"
                if verified else "")
            lines.append(
                f"- `{header['path']}`: git `{prov.get('git_sha', 'unknown')}`"
                f"{' (dirty)' if prov.get('git_dirty') else ''}; "
                f"evidence `{header.get('evidence')}`{verification_text}")
        lines.append(
            f"- boundary: {headers[0].get('claim_boundary', 'not recorded')}")

    return "\n".join(lines) + "\n", groups


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, nargs="+",
        default=[Path("benchmarks/results/packet_seed_ensemble.jsonl")],
        help="one or more non-overlapping ensemble JSONL records")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional markdown output path")
    args = parser.parse_args(argv)

    text, _ = summarize(args.path)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
