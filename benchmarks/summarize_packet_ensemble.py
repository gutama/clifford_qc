"""Paired statistics for the Phase 11B packet seed-ensemble sweep.

Reads ``packet_seed_ensemble.jsonl`` and answers the question the Phase 10
single-draw record cannot: does the coarse-to-fine packet stage beat the plain
dressed arm at matched budget *reliably*, or was the five-row screen an
accident of one seed and one ordering?

Every statistic here is **paired**.  Each cell ran both arms on the identical
draw, so the unit of evidence is the within-cell difference, not two marginal
distributions.  Three things are reported per group:

``win rate``
    fraction of eligible cells where the packet error is strictly lower.
``sign test``
    exact two-sided binomial p-value against the null "packets and dressed are
    equally likely to win".  Ties are dropped from both numerator and
    denominator, which is the conservative convention.
``median log10 ratio + bootstrap CI``
    effect size.  Negative favours packets; -1.0 means a 10x error reduction.
    The CI is a percentile bootstrap over cells, so it carries the seed spread
    that the single-draw record silently assumed away.

A win rate near 0.5 with a CI straddling zero is a *negative result* and should
be reported as one -- that is the §11C bar, not a disappointment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _binomial_sign_test(wins: int, trials: int) -> float:
    """Exact two-sided binomial p-value at p=0.5."""
    if trials == 0:
        return float("nan")
    tail = min(wins, trials - wins)
    cdf = sum(math.comb(trials, k) for k in range(tail + 1)) / (2.0 ** trials)
    return float(min(1.0, 2.0 * cdf))


def _bootstrap_ci(values, *, replicates: int = 10000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(values, size=(replicates, values.size), replace=True), axis=1)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _summarize(cells: list[dict], label: str) -> dict:
    eligible = [c for c in cells if c.get("packet_eligible")]
    ineligible = len(cells) - len(eligible)
    matched = [c for c in eligible if c.get("matched_M", True)]
    ratios = [c["log_ratio"] for c in matched]
    wins = sum(1 for c in matched if c["packet_error"] < c["dressed_error"])
    losses = sum(1 for c in matched if c["packet_error"] > c["dressed_error"])
    decided = wins + losses
    lo, hi = _bootstrap_ci(ratios)
    return {
        "group": label,
        "cells": len(cells),
        "ineligible": ineligible,
        "unmatched_M": len(eligible) - len(matched),
        "compared": len(matched),
        "wins": wins,
        "losses": losses,
        "ties": len(matched) - decided,
        "win_rate": (wins / decided) if decided else float("nan"),
        "sign_test_p": _binomial_sign_test(wins, decided),
        "median_log_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "median_packet_directions": (
            float(np.median([c.get("packet_directions", 0) for c in matched]))
            if matched else float("nan")),
        "median_extra_selection_work": (
            float(np.median([c["packet_selection_work"]
                             - c["dressed_selection_work"] for c in matched]))
            if matched else float("nan")),
    }


def _verdict(row: dict) -> str:
    """The §11C bar, applied mechanically so the reader is not talked into it."""
    if row["compared"] < 8:
        return "insufficient"
    if math.isnan(row["ci_high"]):
        return "insufficient"
    if row["ci_high"] < 0.0:
        return "packets better"
    if row["ci_low"] > 0.0:
        return "packets worse"
    return "no effect"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path", type=Path,
        default=Path("benchmarks/results/packet_seed_ensemble.jsonl"))
    parser.add_argument("--out", type=Path, default=None,
                        help="optional markdown output path")
    args = parser.parse_args(argv)

    cells, header = [], None
    for line in args.path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record") == "header":
            header = row
            continue
        cells.append(row)
    if not cells:
        raise SystemExit(f"no cells in {args.path}")

    systems = sorted({c["system"] for c in cells})
    orderings = sorted({c["ordering"] for c in cells})
    shot_grid = sorted({c["shots"] for c in cells})

    groups = [_summarize(cells, "ALL")]
    for ordering in orderings:
        groups.append(_summarize(
            [c for c in cells if c["ordering"] == ordering], f"order={ordering}"))
    for system in systems:
        groups.append(_summarize(
            [c for c in cells if c["system"] == system], f"system={system}"))
    for shots in shot_grid:
        groups.append(_summarize(
            [c for c in cells if c["shots"] == shots], f"shots={shots}"))
    for system in systems:
        for ordering in orderings:
            subset = [c for c in cells
                      if c["system"] == system and c["ordering"] == ordering]
            if subset:
                groups.append(_summarize(subset, f"{system} / {ordering}"))

    lines = [
        "# Packet seed-ensemble: paired dressed-vs-packet comparison",
        "",
        f"Source: `{args.path}` — {len(cells)} cells "
        f"({len(systems)} systems x {len(orderings)} orderings x "
        f"{len(shot_grid)} shot settings).",
        "",
        "Negative `median log10 ratio` favours the packet arm. A confidence "
        "interval straddling zero is a no-effect result, which is the "
        "§11C answer, not a missing one.",
        "",
        "| Group | n | win rate | sign p | median log10 ratio | 95% CI | verdict |",
        "| :--- | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for row in groups:
        if row["compared"] == 0:
            lines.append(f"| {row['group']} | 0 | — | — | — | — | insufficient |")
            continue
        lines.append(
            f"| {row['group']} | {row['compared']} | {row['win_rate']:.2f} | "
            f"{row['sign_test_p']:.3f} | {row['median_log_ratio']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {_verdict(row)} |")

    overall = groups[0]
    lines += [
        "",
        "## Cost side",
        "",
        f"- median extra frontier scoring for the packet stage: "
        f"`{overall['median_extra_selection_work']:+.0f}` candidates per run",
        f"- median packet directions retained: "
        f"`{overall['median_packet_directions']:.1f}`",
        f"- packet-ineligible cells (fewer than two non-reference "
        f"configurations): `{overall['ineligible']}`",
        f"- cells dropped for unmatched M: `{overall['unmatched_M']}`",
    ]
    if header is not None:
        prov = header.get("provenance", {})
        lines += [
            "",
            "## Provenance",
            "",
            f"- git sha: `{prov.get('git_sha', 'unknown')}`"
            f"{' (dirty)' if prov.get('git_dirty') else ''}",
            f"- evidence: `{header.get('evidence')}`",
            f"- boundary: {header.get('claim_boundary')}",
        ]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
