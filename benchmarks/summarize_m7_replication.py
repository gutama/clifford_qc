"""Seed-clustered inference for the Phase 12 matched-budget M arms.

Answers the two questions PR #48 could not, because it ran one seed:

1. does the packet arm's separation from the plain dressed arm survive
   replication at the ladder's own budget, and under which orderings; and
2. do the classical comparators and the dressed arm move with the draw at all?

Inference is clustered on seed for the same reason as the Phase 11 packet
ensemble: orderings are repeated conditions within a seed, so cells sharing a
seed share a draw and are not independent.  Each group reduces to one median
paired log10 error ratio per seed, the exact sign test is applied to those, and
whole seeds are resampled for the interval.

`random` is the predeclared §11C control and is reported separately from the
candidate policies -- the hierarchy is *expected* to lose under an
uninformative order, and pooling that in manufactures a null.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTROL_ORDERING = "random"
MIN_EFFECT_LOG10 = 0.01
TREATMENT = "qsci_haar_dressed_acase"
BASELINE = "qsci_dressed_acase"
ARMS = ("budget_selected_ci", "matched_selected_ci", "acase",
        "qsci_dressed_acase", "qsci_haar_dressed_acase")


def _sign_p(wins: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    tail = min(wins, trials - wins)
    return float(min(1.0, 2.0 * sum(math.comb(trials, k)
                                    for k in range(tail + 1)) / 2.0 ** trials))


def _cluster_ci(effects: np.ndarray, *, replicates: int = 10000,
                seed: int = 0) -> tuple[float, float]:
    if effects.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(effects, size=(replicates, effects.size), replace=True), axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return (float(lo), float(hi))


def _seed_effects(cells: list[dict]) -> np.ndarray:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for cell in cells:
        errors = cell["errors"]
        if TREATMENT not in errors or BASELINE not in errors:
            continue
        by_seed[int(cell["seed"])].append(
            math.log10(max(errors[TREATMENT], 1e-15)
                       / max(errors[BASELINE], 1e-15)))
    return np.asarray([float(np.median(v)) for v in by_seed.values()], dtype=float)


def _summarize(cells: list[dict], label: str) -> dict:
    paired = [c for c in cells
              if TREATMENT in c["errors"] and BASELINE in c["errors"]]
    cell_wins = sum(1 for c in paired
                    if c["errors"][TREATMENT] < c["errors"][BASELINE])
    cell_losses = sum(1 for c in paired
                      if c["errors"][TREATMENT] > c["errors"][BASELINE])
    effects = _seed_effects(paired)
    seed_wins = int(np.sum(effects < 0.0))
    seed_losses = int(np.sum(effects > 0.0))
    lo, hi = _cluster_ci(effects)
    decided_cells = cell_wins + cell_losses
    decided_seeds = seed_wins + seed_losses
    return {
        "group": label,
        "compared": len(paired),
        "ineligible": len(cells) - len(paired),
        "seed_clusters": int(effects.size),
        "cell_win_rate": (cell_wins / decided_cells) if decided_cells else float("nan"),
        "seed_win_rate": (seed_wins / decided_seeds) if decided_seeds else float("nan"),
        "sign_test_p": _sign_p(seed_wins, decided_seeds),
        "median_log_ratio": float(np.median(effects)) if effects.size else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
    }


def _verdict(row: dict) -> str:
    if row["seed_clusters"] < 8 or math.isnan(row["ci_high"]):
        return "insufficient"
    median = row["median_log_ratio"]
    significant = row["sign_test_p"] < 0.05
    if significant and row["ci_high"] < 0.0:
        return "packets better" if -median >= MIN_EFFECT_LOG10 else "negligible"
    if significant and row["ci_low"] > 0.0:
        return "packets worse" if median >= MIN_EFFECT_LOG10 else "negligible"
    return "no effect"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--path", type=Path,
        default=Path("benchmarks/results/m7_seed_replication.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    header = None
    cells = []
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
    policies = [o for o in orderings if o != CONTROL_ORDERING]

    lines = [
        "# Phase 12 matched-budget arms: seed-clustered replication",
        "",
        f"Source: `{args.path}` — {len(cells)} cells across {len(systems)} "
        f"systems x {len(orderings)} orderings x "
        f"{len({c['seed'] for c in cells})} seeds, at the ladder's own "
        f"determinant budget.",
        "",
        "## Arm errors (Ha), median over cells",
        "",
        "An arm with identical min and max does not move with the draw. "
        "`matched_selected_ci` is sample-independent by construction, so its "
        "constancy is a correctness check rather than a finding.",
        "",
        "| system | arm | median | min | max |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for system in systems:
        subset = [c for c in cells if c["system"] == system]
        for arm in ARMS:
            values = np.array([c["errors"][arm] for c in subset
                               if arm in c["errors"]])
            if values.size:
                lines.append(
                    f"| {system} | `{arm}` | {np.median(values):.5f} | "
                    f"{values.min():.5f} | {values.max():.5f} |")

    lines += [
        "",
        f"## `{TREATMENT}` vs `{BASELINE}`, paired within draw",
        "",
        "Negative median log10 ratio favours the packet arm. Pooled rows "
        f"exclude the `{CONTROL_ORDERING}` control.",
        "",
        "| group | n | seeds | cell win | seed win | seed p | median log10 | 95% CI | verdict |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    groups = []
    for system in systems:
        subset = [c for c in cells if c["system"] == system]
        groups.append(_summarize(
            [c for c in subset if c["ordering"] in policies],
            f"{system} — pooled policy"))
        for ordering in orderings:
            tag = "control" if ordering == CONTROL_ORDERING else "order"
            groups.append(_summarize(
                [c for c in subset if c["ordering"] == ordering],
                f"{system} — {tag}={ordering}"))
    for row in groups:
        if row["seed_clusters"] == 0:
            lines.append(f"| {row['group']} | 0 | 0 | — | — | — | — | — | insufficient |")
            continue
        lines.append(
            f"| {row['group']} | {row['compared']} | {row['seed_clusters']} | "
            f"{row['cell_win_rate']:.2f} | {row['seed_win_rate']:.2f} | "
            f"{row['sign_test_p']:.3f} | {row['median_log_ratio']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {_verdict(row)} |")

    if header is not None:
        prov = header.get("provenance", {})
        lines += [
            "",
            "## Provenance",
            "",
            f"- git `{prov.get('git_sha', 'unknown')}`"
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
