"""How much of the hierarchy's setting count is the colouring, not the rule.

``run_clifford_hierarchy.py`` prices each ``k`` rung over one constructive
colouring of the block-commuting rule: largest conflict degree first, then
first fit, over *contiguous* blocks. Both halves of that are choices, and
neither is the rule. This probe holds the rule, the bank, and the block size
fixed and varies only the two choices, using
:mod:`clifford_qc.measurement.regrouping`:

* the colouring -- DSATUR and iterated greedy instead of one fixed order, plus
  span recovery, which charges each setting for everything its own
  diagonalizer reads rather than only the words the colouring inserted;
* the frame -- which qubits share a block, searched over the block assignments
  rather than taken from the register's labelling. Under the declared
  all-to-all logical model a relabelling costs nothing, so a frame that halves
  the settings is a free reduction and not a new protocol.

Every partition is re-synthesized by ``synthesize_block_settings``, so the CX
counts and depths printed here are the same quantities the frozen record's
columns are, on partitions that pass the same ``Z``-only invariant.

    python benchmarks/probe_grouping_refinement.py
    python benchmarks/probe_grouping_refinement.py --system h4 --block-size 4

This is a probe, not a producer: it freezes no record and gates nothing. It
exists so the claim "the hierarchy's interior rungs are colouring-limited and
frame-limited, not rule-limited" can be re-derived rather than believed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.measurement.block_synthesis import synthesize_block_settings
from clifford_qc.measurement.regrouping import (
    counting_bound,
    dsatur_partition,
    isotropic_cover,
    iterated_greedy,
    permuted_codes,
    reduce_settings,
    search_block_order,
    span_recover,
)

from run_clifford_hierarchy import BLOCK_SIZES, SYSTEMS

HERE = Path(__file__).resolve().parent


def _priced(n, codes, groups, block_size) -> dict:
    synthesis = synthesize_block_settings(n, codes, groups, block_size)
    settings = synthesis.n_settings
    return {
        "settings": settings,
        "logical_cx_per_sweep": synthesis.logical_cx_per_sweep,
        "mean_logical_cx_per_setting": synthesis.logical_cx_per_sweep / settings,
        "max_logical_cx_depth": max(synthesis.cx_depth_per_setting),
        "words_per_setting": len(codes) / settings,
        "mean_words_read_per_setting": float(synthesis.compatibility.sum()) / settings,
    }


def strategy_rows(n, codes, block_size, *, rounds: int, seed: int) -> list[dict]:
    """One row per strategy, all in the identity frame, all priced."""
    greedy = block_commuting_partition(n, codes, block_size)
    dsatur = dsatur_partition(n, codes, block_size)
    rows = [
        ("shipped greedy", greedy),
        ("DSATUR", dsatur),
        ("greedy + iterated", iterated_greedy(
            n, codes, block_size, greedy, rounds=rounds, seed=seed)),
        ("DSATUR + iterated", iterated_greedy(
            n, codes, block_size, dsatur, rounds=rounds, seed=seed)),
        ("isotropic cover", isotropic_cover(n, codes, block_size)),
    ]
    best = min((groups for _, groups in rows), key=len)
    rows.append(("+ span recovery", span_recover(n, codes, block_size, best)))
    return [{"strategy": name, **_priced(n, codes, groups, block_size)}
            for name, groups in rows]


def frame_rows(n, codes, block_size) -> list[dict]:
    """The block assignments, ranked by the shipped greedy's own count."""
    ranked = search_block_order(n, codes, block_size)
    if len(ranked) == 1:
        return []
    identity = tuple(range(n))
    picked = ranked[:3]
    if identity not in [order for order, _ in picked]:
        picked.append(next(row for row in ranked if row[0] == identity))
    out = []
    for order, settings in picked:
        moved = permuted_codes(n, codes, order)
        groups = block_commuting_partition(n, moved, block_size)
        out.append({"frame": list(order), "greedy_settings": settings,
                    **_priced(n, moved, groups, block_size)})
    out.append({"frame": "worst", "greedy_settings": ranked[-1][1]})
    return out


def probe(system: str, block_sizes, *, rounds: int, seed: int) -> dict:
    bank = SYSTEMS[system]()
    n, codes = bank["n_qubits"], bank["codes"]
    record = {
        "system": system,
        "label": bank["label"],
        "n_qubits": n,
        "word_universe": len(codes),
        "basis_labels": bank["basis_labels"],
        "counting_bound": counting_bound(n, codes),
        "rungs": [],
    }
    for block_size in block_sizes:
        frozen = block_commuting_partition(n, codes, block_size)
        reduced = reduce_settings(n, codes, block_size, rounds=rounds, seed=seed)
        record["rungs"].append({
            "block_size": block_size,
            "frozen": _priced(n, codes, frozen, block_size),
            "reduced": _priced(n, list(reduced.codes),
                               [list(group) for group in reduced.groups],
                               block_size),
            "reduced_frame": list(reduced.order),
            "reduced_provenance": reduced.provenance,
            "strategies": strategy_rows(n, codes, block_size,
                                        rounds=rounds, seed=seed),
            "frames": frame_rows(n, codes, block_size),
        })
    return record


def report(record: dict) -> None:
    print(f"\n=== {record['system']}: {record['word_universe']} words on "
          f"{record['n_qubits']} qubits, counting bound "
          f"{record['counting_bound']} settings")
    for rung in record["rungs"]:
        frozen, reduced = rung["frozen"], rung["reduced"]
        print(f"\n  k = {rung['block_size']}")
        print(f"    {'strategy':<22} {'G':>6} {'CX/sweep':>10} {'max D_CX':>9} "
              f"{'W/G':>8} {'read/G':>8}")
        for row in rung["strategies"]:
            print(f"    {row['strategy']:<22} {row['settings']:>6} "
                  f"{row['logical_cx_per_sweep']:>10} "
                  f"{row['max_logical_cx_depth']:>9} "
                  f"{row['words_per_setting']:>8.1f} "
                  f"{row['mean_words_read_per_setting']:>8.1f}")
        for row in rung["frames"]:
            if row["frame"] == "worst":
                print(f"    {'frame: worst of all':<22} {row['greedy_settings']:>6}")
                continue
            print(f"    frame {str(row['frame']):<16} {row['settings']:>6} "
                  f"{row['logical_cx_per_sweep']:>10} "
                  f"{row['max_logical_cx_depth']:>9} "
                  f"{row['words_per_setting']:>8.1f} "
                  f"{row['mean_words_read_per_setting']:>8.1f}")
        print(f"    -> best {reduced['settings']} settings "
              f"({frozen['settings']} frozen, "
              f"{frozen['settings'] / reduced['settings']:.2f}x), "
              f"{reduced['logical_cx_per_sweep']} CX "
              f"({frozen['logical_cx_per_sweep']} frozen), "
              f"frame {rung['reduced_frame']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), action="append",
                        help="probe one system; repeatable, default all")
    parser.add_argument("--block-size", type=int, action="append",
                        help="probe one rung; repeatable, default the ladder")
    parser.add_argument("--rounds", type=int, default=96,
                        help="iterated-greedy rounds per seed")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the iterated-greedy shuffle schedule")
    parser.add_argument("--output", type=Path,
                        help="write the probe record as JSON")
    args = parser.parse_args(argv)

    systems = args.system or sorted(SYSTEMS)
    block_sizes = args.block_size or list(BLOCK_SIZES)
    records = [probe(system, block_sizes, rounds=args.rounds, seed=args.seed)
               for system in systems]
    for record in records:
        report(record)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
