"""R3S -- which candidates a declared screen admits for a sampling probe.

The R3 cost layer prices one bank.  ``h4_converged`` is right-censored at the
frozen ``65536`` endpoint and the QR3b LiH candidate was rejected for the same
reason, so every accuracy-matched question downstream -- QR3, QR3b, and R4a's
four-arm interaction test, which needs three finite cost ratios -- is waiting on
a second priceable instance.

``PLAN.md`` frames the way out as endpoints above ``65536``, a joint R1/R3
regeneration it deferred.  This producer takes the other route.  The QR3b record
already carries a word-universe ceiling and the admission
``screen_would_have_rejected_before_probe: true``: the gate existed, but it was
evaluated *after* the probe had spent all forty of its cells.  Evaluating the
same gate first, over every declared candidate, costs seconds.

**What the screen changes, and what it must not.**  The frozen selection rule
runs the A-CASE greedy to its own predicted-lowering threshold.  That rule is
accuracy-maximizing while the exact-tier price is resolution-limited, and the
two conflict: on LiH the greedy spends four orders of magnitude of bias headroom
(down to ``0.0002`` mHa) buying ``5.4x`` the word universe.  So the screen
declares a second stopping rule -- the smallest prefix clearing the target with
margin -- and reports both.  The rule is a function of the accuracy target
alone.  It may not see a mapping, an arm, or a word count, because choosing the
prefix that makes an instance cheap to measure is precisely the cost-direction
selection the QR3b preregistration forbids; the record carries the per-arm bias
agreement so a checker can confirm the choice was mapping-blind rather than take
it on trust.

**Two monotonicities make the search well-posed**, and both are asserted per
candidate rather than assumed.  Bias is non-increasing along the greedy prefix
(a Ritz value can only fall as the span grows), so the first prefix clearing the
margin is the unique smallest one.  The word universe is non-decreasing (a
longer prefix adds matrix-element pairs and removes none), so once ``W`` passes
the ceiling with the bias gate still unmet, no longer prefix can pass either and
the walk stops there with its reason recorded.

**What this record may claim.**  It is ``structural``: biases and word counts in
deterministic double-precision arithmetic -- dense eigensolves, reproducible run
to run on fixed BLAS threading, but floating-point and compared to tolerance by
the checker rather than exact.  Nothing is sampled.  It can authorize or
withhold a probe.  It cannot price ``C(epsilon)``, answer QR3b, or reclassify a
frozen censoring decision.  Candidates are admitted or rejected *under this
declared screen*: the ceiling is an operational threshold calibrated on one
priced bank, not a demonstrated necessary condition, so an admission is not a
demonstration that a bank will resolve and a rejection is not a demonstration
that it cannot.

**The margin factor is declared here, not preregistered.**  This producer and
the first result it reports land in the same commit, so factor 3 is exploratory
with respect to this run.  Each candidate therefore carries a
``margin_sensitivity`` range, re-derived from its own walked rows, showing over
which margin factors its verdict is unchanged -- which is what says whether the
number is load-bearing.  The rule is frozen from here; the preregistered use is
the next candidate screened under it.

    python benchmarks/run_priceability_screen.py
    python benchmarks/check_priceability_screen.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.fermion_mapping import FERMION_ENCODINGS, fermion_encoding
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import qwc_basis_cover, qwc_groups
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import Generator
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.elements import MatrixElementBank

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
    )
    from benchmarks.run_qr3b_instance_preflight import (
        CONFIG as QR3B_CONFIG,
        load_config as load_qr3b_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
    )
    from run_qr3b_instance_preflight import (
        CONFIG as QR3B_CONFIG,
        load_config as load_qr3b_config,
    )

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "priceability_screen.json"
REFERENCE = HERE / "reference_results" / "priceability_screen.json"
SCHEMA = "clifford_qc.priceability_screen.v1"
CONFIG_SCHEMA = "clifford_qc.priceability_screen_config.v1"

GROUPERS = {"qwc_groups": qwc_groups, "qwc_basis_cover": qwc_basis_cover}


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported priceability-screen config schema")
    gates = config.get("acceptance_gates", {})
    margin = gates.get("margin_factor")
    if not isinstance(margin, (int, float)) or margin < 1.0:
        raise ValueError("margin_factor must be a number at or above 1")
    if not gates.get("margin_factor_basis"):
        raise ValueError("the margin factor must declare what calibrates it")
    ceiling = gates.get("word_universe_ceiling")
    if not isinstance(ceiling, int) or ceiling <= 0:
        raise ValueError("the screen must declare a word-universe ceiling")
    if not gates.get("word_universe_ceiling_basis"):
        raise ValueError("the word-universe ceiling must declare what calibrates it")
    if gates.get("selection_may_not_use_mapping_cost_direction") is not True:
        raise ValueError("the screen may not relax the QR3b cost-direction gate")
    if gates.get("screen_authorizes_a_probe_not_a_price") is not True:
        raise ValueError("the screen must disclaim cost-record status")
    rule = config.get("selection_rule", {})
    if int(rule.get("minimum_prefix_size", 0)) < 2:
        raise ValueError(
            "a one-generator prefix is the identity alone and spans no subspace"
        )
    if not rule.get("minimum_prefix_size_basis"):
        raise ValueError("the minimum prefix size must declare its reason")
    if gates.get("word_universe_convention") != WORD_UNIVERSE_CONVENTION:
        raise ValueError(
            "the ceiling is calibrated on the "
            f"{WORD_UNIVERSE_CONVENTION!r} word count; the config declares "
            f"{gates.get('word_universe_convention')!r}"
        )
    if not gates.get("word_universe_convention_basis"):
        raise ValueError("the word-universe convention must declare its reason")
    if not gates.get("margin_factor_status_basis"):
        raise ValueError("the margin factor must declare its evidence status")

    # Everything below is consumed by ``build_record`` and ``_arm_rows`` by
    # direct indexing. Left unchecked it surfaces as a KeyError or a TypeError
    # somewhere inside a candidate walk, which names neither the field nor the
    # file that carries it -- so the config boundary rejects it here instead.
    target = gates.get("accuracy_target_millihartree")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        raise ValueError("accuracy_target_millihartree must be a number")
    if not math.isfinite(float(target)) or float(target) <= 0.0:
        raise ValueError("accuracy_target_millihartree must be finite and positive")

    arms = config.get("mapping_arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("the screen must declare a non-empty mapping_arms list")
    if len(set(arms)) != len(arms):
        raise ValueError("mapping_arms contains a duplicate")
    unknown = sorted(set(arms) - FERMION_ENCODINGS)
    if unknown:
        raise ValueError(
            f"unsupported mapping arms {unknown}; choose from {sorted(FERMION_ENCODINGS)}"
        )

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("the screen must declare at least one candidate")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates contains a duplicate")
    known = set(candidate_specs())
    missing = sorted(set(candidates) - known)
    if missing:
        raise ValueError(
            f"candidates {missing} have no frozen spec; known keys are {sorted(known)}"
        )

    grouping = config.get("grouping_protocol_by_system")
    if not isinstance(grouping, dict):
        raise ValueError("grouping_protocol_by_system must be an object")
    ungrouped = sorted(key for key in candidates if key not in grouping)
    if ungrouped:
        raise ValueError(f"candidates {ungrouped} declare no grouping protocol")
    bad = sorted(
        f"{key}={grouping[key]!r}" for key in candidates if grouping[key] not in GROUPERS
    )
    if bad:
        raise ValueError(
            f"unsupported grouping protocol for {bad}; choose from {sorted(GROUPERS)}"
        )

    if not isinstance(config.get("omitted_candidates"), dict):
        raise ValueError("omitted_candidates must be an object, empty if nothing is omitted")
    return config


def candidate_specs() -> dict[str, dict]:
    """Frozen instance specs, each read from the preregistration that owns it.

    Nothing is redeclared here.  The four R2b banks keep their ``mapping_axis``
    specs and LiH keeps the QR3b one, so the labels this screen walks are the
    labels those records froze, and their FCIDUMP digests stay bound to the
    provenance files those producers already check.
    """
    specs = {spec["key"]: spec for spec in load_mapping_config()["systems"]}
    qr3b = load_qr3b_config(QR3B_CONFIG)["candidate"]
    if qr3b["key"] in specs:
        raise ValueError("the QR3b candidate collides with a frozen R2b bank key")
    specs[qr3b["key"]] = qr3b
    return specs


# Two word-universe conventions are in the tree and they differ by one. The
# ``mapping_axis`` invariants count the identity word (BeH2 full-width: 1815);
# ``protocol_axis`` and the QR3b probe do not (1814). The screen counts as those
# two do, because that is the convention the 2048 ceiling was calibrated in and
# because the identity is the one word a shot budget never buys -- its
# expectation is 1 by normalization, so it contributes nothing to the
# reconstruction variance the ceiling is a proxy for.
WORD_UNIVERSE_CONVENTION = "non_identity_words_only"


def _solved(bank: MatrixElementBank, exact_energy: float) -> tuple[float, int]:
    """Solve one bank, then read its word universe. The order is load-bearing.

    ``MatrixElementBank`` fills its word cache in ``_build_pair``, which the
    solve triggers; reading ``word_set()`` first returns an empty frozenset and
    every gate downstream then compares against zero and passes. That is the
    silent failure this helper exists to prevent, so the populated cache is
    asserted rather than left to the call order -- a solved bank over any
    prefix carries at least the identity word.
    """
    bias = (float(bank.solve().ground_energy) - exact_energy) * 1e3
    words = bank.word_set()
    if not words:
        raise AssertionError(
            "empty word universe after a solve; the bank cache did not populate"
        )
    return bias, sum(1 for code in words if code != 0)


def _walk(
    reference,
    hamiltonian,
    selected: Sequence[Generator],
    exact_energy: float,
    *,
    admissible_bias: float,
    ceiling: int,
    minimum_prefix: int,
) -> tuple[list[dict], int | None, str]:
    """Walk greedy prefixes until the margin is cleared or the ceiling is passed.

    Returns the walked rows, the chosen prefix size (``None`` if none clears),
    and the reason the walk ended.  The early exit is sound only because ``W``
    is non-decreasing and the bias non-increasing along the ordering; both are
    recorded per row so ``check_priceability_screen.py`` re-derives them instead
    of trusting this docstring.
    """
    rows: list[dict] = []
    for size in range(minimum_prefix, len(selected) + 1):
        bank = MatrixElementBank(reference, hamiltonian, list(selected[:size]))
        bias, words = _solved(bank, exact_energy)
        rows.append(
            {
                "basis_size": size,
                "bias_millihartree": bias,
                "source_word_universe": words,
                "clears_margin": bias <= admissible_bias,
                "within_ceiling": words <= ceiling,
            }
        )
        if bias <= admissible_bias:
            return rows, size, "margin_cleared"
        if words > ceiling:
            return rows, None, "word_universe_exceeded_before_bias_cleared"
    return rows, None, "bias_margin_unreachable_within_the_frozen_ordering"


def _arm_rows(
    model,
    reference,
    prefix: Sequence[Generator],
    exact_energy: float,
    arms: Sequence[str],
    grouping: str,
    *,
    count_settings: bool,
) -> list[dict]:
    """Transport one prefix through every declared encoding, as R2b does."""
    rows = []
    for name in arms:
        reduced = name.endswith("+2q")
        encoding = fermion_encoding(
            name,
            model.n,
            n_electrons=int(model.metadata["n_electrons"]) if reduced else None,
            sz=float(model.metadata["sz"]) if reduced else None,
            spin_ordering=model.metadata.get("spin_convention", "interleaved"),
        )
        transported = encoding.restriction().transport(
            hamiltonian=as_multivector(model.hamiltonian),
            reference=reference,
            generators=(generator.mv for generator in prefix),
        )
        mapped = [
            Generator(generator.label, image)
            for generator, image in zip(prefix, transported.generators)
        ]
        bank = MatrixElementBank(
            transported.reference, transported.hamiltonian, mapped
        )
        bias, universe = _solved(bank, exact_energy)
        row = {
            "mapping": name,
            "measured_qubits": transported.n,
            "word_universe": universe,
            "bias_millihartree": bias,
        }
        if count_settings:
            codes = sorted(code for code in bank.word_set() if code != 0)
            row["grouping_protocol"] = grouping
            row["settings"] = len(
                GROUPERS[grouping]([PauliWord(transported.n, c) for c in codes])
            )
        rows.append(row)
    return rows


def _margin_sensitivity(
    rows: Sequence[dict],
    chosen: int | None,
    *,
    target: float,
    ceiling: int,
    margin: float,
) -> dict:
    """Over which margin factors the recorded verdict survives.

    The margin is declared, not derived, so the useful question is not whether
    3 is the right number but whether the verdict turns on it. Both ends are
    re-derived from the walked rows alone, which is what lets
    ``check_priceability_screen.py`` recompute them.

    For an admitted candidate the upper end is where the chosen prefix stops
    clearing; it is a *lower bound* on the true upper end, because a stricter
    margin may still be met by a longer prefix the walk never needed to reach --
    one that carries more words and may or may not stay under the ceiling. For a
    rejected candidate the upper end is unbounded: a stricter margin only
    rejects more.
    """
    if chosen is not None:
        stop = rows[-1]
        upper = target / stop["bias_millihartree"]
        earlier = [row for row in rows[:-1] if row["bias_millihartree"] <= target]
        basis = (
            "admitted; the upper end is where the chosen prefix stops clearing and "
            "a longer prefix may extend it"
        )
    else:
        upper = None
        earlier = [
            row
            for row in rows
            if row["bias_millihartree"] <= target
            and row["source_word_universe"] <= ceiling
        ]
        basis = (
            "rejected; a stricter margin only rejects more, so the upper end is "
            "unbounded"
        )
    lower = (
        target / min(row["bias_millihartree"] for row in earlier) if earlier else 1.0
    )
    return {
        "declared_margin_factor": margin,
        "verdict_unchanged_from_margin_factor": lower,
        "verdict_unchanged_to_margin_factor": upper,
        "basis": basis,
    }


def build_candidate_record(
    spec: dict,
    *,
    arms: Sequence[str],
    grouping: str,
    target: float,
    margin: float,
    admissible_bias: float,
    ceiling: int,
    minimum_prefix: int,
) -> dict:
    model, construction = _build_model(spec)
    _, selected = _selected_generators(model, spec)
    reference = ExactMVBackend().state(model.reference, ())
    backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    # ``method='dense'``, for the reason run_protocol_cost.py states at length:
    # ARPACK's last bit is process-dependent and every bias here is a
    # millihartree residue taken against this reference.
    exact_energy = float(
        backend.ground_state(model.hamiltonian, k=1, method="dense")[0][0]
    )

    rows, chosen, reason = _walk(
        reference,
        model.hamiltonian,
        selected,
        exact_energy,
        admissible_bias=admissible_bias,
        ceiling=ceiling,
        minimum_prefix=minimum_prefix,
    )

    margin_stop = None
    if chosen is not None:
        arm_rows = _arm_rows(
            model, reference, selected[:chosen], exact_energy, arms, grouping,
            count_settings=True,
        )
        binding = max(row["word_universe"] for row in arm_rows)
        margin_stop = {
            "basis_size": chosen,
            "labels": [generator.label for generator in selected[:chosen]],
            "bias_millihartree": rows[-1]["bias_millihartree"],
            "binding_word_universe": binding,
            "binding_arms": [
                row["mapping"] for row in arm_rows if row["word_universe"] == binding
            ],
            "within_ceiling": binding <= ceiling,
            "arms": arm_rows,
        }

    # The frozen rule, priced the same way but without a grouping pass: the
    # contrast only needs W, and on H2O the intrinsic bank carries six figures
    # of words, where a QWC cover would dominate the screen's whole runtime.
    intrinsic_arms = _arm_rows(
        model, reference, selected, exact_energy, arms, grouping,
        count_settings=False,
    )
    intrinsic_binding = max(row["word_universe"] for row in intrinsic_arms)
    intrinsic_bias, _ = _solved(
        MatrixElementBank(reference, model.hamiltonian, list(selected)), exact_energy
    )

    admissible = margin_stop is not None and margin_stop["within_ceiling"]
    if admissible:
        verdict_reason = "clears the margin at a binding word universe within the ceiling"
    elif margin_stop is None:
        verdict_reason = reason
    else:
        verdict_reason = "clears the margin but its binding word universe exceeds the ceiling"

    return {
        "candidate": spec["key"],
        "label": spec["label"],
        "construction": construction,
        "n_qubits": model.n,
        "exact_sector_energy": exact_energy,
        "frozen_ordering_size": len(selected),
        "prefix_walk": rows,
        "walk_terminated_because": reason,
        "margin_sensitivity": _margin_sensitivity(
            rows, chosen, target=target, ceiling=ceiling, margin=margin
        ),
        "margin_stop": margin_stop,
        "intrinsic_stop": {
            "basis_size": len(selected),
            "bias_millihartree": intrinsic_bias,
            "binding_word_universe": intrinsic_binding,
            "within_ceiling": intrinsic_binding <= ceiling,
            "arms": intrinsic_arms,
        },
        "verdict": {
            "admissible_for_a_probe": admissible,
            "reason": verdict_reason,
            # True only where the frozen rule would have put this same instance
            # above the ceiling -- the case where the stopping rule, not the
            # instance, is what the earlier record rejected.
            "rule_change_is_what_admits_it": bool(
                admissible and intrinsic_binding > ceiling
            ),
        },
    }


def build_record(config: dict | None = None) -> dict:
    config = load_config() if config is None else config
    gates = config["acceptance_gates"]
    target = float(gates["accuracy_target_millihartree"])
    margin = float(gates["margin_factor"])
    admissible_bias = target / margin
    ceiling = int(gates["word_universe_ceiling"])
    specs = candidate_specs()
    grouping = config["grouping_protocol_by_system"]
    minimum_prefix = int(config["selection_rule"]["minimum_prefix_size"])

    candidates = [
        build_candidate_record(
            specs[key],
            arms=config["mapping_arms"],
            grouping=grouping[key],
            target=target,
            margin=margin,
            admissible_bias=admissible_bias,
            ceiling=ceiling,
            minimum_prefix=minimum_prefix,
        )
        for key in config["candidates"]
    ]

    admitted = [c["candidate"] for c in candidates if c["verdict"]["admissible_for_a_probe"]]
    record = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_tier": "structural",
        "margin_factor_status": gates["margin_factor_status"],
        "word_universe_convention": WORD_UNIVERSE_CONVENTION,
        "estimand": config["estimand"],
        "selection_rule": config["selection_rule"],
        "acceptance_gates": gates,
        "derived_gates": {
            "admissible_bias_millihartree": admissible_bias,
            "statistical_allowance_millihartree": math.sqrt(
                max(target**2 - admissible_bias**2, 0.0)
            ),
            # Two different fractions, easy to conflate and previously conflated
            # here. RMSE combines bias and sampling scatter in quadrature, so a
            # bank at the admissible bias takes (bias/target)^2 = 1/margin^2 of
            # the *MSE* budget -- 11.1% at margin 3. What is left for the
            # statistical component is an RMSE, and it falls from the target by
            # only 5.7%, because the square root pulls the two much closer
            # together than the MSE share suggests.
            "bias_share_of_mse_budget": (admissible_bias / target) ** 2,
            "statistical_allowance_reduction_fraction": 1.0
            - math.sqrt(max(target**2 - admissible_bias**2, 0.0)) / target,
            "statistical_allowance_note": (
                "The bank's share of the MSE budget is "
                "bias_share_of_mse_budget; the reduction it forces in the "
                "remaining statistical RMSE allowance is "
                "statistical_allowance_reduction_fraction. The second is the "
                "smaller number and is not a fraction of a shot budget consumed."
            ),
        },
        "mapping_arms": list(config["mapping_arms"]),
        "omitted_candidates": config["omitted_candidates"],
        "candidates": candidates,
        "summary": {
            "admissible_for_a_probe": admitted,
            "rejected": {
                c["candidate"]: c["verdict"]["reason"]
                for c in candidates
                if not c["verdict"]["admissible_for_a_probe"]
            },
            "admitted_only_by_the_rule_change": [
                c["candidate"]
                for c in candidates
                if c["verdict"]["rule_change_is_what_admits_it"]
            ],
        },
        "claim_boundary": config["claim_boundary"],
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record()
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for candidate in record["candidates"]:
        stop = candidate["margin_stop"]
        verdict = "ADMIT" if candidate["verdict"]["admissible_for_a_probe"] else "reject"
        intrinsic = candidate["intrinsic_stop"]
        if stop is None:
            detail = f"no prefix clears {candidate['walk_terminated_because']}"
        else:
            detail = (
                f"M={stop['basis_size']} bias={stop['bias_millihartree']:.4f} mHa "
                f"W={stop['binding_word_universe']}"
            )
        print(
            f"{verdict:>6}  {candidate['candidate']:14s} {detail}"
            f"  (intrinsic M={intrinsic['basis_size']} "
            f"bias={intrinsic['bias_millihartree']:.4f} mHa "
            f"W={intrinsic['binding_word_universe']})"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
