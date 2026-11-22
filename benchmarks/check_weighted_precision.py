"""Independent verifier for the weighted measurement & precision acquisition study.

Supports:
- ``--declaration-only``: validates the result-free declaration, historical
  digests, FCIDUMP hashes, seed uniqueness, schedule rules, and independent
  occupation-space anchors (without drawing any outcomes).
- ``--full-anchor``: additionally cross-checks native Pauli-bank word counts
  and exact expectation matrices against the independent occupation-space
  matrices.
- ``--report``: audits every committed Part A and Part B row, independently
  recomputes all 36 noisy infinite-shot anchors, enforces shot/schedule/guard
  accounting, computes unconditional and stop-conditional 68%/95% coverage and
  widths, and evaluates the frozen P1a--P3 endpoints and shot-ratio CIs.
- ``--spot-replay``: replays designated replica-0 attempts from scratch and
  checks exact agreement with the committed record.
- ``--verify-history``: enforces strict Git chronology between declaration
  and result commits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_weighted_precision as producer  # noqa: E402
from benchmarks.experiment_artifacts import read_json  # noqa: E402
from benchmarks.noisy_readout_sim import (  # noqa: E402
    NoisyReadoutSimulator,
    ReadoutNoiseModel,
)
from clifford_qc.measurement import DeviceCard, NoiseGatePolicy  # noqa: E402
from clifford_qc.sampled_reference import PreparedSampledReference  # noqa: E402
from clifford_qc.subspace import RankStopPolicy  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, *, rel_tol: float = 1e-8, abs_tol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def z_score(level: float, family: int) -> float:
    return NormalDist().inv_cdf(1.0 - (1.0 - level) / (2.0 * family))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact P(Binomial(n, p) <= k) using stdlib math.comb (no SciPy required)."""
    return float(sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1)))


def check_interval(
    entry: dict[str, Any],
    estimate: float,
    scale: float,
    delta: float,
    family: int,
    evidence: str,
) -> None:
    require(entry["evidence"] == evidence and close(entry["delta"], delta), "interval metadata mismatch")
    require(close(entry["estimate"], estimate), "interval center mismatch")
    radius = z_score(1.0 - delta, family) * scale
    require(
        close(entry["lower"], estimate - radius) and close(entry["upper"], estimate + radius),
        "interval radius mismatch",
    )


def validate_declaration_structure(d: dict[str, Any]) -> dict[str, Any]:
    """Validate internal declaration consistency, arm/case bindings, and seed uniqueness."""
    case_names = [c["name"] for c in d["cases"]]
    require(len(case_names) == len(set(case_names)) and len(case_names) >= 1, "duplicate or empty cases")
    case_set = set(case_names)

    a_keys = [a["key"] for a in d["part_a_arms"]]
    require(len(a_keys) == len(set(a_keys)), "duplicate Part A arms")
    for arm in d["part_a_arms"]:
        require(set(arm["cases"]).issubset(case_set), f"unknown case in arm {arm['key']}")
        if arm["grouping"] == "ritz_replan":
            require(arm["pilot_arm"] in a_keys, f"unknown pilot_arm in {arm['key']}")
            require(float(arm["overlap_weight"]) >= 0.0, "negative overlap_weight")

    b_keys = [a["key"] for a in d["part_b_arms"]]
    require(len(b_keys) == len(set(b_keys)), "duplicate Part B arms")
    reg_keys = [r["key"] for r in d["part_b_noise_regimes"]]
    require(len(reg_keys) == len(set(reg_keys)), "duplicate Part B noise regimes")

    card = DeviceCard.load(ROOT / d["device_card"])
    for regime in d["part_b_noise_regimes"]:
        ReadoutNoiseModel(
            p_1q=regime["p_1q"],
            p_2q=regime["p_2q"],
            routing_2q_multiplier=regime["routing_2q_multiplier"],
            p_readout=regime["p_readout"],
            label=regime["key"],
        )
        for tier_key in ("galic_tight_eps", "galic_moderate_eps"):
            NoiseGatePolicy(
                two_qubit_error=regime["p_2q"],
                max_relative_bias=regime[tier_key],
                routing_2q_multiplier=regime["routing_2q_multiplier"],
                device_card=card,
            )

    # Verify global seed uniqueness across every declared job and stream.
    jobs = producer.expected_job_keys(d)
    seen_seeds: set[int] = set()
    for part, ci, ai, reg_idx, rep in jobs:
        part_id = 0 if part == "A" else 1
        streams = (
            (0, 1, 2)
            if (part == "A" and d["part_a_arms"][ai]["grouping"] == "ritz_replan")
            else (0, 1)
        )
        for stream in streams:
            s = producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, stream)
            require(s not in seen_seeds, f"seed collision detected for {(part, ci, ai, reg_idx, rep, stream)}")
            seen_seeds.add(s)

    return {
        "cases": len(case_names),
        "part_a_jobs": sum(1 for j in jobs if j[0] == "A"),
        "part_b_jobs": sum(1 for j in jobs if j[0] == "B"),
        "total_jobs": len(jobs),
        "unique_seeds": len(seen_seeds),
    }


def anchors(d: dict[str, Any], *, full: bool = False) -> list[dict[str, Any]]:
    """Independently compute occupation-space anchors for every declared case."""
    from scipy.linalg import eigh

    from benchmarks.check_cross_domain_ratio import case_data
    from benchmarks.check_stretched_molecular_stop import independent_anchor

    systems: list[dict[str, Any]] = []
    for ci, case in enumerate(d["cases"]):
        if case["kind"] == "molecule_8q":
            indep = case_data({**case, "kind": "molecule"})
            s_mat, h_mat = indep["overlap"], indep["projected"]
            exact = float(eigh(h_mat, s_mat, eigvals_only=True)[0])
            diag = np.sqrt(np.real(np.diag(s_mat)))
            s_norm = s_mat / np.outer(diag, diag)
            spectrum = sorted(np.linalg.eigvalsh(0.5 * (s_norm + s_norm.conj().T)).tolist(), reverse=True)
            require(exact >= float(indep["ground"]) - 1e-9, f"{case['name']}: projected energy below singlet ground")
            require(len(spectrum) == int(case["expected_rank"]), f"{case['name']}: rank mismatch")
            require(spectrum[-1] > float(case["solver"]["tau_s"]), f"{case['name']}: weak mode below tau_s")
            anchor: dict[str, Any] = {
                "case": ci,
                "name": case["name"],
                "independent_target": exact,
                "exact_singlet_ground": float(indep["ground"]),
                "rank": len(spectrum),
                "overlap_spectrum": spectrum,
            }
        elif case["kind"] == "molecule_10q_stretched":
            _words, _h_occ, s_mat, h_mat, spec_asc, target, ground = independent_anchor(case)
            spectrum = sorted(spec_asc.tolist(), reverse=True)
            tau = float(case["solver"]["tau_s"])
            require(spectrum[-1] > tau and spectrum[-1] - tau < 0.001, "stress case does not straddle cutoff")
            require(len(spectrum) == int(case["expected_rank"]), "stretched rank mismatch")
            anchor = {
                "case": ci,
                "name": case["name"],
                "independent_target": float(target),
                "exact_singlet_ground": float(ground),
                "rank": len(spectrum),
                "overlap_spectrum": spectrum,
            }
        else:
            raise ValueError(f"unsupported case kind: {case['kind']!r}")

        if full:
            h, generators, state = producer.build_case_inputs(case)
            sim = NoisyReadoutSimulator(state, int(case["n"]), ReadoutNoiseModel.ideal())
            ref = PreparedSampledReference(int(case["n"]), f"{case['name']}-anchor", sim.draw)
            prov = producer.build_arm_provider(
                ref, h, generators, int(case["n"]), block_size=2, grouping="sorted_insertion"
            )
            require(len(prov.session.words) == int(case["expected_words"]), f"{case['name']}: word count mismatch")
            exps = sim.exact_word_expectations(prov.plan.settings)
            native_e, native_rank, native_spec = producer.exact_projected_from_expectations(
                prov, exps, case["solver"]
            )
            require(close(native_e, anchor["independent_target"], abs_tol=1e-9), f"{case['name']}: target mismatch")
            require(native_rank == anchor["rank"], f"{case['name']}: native rank mismatch")
            require(
                bool(np.allclose(native_spec, anchor["overlap_spectrum"], atol=1e-9, rtol=0.0)),
                f"{case['name']}: overlap spectrum mismatch",
            )
            anchor["words"] = len(prov.session.words)
        systems.append(anchor)
    return systems


def verify_noisy_anchors(
    record: dict[str, Any],
    d: dict[str, Any],
    systems: list[dict[str, Any]],
    *,
    case_inputs_fn: Callable[[dict[str, Any]], tuple[Any, Any, np.ndarray]] | None = None,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Independently recompute and audit every declared Part B noisy anchor."""
    build_inputs = producer.build_case_inputs if case_inputs_fn is None else case_inputs_fn
    expected_keys = {
        (ci, regime["key"], arm["key"])
        for ci, case in enumerate(d["cases"])
        if "B" in case["parts"]
        for regime in d["part_b_noise_regimes"]
        for arm in d["part_b_arms"]
    }
    raw_anchors = record.get("noisy_anchors")
    require(isinstance(raw_anchors, list), "record missing noisy_anchors list")
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for entry in raw_anchors:
        key = (int(entry["case"]), str(entry["regime"]), str(entry["arm"]))
        require(key in expected_keys and key not in by_key, f"unexpected or duplicate noisy anchor: {key}")
        by_key[key] = entry
    require(set(by_key) == expected_keys, f"incomplete noisy anchors: got {len(by_key)} of {len(expected_keys)}")

    card = DeviceCard.load(ROOT / d["device_card"])
    for ci, case in enumerate(d["cases"]):
        if "B" not in case["parts"]:
            continue
        n = int(case["n"])
        exact_e = float(systems[ci]["independent_target"])
        h, generators, state = build_inputs(case)
        # Recompile each unique (block_size, grouping, max_logical_2q) plan independently.
        plan_cache: dict[tuple[int, str, int | None], Any] = {}
        for regime in d["part_b_noise_regimes"]:
            nmodel = ReadoutNoiseModel(
                p_1q=regime["p_1q"],
                p_2q=regime["p_2q"],
                routing_2q_multiplier=regime["routing_2q_multiplier"],
                p_readout=regime["p_readout"],
                label=regime["key"],
            )
            nsim = NoisyReadoutSimulator(state, n, nmodel)
            nref = PreparedSampledReference(n, f"{case['name']}-{regime['key']}-check", nsim.draw)
            for arm in d["part_b_arms"]:
                key = (ci, regime["key"], arm["key"])
                recorded = by_key[key]
                require(recorded["case_name"] == case["name"], f"noisy anchor case_name mismatch: {key}")
                bs = n if arm["block_size"] == "full" else int(arm["block_size"])
                tier = arm["gate_tier"]
                gate: NoiseGatePolicy | None = None
                if tier == "tight":
                    gate = NoiseGatePolicy(
                        two_qubit_error=regime["p_2q"],
                        max_relative_bias=regime["galic_tight_eps"],
                        routing_2q_multiplier=regime["routing_2q_multiplier"],
                        device_card=card,
                    )
                elif tier == "moderate":
                    gate = NoiseGatePolicy(
                        two_qubit_error=regime["p_2q"],
                        max_relative_bias=regime["galic_moderate_eps"],
                        routing_2q_multiplier=regime["routing_2q_multiplier"],
                        device_card=card,
                    )
                elif tier != "none":
                    raise ValueError(f"unknown gate_tier {tier!r}")

                ck = (bs, arm["grouping"], None if gate is None else gate.max_logical_two_qubit_gates)
                if ck not in plan_cache:
                    plan_cache[ck] = producer.build_arm_provider(
                        nref,
                        h,
                        generators,
                        n,
                        block_size=bs,
                        grouping=arm["grouping"],
                        noise_gate=gate,
                    )
                prov = plan_cache[ck]
                # Re-evaluate metrics under this regime's exact gate policy.
                from clifford_qc.measurement import evaluate_plan_variance

                bm = evaluate_plan_variance(
                    prov.plan,
                    prov.session.matrix_functionals(),
                    strategy=arm["grouping"],
                    noise_gate=gate,
                )
                nexps = nsim.exact_word_expectations(prov.plan.settings)
                ne, nrank, nevals = producer.exact_projected_from_expectations(
                    prov, nexps, case["solver"]
                )

                require(recorded["noise_model"] == nmodel.as_dict(), f"noise_model mismatch: {key}")
                require(
                    recorded["noise_gate"] == (None if gate is None else gate.as_dict()),
                    f"noise_gate mismatch: {key}",
                )
                require(recorded["settings"] == bm.n_settings, f"noisy anchor settings mismatch: {key}")
                require(close(recorded["r_hat"], bm.r_hat, rel_tol=1e-7, abs_tol=1e-9), f"noisy anchor r_hat mismatch: {key}")
                require(recorded["cz_total"] == int(bm.two_qubit_gates_total), f"noisy anchor cz_total mismatch: {key}")
                require(recorded["cz_max"] == int(bm.two_qubit_gates_max), f"noisy anchor cz_max mismatch: {key}")
                require(
                    recorded["admissible_settings"] == bm.noise_gate_admissible_settings,
                    f"noisy anchor admissible_settings mismatch: {key}",
                )
                if gate is not None:
                    require(
                        recorded["admissible_settings"] == recorded["settings"],
                        f"gated noisy anchor contains inadmissible settings: {key}",
                    )
                    if gate.max_logical_two_qubit_gates is not None:
                        require(
                            recorded["cz_max"] <= gate.max_logical_two_qubit_gates,
                            f"gated noisy anchor exceeds max_logical_two_qubit_gates: {key}",
                        )
                require(
                    close(recorded["noisy_exact_energy_Ha"], ne, abs_tol=1e-9),
                    f"noisy_exact_energy_Ha mismatch: {key}",
                )
                require(
                    close(recorded["bias_Ha"], ne - exact_e, abs_tol=1e-9),
                    f"bias_Ha mismatch: {key}",
                )
                require(recorded["noisy_rank"] == int(nrank), f"noisy_rank mismatch: {key}")
                require(
                    bool(np.allclose(recorded["noisy_overlap_spectrum"], nevals, atol=1e-9, rtol=0.0)),
                    f"noisy_overlap_spectrum mismatch: {key}",
                )
    return by_key


def _check_rank_guard(
    guard: dict[str, Any],
    look: dict[str, Any],
    case: dict[str, Any],
    required_rank: int,
    consecutive_required: int,
    prev_streak: int,
) -> tuple[bool, bool, int]:
    """Validate rank_guard diagnostics and return (resolved, rank_ok, new_streak)."""
    require(isinstance(guard, dict), "missing rank_guard dictionary")
    require(
        guard.get("required_rank") == required_rank and guard.get("evidence") == "heuristic",
        "rank_guard metadata mismatch",
    )
    live_rank = guard.get("live_rank")
    require(isinstance(live_rank, int) and 0 <= live_rank <= required_rank, "invalid live_rank")
    comparisons = guard.get("comparisons")
    require(isinstance(comparisons, list), "invalid rank_guard comparisons")
    if comparisons:
        require(len(comparisons) == 3, "rank_guard must report 3 cutoff comparisons")
        weakest = float(guard["weakest"])
        require(math.isfinite(weakest), "nonfinite rank_guard weakest")
        expected_names = ("absolute", "relative", "condition")
        for comp, name in zip(comparisons, expected_names, strict=True):
            require(comp["cutoff"] == name, f"unexpected rank_guard cutoff name: {comp['cutoff']!r}")
            thresh = float(comp["threshold"])
            radius = float(comp["radius"])
            margin = float(comp["margin"])
            lower = float(comp["lower_margin"])
            require(
                all(math.isfinite(x) for x in (thresh, radius, margin, lower)) and radius >= 0.0,
                "invalid rank_guard comparison scalars",
            )
            if name == "absolute":
                require(close(thresh, float(case["solver"]["tau_s"])), "absolute cutoff threshold mismatch")
            elif name == "relative":
                require(close(thresh, float(case["solver"].get("relative_tau", 0.0))), "relative cutoff threshold mismatch")
            require(close(margin, weakest - thresh), "rank_guard margin arithmetic mismatch")
            require(close(lower, margin - radius), "rank_guard lower_margin arithmetic mismatch")
        min_lower = min(float(c["lower_margin"]) for c in comparisons)
        require(close(float(guard["lower_margin"]), min_lower), "rank_guard minimum lower_margin mismatch")
        resolved = (int(look["retained_rank"]) == required_rank) and (min_lower > 0.0)
    else:
        require(guard.get("lower_margin") is None, "empty comparisons with non-None lower_margin")
        resolved = False
    new_streak = prev_streak + 1 if resolved else 0
    rank_ok = new_streak >= consecutive_required
    require(
        guard.get("resolved") is resolved
        and guard.get("consecutive_resolved") == new_streak
        and guard.get("passed") is rank_ok,
        "rank_guard decision/streak mismatch",
    )
    return resolved, rank_ok, new_streak


def check_row(
    row: dict[str, Any],
    d: dict[str, Any],
    case: dict[str, Any],
    system: dict[str, Any],
    *,
    expected_plan: dict[str, Any] | None = None,
    pilot_settings: int | None = None,
) -> None:
    """Strictly audit one recorded attempt's seeds, shot billing, schedule, guards, and intervals."""
    part = row["part"]
    require(part in ("A", "B"), f"unknown part: {part!r}")
    part_id = 0 if part == "A" else 1
    ci, ai, reg_idx, rep = row["case"], row["arm_index"], row["regime_index"], row["replica"]
    require(row.get("case_name") == case["name"], "row case_name mismatch")

    if part == "A":
        require(0 <= ai < len(d["part_a_arms"]), "invalid Part A arm_index")
        arm = d["part_a_arms"][ai]
        require(row["arm"] == arm["key"], "Part A row arm label mismatch")
        require(case["name"] in arm["cases"], "Part A arm not declared for case")
        require(row["regime"] == "ideal" and reg_idx == 0, "Part A regime mismatch")
        is_replan = arm["grouping"] == "ritz_replan"
        width_guard = bool(arm["width_guard"])
    else:
        require("B" in case["parts"], "Part B not declared for case")
        require(0 <= ai < len(d["part_b_arms"]), "invalid Part B arm_index")
        require(0 <= reg_idx < len(d["part_b_noise_regimes"]), "invalid Part B regime_index")
        arm = d["part_b_arms"][ai]
        regime = d["part_b_noise_regimes"][reg_idx]
        require(row["arm"] == arm["key"], "Part B row arm label mismatch")
        require(row["regime"] == regime["key"], "Part B row regime label mismatch")
        is_replan = False
        width_guard = False

    require(
        row["acq_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 0),
        "acq_seed mismatch",
    )
    require(
        row["boot_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 1),
        "boot_seed mismatch",
    )

    n_settings = row.get("settings")
    require(isinstance(n_settings, int) and n_settings >= 1, "invalid row settings count")
    if expected_plan is not None and not is_replan:
        require(n_settings == int(expected_plan["settings"]), "row settings mismatch with compiled plan")
        if "cz_total" in expected_plan and "cz_total" in row:
            require(int(row["cz_total"]) == int(expected_plan["cz_total"]), "row cz_total mismatch")
        if "cz_max" in expected_plan and "cz_max" in row:
            require(int(row["cz_max"]) == int(expected_plan["cz_max"]), "row cz_max mismatch")

    if is_replan:
        require(
            row["pilot_seed"] == producer.derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, rep, 2),
            "pilot_seed mismatch",
        )
        resolved_pilot_settings = pilot_settings
        if resolved_pilot_settings is None and "part_a_plans" in system:
            pilot_arm_key = arm["pilot_arm"]
            if pilot_arm_key in system["part_a_plans"]:
                resolved_pilot_settings = int(system["part_a_plans"][pilot_arm_key]["settings"])
        if resolved_pilot_settings is not None:
            pilot_pg0 = max(2, int(case["replan_pilot_total_shots"]) // resolved_pilot_settings)
            expected_pilot_shots = pilot_pg0 * resolved_pilot_settings
            if row["failure"] is None:
                require(row["pilot_shots"] == expected_pilot_shots, "paid pilot_shots mismatch")
            else:
                require(row["pilot_shots"] in (0, expected_pilot_shots), "failed replan pilot_shots mismatch")
        else:
            require(row["pilot_shots"] > 0 if row["failure"] is None else row["pilot_shots"] >= 0, "invalid pilot_shots")
    else:
        require(
            row["pilot_seed"] is None and row["pilot_shots"] == 0 and row.get("pilot_fingerprint") is None,
            "unexpected pilot fields on non-replan arm",
        )

    require(
        isinstance(row["pilot_shots"], int)
        and isinstance(row["acquisition_shots"], int)
        and isinstance(row["total_shots"], int)
        and row["pilot_shots"] >= 0
        and row["acquisition_shots"] >= 0
        and row["total_shots"] == row["pilot_shots"] + row["acquisition_shots"],
        "shot sum mismatch",
    )
    require(row["device_seconds"] >= 0.0 and math.isfinite(row["device_seconds"]), "invalid device_seconds")

    if row["failure"] is not None:
        null_fields = (
            "estimate",
            "std_error",
            "delta_interval",
            "naive_interval",
            "bootstrap_interval",
            "final_fingerprint",
            "schedule",
        )
        require(
            row["status"] == "failed" and all(row[k] is None for k in null_fields),
            "invalid failure row",
        )
        return

    totals = [int(x) for x in case["cumulative_total_shots"]]
    family = len(totals)
    if is_replan:
        rem_totals = [max(2 * n_settings, tot - row["pilot_shots"]) for tot in totals]
        exp_pilot_pg, exp_rounds = producer.make_schedule(n_settings, rem_totals)
    else:
        exp_pilot_pg, exp_rounds = producer.make_schedule(n_settings, totals)

    exp_cum: list[int] = [exp_pilot_pg * n_settings]
    for budget in exp_rounds:
        exp_cum.append(exp_cum[-1] + budget)
    require(all(x > 0 for x in exp_cum), "expected cumulative shots must be positive")

    # Validate recorded schedule against declared schedule rule.
    expected_schedule: dict[str, Any] = {
        "pilot_shots_per_group": exp_pilot_pg,
        "round_budgets": list(exp_rounds),
        "target_std_error": float(d["target_std_error_Ha"]),
        "delta": float(d["delta"]),
        "interval_family": family,
        "solver_kwargs": case["solver"],
        "rank_stop": asdict(RankStopPolicy(**d["rank_policy"])),
        "bootstrap": {"replicates": int(case["bootstrap_replicas"]), "seed": row["boot_seed"]},
    }
    if width_guard:
        expected_schedule["max_bootstrap_half_width"] = float(d["bootstrap_half_width_Ha"])
    require(row["schedule"] == expected_schedule, "recorded schedule differs from declared schedule")

    looks = row["looks"]
    require(1 <= len(looks) <= family, "invalid look count")
    require(is_sha256(row["final_fingerprint"]), "invalid final_fingerprint")
    if is_replan:
        require(is_sha256(row["pilot_fingerprint"]), "invalid pilot_fingerprint")

    # Enforce per-look cumulative shots, rank guard arithmetic, and bootstrap width guard.
    target_se = float(d["target_std_error_Ha"])
    delta = float(d["delta"])
    consec_req = int(d["rank_policy"]["consecutive_looks"])
    boot_reps = int(case["bootstrap_replicas"])
    streak = 0
    eligible: list[bool] = []
    for idx, look in enumerate(looks):
        require(look["look"] == idx, "look index mismatch")
        require(look["shots"] == exp_cum[idx], f"look {idx} cumulative shots mismatch: {look['shots']} != {exp_cum[idx]}")
        require(1 <= int(look["retained_rank"]) <= int(system["rank"]), "invalid retained_rank")
        require(
            math.isfinite(float(look["energy"]))
            and math.isfinite(float(look["std_error"]))
            and float(look["std_error"]) >= 0.0,
            "invalid look energy or std_error",
        )
        resolved, rank_ok, streak = _check_rank_guard(
            look.get("rank_guard"), look, case, int(system["rank"]), consec_req, streak
        )
        se_ok = float(look["std_error"]) <= target_se
        if not resolved and idx < len(looks) - 1:
            expected_rank_budget = int(exp_rounds[idx]) if se_ok else (int(exp_rounds[idx]) + 1) // 2
            require(look.get("next_rank_budget") == expected_rank_budget, "next_rank_budget mismatch")
        else:
            require("next_rank_budget" not in look, "unexpected next_rank_budget")

        if width_guard and "bootstrap_seed" in look:
            exp_bseed = int(np.random.SeedSequence([row["boot_seed"], idx]).generate_state(1, np.uint64)[0])
            require(
                look["bootstrap_seed"] == exp_bseed
                and look.get("bootstrap_attempted_replicates") == boot_reps,
                "width-guard bootstrap seed or replicate count mismatch",
            )
        if width_guard and se_ok and rank_ok:
            require("bootstrap_seed" in look and "bootstrap_width_passed" in look, "missing width guard assessment")
            if "bootstrap_spread" in look:
                spread_val = float(look["bootstrap_spread"])
                require(math.isfinite(spread_val) and spread_val >= 0.0, "invalid look bootstrap_spread")
                width_ok = z_score(1.0 - delta, family) * spread_val <= float(d["bootstrap_half_width_Ha"]) + 1e-13
            else:
                require("bootstrap_failure" in look, "width check missing both bootstrap_spread and bootstrap_failure")
                width_ok = False
            require(look["bootstrap_width_passed"] is width_ok, "bootstrap_width_passed decision mismatch")
        else:
            require("bootstrap_width_passed" not in look, "spurious bootstrap_width_passed on ineligible look")
            width_ok = not width_guard

        eligible.append(se_ok and rank_ok and width_ok)
        if idx < len(looks) - 1:
            require(not look["stopped"] and not eligible[-1], "missed earlier eligible stop")

    last = looks[-1]
    require(
        last["stopped"] is True
        and close(last["energy"], row["estimate"])
        and close(last["std_error"], row["std_error"]),
        "terminal look mismatch",
    )
    require(
        row["acquisition_shots"] == exp_cum[len(looks) - 1],
        f"acquisition_shots {row['acquisition_shots']} != declared look total {exp_cum[len(looks) - 1]}",
    )
    require(
        row["total_shots"] == row["pilot_shots"] + row["acquisition_shots"],
        "terminal total_shots mismatch",
    )
    require(
        row["measured_solves"] >= len(looks) + (1 if is_replan else 0),
        "measured_solves undercounted",
    )
    require(row["status"] in ("target_met", "schedule_exhausted"), f"unexpected terminal status: {row['status']!r}")
    require((row["status"] == "target_met") == eligible[-1], "terminal status inconsistent with stop eligibility")
    if row["status"] == "schedule_exhausted":
        require(len(looks) == family and not eligible[-1], "premature or invalid schedule_exhausted")

    # Verify delta & bootstrap interval formulas.
    naive = row["naive_interval"]
    scale = (float(naive["upper"]) - float(naive["lower"])) / (2.0 * z_score(1.0 - delta, 1))
    require(math.isfinite(scale) and scale >= 0.0, "invalid delta scale")
    require(close(naive["std_error"], row["std_error"]) and close(row["delta_interval"]["std_error"], row["std_error"]), "delta SE mismatch")
    check_interval(naive, row["estimate"], scale, delta, 1, "asymptotic")
    check_interval(row["delta_interval"], row["estimate"], scale, delta, family, "asymptotic")
    if row["bootstrap_interval"] is not None:
        spread = float(last["bootstrap_spread"])
        require(close(row["bootstrap_interval"]["std_error"], spread), "bootstrap spread mismatch")
        check_interval(row["bootstrap_interval"], row["estimate"], spread, delta, family, "heuristic")
    else:
        require("bootstrap_failure" in last, "missing bootstrap_interval without bootstrap_failure")


def _interval_scale(row: dict[str, Any], kind: str, delta: float) -> float | None:
    if row["failure"] is not None:
        return None
    if kind == "delta":
        naive = row["naive_interval"]
        if naive is None:
            return None
        return (float(naive["upper"]) - float(naive["lower"])) / (2.0 * z_score(1.0 - delta, 1))
    boot = row["bootstrap_interval"]
    if boot is None:
        return None
    return float(boot["std_error"])


def _coverage_block(
    rows: list[dict[str, Any]],
    exact: float,
    family: int,
    delta: float,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    """Compute unconditional (target_met / all attempts) and stop-conditional coverage & widths."""
    n = len(rows)
    met = [r for r in rows if r["failure"] is None and r["status"] == "target_met"]
    nonfailed = [r for r in rows if r["failure"] is None]
    out: dict[str, Any] = {}
    for kind in ("delta", "bootstrap"):
        out[kind] = {}
        for level in levels:
            z = z_score(level, family)
            met_widths: list[float] = []
            covered = 0
            for r in met:
                sc = _interval_scale(r, kind, delta)
                if sc is None:
                    continue
                rad = z * sc
                met_widths.append(2.0 * rad)
                if r["estimate"] - rad <= exact <= r["estimate"] + rad:
                    covered += 1
            all_widths = [
                2.0 * z * sc
                for r in nonfailed
                for sc in [_interval_scale(r, kind, delta)]
                if sc is not None
            ]
            key = f"{int(round(level * 100))}"
            out[kind][key] = {
                "unconditional_covered": covered,
                "unconditional_n": n,
                "unconditional_coverage": (covered / n) if n else 0.0,
                "unconditional_wilson95": wilson_interval(covered, n),
                "unconditional_p_lower": exact_binomial_cdf(covered, n, level) if n else 1.0,
                "conditional_covered": covered,
                "conditional_n": len(met),
                "conditional_coverage": (covered / len(met)) if met else None,
                "median_width_mHa_among_stops": (
                    float(np.median(met_widths) * 1000.0) if met_widths else None
                ),
                "median_width_mHa_all_nonfailed": (
                    float(np.median(all_widths) * 1000.0) if all_widths else None
                ),
            }
    return out


def bootstrap_shot_ratio_ci(
    treat_shots: list[int],
    comp_shots: list[int],
    *,
    seed_root: int,
    comp_index: int,
    replicates: int,
    alpha: float,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap CI for median and mean shot ratios over all attempts."""
    t_arr = np.asarray(treat_shots, dtype=float)
    c_arr = np.asarray(comp_shots, dtype=float)
    require(t_arr.size >= 1 and c_arr.size >= 1, "empty shot sample for ratio CI")
    require(bool(np.all(c_arr > 0.0)), "nonpositive comparator shots")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed_root), int(comp_index)]))
    idx_t = rng.integers(0, t_arr.size, size=(int(replicates), t_arr.size))
    idx_c = rng.integers(0, c_arr.size, size=(int(replicates), c_arr.size))
    t_boot = t_arr[idx_t]
    c_boot = c_arr[idx_c]
    med_ratios = np.median(t_boot, axis=1) / np.median(c_boot, axis=1)
    mean_ratios = np.mean(t_boot, axis=1) / np.mean(c_boot, axis=1)
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        "median_ratio": float(np.median(t_arr) / np.median(c_arr)),
        "median_ci": [float(np.percentile(med_ratios, lo_q)), float(np.percentile(med_ratios, hi_q))],
        "mean_ratio": float(np.mean(t_arr) / np.mean(c_arr)),
        "mean_ci": [float(np.percentile(mean_ratios, lo_q)), float(np.percentile(mean_ratios, hi_q))],
        "p_mean_ge_one": float(np.mean(mean_ratios >= 1.0)),
        "p_median_ge_one": float(np.mean(med_ratios >= 1.0)),
    }


def summarize(record: dict[str, Any], d: dict[str, Any], systems: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute unconditional & stop-conditional cell summaries plus frozen P1a--P3 verdicts."""
    target = float(d["target_std_error_Ha"])
    delta = float(d["delta"])
    analysis = d["analysis"]
    levels = tuple(float(x) for x in analysis["coverage_levels"])

    part_a_summary: list[dict[str, Any]] = []
    part_a_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ci, case in enumerate(d["cases"]):
        exact = float(systems[ci]["independent_target"])
        rank = int(systems[ci]["rank"])
        family = len(case["cumulative_total_shots"])
        for arm in d["part_a_arms"]:
            if case["name"] not in arm["cases"]:
                continue
            rows = [r for r in record["rows"] if r["part"] == "A" and r["case"] == ci and r["arm"] == arm["key"]]
            part_a_rows_by_key[(case["name"], arm["key"])] = rows
            n = len(rows)
            nonfailed = [r for r in rows if r["failure"] is None]
            met = [r for r in nonfailed if r["status"] == "target_met"]
            errors_all = [r["estimate"] - exact for r in nonfailed]
            errors_met = [r["estimate"] - exact for r in met]
            true_1s = sum(abs(e) <= target for e in errors_met)
            false_1s = len(met) - true_1s
            within_2s = sum(abs(e) <= 2.0 * target for e in errors_met)
            cov = _coverage_block(rows, exact, family, delta, levels)
            cal_ratios = [
                r["r_emp_ritz"] / r["r_hat_ritz"]
                for r in nonfailed
                if r.get("r_emp_ritz") is not None
                and r.get("r_hat_ritz") is not None
                and float(r["r_hat_ritz"]) > 0.0
                and math.isfinite(float(r["r_emp_ritz"]))
            ]
            part_a_summary.append(
                {
                    "case": case["name"],
                    "arm": arm["key"],
                    "attempts": n,
                    "target_met": len(met),
                    "true_precision_stops": true_1s,
                    "false_precision_stops": false_1s,
                    "within_2sigma_stops": within_2s,
                    "reduced_rank_stops": sum(r["looks"][-1]["retained_rank"] < rank for r in met),
                    "exhausted": sum(r["status"] == "schedule_exhausted" for r in rows),
                    "failed": n - len(nonfailed),
                    "median_total_shots": float(np.median([r["total_shots"] for r in rows])) if rows else None,
                    "mean_total_shots": float(np.mean([r["total_shots"] for r in rows])) if rows else None,
                    "median_device_seconds": float(np.median([r["device_seconds"] for r in rows])) if rows else None,
                    "total_measured_solves": sum(int(r["measured_solves"]) for r in rows),
                    "total_bootstrap_solves": sum(int(r["bootstrap_solves"]) for r in rows),
                    "rmse_mHa_all_nonfailed": float(np.sqrt(np.mean(np.square(errors_all))) * 1000.0) if errors_all else None,
                    "rmse_mHa_among_stops": float(np.sqrt(np.mean(np.square(errors_met))) * 1000.0) if errors_met else None,
                    "bias_mHa_among_stops": float(np.mean(errors_met) * 1000.0) if errors_met else None,
                    "coverage": cov,
                    "median_r_hat_ritz": (
                        float(np.median([r["r_hat_ritz"] for r in nonfailed if r.get("r_hat_ritz") is not None]))
                        if any(r.get("r_hat_ritz") is not None for r in nonfailed)
                        else None
                    ),
                    "median_r_emp_ritz": (
                        float(np.median([r["r_emp_ritz"] for r in nonfailed if r.get("r_emp_ritz") is not None]))
                        if any(r.get("r_emp_ritz") is not None for r in nonfailed)
                        else None
                    ),
                    "median_r_emp_over_r_hat_ritz": float(np.median(cal_ratios)) if cal_ratios else None,
                }
            )

    anchor_by_key = {(a["case"], a["regime"], a["arm"]): a for a in record.get("noisy_anchors", [])}
    part_b_summary: list[dict[str, Any]] = []
    for ci, case in enumerate(d["cases"]):
        if "B" not in case["parts"]:
            continue
        exact = float(systems[ci]["independent_target"])
        family = len(case["cumulative_total_shots"])
        for regime in d["part_b_noise_regimes"]:
            for arm in d["part_b_arms"]:
                rows = [
                    r
                    for r in record["rows"]
                    if r["part"] == "B"
                    and r["case"] == ci
                    and r["regime"] == regime["key"]
                    and r["arm"] == arm["key"]
                ]
                n = len(rows)
                nonfailed = [r for r in rows if r["failure"] is None]
                met = [r for r in nonfailed if r["status"] == "target_met"]
                errors_all = [r["estimate"] - exact for r in nonfailed]
                errors_met = [r["estimate"] - exact for r in met]
                true_1s = sum(abs(e) <= target for e in errors_met)
                false_1s = len(met) - true_1s
                na = anchor_by_key.get((ci, regime["key"], arm["key"]), {})
                emp_bias_mha = float(np.mean(errors_all) * 1000.0) if errors_all else None
                emp_std_mha = float(np.std(errors_all, ddof=0) * 1000.0) if errors_all else None
                rmse_mha = float(np.sqrt(np.mean(np.square(errors_all))) * 1000.0) if errors_all else None
                bias_sq_share = (
                    float((emp_bias_mha**2) / (rmse_mha**2))
                    if emp_bias_mha is not None and rmse_mha is not None and rmse_mha > 0.0
                    else None
                )
                part_b_summary.append(
                    {
                        "case": case["name"],
                        "regime": regime["key"],
                        "arm": arm["key"],
                        "attempts": n,
                        "target_met": len(met),
                        "true_precision_stops": true_1s,
                        "false_precision_stops": false_1s,
                        "exhausted": sum(r["status"] == "schedule_exhausted" for r in rows),
                        "failed": n - len(nonfailed),
                        "exact_inf_bias_mHa": None if "bias_Ha" not in na else float(na["bias_Ha"] * 1000.0),
                        "emp_bias_mHa": emp_bias_mha,
                        "emp_std_mHa": emp_std_mha,
                        "rmse_mHa": rmse_mha,
                        "bias_sq_share": bias_sq_share,
                        "coverage": _coverage_block(rows, exact, family, delta, levels),
                        "median_total_shots": float(np.median([r["total_shots"] for r in rows])) if rows else None,
                        "mean_total_shots": float(np.mean([r["total_shots"] for r in rows])) if rows else None,
                        "median_device_seconds": float(np.median([r["device_seconds"] for r in rows])) if rows else None,
                        "cz_max": na.get("cz_max"),
                        "settings": na.get("settings"),
                    }
                )

    criteria = evaluate_criteria(d, part_a_summary, part_a_rows_by_key, part_b_summary)
    return {"part_a": part_a_summary, "part_b": part_b_summary, "criteria": criteria}


def evaluate_criteria(
    d: dict[str, Any],
    part_a_summary: list[dict[str, Any]],
    part_a_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    part_b_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministically evaluate frozen P1a, P1b, P2, P3, and width-guard endpoints."""
    analysis = d["analysis"]
    b_reps = int(analysis["shot_ratio_bootstrap_replicates"])
    b_seed = int(analysis["shot_ratio_bootstrap_seed"])
    alpha = float(analysis["shot_ratio_alpha"])

    a_map = {(s["case"], s["arm"]): s for s in part_a_summary}
    b_map = {(s["case"], s["regime"], s["arm"]): s for s in part_b_summary}
    comp_idx = 0

    # P1a: weighted SI vs unweighted LDF across 8q cases x k in {1, 2, 8}.
    cases_8q = [c["name"] for c in d["cases"] if c["kind"] == "molecule_8q"]
    p1a_pairs = (("si_qwc", "ldf_qwc", 1), ("si_k2", "ldf_k2", 2), ("si_fc", "ldf_fc", 8))
    p1a_cells: list[dict[str, Any]] = []
    n_p1a = len(cases_8q) * len(p1a_pairs)
    bonf_cov_alpha = 0.05 / max(1, n_p1a)
    for cname in cases_8q:
        for treat, comp, k_val in p1a_pairs:
            if (cname, treat) not in a_map or (cname, comp) not in a_map:
                continue
            ts = a_map[(cname, treat)]
            ratio = bootstrap_shot_ratio_ci(
                [r["total_shots"] for r in part_a_rows_by_key[(cname, treat)]],
                [r["total_shots"] for r in part_a_rows_by_key[(cname, comp)]],
                seed_root=b_seed,
                comp_index=comp_idx,
                replicates=b_reps,
                alpha=alpha,
            )
            comp_idx += 1
            cov95 = ts["coverage"]["bootstrap"]["95"]
            median_tied = close(ratio["median_ratio"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
            cost_ok = (ratio["median_ratio"] <= 0.80) or (median_tied and ratio["mean_ratio"] < 1.0)
            min_stops = int(math.ceil(30.0 * ts["attempts"] / 32.0)) if ts["attempts"] < 32 else 30
            passed = (
                cost_ok
                and ts["target_met"] >= min_stops
                and ts["reduced_rank_stops"] == 0
                and cov95["unconditional_p_lower"] >= bonf_cov_alpha
            )
            p1a_cells.append(
                {
                    "case": cname,
                    "k": k_val,
                    "treatment": treat,
                    "comparator": comp,
                    "target_met": ts["target_met"],
                    "attempts": ts["attempts"],
                    "reduced_rank_stops": ts["reduced_rank_stops"],
                    "boot95_p_lower": cov95["unconditional_p_lower"],
                    "bonferroni_alpha": bonf_cov_alpha,
                    "shot_ratio": ratio,
                    "cost_ok": cost_ok,
                    "passed": passed,
                }
            )
    # Apply Holm-Bonferroni step-down on mean-ratio p-values across P1a cells.
    ordered = sorted(range(len(p1a_cells)), key=lambda i: p1a_cells[i]["shot_ratio"]["p_mean_ge_one"])
    holm_active = True
    for rank_i, idx in enumerate(ordered):
        threshold = alpha / max(1, len(p1a_cells) - rank_i)
        sig = holm_active and (p1a_cells[idx]["shot_ratio"]["p_mean_ge_one"] <= threshold)
        if not sig:
            holm_active = False
        p1a_cells[idx]["holm_mean_ratio_lt_one"] = sig

    # P1b: paid-pilot ritz_si_fc vs si_fc and si_qwc (+ si_fc vs si_qwc).
    p1b_cells: list[dict[str, Any]] = []
    for cname in cases_8q:
        for treat, comp in (("ritz_si_fc", "si_fc"), ("ritz_si_fc", "si_qwc"), ("si_fc", "si_qwc")):
            if (cname, treat) not in a_map or (cname, comp) not in a_map:
                continue
            ts = a_map[(cname, treat)]
            ratio = bootstrap_shot_ratio_ci(
                [r["total_shots"] for r in part_a_rows_by_key[(cname, treat)]],
                [r["total_shots"] for r in part_a_rows_by_key[(cname, comp)]],
                seed_root=b_seed,
                comp_index=comp_idx,
                replicates=b_reps,
                alpha=alpha,
            )
            comp_idx += 1
            p1b_cells.append(
                {
                    "case": cname,
                    "treatment": treat,
                    "comparator": comp,
                    "shot_ratio": ratio,
                    "rmse_mHa_among_stops": ts["rmse_mHa_among_stops"],
                    "boot95_unconditional": ts["coverage"]["bootstrap"]["95"]["unconditional_coverage"],
                    "boot95_conditional": ts["coverage"]["bootstrap"]["95"]["conditional_coverage"],
                    "median_r_emp_over_r_hat_ritz": ts["median_r_emp_over_r_hat_ritz"],
                }
            )

    # Width-guard overhead: si_fc_width vs si_fc.
    width_overhead: list[dict[str, Any]] = []
    for cname in cases_8q:
        if (cname, "si_fc_width") in a_map and (cname, "si_fc") in a_map:
            ws = a_map[(cname, "si_fc_width")]
            bs = a_map[(cname, "si_fc")]
            ratio = bootstrap_shot_ratio_ci(
                [r["total_shots"] for r in part_a_rows_by_key[(cname, "si_fc_width")]],
                [r["total_shots"] for r in part_a_rows_by_key[(cname, "si_fc")]],
                seed_root=b_seed,
                comp_index=comp_idx,
                replicates=b_reps,
                alpha=alpha,
            )
            comp_idx += 1
            width_overhead.append(
                {
                    "case": cname,
                    "shot_ratio": ratio,
                    "bootstrap_solve_ratio": (
                        ws["total_bootstrap_solves"] / bs["total_bootstrap_solves"]
                        if bs["total_bootstrap_solves"]
                        else None
                    ),
                    "width_true_stops": ws["true_precision_stops"],
                    "base_true_stops": bs["true_precision_stops"],
                }
            )

    # P2: stretched 10q rank transfer.
    p2_cells: list[dict[str, Any]] = []
    st_cases = [c["name"] for c in d["cases"] if c["kind"] == "molecule_10q_stretched"]
    for cname in st_cases:
        for arm_key in ("si_k2", "ritz_si_k2"):
            if (cname, arm_key) not in a_map:
                continue
            s = a_map[(cname, arm_key)]
            passed = (
                s["true_precision_stops"] >= 12
                and s["false_precision_stops"] <= 1
                and s["reduced_rank_stops"] == 0
            )
            ratios: dict[str, Any] = {}
            for comp in ("ldf_qwc", "si_qwc"):
                if (cname, comp) in part_a_rows_by_key:
                    ratios[f"vs_{comp}"] = bootstrap_shot_ratio_ci(
                        [r["total_shots"] for r in part_a_rows_by_key[(cname, arm_key)]],
                        [r["total_shots"] for r in part_a_rows_by_key[(cname, comp)]],
                        seed_root=b_seed,
                        comp_index=comp_idx,
                        replicates=b_reps,
                        alpha=alpha,
                    )
                    comp_idx += 1
            p2_cells.append(
                {
                    "case": cname,
                    "arm": arm_key,
                    "true_precision_stops": s["true_precision_stops"],
                    "false_precision_stops": s["false_precision_stops"],
                    "reduced_rank_stops": s["reduced_rank_stops"],
                    "passed": passed,
                    "shot_ratios": ratios,
                }
            )

    # P3: GALIC noise gate bias & false-stop protection.
    p3_cells: list[dict[str, Any]] = []
    p3_diagnostics: list[dict[str, Any]] = []
    target_mha = float(d["target_std_error_Ha"]) * 1000.0
    for cname in cases_8q:
        for regime in d["part_b_noise_regimes"]:
            rkey = regime["key"]
            if (cname, rkey, "si_fc_unconstrained") not in b_map or (cname, rkey, "si_fc_galic_tight") not in b_map:
                continue
            unc = b_map[(cname, rkey, "si_fc_unconstrained")]
            tight = b_map[(cname, rkey, "si_fc_galic_tight")]
            mod = b_map.get((cname, rkey, "si_fc_galic_moderate"))
            qwc = b_map.get((cname, rkey, "si_qwc"))
            if rkey == "sc_card_full":
                p3_diagnostics.append(
                    {
                        "case": cname,
                        "regime": rkey,
                        "qwc_inf_bias_mHa": None if qwc is None else qwc["exact_inf_bias_mHa"],
                        "tight_inf_bias_mHa": tight["exact_inf_bias_mHa"],
                        "moderate_inf_bias_mHa": None if mod is None else mod["exact_inf_bias_mHa"],
                        "unconstrained_inf_bias_mHa": unc["exact_inf_bias_mHa"],
                    }
                )
                continue
            bias_ok = (
                tight["exact_inf_bias_mHa"] is not None
                and unc["exact_inf_bias_mHa"] is not None
                and abs(tight["exact_inf_bias_mHa"]) < abs(unc["exact_inf_bias_mHa"])
            )
            rmse_ok = (
                tight["rmse_mHa"] is not None
                and unc["rmse_mHa"] is not None
                and tight["rmse_mHa"] < unc["rmse_mHa"]
            )
            unc_biased = unc["exact_inf_bias_mHa"] is not None and abs(unc["exact_inf_bias_mHa"]) > target_mha
            false_stop_ok = (
                (tight["false_precision_stops"] < unc["false_precision_stops"])
                if unc_biased
                else True
            )
            p3_cells.append(
                {
                    "case": cname,
                    "regime": rkey,
                    "tight_inf_bias_mHa": tight["exact_inf_bias_mHa"],
                    "unconstrained_inf_bias_mHa": unc["exact_inf_bias_mHa"],
                    "moderate_inf_bias_mHa": None if mod is None else mod["exact_inf_bias_mHa"],
                    "tight_rmse_mHa": tight["rmse_mHa"],
                    "unconstrained_rmse_mHa": unc["rmse_mHa"],
                    "tight_false_stops": tight["false_precision_stops"],
                    "unconstrained_false_stops": unc["false_precision_stops"],
                    "bias_ok": bias_ok,
                    "rmse_ok": rmse_ok,
                    "false_stop_ok": false_stop_ok,
                    "passed": bias_ok and rmse_ok and false_stop_ok,
                }
            )

    return {
        "P1a_weighted_vs_unweighted": {
            "cells": p1a_cells,
            "all_passed": bool(p1a_cells) and all(c["passed"] for c in p1a_cells),
        },
        "P1b_ritz_replan_vs_bank_fc": {"cells": p1b_cells},
        "P2_stretched_rank_transfer": {
            "cells": p2_cells,
            "all_passed": bool(p2_cells) and all(c["passed"] for c in p2_cells),
        },
        "P3_galic_noise_gate": {
            "cells": p3_cells,
            "sc_card_full_diagnostics": p3_diagnostics,
            "all_passed": bool(p3_cells) and all(c["passed"] for c in p3_cells),
        },
        "width_guard_overhead": width_overhead,
    }


def check_record(
    record: dict[str, Any],
    *,
    verify_history: bool = False,
    full: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    d = producer.declaration()
    validate_declaration_structure(d)
    require(record["schema"] == producer.SCHEMA and record["declaration"] == d, "declaration mismatch")
    require(record["environment"] == d["environment"], "environment mismatch")
    commit = record["declaration_commit"]
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid declaration_commit")
    decl_bytes = (ROOT / producer.DECLARATION).read_bytes()
    require(
        record["declaration_sha256"] == producer.digest(decl_bytes)
        and producer.git("show", f"{commit}:{producer.DECLARATION}") == decl_bytes,
        "declaration commit binding mismatch",
    )
    require(
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{producer.RESULT}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        != 0,
        "result already existed at declaration commit",
    )
    require(record["source_sha256"] == producer.manifest(commit, d), "source manifest mismatch")
    if verify_history:
        final = producer.git("log", "-1", "--format=%H", "--", producer.RESULT).decode().strip()
        require(
            bool(final)
            and final != commit
            and subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, final], cwd=ROOT
            ).returncode
            == 0,
            "result is not in a separate descendant commit",
        )
        require(
            producer.git("show", f"{final}:{producer.RESULT}") == (ROOT / producer.RESULT).read_bytes(),
            "uncommitted result bytes",
        )

    systems = anchors(d, full=full)
    for case, anchor, actual in zip(d["cases"], systems, record["systems"], strict=True):
        require(
            close(actual["exact_energy_Ha"], anchor["independent_target"], abs_tol=1e-9)
            and actual["exact_rank"] == anchor["rank"]
            and actual["words"] == int(case["expected_words"]),
            f"system anchor mismatch for {case['name']}",
        )
        require(
            bool(np.allclose(actual["overlap_spectrum"], anchor["overlap_spectrum"], atol=1e-9, rtol=0.0)),
            f"overlap spectrum mismatch for {case['name']}",
        )
        anchor["part_a_plans"] = actual.get("part_a_plans", {})

    verified_noisy = verify_noisy_anchors(record, d, systems)

    expected = set(producer.expected_job_keys(d))
    seen: set[tuple[str, int, int, int, int]] = set()
    for row in record["rows"]:
        key = (row["part"], row["case"], row["arm_index"], row["regime_index"], row["replica"])
        require(key in expected and key not in seen, f"unexpected or duplicate row: {key}")
        seen.add(key)
        ci = row["case"]
        exp_plan: dict[str, Any] | None = None
        if row["part"] == "A":
            arm = d["part_a_arms"][row["arm_index"]]
            if arm["grouping"] != "ritz_replan":
                exp_plan = systems[ci]["part_a_plans"].get(arm["key"])
        else:
            exp_plan = verified_noisy[(ci, row["regime"], row["arm"])]
        check_row(row, d, d["cases"][ci], systems[ci], expected_plan=exp_plan)
    require(seen == expected, "incomplete record")
    return d, systems, len(verified_noisy)


def spot_replay(record: dict[str, Any], d: dict[str, Any]) -> int:
    """Replay replica 0 of beh2_8q Part A (si_fc, ritz_si_fc) & Part B (depol2q_p010_r3 si_fc_unconstrained)."""
    case = d["cases"][0]
    n = int(case["n"])
    h, generators, state = producer.build_case_inputs(case)
    replayed = 0

    ideal_sim = NoisyReadoutSimulator(state, n, ReadoutNoiseModel.ideal())
    ideal_ref = PreparedSampledReference(n, f"{case['name']}-ideal", ideal_sim.draw)
    si_fc_idx = next(i for i, a in enumerate(d["part_a_arms"]) if a["key"] == "si_fc")
    ritz_idx = next(i for i, a in enumerate(d["part_a_arms"]) if a["key"] == "ritz_si_fc")

    prov_si_fc = producer.build_arm_provider(
        ideal_ref, h, generators, n, block_size=n, grouping="sorted_insertion"
    )
    for ai, is_replan in ((si_fc_idx, False), (ritz_idx, True)):
        recorded = next(
            r
            for r in record["rows"]
            if r["part"] == "A" and r["case"] == 0 and r["arm_index"] == ai and r["replica"] == 0
        )
        fresh = producer.run_single_attempt(
            prov_si_fc,
            case,
            d,
            acq_seed=recorded["acq_seed"],
            boot_seed=recorded["boot_seed"],
            pilot_seed=recorded["pilot_seed"],
            replan_spec=d["part_a_arms"][ai] if is_replan else None,
        )
        require(
            fresh["status"] == recorded["status"]
            and fresh["total_shots"] == recorded["total_shots"]
            and fresh["final_fingerprint"] == recorded["final_fingerprint"]
            and (
                recorded["estimate"] is None
                or close(fresh["estimate"], recorded["estimate"], abs_tol=1e-9)
            ),
            f"spot replay mismatch for Part A arm {d['part_a_arms'][ai]['key']}",
        )
        replayed += 1

    reg_idx = next(i for i, r in enumerate(d["part_b_noise_regimes"]) if r["key"] == "depol2q_p010_r3")
    regime = d["part_b_noise_regimes"][reg_idx]
    b_idx = next(i for i, a in enumerate(d["part_b_arms"]) if a["key"] == "si_fc_unconstrained")
    nsim = NoisyReadoutSimulator(
        state,
        n,
        ReadoutNoiseModel(
            p_1q=regime["p_1q"],
            p_2q=regime["p_2q"],
            routing_2q_multiplier=regime["routing_2q_multiplier"],
            p_readout=regime["p_readout"],
            label=regime["key"],
        ),
    )
    nref = PreparedSampledReference(n, f"{case['name']}-{regime['key']}", nsim.draw)
    prov_b = producer.build_arm_provider(
        nref, h, generators, n, block_size=n, grouping="sorted_insertion"
    )
    recorded_b = next(
        r
        for r in record["rows"]
        if r["part"] == "B"
        and r["case"] == 0
        and r["regime_index"] == reg_idx
        and r["arm_index"] == b_idx
        and r["replica"] == 0
    )
    fresh_b = producer.run_single_attempt(
        prov_b, case, d, acq_seed=recorded_b["acq_seed"], boot_seed=recorded_b["boot_seed"]
    )
    require(
        fresh_b["status"] == recorded_b["status"]
        and fresh_b["total_shots"] == recorded_b["total_shots"]
        and fresh_b["final_fingerprint"] == recorded_b["final_fingerprint"]
        and (
            recorded_b["estimate"] is None
            or close(fresh_b["estimate"], recorded_b["estimate"], abs_tol=1e-9)
        ),
        "spot replay mismatch for Part B depol2q_p010_r3 si_fc_unconstrained",
    )
    replayed += 1
    return replayed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration-only", action="store_true")
    parser.add_argument("--full-anchor", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--spot-replay", action="store_true")
    parser.add_argument("--verify-history", action="store_true")
    parser.add_argument("--record", type=Path, default=ROOT / producer.RESULT)
    args = parser.parse_args(argv)

    if args.declaration_only:
        d = producer.declaration()
        stats = validate_declaration_structure(d)
        systems = anchors(d, full=args.full_anchor)
        print(
            json.dumps(
                {
                    "status": "declaration_ok",
                    "result_exists": (ROOT / producer.RESULT).exists(),
                    "stats": stats,
                    "anchors": systems,
                },
                indent=2,
            )
        )
        return 0

    record = read_json(args.record)
    d, systems, noisy_count = check_record(
        record, verify_history=args.verify_history, full=args.full_anchor
    )
    out: dict[str, Any] = {
        "status": "record_ok",
        "rows_verified": len(record["rows"]),
        "noisy_anchors_verified": noisy_count,
    }
    if args.spot_replay:
        out["spot_replays_verified"] = spot_replay(record, d)
    if args.report:
        out["summary"] = summarize(record, d, systems)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
