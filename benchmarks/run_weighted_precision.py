"""Preregistered weighted measurement planning & unified adaptive precision runner.

Executes Part A (ideal-shot weighted k-commuting grouping and Ritz-weighted
replanning under density-free ``sampled_precision``) and Part B (gate-level
noisy-readout bias-vs-variance validation of GALIC ``NoiseGatePolicy``).

Execution requires the pinned environment declared in
``docs/restart/weighted_precision_declaration.json`` at the result-free
declaration commit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json, write_json  # noqa: E402
from benchmarks.noisy_readout_sim import (  # noqa: E402
    NoisyReadoutSimulator,
    ReadoutNoiseModel,
)
from clifford_qc.fermion import c_op, cdag_op  # noqa: E402
from clifford_qc.measurement import DeviceCard, NoiseGatePolicy  # noqa: E402
from clifford_qc.measurement.functionals import ritz_functional  # noqa: E402
from clifford_qc.measurement.graph_state import native_stats  # noqa: E402
from clifford_qc.measurement.session import grouped_cache_fingerprint  # noqa: E402
from clifford_qc.models.fcidump import model_from_fcidump  # noqa: E402
from clifford_qc.multivector import MV  # noqa: E402
from clifford_qc.sampled_reference import PreparedSampledReference  # noqa: E402
from clifford_qc.subspace import PrecisionAcquisitionConfig, RankStopPolicy  # noqa: E402
from clifford_qc.subspace.linalg import solve_projected  # noqa: E402
from clifford_qc.subspace.rank_stopping import rank_margin  # noqa: E402
from clifford_qc.subspace.sampled import SampledProjectedProvider  # noqa: E402

DECLARATION = "docs/restart/weighted_precision_declaration.json"
RESULT = "docs/restart/weighted_precision_results.json.gz"
SCHEMA = "clifford_qc.weighted_precision_results.v1"
DECLARATION_SCHEMA = "clifford_qc.weighted_precision_declaration.v1"

EXTRA_MANIFEST_FILES = (
    DECLARATION,
    "tests/test_noisy_readout_sim.py",
    "tests/test_weighted_precision.py",
)

# Worker-local state populated before fork per case.
_WORKER_STATE: dict[str, Any] = {}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def require_clean_worktree(*, allow_paths: set[str] | None = None) -> None:
    """Refuse execution if any tracked or untracked file dirties the checkout."""
    allowed = set() if allow_paths is None else set(allow_paths)
    status = git("status", "--porcelain", "--untracked-files=all").decode("utf-8")
    dirty: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        rel = line[3:].split(" -> ")[-1].strip()
        if rel not in allowed:
            dirty.append(line)
    if dirty:
        raise ValueError(
            "execution requires a clean Git worktree; dirty entries: " + "; ".join(dirty[:5])
        )


def current_environment() -> dict[str, Any]:
    import scipy

    threads_raw = {
        os.environ.get(name, "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    threads = int(next(iter(threads_raw))) if len(threads_raw) == 1 and next(iter(threads_raw)).isdigit() else -1
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": threads,
    }


def declaration(path: Path | None = None) -> dict[str, Any]:
    d = read_json(ROOT / DECLARATION if path is None else Path(path))
    if d.get("schema") != DECLARATION_SCHEMA:
        raise ValueError(f"invalid declaration schema: {d.get('schema')!r}")
    if d.get("compiler") != "graph_state" or d.get("storage") != "packed" or d.get("pooling") != "assigned":
        raise ValueError("declaration compiler/storage/pooling contract changed")
    if d.get("seed_root") == d.get("development_seed_root"):
        raise ValueError("evaluation seed_root must differ from development_seed_root")
    analysis = d.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("declaration missing frozen analysis block")
    if analysis.get("coverage_levels") != [0.68, 0.95]:
        raise ValueError("declaration analysis.coverage_levels must be [0.68, 0.95]")
    if int(analysis.get("shot_ratio_bootstrap_replicates", 0)) < 100:
        raise ValueError("declaration analysis.shot_ratio_bootstrap_replicates must be >= 100")
    if analysis.get("shot_ratio_bootstrap_seed") in (None, d.get("seed_root"), d.get("development_seed_root")):
        raise ValueError("declaration analysis.shot_ratio_bootstrap_seed must be a distinct integer")
    for rel, expected in d["historical_bindings"].items():
        if digest((ROOT / rel).read_bytes()) != expected:
            raise ValueError(f"historical binding changed: {rel}")
    for case in d["cases"]:
        if digest((ROOT / case["fcidump"]).read_bytes()) != case["fcidump_sha256"]:
            raise ValueError(f"FCIDUMP bytes changed: {case['fcidump']}")
        totals = case["cumulative_total_shots"]
        if len(totals) < 2 or any(not isinstance(x, int) or x <= 0 for x in totals):
            raise ValueError(f"invalid cumulative_total_shots for {case['name']}")
        if any(b <= a for a, b in zip(totals, totals[1:])):
            raise ValueError(f"cumulative_total_shots must strictly increase for {case['name']}")
        if case["replan_pilot_total_shots"] <= 0 or case["replan_pilot_total_shots"] >= totals[0]:
            raise ValueError(f"invalid replan_pilot_total_shots for {case['name']}")
    return d


def manifest(commit: str, d: dict[str, Any], *, live: bool = False) -> dict[str, str]:
    """Bind the complete tracked clifford_qc/ and benchmarks/ Python tree plus inputs."""
    tracked_py = {
        path
        for path in git("ls-tree", "-r", "--name-only", commit, "--", "clifford_qc", "benchmarks")
        .decode("utf-8")
        .splitlines()
        if path.endswith(".py")
    }
    if not tracked_py:
        raise ValueError("failed to enumerate tracked clifford_qc/ and benchmarks/ sources")
    paths = sorted(
        tracked_py
        | set(EXTRA_MANIFEST_FILES)
        | set(d["historical_bindings"])
        | {c["fcidump"] for c in d["cases"]}
    )
    out: dict[str, str] = {}
    for rel in paths:
        committed = git("show", f"{commit}:{rel}")
        if live and (ROOT / rel).read_bytes() != committed:
            raise ValueError(f"live source differs from declaration commit: {rel}")
        out[rel] = digest(committed)
    return out


def derived_seed(
    seed_root: int,
    part_id: int,
    case_index: int,
    arm_index: int,
    regime_index: int,
    replica: int,
    stream: int,
) -> int:
    seq = np.random.SeedSequence(
        [seed_root, part_id, case_index, arm_index, regime_index, replica, stream]
    )
    return int(seq.generate_state(1, np.uint64)[0])


def make_schedule(n_settings: int, cumulative_totals: list[int]) -> tuple[int, tuple[int, ...]]:
    """Build matched total-physical-shot schedule across differing setting counts."""
    if n_settings < 1:
        raise ValueError("n_settings must be positive")
    pilot_per_group = max(2, int(cumulative_totals[0]) // n_settings)
    spent = pilot_per_group * n_settings
    rounds: list[int] = []
    for target_total in cumulative_totals[1:]:
        budget = max(n_settings, int(target_total) - spent)
        rounds.append(budget)
        spent += budget
    return pilot_per_group, tuple(rounds)


def build_case_inputs(case: dict[str, Any]) -> tuple[MV, list[tuple[str, MV]], np.ndarray]:
    kind = case["kind"]
    if kind == "molecule_8q":
        from benchmarks.run_cross_domain_ratio import build_case as build_8q_case

        adapted = {**case, "kind": "molecule"}
        return build_8q_case(adapted)
    if kind == "molecule_10q_stretched":
        from benchmarks.run_stretched_molecular_stop import active_integrals

        n = int(case["n"])
        h = model_from_fcidump(active_integrals(case)).hamiltonian.to_mv()
        state = np.zeros(1 << n, dtype=complex)
        a, b = (int(bits, 2) for bits in case["reference_bits"])
        state[a] = math.cos(case["theta"])
        state[b] = np.exp(1j * case["phi"]) * math.sin(case["theta"])
        ident = MV.scalar(n, 1)
        generators: list[tuple[str, MV]] = [("I", ident)]
        src = int(case["pair_source"])
        for i, tgt in enumerate(case["pair_targets"]):
            t = (
                cdag_op(n, 2 * tgt)
                * cdag_op(n, 2 * tgt + 1)
                * c_op(n, 2 * src + 1)
                * c_op(n, 2 * src)
            )
            g = t - t.dagger()
            generators.append((f"G{src}{tgt}", ident + case["weak_mix"] * g if i == 0 else g))
        return h, generators, state
    raise ValueError(f"unsupported case kind: {kind!r}")


def exact_projected_from_expectations(
    provider: SampledProjectedProvider,
    expectations: dict[int, float],
    solver_kwargs: dict[str, Any],
) -> tuple[float, int, list[float]]:
    s_mat, h_mat = provider.session._assemble(
        lambda f: float(f.constant)
        + sum(float(c) * expectations.get(int(code), 0.0) for code, c in f.coefficients.items())
    )
    solved = solve_projected(s_mat, h_mat, **solver_kwargs)
    diag = np.sqrt(np.maximum(np.real(np.diag(s_mat)), 0.0))
    s_norm = s_mat / np.outer(diag, diag)
    evals = sorted(np.linalg.eigvalsh(0.5 * (s_norm + s_norm.conj().T)).tolist(), reverse=True)
    return float(solved.ground_energy), int(solved.effective_rank), evals


def device_seconds_from_observations(
    observations: tuple[Any, ...],
    settings_by_key: dict[tuple[Any, ...], Any],
    device_scenario_us: dict[str, float],
    n: int,
) -> float:
    depth_cache: dict[str, tuple[int, int]] = {}
    total_us = 0.0
    prep_us = n * float(device_scenario_us["preparation_per_qubit"])
    ro_us = float(device_scenario_us["readout_reset"])
    d1_us = float(device_scenario_us["one_qubit_layer"])
    d2_us = float(device_scenario_us["two_qubit_layer"])
    for obs in observations:
        for group in obs.batch.groups:
            setting = settings_by_key[group.setting_key]
            cid = setting.clifford.identity
            if cid not in depth_cache:
                _, _, d1, d2 = native_stats(setting.clifford)
                depth_cache[cid] = (int(d1), int(d2))
            d1, d2 = depth_cache[cid]
            total_us += group.shots * (prep_us + ro_us + d1 * d1_us + d2 * d2_us)
    return total_us * 1e-6


def _block_size_int(spec: Any, n: int) -> int:
    return n if spec == "full" else int(spec)


def build_arm_provider(
    ref: PreparedSampledReference,
    h: MV,
    generators: list[tuple[str, MV]],
    n: int,
    *,
    block_size: int,
    grouping: str,
    noise_gate: NoiseGatePolicy | None = None,
) -> SampledProjectedProvider:
    return SampledProjectedProvider(
        ref,
        h,
        generators,
        storage="packed",
        block_size=block_size,
        method="graph_state",
        grouping=grouping,
        noise_gate=noise_gate,
    )


def run_single_attempt(
    provider: SampledProjectedProvider,
    case: dict[str, Any],
    d: dict[str, Any],
    *,
    acq_seed: int,
    boot_seed: int,
    pilot_seed: int | None = None,
    replan_spec: dict[str, Any] | None = None,
    width_guard: bool = False,
) -> dict[str, Any]:
    """Execute one preregistered attempt (single-stage or paid-pilot Ritz replan)."""
    totals = list(case["cumulative_total_shots"])
    n = int(case["n"])
    rank_policy = RankStopPolicy(**d["rank_policy"])
    boot_reps = int(case["bootstrap_replicas"])
    target_se = float(d["target_std_error_Ha"])
    delta = float(d["delta"])

    pilot_shots = 0
    pilot_obs_list: list[Any] = []
    pilot_fingerprint: str | None = None
    replan_seconds = 0.0
    active_provider = provider
    replan_error: str | None = None
    settings_by_key = {s.key: s for s in provider.plan.settings}

    if replan_spec is not None:
        assert pilot_seed is not None
        t_replan = time.perf_counter()
        n_pilot_settings = len(provider.plan.settings)
        pilot_pg0 = max(2, int(case["replan_pilot_total_shots"]) // n_pilot_settings)
        pilot_counts = [pilot_pg0] * n_pilot_settings
        try:
            pilot_obs = provider.reference.sample(
                provider.plan.settings, pilot_counts, seed=pilot_seed
            )
            pilot_obs_list.append(pilot_obs)
            pilot_shots = int(pilot_obs.batch.hardware_shots)
            recon = provider.reconstruct(pilot_obs, delta=delta)
            pilot_fingerprint = grouped_cache_fingerprint(recon.cache)
            psolved = provider.session.solve(recon.cache, **case["solver"])
            _, rank_func = rank_margin(
                provider.session, recon.cache, psolved, rank_policy, family=len(totals)
            )
            active_provider = provider.with_ritz_plan(
                psolved.ritz_vector(0),
                float(psolved.ground_energy),
                block_size=_block_size_int(replan_spec["block_size"], n),
                overlap_functional=rank_func,
                overlap_weight=float(replan_spec["overlap_weight"]),
            )
            settings_by_key.update({s.key: s for s in active_provider.plan.settings})
        except Exception as exc:  # pragma: no cover - retained as honest failure
            replan_error = f"{type(exc).__name__}: {exc}"
        replan_seconds = time.perf_counter() - t_replan

    if replan_error is not None:
        return {
            "status": "failed",
            "failure": replan_error,
            "pilot_shots": pilot_shots,
            "acquisition_shots": 0,
            "total_shots": pilot_shots,
            "settings": len(active_provider.plan.settings),
            "looks": [],
            "estimate": None,
            "std_error": None,
            "delta_interval": None,
            "naive_interval": None,
            "bootstrap_interval": None,
            "final_fingerprint": None,
            "pilot_fingerprint": pilot_fingerprint,
            "schedule": None,
            "measured_solves": 1 if pilot_shots else 0,
            "bootstrap_solves": 0,
            "r_hat_bank": None,
            "r_emp_bank": None,
            "r_hat_ritz": None,
            "r_emp_ritz": None,
            "device_seconds": device_seconds_from_observations(
                tuple(pilot_obs_list), settings_by_key, d["device_scenario_us"], n
            ),
            "sampling_seconds": 0.0,
            "inference_seconds": replan_seconds,
            "replan_seconds": replan_seconds,
        }

    if pilot_shots > 0:
        n_active = len(active_provider.plan.settings)
        rem_totals = [max(2 * n_active, tot - pilot_shots) for tot in totals]
        pilot_pg, rounds = make_schedule(n_active, rem_totals)
    else:
        pilot_pg, rounds = make_schedule(len(active_provider.plan.settings), totals)

    cfg = PrecisionAcquisitionConfig(
        pilot_shots_per_group=pilot_pg,
        round_budgets=rounds,
        target_std_error=target_se,
        allocation="ritz_targeted",
        rank_stop=rank_policy,
        max_bootstrap_half_width=float(d["bootstrap_half_width_Ha"]) if width_guard else None,
        bootstrap_replicates=boot_reps,
        bootstrap_seed=boot_seed,
    )
    run = active_provider.acquire_precision(
        cfg, seed=acq_seed, solve_options=case["solver"], delta=delta
    )
    all_obs = tuple(pilot_obs_list) + run.observations
    dev_s = device_seconds_from_observations(
        all_obs, settings_by_key, d["device_scenario_us"], n
    )
    bank_metrics_state_free = active_provider.plan_metrics()
    row: dict[str, Any] = {
        "status": run.status,
        "failure": None if run.error is None else f"{run.error['type']}: {run.error['message']}",
        "pilot_shots": pilot_shots,
        "acquisition_shots": int(run.confirmed_shots),
        "total_shots": int(pilot_shots + run.confirmed_shots),
        "settings": len(active_provider.plan.settings),
        "cz_total": int(bank_metrics_state_free.two_qubit_gates_total),
        "cz_max": int(bank_metrics_state_free.two_qubit_gates_max),
        "looks": list(run.looks),
        "pilot_fingerprint": pilot_fingerprint,
        "measured_solves": int(run.measured_solve_attempts + (1 if pilot_shots else 0)),
        "bootstrap_solves": int(run.bootstrap_solve_attempts),
        "r_hat_bank": float(bank_metrics_state_free.r_hat),
        "r_emp_bank": None,
        "r_hat_ritz": None,
        "r_emp_ritz": None,
        "device_seconds": float(dev_s),
        "sampling_seconds": float(run.sampling_seconds),
        "inference_seconds": float(run.inference_seconds + replan_seconds),
        "replan_seconds": float(replan_seconds),
    }

    acq = run.acquisition
    if acq is None:
        row.update(
            estimate=None,
            std_error=None,
            delta_interval=None,
            naive_interval=None,
            bootstrap_interval=None,
            final_fingerprint=None,
            schedule=None,
        )
    else:
        bank_emp = active_provider.plan_metrics(cache=acq.cache)
        rfunc = ritz_functional(
            active_provider.bank,
            active_provider.session.indices,
            acq.result.ritz_vector(0),
            float(acq.energy),
        )
        ritz_emp = active_provider.plan_metrics(target=rfunc, cache=acq.cache)
        row.update(
            estimate=float(acq.energy),
            std_error=float(acq.std_error),
            delta_interval=asdict(acq.interval),
            naive_interval=asdict(acq.naive_interval),
            bootstrap_interval=(
                None if acq.bootstrap_interval is None else asdict(acq.bootstrap_interval)
            ),
            final_fingerprint=grouped_cache_fingerprint(acq.cache),
            schedule=dict(acq.schedule),
            r_emp_bank=bank_emp.state_dependent_r,
            r_hat_ritz=float(ritz_emp.r_hat),
            r_emp_ritz=ritz_emp.state_dependent_r,
        )
    return row


def _worker_task(job: tuple[str, int, int, int, int]) -> dict[str, Any]:
    part, ci, ai, reg_idx, replica = job
    d = _WORKER_STATE["declaration"]
    case = d["cases"][ci]
    part_id = 0 if part == "A" else 1
    acq_seed = derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, replica, 0)
    boot_seed = derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, replica, 1)

    if part == "A":
        arm = d["part_a_arms"][ai]
        if arm["grouping"] == "ritz_replan":
            pilot_seed = derived_seed(d["seed_root"], part_id, ci, ai, reg_idx, replica, 2)
            provider = _WORKER_STATE["providers_a"][arm["pilot_arm"]]
            out = run_single_attempt(
                provider,
                case,
                d,
                acq_seed=acq_seed,
                boot_seed=boot_seed,
                pilot_seed=pilot_seed,
                replan_spec=arm,
                width_guard=bool(arm["width_guard"]),
            )
        else:
            pilot_seed = None
            provider = _WORKER_STATE["providers_a"][arm["key"]]
            out = run_single_attempt(
                provider,
                case,
                d,
                acq_seed=acq_seed,
                boot_seed=boot_seed,
                width_guard=bool(arm["width_guard"]),
            )
        out.update(
            part="A",
            case=ci,
            case_name=case["name"],
            arm=arm["key"],
            arm_index=ai,
            regime="ideal",
            regime_index=0,
            replica=replica,
            acq_seed=acq_seed,
            boot_seed=boot_seed,
            pilot_seed=pilot_seed,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        )
        return out

    arm = d["part_b_arms"][ai]
    regime = d["part_b_noise_regimes"][reg_idx]
    provider = _WORKER_STATE["providers_b"][(regime["key"], arm["key"])]
    out = run_single_attempt(
        provider,
        case,
        d,
        acq_seed=acq_seed,
        boot_seed=boot_seed,
    )
    out.update(
        part="B",
        case=ci,
        case_name=case["name"],
        arm=arm["key"],
        arm_index=ai,
        regime=regime["key"],
        regime_index=reg_idx,
        replica=replica,
        acq_seed=acq_seed,
        boot_seed=boot_seed,
        pilot_seed=None,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    )
    return out


def expected_job_keys(d: dict[str, Any]) -> list[tuple[str, int, int, int, int]]:
    jobs: list[tuple[str, int, int, int, int]] = []
    for ci, case in enumerate(d["cases"]):
        if "A" in case["parts"]:
            for ai, arm in enumerate(d["part_a_arms"]):
                if case["name"] in arm["cases"]:
                    for rep in range(int(case["outer_replicas"])):
                        jobs.append(("A", ci, ai, 0, rep))
        if "B" in case["parts"]:
            for reg_idx in range(len(d["part_b_noise_regimes"])):
                for ai in range(len(d["part_b_arms"])):
                    for rep in range(int(d["part_b_outer_replicas"])):
                        jobs.append(("B", ci, ai, reg_idx, rep))
    return jobs


def execute(d: dict[str, Any], args: argparse.Namespace) -> None:
    commit = git("log", "-1", "--format=%H", "--", DECLARATION).decode().strip()
    if git("rev-parse", "HEAD").decode().strip() != commit or current_environment() != d["environment"]:
        raise ValueError("execute only at the pinned, result-free declaration commit")
    if (ROOT / RESULT).exists():
        raise ValueError("result already exists")
    require_clean_worktree()
    bindings = manifest(commit, d, live=True)
    meta = {
        "schema": SCHEMA,
        "declaration": d,
        "declaration_commit": commit,
        "declaration_sha256": digest((ROOT / DECLARATION).read_bytes()),
        "source_sha256": bindings,
        "environment": current_environment(),
    }

    folder = args.checkpoint_dir.resolve()
    if folder == ROOT or ROOT in folder.parents:
        raise ValueError("checkpoint_dir must live outside the repository checkout")
    folder.mkdir(parents=True, exist_ok=args.resume)
    with (folder / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = folder / "meta.json"
        if args.resume:
            if read_json(stamp) != meta:
                raise ValueError("checkpoint metadata mismatch")
        else:
            write_json(stamp, meta)

        expected = expected_job_keys(d)
        expected_set = set(expected)
        completed: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
        for path in folder.glob("row-*.json"):
            row = read_json(path)
            key = (row["part"], row["case"], row["arm_index"], row["regime_index"], row["replica"])
            if key not in expected_set or key in completed:
                raise ValueError(f"unexpected or duplicate checkpoint row: {key}")
            completed[key] = row

        card = DeviceCard.load(ROOT / d["device_card"])
        systems: list[dict[str, Any]] = []
        noisy_anchors: list[dict[str, Any]] = []

        for ci, case in enumerate(d["cases"]):
            n = int(case["n"])
            h, generators, state = build_case_inputs(case)
            ideal_sim = NoisyReadoutSimulator(state, n, ReadoutNoiseModel.ideal())
            ideal_ref = PreparedSampledReference(n, f"{case['name']}-ideal", ideal_sim.draw)

            providers_a: dict[str, SampledProjectedProvider] = {}
            arm_plans_a: dict[str, dict[str, Any]] = {}
            for arm in d["part_a_arms"]:
                if case["name"] not in arm["cases"] or arm["grouping"] == "ritz_replan":
                    continue
                bs = _block_size_int(arm["block_size"], n)
                prov = build_arm_provider(
                    ideal_ref, h, generators, n, block_size=bs, grouping=arm["grouping"]
                )
                providers_a[arm["key"]] = prov
                m = prov.plan_metrics()
                arm_plans_a[arm["key"]] = {
                    "settings": m.n_settings,
                    "r_hat": float(m.r_hat),
                    "cz_total": int(m.two_qubit_gates_total),
                    "cz_max": int(m.two_qubit_gates_max),
                    "depth_2q_max": int(m.two_qubit_depth_max),
                }

            ref_prov = next(iter(providers_a.values()))
            if len(ref_prov.session.words) != int(case["expected_words"]):
                raise ValueError(f"word count drifted for {case['name']}")
            exact_exps = ideal_sim.exact_word_expectations(ref_prov.plan.settings)
            exact_e, exact_rank, exact_evals = exact_projected_from_expectations(
                ref_prov, exact_exps, case["solver"]
            )
            if exact_rank != int(case["expected_rank"]):
                raise ValueError(f"exact rank drifted for {case['name']}")

            providers_b: dict[tuple[str, str], SampledProjectedProvider] = {}
            if "B" in case["parts"]:
                # Cache noise-gated partitions by (max_logical_2q) across regimes.
                plan_cache: dict[tuple[int, str, int | None], SampledProjectedProvider] = {}
                for regime in d["part_b_noise_regimes"]:
                    nmodel = ReadoutNoiseModel(
                        p_1q=regime["p_1q"],
                        p_2q=regime["p_2q"],
                        routing_2q_multiplier=regime["routing_2q_multiplier"],
                        p_readout=regime["p_readout"],
                        label=regime["key"],
                    )
                    nsim = NoisyReadoutSimulator(state, n, nmodel)
                    nref = PreparedSampledReference(n, f"{case['name']}-{regime['key']}", nsim.draw)
                    for arm in d["part_b_arms"]:
                        bs = _block_size_int(arm["block_size"], n)
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
                        cache_key = (
                            bs,
                            arm["grouping"],
                            None if gate is None else gate.max_logical_two_qubit_gates,
                        )
                        if cache_key in plan_cache:
                            base = plan_cache[cache_key]
                            prov_b = SampledProjectedProvider(
                                nref,
                                h,
                                generators,
                                storage="packed",
                                block_size=bs,
                                method="graph_state",
                                grouping="largest_degree",
                            )
                            prov_b.plan = base.plan
                            prov_b.grouping = base.grouping
                            prov_b.noise_gate = gate
                            prov_b.circuit_identity = base.circuit_identity
                            from clifford_qc.reference_states import _digest, operator_identity

                            prov_b.cache_identity = _digest(
                                [
                                    "sampled_projection.v1",
                                    prov_b.reference_identity,
                                    prov_b.hamiltonian_identity,
                                    [
                                        operator_identity(prov_b.bank.generator(i).mv)
                                        for i in range(len(prov_b.bank))
                                    ],
                                    "packed",
                                    prov_b.circuit_identity,
                                ]
                            )
                        else:
                            prov_b = build_arm_provider(
                                nref,
                                h,
                                generators,
                                n,
                                block_size=bs,
                                grouping=arm["grouping"],
                                noise_gate=gate,
                            )
                            plan_cache[cache_key] = prov_b
                        providers_b[(regime["key"], arm["key"])] = prov_b
                        nexps = nsim.exact_word_expectations(prov_b.plan.settings)
                        ne, nrank, nevals = exact_projected_from_expectations(
                            prov_b, nexps, case["solver"]
                        )
                        bm = prov_b.plan_metrics()
                        noisy_anchors.append(
                            {
                                "case": ci,
                                "case_name": case["name"],
                                "regime": regime["key"],
                                "arm": arm["key"],
                                "settings": bm.n_settings,
                                "r_hat": float(bm.r_hat),
                                "cz_total": int(bm.two_qubit_gates_total),
                                "cz_max": int(bm.two_qubit_gates_max),
                                "admissible_settings": bm.noise_gate_admissible_settings,
                                "noisy_exact_energy_Ha": float(ne),
                                "bias_Ha": float(ne - exact_e),
                                "noisy_rank": int(nrank),
                                "noisy_overlap_spectrum": nevals,
                                "noise_model": nmodel.as_dict(),
                                "noise_gate": None if gate is None else gate.as_dict(),
                            }
                        )

            systems.append(
                {
                    "case": ci,
                    "name": case["name"],
                    "n": n,
                    "words": len(ref_prov.session.words),
                    "exact_energy_Ha": exact_e,
                    "exact_rank": exact_rank,
                    "overlap_spectrum": exact_evals,
                    "part_a_plans": arm_plans_a,
                }
            )
            print(f"Prepared {case['name']}: words={len(ref_prov.session.words)} exact_E={exact_e:.10f}", flush=True)

            _WORKER_STATE.clear()
            _WORKER_STATE.update(
                declaration=d, providers_a=providers_a, providers_b=providers_b
            )
            case_jobs = [k for k in expected if k[1] == ci and k not in completed]
            if case_jobs:
                with mp.get_context("fork").Pool(args.jobs) as pool:
                    for row in pool.imap_unordered(_worker_task, case_jobs, chunksize=1):
                        key = (row["part"], row["case"], row["arm_index"], row["regime_index"], row["replica"])
                        write_json(
                            folder / f"row-{key[0]}-{key[1]}-{key[2]}-{key[3]}-{key[4]}.json",
                            row,
                        )
                        completed[key] = row
                        if len(completed) % 16 == 0 or len(completed) == len(expected):
                            print(f"{len(completed)}/{len(expected)} completed", flush=True)
            _WORKER_STATE.clear()

        require_clean_worktree()
        if manifest(commit, d, live=True) != bindings:
            raise ValueError("source changed during evaluation")
        write_json(
            ROOT / RESULT,
            {
                **meta,
                "systems": systems,
                "noisy_anchors": noisy_anchors,
                "rows": [completed[k] for k in expected],
            },
        )
        print(f"wrote {RESULT}: {len(completed)} acquisitions", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--jobs", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error("jobs must be in [1, 8]")
    execute(declaration(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
