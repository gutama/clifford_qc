"""Comprehensive research benchmark for geometric commutator subspaces and
resource-/information-aware adaptive selection & closed-loop acquisition.

Part A tests hierarchical commutator operator generation (A_0, A_1={[H,A]},
A_2={[H,[H,A]]}, ...) and lazy demand-driven CommutatorFrontier against Krylov,
excitation/Pauli pools, and classical selected-CI controls across spin chains,
Hubbard lattices, and molecular Hamiltonians (H4, LiH CAS(4e,4o)), on both
determinant and correlated references.

Part B tests resource-aware selection argmax E[dE_A] / (C_incremental(A) + eps),
7D finite-shot delta-method uncertainty calibration, and oracle-free closed-loop
acquisition (acquire_to_accuracy) against both UniformFullRegistry (same
measurement groups) and UniformBasisOnly (final-basis lower bound) across five
seeds and declared hardware DeviceCards.

Run:
    python3 benchmarks/run_geometric_information_study.py
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clifford_qc import X, Z
from clifford_qc.backends.exact_mv import ExactMVBackend
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.backends.sector_statevector import SectorOperator, SectorStatevectorBackend
from clifford_qc.dense_reference import exact_ground
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.cost import DeviceCard, setting_duration_us
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.spin import tfim, xxz
from clifford_qc.reproducibility import execution_provenance
from clifford_qc.subspace import (
    Generator,
    IncrementalCostModel,
    MatrixElementBank,
    acquire_to_accuracy,
    basis_dict_to_code,
    commutator_apply,
    dense_basis,
    determinant_excitations,
    expected_lowering_estimate,
    hierarchy_span_diagnostics,
    identity_generator,
    krylov_response,
    nested_commutator_hierarchy,
    occupied_spin_orbitals,
    pauli_orbit,
    qwc_bases_for_words,
    qwc_setting_resources,
    reference_sector_leakage,
    ritz_uncertainty,
    run_acase,
    run_control,
    run_dynamic_commutator_acase,
    run_resource_aware_acase,
    score_information_candidate,
    sector_leakage,
    span_comparison,
    trotter_program,
)

DEVICE_DIR = ROOT / "benchmarks" / "configs" / "device_cards"
DEFAULT_OUT = ROOT / "benchmarks" / "geometric_information_study.json"
MHA = 1e3


def _qwc_count(n: int, word_codes) -> int:
    return len(qwc_bases_for_words(n, word_codes))


def _total_qwc_time_us(card: DeviceCard, n: int, word_codes) -> float:
    bases = qwc_bases_for_words(n, word_codes)
    return float(sum(setting_duration_us(card, qwc_setting_resources(n, b)) for b in bases))


def _word_readable_by_basis(n: int, word_code: int, basis_code: int) -> bool:
    if word_code == 0:
        return True
    for q in range(n):
        wl = (word_code >> (2 * q)) & 3
        if wl != 0 and ((basis_code >> (2 * q)) & 3) != wl:
            return False
    return True


def _registry_groups_for_acq(acq) -> list[list[PauliWord]]:
    """Reconstruct exact group list matching acq.cache's discovered QWC settings."""
    n = acq.bank.n
    req = sorted(c for c in acq.bank.word_set(acq.indices) if c != 0)
    gstates = acq.cache.group_states()
    wsets = [set(gs["word_codes"]) for gs in gstates]
    bcodes = [basis_dict_to_code(dict(gs["basis"])) for gs in gstates]
    assigned = set().union(*wsets) if wsets else set()
    for code in req:
        if code in assigned:
            continue
        for pos, bcode in enumerate(bcodes):
            if _word_readable_by_basis(n, code, bcode):
                wsets[pos].add(code)
                assigned.add(code)
                break
    return [[PauliWord(n, c) for c in sorted(ws)] for ws in wsets]


def _adaptive_summary(res, exact_energy: float, n: int, wall_s: float) -> dict[str, Any]:
    word_codes = res.bank.word_set(res.indices)
    depth_counts = res.resources.get("depth_counts_in_basis")
    return {
        "basis_size": int(len(res.indices)),
        "effective_rank": int(res.result.effective_rank),
        "energy": float(res.energy),
        "error_mHa": float((res.energy - exact_energy) * MHA),
        "condition_number": float(res.result.condition_number),
        "word_universe": int(res.resources["word_universe"]),
        "qwc_groups": int(_qwc_count(n, word_codes)),
        "max_generator_support": int(res.resources.get("max_generator_support", 0)),
        "commutators_generated": int(res.resources.get("commutators_generated", 0)),
        "depth_counts": ({str(k): int(v) for k, v in sorted(depth_counts.items())} if depth_counts else None),
        "wall_seconds": float(wall_s),
        "stopped_reason": str(res.stopped_reason),
    }


def _fixed_subspace_summary(rho, H, gens, exact_energy: float, n: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    bank = MatrixElementBank(rho, H, gens)
    solved = bank.solve()
    dt = time.perf_counter() - t0
    word_codes = bank.word_set(solved.indices)
    res = bank.resources(solved.indices)
    return {
        "basis_size": int(len(solved.indices)),
        "effective_rank": int(solved.effective_rank),
        "energy": float(solved.ground_energy),
        "error_mHa": float((solved.ground_energy - exact_energy) * MHA),
        "condition_number": float(solved.condition_number),
        "word_universe": int(res["word_universe"]),
        "qwc_groups": int(_qwc_count(n, word_codes)),
        "max_generator_support": int(res["max_generator_support"]),
        "commutators_generated": 0,
        "depth_counts": None,
        "wall_seconds": float(dt),
        "stopped_reason": "fixed basis",
    }


def part_a1_growth_closure_and_symmetry() -> dict[str, Any]:
    print("[A1] Operator growth, Lie closure, truncation & fermionic symmetry...", flush=True)
    out: dict[str, Any] = {}

    # 1. Exact Lie closure on 1q sl(2)
    h_1q = X(1, 0) + 0.5 * Z(1, 0)
    hier_1q = nested_commutator_hierarchy(h_1q, [Z(1, 0)], max_depth=5, phase_mode="hermitian")
    out["sl2_1q"] = {
        "closure_reached": bool(hier_1q.closure_reached),
        "closure_depth": hier_1q.closure_depth,
        "summary_table": hier_1q.summary_table(),
    }

    # 2. Spin chains depth 0..3 growth + span diagnostics on det & correlated refs
    spin_growth: dict[str, Any] = {}
    for label, mod in [
        ("tfim6", tfim(6, J=1.0, h=1.0)),
        ("xxz6", xxz(6, J=1.0, delta=1.0)),
    ]:
        e0, _ = exact_ground(mod.hamiltonian.to_mv())
        h_words = [w for w, _ in mod.hamiltonian.items() if w.code != 0]
        seeds = pauli_orbit(h_words)
        hier = nested_commutator_hierarchy(mod.hamiltonian, seeds, max_depth=3, phase_mode="hermitian")
        rho_det = ExactMVBackend().state(mod.reference, ())
        rho_corr = ExactMVBackend().state(trotter_program(mod.hamiltonian, 0.35, steps=6), (), initial_state=rho_det)
        diag_det = hierarchy_span_diagnostics(rho_det, mod.hamiltonian, hier)
        diag_corr = hierarchy_span_diagnostics(rho_corr, mod.hamiltonian, hier)
        for row in diag_det:
            row["error_mHa"] = float((row["ground_energy"] - e0) * MHA)
        for row in diag_corr:
            row["error_mHa"] = float((row["ground_energy"] - e0) * MHA)
        spin_growth[label] = {
            "exact_ground_energy": float(e0),
            "closure_reached": bool(hier.closure_reached),
            "closure_depth": hier.closure_depth,
            "summary_table": hier.summary_table(),
            "span_det": diag_det,
            "span_corr": diag_corr,
        }
        print(
            f"  {label}: exact E0={e0:.6f} | gens d0..d3={[s['kept'] for s in hier.summary_table()]} | "
            f"dE_corr(d3)={diag_corr[-1]['error_mHa']:.2f} mHa W={diag_corr[-1]['word_universe']}",
            flush=True,
        )
    out["spin_growth"] = spin_growth

    # 3. Support truncation ablation on xxz(4)
    mod_xxz4 = xxz(4, J=1.0, delta=1.0)
    e0_xxz4, _ = exact_ground(mod_xxz4.hamiltonian.to_mv())
    seeds_xxz4 = pauli_orbit([w for w, _ in mod_xxz4.hamiltonian.items() if w.code != 0])
    rho_x4_det = ExactMVBackend().state(mod_xxz4.reference, ())
    rho_x4_corr = ExactMVBackend().state(trotter_program(mod_xxz4.hamiltonian, 0.35, steps=6), (), initial_state=rho_x4_det)
    trunc_rows = []
    for ref_name, rref in [("det", rho_x4_det), ("corr_trotter", rho_x4_corr)]:
        for pol_name, kw in [
            ("exact_drop", dict(max_depth=2, phase_mode="hermitian")),
            ("truncate_s8_l2_98", dict(max_depth=2, phase_mode="hermitian", max_support=8, support_policy="truncate", truncation_l2_keep=0.98)),
        ]:
            h_loc = nested_commutator_hierarchy(mod_xxz4.hamiltonian, seeds_xxz4, **kw)
            d_loc = hierarchy_span_diagnostics(rref, mod_xxz4.hamiltonian, h_loc)[-1]
            res_loc = run_acase(rref, mod_xxz4.hamiltonian, h_loc.generators(), max_size=6, min_lowering=1e-10)
            w_codes = res_loc.bank.word_set(res_loc.indices)
            min_l2 = min(node.retained_l2_fraction for node in h_loc.all_nodes)
            trunc_rows.append(
                {
                    "ref": ref_name,
                    "policy": pol_name,
                    "min_retained_l2": float(min_l2),
                    "hier_d2_max_support": int(d_loc["max_generator_support"]),
                    "hier_d2_word_universe": int(d_loc["word_universe"]),
                    "hier_d2_error_mHa": float((d_loc["ground_energy"] - e0_xxz4) * MHA),
                    "acase_M6_error_mHa": float((res_loc.energy - e0_xxz4) * MHA),
                    "acase_M6_words": int(res_loc.resources["word_universe"]),
                    "acase_M6_qwc_groups": int(_qwc_count(4, w_codes)),
                }
            )
    out["xxz4_truncation"] = trunc_rows

    # 4. Fermionic symmetry preservation vs naive Pauli-tail truncation
    sym_rows = []
    hub1 = hubbard(2, t=1.0, U=4.0)
    s_h1 = determinant_excitations(hub1.n, occupied_spin_orbitals(hub1), max_rank=1)
    hier_h1 = nested_commutator_hierarchy(hub1.hamiltonian, s_h1, max_depth=2, phase_mode="hermitian")
    sym_rows.append(
        {
            "system": "hubbard(2)_d1..2",
            "mode": "exact",
            "nodes": len(hier_h1.all_nodes),
            "max_operator_leakage": float(max(max(sector_leakage(nd.generator).values()) for nd in hier_h1.all_nodes)),
        }
    )

    h4 = fcidump_model(ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP")
    rho_h4 = ExactMVBackend().state(h4.reference, ())
    s_h4 = determinant_excitations(h4.n, occupied_spin_orbitals(h4), max_rank=1)
    h4_d1_nodes = []
    h4_l2_90_leaks = []
    h4_top16_leaks = []
    for s in s_h4:
        g_raw, _ = commutator_apply(h4.hamiltonian, s, phase_mode="hermitian")
        if g_raw is not None:
            h4_d1_nodes.append(max(sector_leakage(g_raw).values()))
        g_90, _ = commutator_apply(h4.hamiltonian, s, phase_mode="hermitian", truncation_l2_keep=0.90)
        if g_90 is not None:
            h4_l2_90_leaks.append((max(sector_leakage(g_90).values()), max(reference_sector_leakage(g_90, rho_h4).values()), g_90.support()))
        g_16, _ = commutator_apply(h4.hamiltonian, s, phase_mode="hermitian", truncation_top_k=16)
        if g_16 is not None:
            h4_top16_leaks.append((max(sector_leakage(g_16).values()), max(reference_sector_leakage(g_16, rho_h4).values()), g_16.support()))
    sym_rows.append(
        {
            "system": "h4_d1",
            "mode": "exact",
            "nodes": len(h4_d1_nodes),
            "max_operator_leakage": float(max(h4_d1_nodes)),
        }
    )
    sym_rows.append(
        {
            "system": "h4_d1",
            "mode": "truncation_l2_keep=0.90",
            "nodes": len(h4_l2_90_leaks),
            "max_operator_leakage": float(max(x[0] for x in h4_l2_90_leaks)),
            "max_ref_leakage": float(max(x[1] for x in h4_l2_90_leaks)),
            "max_support": int(max(x[2] for x in h4_l2_90_leaks)),
        }
    )
    sym_rows.append(
        {
            "system": "h4_d1",
            "mode": "truncation_top_k=16",
            "nodes": len(h4_top16_leaks),
            "max_operator_leakage": float(max(x[0] for x in h4_top16_leaks)),
            "max_ref_leakage": float(max(x[1] for x in h4_top16_leaks)),
            "max_support": int(max(x[2] for x in h4_top16_leaks)),
        }
    )

    lih = fcidump_model(ROOT / "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP")
    s_lih = determinant_excitations(lih.n, occupied_spin_orbitals(lih), max_rank=1)
    hier_lih = nested_commutator_hierarchy(lih.hamiltonian, s_lih, max_depth=1, phase_mode="hermitian")
    sym_rows.append(
        {
            "system": "lih_d1",
            "mode": "exact",
            "nodes": len(hier_lih.all_nodes),
            "max_operator_leakage": float(max(max(sector_leakage(nd.generator).values()) for nd in hier_lih.all_nodes)),
        }
    )
    out["fermionic_symmetry"] = sym_rows
    print("  sym rows:", [(r["system"], r["mode"], round(r["max_operator_leakage"], 4)) for r in sym_rows], flush=True)
    return out


def part_a2_spin_chains() -> dict[str, Any]:
    print("[A2] Spin chains (tfim6, xxz6) static vs dynamic frontier vs Krylov6...", flush=True)
    results: dict[str, Any] = {}
    max_size = 7

    for label, mod in [("tfim6", tfim(6, J=1.0, h=1.0)), ("xxz6", xxz(6, J=1.0, delta=1.0))]:
        e0, _ = exact_ground(mod.hamiltonian.to_mv())
        h_words = [w for w, _ in mod.hamiltonian.items() if w.code != 0]
        seeds = pauli_orbit(h_words)
        hier3 = nested_commutator_hierarchy(mod.hamiltonian, seeds, max_depth=3, phase_mode="hermitian")
        rho_det = ExactMVBackend().state(mod.reference, ())
        rho_corr = ExactMVBackend().state(trotter_program(mod.hamiltonian, 0.35, steps=6), (), initial_state=rho_det)
        k_gens = [identity_generator(mod.n)] + krylov_response(mod.hamiltonian, 6)

        ref_dict: dict[str, Any] = {}
        for ref_name, rho in [("det", rho_det), ("corr_trotter", rho_corr)]:
            arms: dict[str, Any] = {}
            for arm_name, max_d in [
                ("Orbit_d0", 0),
                ("Static_d<=1", 1),
                ("Static_d<=2", 2),
                ("Static_d<=3", 3),
            ]:
                t0 = time.perf_counter()
                res = run_acase(rho, mod.hamiltonian, hier3.generators(max_depth=max_d), max_size=max_size, min_lowering=1e-10)
                dt = time.perf_counter() - t0
                summary = _adaptive_summary(res, e0, mod.n, dt)
                if max_d >= 1:
                    static_comm = sum(s.kept + s.deduplicated + s.vanished for s in hier3.summaries[1 : max_d + 1])
                    summary["commutators_generated"] = int(static_comm)
                arms[arm_name] = summary

            for arm_name, mode in [
                ("Dyn_on_select_or_stall", "on_select_or_stall"),
                ("Dyn_on_stall", "on_stall"),
            ]:
                t0 = time.perf_counter()
                res = run_dynamic_commutator_acase(
                    rho, mod.hamiltonian, seeds, max_depth=3, expand_mode=mode, expand_top_k=2,
                    phase_mode="hermitian", max_size=max_size, min_lowering=1e-10,
                )
                dt = time.perf_counter() - t0
                arms[arm_name] = _adaptive_summary(res, e0, mod.n, dt)

            arms["Krylov6"] = _fixed_subspace_summary(rho, mod.hamiltonian, k_gens, e0, mod.n)
            ref_dict[ref_name] = arms
            print(
                f"  {label} ({ref_name}): dE[Orbit,d1,d2,d3,DynOSS,Krylov6] = "
                + ", ".join(f"{k}={arms[k]['error_mHa']:.2f}" for k in ("Orbit_d0", "Static_d<=1", "Static_d<=2", "Static_d<=3", "Dyn_on_select_or_stall", "Krylov6")),
                flush=True,
            )
        results[label] = {"exact_ground_energy": float(e0), "references": ref_dict}
    return results


def part_a3_fermionic_and_chemistry() -> dict[str, Any]:
    print("[A3] Fermionic & chemistry comparisons (Hubbard, H4, LiH) + SCI & span...", flush=True)
    out: dict[str, Any] = {}

    def _run_controls(mod, exact_energy: float, max_det: int) -> dict[str, Any]:
        sector = SectorStatevectorBackend(mod.n, mod.metadata["n_electrons"], mod.metadata["sz"])
        sop = SectorOperator(sector, mod.hamiltonian)
        ref_idx = int(np.flatnonzero(sector.state_from_program(mod.reference))[0])
        sampled = np.array([ref_idx], dtype=np.int64)
        controls = {}
        for cname, kind, extra in [
            ("MatchedSCI", "matched_selected_ci", {}),
            ("IterSCI", "iterated_selected_ci", {}),
            ("RandomCI", "random", {"seed": 7}),
        ]:
            c_res = run_control(
                sop, sampled, name=cname, kind=kind, max_determinants=max_det,
                score="epstein_nesbet", exact_energy=exact_energy, **extra,
            )
            rec = c_res.to_record()
            controls[cname] = {
                "basis_size": int(rec["determinant_count"]),
                "energy": float(rec["energy"]),
                "error_mHa": float(rec["error"] * MHA),
                "matrix_nonzeros": int(rec["matrix_nonzeros"]),
                "selection_work": int(rec["selection_work"]),
                "total_seconds": float(rec["build_seconds"] + rec["solve_seconds"]),
            }
        return controls

    # 1. Hubbard dimer (2 sites, 4 qubits)
    hub1 = hubbard(2, t=1.0, U=4.0)
    e0_h1, _ = exact_ground(hub1.hamiltonian.to_mv())
    rho_h1_det = ExactMVBackend().state(hub1.reference, ())
    rho_h1_corr = ExactMVBackend().state(trotter_program(hub1.hamiltonian, 0.25, steps=4), (), initial_state=rho_h1_det)
    occ1 = occupied_spin_orbitals(hub1)
    s1 = determinant_excitations(hub1.n, occ1, max_rank=1)
    sd1 = determinant_excitations(hub1.n, occ1, max_rank=2)
    s1_det_cols = dense_basis(rho_h1_det, [identity_generator(hub1.n)] + s1)
    sd1_det_cols = dense_basis(rho_h1_det, [identity_generator(hub1.n)] + sd1)

    hub1_refs = {}
    for ref_name, rho in [("det", rho_h1_det), ("corr_trotter", rho_h1_corr)]:
        arms = {}
        t0 = time.perf_counter()
        res_s = run_acase(rho, hub1.hamiltonian, s1, max_size=4, leakage_tol=1e-9)
        arms["Singles"] = _adaptive_summary(res_s, e0_h1, hub1.n, time.perf_counter() - t0)

        t0 = time.perf_counter()
        res_dyn = run_dynamic_commutator_acase(
            rho, hub1.hamiltonian, s1, max_depth=2, max_size=4, leakage_tol=1e-9, phase_mode="hermitian"
        )
        arms["DynComm(S)_d2"] = _adaptive_summary(res_dyn, e0_h1, hub1.n, time.perf_counter() - t0)

        t0 = time.perf_counter()
        res_sd = run_acase(rho, hub1.hamiltonian, sd1, max_size=4, leakage_tol=1e-9)
        arms["SD_Pool"] = _adaptive_summary(res_sd, e0_h1, hub1.n, time.perf_counter() - t0)

        arms["Krylov3"] = _fixed_subspace_summary(rho, hub1.hamiltonian, [identity_generator(hub1.n)] + krylov_response(hub1.hamiltonian, 3), e0_h1, hub1.n)
        arms["controls"] = _run_controls(hub1, e0_h1, 4)

        dyn_gens = [res_dyn.bank.generator(i) for i in res_dyn.indices]
        dyn_cols = dense_basis(rho, dyn_gens)
        sc_vs_s = span_comparison(dyn_cols, s1_det_cols)
        sc_vs_sd = span_comparison(dyn_cols, sd1_det_cols)
        hub1_refs[ref_name] = {
            "arms": arms,
            "span_vs_S": sc_vs_s.to_record(),
            "span_vs_SD": sc_vs_sd.to_record(),
        }
    out["hubbard_2x1"] = {"exact_ground_energy": float(e0_h1), "references": hub1_refs}

    # 2. Hubbard 2x2 (8 qubits, 36 sector states)
    hub2 = hubbard((2, 2), t=1.0, U=4.0)
    sec2 = SectorStatevectorBackend(hub2.n, hub2.metadata["n_electrons"], hub2.metadata["sz"])
    e0_h2 = float(sec2.ground_state(hub2.hamiltonian, k=1, method="lanczos")[0][0])
    rho_h2 = ExactMVBackend().state(hub2.reference, ())
    occ2 = occupied_spin_orbitals(hub2)
    s2 = determinant_excitations(hub2.n, occ2, max_rank=1)
    sd2 = determinant_excitations(hub2.n, occ2, max_rank=2)
    sd2_det_cols = dense_basis(rho_h2, [identity_generator(hub2.n)] + sd2)

    h2_arms = {}
    t0 = time.perf_counter()
    h2_arms["Singles"] = _adaptive_summary(run_acase(rho_h2, hub2.hamiltonian, s2, max_size=8, leakage_tol=1e-9), e0_h2, hub2.n, time.perf_counter() - t0)
    t0 = time.perf_counter()
    res_h2_d1 = run_dynamic_commutator_acase(rho_h2, hub2.hamiltonian, s2, max_depth=1, max_size=8, leakage_tol=1e-9, phase_mode="hermitian")
    h2_arms["DynComm(S)_d1"] = _adaptive_summary(res_h2_d1, e0_h2, hub2.n, time.perf_counter() - t0)
    t0 = time.perf_counter()
    res_h2_d2 = run_dynamic_commutator_acase(
        rho_h2, hub2.hamiltonian, s2, max_depth=2, expand_mode="on_stall", expand_top_k=1, max_children_total=20,
        max_size=8, leakage_tol=1e-9, phase_mode="hermitian",
    )
    h2_arms["DynComm(S)_d2_stall"] = _adaptive_summary(res_h2_d2, e0_h2, hub2.n, time.perf_counter() - t0)
    t0 = time.perf_counter()
    res_h2_sd = run_acase(rho_h2, hub2.hamiltonian, sd2, max_size=8, leakage_tol=1e-9)
    h2_arms["SD_Pool"] = _adaptive_summary(res_h2_sd, e0_h2, hub2.n, time.perf_counter() - t0)
    h2_arms["Krylov3"] = _fixed_subspace_summary(rho_h2, hub2.hamiltonian, [identity_generator(hub2.n)] + krylov_response(hub2.hamiltonian, 3), e0_h2, hub2.n)
    h2_controls = _run_controls(hub2, e0_h2, 8)
    span_h2_d1 = span_comparison(dense_basis(rho_h2, [res_h2_d1.bank.generator(i) for i in res_h2_d1.indices]), sd2_det_cols).to_record()
    span_h2_d2 = span_comparison(dense_basis(rho_h2, [res_h2_d2.bank.generator(i) for i in res_h2_d2.indices]), sd2_det_cols).to_record()
    out["hubbard_2x2"] = {
        "exact_ground_energy": float(e0_h2),
        "arms": h2_arms,
        "controls": h2_controls,
        "span_d1_vs_SD": span_h2_d1,
        "span_d2_vs_SD": span_h2_d2,
    }
    print(
        f"  hub2x2: S={h2_arms['Singles']['error_mHa']:.2f}, d1={h2_arms['DynComm(S)_d1']['error_mHa']:.2f}, "
        f"d2_stall={h2_arms['DynComm(S)_d2_stall']['error_mHa']:.2f}, SD={h2_arms['SD_Pool']['error_mHa']:.2f}, "
        f"K3={h2_arms['Krylov3']['error_mHa']:.2f}, IterSCI={h2_controls['IterSCI']['error_mHa']:.2f} mHa",
        flush=True,
    )

    # 3. H4 STO-3G r=0.9 & LiH CAS(4e,4o)
    for mname, path in [
        ("h4_sto3g_r0.9", ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP"),
        ("lih_cas4e4o", ROOT / "benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP"),
    ]:
        mod = fcidump_model(path)
        sec = SectorStatevectorBackend(mod.n, mod.metadata["n_electrons"], mod.metadata["sz"])
        e0 = float(sec.ground_state(mod.hamiltonian, k=1, method="lanczos")[0][0])
        rho = ExactMVBackend().state(mod.reference, ())
        occ = occupied_spin_orbitals(mod)
        s_ops = determinant_excitations(mod.n, occ, max_rank=1)
        sd_ops = determinant_excitations(mod.n, occ, max_rank=2)
        sd_det_cols = dense_basis(rho, [identity_generator(mod.n)] + sd_ops)

        arms = {}
        t0 = time.perf_counter()
        arms["Singles"] = _adaptive_summary(run_acase(rho, mod.hamiltonian, s_ops, max_size=8, leakage_tol=1e-9), e0, mod.n, time.perf_counter() - t0)
        t0 = time.perf_counter()
        res_d1 = run_dynamic_commutator_acase(rho, mod.hamiltonian, s_ops, max_depth=1, max_size=8, leakage_tol=1e-9, phase_mode="hermitian")
        arms["DynComm(S)_d1"] = _adaptive_summary(res_d1, e0, mod.n, time.perf_counter() - t0)
        t0 = time.perf_counter()
        res_sd = run_acase(rho, mod.hamiltonian, sd_ops, max_size=8, leakage_tol=1e-9)
        arms["SD_Pool"] = _adaptive_summary(res_sd, e0, mod.n, time.perf_counter() - t0)
        arms["Krylov2"] = _fixed_subspace_summary(rho, mod.hamiltonian, [identity_generator(mod.n)] + krylov_response(mod.hamiltonian, 2), e0, mod.n)
        if mname == "lih_cas4e4o":
            t0 = time.perf_counter()
            res_t32 = run_dynamic_commutator_acase(
                rho, mod.hamiltonian, s_ops, max_depth=2, max_children_total=12,
                truncation_top_k=32, leakage_tol=1e-9, leakage_mode="reference",
                max_size=8, phase_mode="hermitian",
            )
            arms["DynComm(S)_d2_top32"] = _adaptive_summary(res_t32, e0, mod.n, time.perf_counter() - t0)
        ctrls = _run_controls(mod, e0, 8)
        span_d1 = span_comparison(dense_basis(rho, [res_d1.bank.generator(i) for i in res_d1.indices]), sd_det_cols).to_record()
        out[mname] = {
            "exact_ground_energy": float(e0),
            "reference_energy": float(arms["Singles"]["energy"]),
            "arms": arms,
            "controls": ctrls,
            "span_d1_vs_SD": span_d1,
        }
        print(
            f"  {mname}: RHF={arms['Singles']['error_mHa']:.3f}, Dyn_d1={arms['DynComm(S)_d1']['error_mHa']:.3f} "
            f"(W={arms['DynComm(S)_d1']['word_universe']}), SD={arms['SD_Pool']['error_mHa']:.3f} "
            f"(W={arms['SD_Pool']['word_universe']}), K2={arms['Krylov2']['error_mHa']:.3f}, "
            f"IterSCI={ctrls['IterSCI']['error_mHa']:.3f} mHa",
            flush=True,
        )
    return out


def part_b1_pareto_sweeps() -> dict[str, Any]:
    print("[B1] Cost-ratio Pareto sweeps & epsilon myopia ablation...", flush=True)
    sc = DeviceCard.load(DEVICE_DIR / "superconducting-like.json")
    ion = DeviceCard.load(DEVICE_DIR / "ion-like.json")
    logical = DeviceCard.load(DEVICE_DIR / "logical-alltoall.json")

    models: list[tuple[str, Any, Any, list[Generator], int]] = []
    for name, mod, m_max in [
        ("tfim4", tfim(4, J=1.0, h=1.0), 5),
        ("tfim5", tfim(5, J=1.0, h=1.0), 5),
        ("xxz4", xxz(4, J=1.0, delta=1.0), 5),
    ]:
        rho = ExactMVBackend().state(mod.reference, ())
        hw = [w for w, _ in mod.hamiltonian.items() if w.code != 0]
        pool = pauli_orbit(hw) + krylov_response(mod.hamiltonian, 3)
        models.append((name, mod, rho, pool, m_max))

    hub1 = hubbard(2, t=1.0, U=4.0)
    rho_h1_det = ExactMVBackend().state(hub1.reference, ())
    rho_h1_corr = ExactMVBackend().state(trotter_program(hub1.hamiltonian, 0.25, steps=4), (), initial_state=rho_h1_det)
    occ1 = occupied_spin_orbitals(hub1)
    s1 = determinant_excitations(hub1.n, occ1, max_rank=1)
    hier_h1 = nested_commutator_hierarchy(hub1.hamiltonian, s1, max_depth=1, phase_mode="hermitian")
    pool_h1 = hier_h1.generators() + krylov_response(hub1.hamiltonian, 2)
    models.append(("hub1_corr", hub1, rho_h1_corr, pool_h1, 4))

    cost_configs = [
        ("EnergyOnly", IncrementalCostModel(epsilon=1.0)),
        ("Word_0.05", IncrementalCostModel(word_weight=0.05, epsilon=1.0)),
        ("Word_0.2", IncrementalCostModel(word_weight=0.2, epsilon=1.0)),
        ("Word_1.0", IncrementalCostModel(word_weight=1.0, epsilon=1.0)),
        ("QWCCost_SC_0.5", IncrementalCostModel(setting_weight=0.5, device_card=sc, epsilon=1.0)),
        ("QWCCost_SC_2.0", IncrementalCostModel(setting_weight=2.0, device_card=sc, epsilon=1.0)),
        ("TimeCost_SC", IncrementalCostModel(time_weight_us=0.05, device_card=sc, shots_per_setting=10.0, epsilon=1.0)),
        ("TimeCost_Ion", IncrementalCostModel(time_weight_us=0.0005, device_card=ion, shots_per_setting=10.0, epsilon=1.0)),
        ("TimeCost_Logical", IncrementalCostModel(time_weight_us=0.05, device_card=logical, shots_per_setting=10.0, epsilon=1.0)),
        ("Classical_1e-3", IncrementalCostModel(classical_product_weight=1e-3, epsilon=1.0)),
        ("Myopia_eps1e-3", IncrementalCostModel(word_weight=0.05, setting_weight=0.2, device_card=sc, epsilon=1e-3)),
    ]

    sweep_results: dict[str, Any] = {}
    for mname, mod, rho, pool, m_max in models:
        e0, _ = exact_ground(mod.hamiltonian.to_mv())
        sys_arms = {}
        for cname, cm in cost_configs:
            res = run_resource_aware_acase(
                rho, mod.hamiltonian, pool, cost_model=cm, max_size=m_max, min_lowering=1e-10
            )
            trajectory = []
            for k in range(2, len(res.indices) + 1):
                prefix_idx = res.indices[:k]
                solved_k = res.bank.solve(prefix_idx)
                words_k = res.bank.word_set(prefix_idx)
                rec_k = res.records[k - 2] if k - 2 < len(res.records) else None
                bd_k = res.resources["cost_breakdowns"][k - 2] if k - 2 < len(res.resources["cost_breakdowns"]) else {}
                trajectory.append(
                    {
                        "M": int(k),
                        "error_mHa": float((solved_k.ground_energy - e0) * MHA),
                        "word_universe": int(len(words_k)),
                        "qwc_groups": int(_qwc_count(mod.n, words_k)),
                        "time_us_sc_1shot": float(_total_qwc_time_us(sc, mod.n, words_k)),
                        "time_us_ion_1shot": float(_total_qwc_time_us(ion, mod.n, words_k)),
                        "step_classical_products": int(bd_k.get("classical_products", 0)),
                        "selected_label": str(rec_k.selected_label) if rec_k else None,
                    }
                )
            sys_arms[cname] = {
                "final_M": int(len(res.indices)),
                "final_error_mHa": float((res.energy - e0) * MHA),
                "final_word_universe": int(res.resources["word_universe"]),
                "final_qwc_groups": int(res.resources["final_qwc_groups"]),
                "final_time_us_sc_1shot": float(_total_qwc_time_us(sc, mod.n, res.bank.word_set(res.indices))),
                "total_classical_products": int(sum(x["classical_products"] for x in res.resources["cost_breakdowns"])),
                "trajectory": trajectory,
            }
        sweep_results[mname] = {"exact_ground_energy": float(e0), "arms": sys_arms}
        print(
            f"  {mname}: E-only(M={sys_arms['EnergyOnly']['final_M']}, {sys_arms['EnergyOnly']['final_error_mHa']:.2f}mHa, "
            f"{sys_arms['EnergyOnly']['final_word_universe']}w/{sys_arms['EnergyOnly']['final_qwc_groups']}g) | "
            f"QWC0.5({sys_arms['QWCCost_SC_0.5']['final_error_mHa']:.2f}mHa, "
            f"{sys_arms['QWCCost_SC_0.5']['final_word_universe']}w/{sys_arms['QWCCost_SC_0.5']['final_qwc_groups']}g) | "
            f"Myopia({sys_arms['Myopia_eps1e-3']['final_error_mHa']:.2f}mHa, "
            f"{sys_arms['Myopia_eps1e-3']['final_word_universe']}w/{sys_arms['Myopia_eps1e-3']['final_qwc_groups']}g)",
            flush=True,
        )
    return sweep_results


def part_b2_delta_method_mc() -> dict[str, Any]:
    print("[B2] 7D delta-method finite-shot MC validation (tfim4, 25 seeds)...", flush=True)
    mod = tfim(4, J=1.0, h=1.0)
    rho = ExactMVBackend().state(mod.reference, ())
    bank = MatrixElementBank(rho, mod.hamiltonian)
    basis = list(bank.extend([identity_generator(4)]))
    h2_idx = bank.extend(krylov_response(mod.hamiltonian, 2))[1]  # H^2
    h1_idx = bank.extend(krylov_response(mod.hamiltonian, 1))[0]  # H
    dup_idx = bank.add(Generator("I+0.05X0", 1.0 * (X(4, 0) * X(4, 0)) + 0.05 * X(4, 0)))

    cands = [("H^2", h2_idx), ("H", h1_idx), ("I+0.05X0", dup_idx)]
    sess = SharedMeasurement(bank, basis + [c[1] for c in cands], pooling="shots")
    basis_sess = SharedMeasurement(bank, basis, groups=sess.groups, pooling="shots")

    # Exact values
    exact_b = basis_sess.bank.solve(basis)
    exact_ref: dict[str, tuple[float, float]] = {}
    for cname, cidx in cands:
        low_ex, _, orth_ex, _, _ = expected_lowering_estimate(bank, basis, exact_b, cidx, cache=None)
        exact_ref[cname] = (float(low_ex), float(orth_ex))

    risk_cm = IncrementalCostModel(expected_improvement_mode="risk_adjusted", risk_aversion=2.0, require_stat_orthogonality=True, orthogonality_z=2.0, epsilon=1.0)
    seeds = list(range(100, 125))
    shots = 3000

    records: dict[str, list[tuple[float, float, float, float, float, bool]]] = {c[0]: [] for c in cands}
    for seed in seeds:
        cache = sess.measure(FiniteShotBackend(seed=seed), shots)
        res_b = basis_sess.solve(cache)
        bw = bank.word_set(basis)
        for cname, cidx in cands:
            low, lstd, orth, ostd, _ = expected_lowering_estimate(bank, basis, res_b, cidx, cache=cache)
            sc_risk = score_information_candidate(
                bank, basis, res_b, cidx, cost_model=risk_cm, cache=cache, basis_words=bw, min_orthogonality=1e-8
            )
            records[cname].append((low, lstd, orth, ostd, sc_risk.risk_adjusted_lowering, sc_risk.accepted))

    summary = {}
    for cname, _ in cands:
        rows = records[cname]
        lows = np.array([r[0] for r in rows])
        lstds = np.array([r[1] for r in rows])
        orths = np.array([r[2] for r in rows])
        ostds = np.array([r[3] for r in rows])
        ra_lows = np.array([r[4] for r in rows])
        accepted = [r[5] for r in rows]
        emp_lstd = float(np.std(lows, ddof=1))
        emp_ostd = float(np.std(orths, ddof=1))
        med_lstd = float(np.median(lstds))
        med_ostd = float(np.median(ostds))
        summary[cname] = {
            "exact_lowering": exact_ref[cname][0],
            "exact_orth": exact_ref[cname][1],
            "mc_mean_lowering": float(np.mean(lows)),
            "empirical_lowering_std": emp_lstd,
            "predicted_lowering_std_median": med_lstd,
            "predicted_lowering_std_iqr": [float(np.percentile(lstds, 25)), float(np.percentile(lstds, 75))],
            "lowering_std_ratio": float(med_lstd / emp_lstd) if emp_lstd > 0 else None,
            "empirical_orth_std": emp_ostd,
            "predicted_orth_std_median": med_ostd,
            "orth_std_ratio": float(med_ostd / emp_ostd) if emp_ostd > 0 else None,
            "risk_adjusted_zeroed_fraction": float(np.mean(ra_lows == 0.0)),
            "stat_orth_accepted_fraction": float(np.mean(accepted)),
        }
        print(
            f"  {cname:<9}: exact_dE={exact_ref[cname][0]*1e3:6.2f}mHa | emp_std={emp_lstd*1e3:5.2f} vs "
            f"pred_med={med_lstd*1e3:5.2f}mHa (ratio={med_lstd/emp_lstd if emp_lstd else 0:.2f}) | "
            f"orth emp={emp_ostd:.4f} vs pred={med_ostd:.4f} | zeroed={summary[cname]['risk_adjusted_zeroed_fraction']:.2f}",
            flush=True,
        )
    return {"shots_per_group": shots, "seeds": seeds, "candidates": summary}


def part_b3_closed_loop_acquisition() -> dict[str, Any]:
    print("[B3] Closed-loop acquire_to_accuracy (5 seeds) vs UniformFullRegistry & UniformBasisOnly...", flush=True)
    sig_params = sorted(inspect.signature(acquire_to_accuracy).parameters.keys())
    oracle_params = [p for p in sig_params if "exact" in p or "oracle" in p or "target_error" in p]
    if oracle_params:
        raise AssertionError(f"Oracle parameters found on acquire_to_accuracy: {oracle_params}")

    sc = DeviceCard.load(DEVICE_DIR / "superconducting-like.json")
    ion = DeviceCard.load(DEVICE_DIR / "ion-like.json")
    seeds = [11, 12, 13, 17, 23]

    configs = [
        ("tfim4_sc_se", tfim(4, J=1.0, h=1.0), sc, "std_error", 0.02),
        ("tfim4_ion_se", tfim(4, J=1.0, h=1.0), ion, "std_error", 0.02),
        ("tfim4_sc_ci", tfim(4, J=1.0, h=1.0), sc, "ci_halfwidth", 0.04),
        ("xxz4_sc_se", xxz(4, J=1.0, delta=1.0), sc, "std_error", 0.02),
    ]

    scenarios: dict[str, Any] = {}
    for cfg_name, mod, card, metric, target_acc in configs:
        e0, _ = exact_ground(mod.hamiltonian.to_mv())
        rho = ExactMVBackend().state(mod.reference, ())
        hw = [w for w, _ in mod.hamiltonian.items() if w.code != 0]
        pool = krylov_response(mod.hamiltonian, 3) + pauli_orbit(hw)[:8]
        cm = IncrementalCostModel(
            setting_weight=0.1,
            time_weight_us=0.001 if card.name == "superconducting-like" else 0.0001,
            device_card=card,
            shots_per_setting=60.0,
            risk_aversion=1.0,
            epsilon=1.0,
        )

        rows = []
        for seed in seeds:
            acq = acquire_to_accuracy(
                rho,
                mod.hamiltonian,
                pool,
                FiniteShotBackend(seed=seed),
                target_accuracy=target_acc,
                stop_metric=metric,
                cost_model=cm,
                max_basis_size=6,
                max_steps=80,
                pilot_shots=60,
                refine_batch_shots=600,
                max_total_shots=60_000,
            )

            # Fair comparator 1: UniformFullRegistry (exact same QWC groups discovered by acq.cache)
            reg_groups = _registry_groups_for_acq(acq)
            sm_uf = SharedMeasurement(acq.bank, acq.indices, groups=reg_groups, pooling="shots")
            shots_uf = max(1, acq.total_shots // max(1, len(reg_groups)))
            cache_uf = sm_uf.measure(FiniteShotBackend(seed=seed), shots_uf)
            res_uf = sm_uf.solve(cache_uf)
            iv_uf = ritz_uncertainty(sm_uf, cache_uf, res_uf, root=0)
            stat_uf = float(iv_uf.std_error or 0.0) if metric == "std_error" else 0.5 * (iv_uf.upper - iv_uf.lower)

            # Fair comparator 2: UniformBasisOnly (final basis QWC groups without candidate screening overhead)
            b_same = MatrixElementBank(rho, mod.hamiltonian, [acq.bank.generator(i) for i in acq.indices])
            sm_ub = SharedMeasurement(b_same, pooling="shots")
            shots_ub = max(1, acq.total_shots // max(1, len(sm_ub.groups)))
            cache_ub = sm_ub.measure(FiniteShotBackend(seed=seed), shots_ub)
            res_ub = sm_ub.solve(cache_ub)
            iv_ub = ritz_uncertainty(sm_ub, cache_ub, res_ub, root=0)
            stat_ub = float(iv_ub.std_error or 0.0) if metric == "std_error" else 0.5 * (iv_ub.upper - iv_ub.lower)

            stat_acq = float(acq.std_error) if metric == "std_error" else 0.5 * (acq.ci_interval.upper - acq.ci_interval.lower)
            rows.append(
                {
                    "seed": int(seed),
                    "converged": bool(acq.converged),
                    "stopped_reason": str(acq.stopped_reason),
                    "basis_size": int(len(acq.indices)),
                    "expand_actions": int(acq.resources["expand_actions"]),
                    "refine_actions": int(acq.resources["refine_actions"]),
                    "total_shots": int(acq.total_shots),
                    "total_time_us": float(acq.total_time_us),
                    "qwc_groups_full": int(acq.qwc_groups),
                    "qwc_groups_basis": int(len(sm_ub.groups)),
                    "acq_stat_mHa": float(stat_acq * MHA),
                    "acq_std_error_mHa": float(acq.std_error * MHA),
                    "acq_ci_halfwidth_mHa": float(0.5 * (acq.ci_interval.upper - acq.ci_interval.lower) * MHA),
                    "acq_est_rem_bias_mHa": float(acq.estimated_remaining_bias * MHA),
                    "acq_true_abs_err_mHa": float(abs(acq.energy - e0) * MHA),
                    "uf_stat_mHa": float(stat_uf * MHA),
                    "uf_true_abs_err_mHa": float(abs(res_uf.ground_energy - e0) * MHA),
                    "ub_stat_mHa": float(stat_ub * MHA),
                    "ub_true_abs_err_mHa": float(abs(res_ub.ground_energy - e0) * MHA),
                    "stat_ratio_vs_uf": float(stat_acq / max(stat_uf, 1e-12)),
                    "stat_ratio_vs_ub": float(stat_acq / max(stat_ub, 1e-12)),
                }
            )

        agg = {
            "converged_count": int(sum(r["converged"] for r in rows)),
            "median_shots": float(np.median([r["total_shots"] for r in rows])),
            "median_time_ms": float(np.median([r["total_time_us"] for r in rows]) / 1e3),
            "median_acq_stat_mHa": float(np.median([r["acq_stat_mHa"] for r in rows])),
            "median_uf_stat_mHa": float(np.median([r["uf_stat_mHa"] for r in rows])),
            "median_ub_stat_mHa": float(np.median([r["ub_stat_mHa"] for r in rows])),
            "median_acq_true_err_mHa": float(np.median([r["acq_true_abs_err_mHa"] for r in rows])),
            "median_uf_true_err_mHa": float(np.median([r["uf_true_abs_err_mHa"] for r in rows])),
            "median_ub_true_err_mHa": float(np.median([r["ub_true_abs_err_mHa"] for r in rows])),
            "wins_vs_uf_stat": int(sum(r["acq_stat_mHa"] < r["uf_stat_mHa"] for r in rows)),
            "wins_vs_ub_stat": int(sum(r["acq_stat_mHa"] < r["ub_stat_mHa"] for r in rows)),
        }
        scenarios[cfg_name] = {
            "device": card.name,
            "stop_metric": metric,
            "target_accuracy_mHa": float(target_acc * MHA),
            "exact_ground_energy": float(e0),
            "seeds": rows,
            "aggregate": agg,
        }
        print(
            f"  {cfg_name:<12}: conv={agg['converged_count']}/5 med_shots={agg['median_shots']:.0f} "
            f"med_stat acq={agg['median_acq_stat_mHa']:.2f} uf={agg['median_uf_stat_mHa']:.2f} ub={agg['median_ub_stat_mHa']:.2f}mHa "
            f"(wins uf={agg['wins_vs_uf_stat']}/5, ub={agg['wins_vs_ub_stat']}/5)",
            flush=True,
        )

    return {
        "oracle_free_signature_checked": True,
        "acquire_to_accuracy_parameters": sig_params,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    t_start = time.perf_counter()
    prov = execution_provenance()
    record: dict[str, Any] = {
        "schema": "clifford_qc.geometric_information_study.v1",
        "metadata": {
            "git_sha": prov["git_sha"],
            "python": prov["python"],
            "numpy": prov["dependencies"].get("numpy"),
            "utc": prov["utc"],
        },
        "part_a1_growth_closure_symmetry": part_a1_growth_closure_and_symmetry(),
        "part_a2_spin_chains": part_a2_spin_chains(),
        "part_a3_fermionic_and_chemistry": part_a3_fermionic_and_chemistry(),
        "part_b1_pareto_sweeps": part_b1_pareto_sweeps(),
        "part_b2_delta_method_mc": part_b2_delta_method_mc(),
        "part_b3_closed_loop_acquisition": part_b3_closed_loop_acquisition(),
    }
    record["total_wall_seconds"] = float(time.perf_counter() - t_start)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nCompleted full study in {record['total_wall_seconds']:.2f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
