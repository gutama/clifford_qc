"""Phase 10 QSCI x A-CASE primary-system driver.

This is an exploratory capability benchmark, not a winner selector. It keeps
the roadmap's distinct controls beside the three hybrid arms:

* full QSCI on every sampled determinant;
* bare A-CASE over the conserving excitation family;
* the Phase 9 exact family-closure control;
* a determinant-count-matched selected-CI control;
* the three Phase 10C hybrid arms and the Phase 10B family projection ledger.

The default sampling state is the exact sector ground state. That makes this a
clean test of subspace construction, but it is ORACLE evidence and must not be
read as an implementable state-preparation result.

Run all Phase 10D primary systems:

    python benchmarks/run_phase10_hybrid.py

The two chemistry-built H4 rungs need the chemistry extra. The FCIDUMP rung is
dependency-light and independently checks the equilibrium H4 mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import (
    configuration_generators_from_words,
    determinant_excitations,
    dressed_family,
    family_projection_report,
    identity_generator,
    occupied_spin_orbitals,
    run_acase,
    run_control,
    run_hybrid,
)
from clifford_qc.subspace.qsci import (
    exact_ground_state_oracle,
    run_qsci,
    sample_state_input,
)


ROOT = Path(__file__).resolve().parent
PRIMARY_SYSTEMS = (
    "hubbard_2x2",
    "hubbard_2x3",
    "h4_equilibrium",
    "h4_stretched",
    "fcidump_h4_equilibrium",
)
FCIDUMP_H4 = ROOT / "data" / "h4_sto3g_r0.9.FCIDUMP"
FCIDUMP_PROVENANCE = ROOT / "data" / "h4_sto3g_r0.9.provenance.json"


def _pyscf_memory_probe_fallback() -> bool:
    """Work around sandboxes that hide /proc/<pid>/statm from PySCF."""
    if Path(f"/proc/{os.getpid()}/statm").exists():
        return False
    try:
        import pyscf.lib
        import pyscf.lib.misc
    except ImportError:
        return False

    def _zero_memory():
        return 0.0, 0.0

    pyscf.lib.current_memory = _zero_memory
    pyscf.lib.misc.current_memory = _zero_memory
    return True


def build_system(name: str):
    """Return (model, construction_metadata) for one Phase 10D system."""
    if name == "hubbard_2x2":
        return hubbard((2, 2), t=1.0, U=4.0), {
            "source": "native_hubbard", "shape": [2, 2], "t": 1.0, "U": 4.0}
    if name == "hubbard_2x3":
        return hubbard((2, 3), t=1.0, U=4.0), {
            "source": "native_hubbard", "shape": [2, 3], "t": 1.0, "U": 4.0}
    if name in ("h4_equilibrium", "h4_stretched"):
        fallback = _pyscf_memory_probe_fallback()
        from clifford_qc.models.chemistry import h4_chain

        spacing = 0.9 if name == "h4_equilibrium" else 1.8
        return h4_chain(spacing=spacing), {
            "source": "pyscf_sto3g",
            "spacing_angstrom": spacing,
            "pyscf_memory_probe_fallback": fallback,
        }
    if name == "fcidump_h4_equilibrium":
        provenance = json.loads(FCIDUMP_PROVENANCE.read_text(encoding="utf-8"))
        model = fcidump_model(FCIDUMP_H4, name=provenance["name"])
        if model.metadata["source_sha256"] != provenance["fcidump_sha256"]:
            raise ValueError("FCIDUMP digest disagrees with its provenance record")
        return model, {
            "source": "committed_fcidump",
            "path": FCIDUMP_H4.relative_to(ROOT.parent).as_posix(),
            "sha256": model.metadata["source_sha256"],
            "external_fci_energy": float(
                provenance["reference_energies"]["fci"]),
        }
    raise ValueError(f"unknown Phase 10 primary system {name!r}")


def _reference_word(backend: SectorStatevectorBackend, model) -> int:
    state = backend.state_from_program(model.reference)
    return int(backend.basis[int(np.argmax(np.abs(state)))])


def _sampled_generators(model, backend, words):
    """One virtual generator per sampled determinant, identity iff sampled."""
    words = np.asarray(words, dtype=np.int64).reshape(-1)
    reference_word = _reference_word(backend, model)
    reference_sampled = bool(np.any(words == reference_word))
    nonreference = words[words != reference_word]
    configurations = (
        configuration_generators_from_words(model, words)
        if nonreference.size else [])
    baseline = list(configurations)
    if reference_sampled:
        baseline.insert(0, identity_generator(model.n))
    if len(baseline) != np.unique(words).size:
        raise AssertionError(
            "sampled determinant/generator count mismatch: the Phase 10 baseline "
            "must span exactly the sampled set")
    return configurations, baseline, reference_sampled


def _acase_record(result, exact_energy: float, *, candidate_pool: int) -> dict:
    resources = result.result.resources
    return {
        "energy": float(result.energy),
        "error": float(result.energy - exact_energy),
        "basis_size": int(result.basis_size),
        "retained_rank": int(result.result.effective_rank),
        "condition_number": float(result.result.condition_number),
        "word_universe": resources.get("word_universe"),
        "max_generator_support": resources.get("max_generator_support"),
        "max_element_support": resources.get(
            "max_hamiltonian_element_support"),
        "candidate_pool": int(candidate_pool),
        "stopped_reason": result.stopped_reason,
        "labels": list(result.labels),
        "evidence": "exact",
    }


def _assert_variational(record: dict, exact_energy: float, label: str) -> None:
    if float(record["energy"]) < exact_energy - 1e-9:
        raise AssertionError(
            f"{label} violates the variational bound: "
            f"{record['energy']} < {exact_energy}")


def run_system(name: str, *, shots: int = 128, seed: int = 0,
               max_size: int = 10, max_generators: int = 256,
               max_support: int = 64, max_packet_support: int = 16,
               project_family: bool = True) -> dict:
    """Run one Phase 10D primary system under one declared screening budget."""
    started = time.perf_counter()
    model, construction = build_system(name)
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]), float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    exact_values, _ = backend.ground_state(model.hamiltonian, k=1)
    exact_energy = float(exact_values[0])
    rho = ExactMVBackend().state(model.reference, ())

    oracle = exact_ground_state_oracle(backend, model.hamiltonian)
    indices, sampling = sample_state_input(oracle, shots=shots, seed=seed)
    words = backend.basis[indices]
    _, sampled_basis, reference_sampled = _sampled_generators(
        model, backend, words)

    # One fixed conserving pool defines bare A-CASE, dressing, and closure.
    partners = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    family = dressed_family(
        sampled_basis, model, kind="excitation", max_support=max_support,
        max_generators=max_generators)

    full_qsci = run_qsci(
        operator, indices, sampling=sampling, exact_energy=exact_energy)
    full_qsci_record = full_qsci.to_record()
    full_qsci_record["error"] = float(full_qsci.variational_gap)
    full_qsci_record["evidence"] = "oracle_sampled_subspace"

    bare_started = time.perf_counter()
    bare_acase = run_acase(
        rho, model.hamiltonian, partners, max_size=max_size,
        exact_ground_energy=exact_energy, leakage_tol=1e-9)
    bare_acase_record = _acase_record(
        bare_acase, exact_energy, candidate_pool=len(partners))
    bare_acase_record["seconds"] = time.perf_counter() - bare_started

    controls = {
        "family_closure": run_control(
            operator, indices, name="family_closure", kind="family_closure",
            exact_energy=exact_energy, generators=partners).to_record(),
        "budget_selected_ci": run_control(
            operator, indices, name="budget_selected_ci", kind="budget_matched",
            exact_energy=exact_energy, max_determinants=max_size + 1).to_record(),
    }

    projection = None
    if project_family:
        projection_started = time.perf_counter()
        projection = family_projection_report(
            rho, model.hamiltonian, family, sampled_basis)
        projection["seconds"] = float(time.perf_counter() - projection_started)

    arms = run_hybrid(
        rho, model, words, max_size=max_size, exact_energy=exact_energy,
        family=family, max_packet_support=max_packet_support)
    arm_records = [arm.to_record() for arm in arms]

    _assert_variational(full_qsci_record, exact_energy, "full QSCI")
    _assert_variational(bare_acase_record, exact_energy, "bare A-CASE")
    for arm in arm_records:
        _assert_variational(arm, exact_energy, arm["arm"])
        if int(arm["basis_size"]) > max_size + 1:
            raise AssertionError(f"{arm['arm']} exceeded the matched growth budget")
    for control_name, control in controls.items():
        _assert_variational(control, exact_energy, control_name)
    if controls["budget_selected_ci"]["determinant_count"] > max_size + 1:
        raise AssertionError("budget-selected CI exceeded its determinant cap")

    if construction["source"] == "committed_fcidump":
        mismatch = exact_energy - construction["external_fci_energy"]
        if abs(mismatch) > 1e-10:
            raise AssertionError(
                "FCIDUMP sector solve disagrees with its external FCI provenance")
        construction["external_fci_error"] = float(mismatch)

    return {
        "system_key": name,
        "system": model.name,
        "n_qubits": int(model.n),
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "sector_dimension": int(backend.dimension),
        "hamiltonian_terms": int(len(model.hamiltonian.terms)),
        "construction": construction,
        "exact_energy": exact_energy,
        "settings": {
            "sampling_input": "exact_ground_state_oracle",
            "sampling_evidence": "oracle",
            "shots": int(shots),
            "seed": int(seed),
            "configuration_order": "ascending_sector_index",
            "max_size_growth_steps": int(max_size),
            "max_retained_directions": int(max_size + 1),
            "family": "configuration_x_excitation",
            "max_family_generators": int(max_generators),
            "max_family_support": int(max_support),
            "max_packet_support": int(max_packet_support),
            "hybrid_leakage_filter": (
                "none: configuration maps preserve the sampled state sector "
                "without commuting with N and Sz as global operators"),
        },
        "sampling": sampling.to_record(),
        "reference_sampled": reference_sampled,
        "sampled_words": [int(word) for word in words.tolist()],
        "full_qsci": full_qsci_record,
        "bare_acase": bare_acase_record,
        "phase9_controls": controls,
        "family": family.to_record(),
        "family_projection": projection,
        "hybrid_arms": arm_records,
        "wall_seconds": float(time.perf_counter() - started),
    }


def _resolve_base_sha(provenance: dict) -> str:
    """The revision this record names, validated before it is written.

    ``execution_provenance`` reports ``"unknown"`` where git is unavailable, so
    ``PHASE10_BASE_SHA`` exists as an override for that case. It was previously
    copied verbatim, which left an unvalidated environment string as the
    record's only revision identifier -- the same failure the Phase 8E
    dirty-tree guard exists to prevent, arriving through a different door.

    An override must therefore look like a git object name, and is rejected
    rather than recorded when it does not.
    """
    override = os.environ.get("PHASE10_BASE_SHA")
    if override is None:
        return provenance["git_sha"]
    candidate = override.strip()
    if not (7 <= len(candidate) <= 40
            and all(character in "0123456789abcdefABCDEF"
                    for character in candidate)):
        raise SystemExit(
            f"PHASE10_BASE_SHA={override!r} is not a git object name; a record "
            "whose revision identifier cannot be resolved names nothing")
    return candidate.lower()


def _all_variational(records: list[dict], tolerance: float = 1e-9) -> bool:
    """Every reported energy sits at or above its exact reference.

    Computed from the records rather than asserted. A cross-check that is a
    hardcoded ``True`` cannot fail, so it certifies nothing while reading in
    the output exactly like one that does.
    """
    for record in records:
        errors = [record["full_qsci"]["error"], record["bare_acase"]["error"]]
        errors += [arm["error"] for arm in record["hybrid_arms"]]
        # `phase9_controls` is keyed by control name, so iterate its values.
        errors += [control["error"]
                   for control in record["phase9_controls"].values()
                   if control.get("error") is not None]
        if any(error < -tolerance for error in errors):
            return False
    return True


def _cross_checks(records: list[dict]) -> dict:
    checks = {
        "all_variational": _all_variational(records),
        "h4_equilibrium_fcidump_energy_delta": None,
        "h4_equilibrium_fcidump_match": None,
    }
    by_name = {record["system_key"]: record for record in records}
    left = by_name.get("h4_equilibrium")
    right = by_name.get("fcidump_h4_equilibrium")
    if left is not None and right is not None:
        delta = float(left["exact_energy"] - right["exact_energy"])
        checks["h4_equilibrium_fcidump_energy_delta"] = delta
        checks["h4_equilibrium_fcidump_match"] = abs(delta) <= 1e-10
        if not checks["h4_equilibrium_fcidump_match"]:
            raise AssertionError(
                "runtime H4 and committed FCIDUMP disagree at equilibrium")
    return checks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(PRIMARY_SYSTEMS))
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=10)
    parser.add_argument("--max-generators", type=int, default=256)
    parser.add_argument("--max-support", type=int, default=64)
    parser.add_argument("--max-packet-support", type=int, default=16)
    parser.add_argument("--skip-family-projection", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results" / "phase10_primary.json")
    args = parser.parse_args(argv)
    if args.shots <= 0 or args.max_size < 1 or args.max_generators < 1:
        raise SystemExit("shots, max-size, and max-generators must be positive")

    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    unknown = sorted(set(systems) - set(PRIMARY_SYSTEMS))
    if unknown:
        raise SystemExit(f"unknown Phase 10 systems: {unknown}")

    records = []
    for name in systems:
        record = run_system(
            name, shots=args.shots, seed=args.seed, max_size=args.max_size,
            max_generators=args.max_generators, max_support=args.max_support,
            max_packet_support=args.max_packet_support,
            project_family=not args.skip_family_projection)
        records.append(record)
        print(
            f"{name:27s} exact={record['exact_energy']:+.9f} "
            f"sampled={record['sampling']['unique_configurations']:3d} "
            f"QSCIerr={record['full_qsci']['error']:+.3e} "
            f"seconds={record['wall_seconds']:.2f}",
            flush=True)
        for arm in record["hybrid_arms"]:
            print(
                f"  {arm['arm']:29s} M={arm['basis_size']:2d} "
                f"err={arm['error']:+.3e} W={arm['word_universe']}",
                flush=True)

    provenance = execution_provenance()
    source_base_sha = _resolve_base_sha(provenance)
    result = stamp_record({
        "schema": "clifford_qc.phase10_primary.v1",
        "source_base_git_sha": source_base_sha,
        "evidence": "oracle_sampling_plus_exact_projected_arithmetic",
        "claim_boundary": (
            "exploratory Phase 10 screen; no winner and no implementable "
            "state-preparation claim"),
        "systems": records,
        "cross_checks": _cross_checks(records),
    }, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} systems to {args.out}", flush=True)


if __name__ == "__main__":
    main()
