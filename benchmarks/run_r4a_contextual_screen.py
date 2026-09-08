"""Execute the one preregistered R4a contextual structural screen.

This producer performs deterministic algebra and dense projected eigensolves
only.  It visits the frozen contextual ladder, evaluates the four declared
BeH2/Jordan--Wigner arms against the common bias and non-identity-word gates,
and selects the largest rung at which every arm passes.  It draws no samples,
prices no device, reports no cost ratio, and cannot classify QR5.

The result-free declaration must already pass before this module will touch the
BeH2 bank.  The resulting record belongs in a separate commit from that
declaration, exactly as its temporal-order contract requires.

    python benchmarks/run_r4a_contextual_screen.py
    python benchmarks/check_r4a_contextual_screen.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import block_commuting_partition
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace import (
    Generator,
    compile_contextual_restriction,
    identity_generator,
    project_contextual_problem,
)
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.generator_core import scalar_free_key

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    import benchmarks.check_r4a_preregistration as preregistration
    from benchmarks.run_mapping_axis import _build_model, load_config as load_mapping_config
except ImportError:  # pragma: no cover - direct script execution
    import check_r4a_preregistration as preregistration
    from run_mapping_axis import _build_model, load_config as load_mapping_config


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "r4a_contextual_interaction.json"
REFERENCE = HERE / "reference_results" / "r4a_contextual_screen.json"
SCHEMA = "clifford_qc.r4a_contextual_screen.v1"
PREREGISTRATION_COMMIT = "fad69c2f8a846c7474dc2f652af19685268640c3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _environment_problems(config: dict) -> list[str]:
    expected = config["conditional_sampled_protocol"]["execution_environment"]
    actual = {
        "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "stim": importlib.metadata.version("stim"),
    }
    return [
        f"execution environment {key}={actual[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]


def load_config(path: Path = CONFIG) -> dict:
    """Load the immutable preregistration and refuse any drift before execution."""
    config = preregistration.load_config(path)
    problems = preregistration.static_problems(config)
    if problems:
        raise ValueError("R4a preregistration failed: " + "; ".join(problems))
    environment = _environment_problems(config)
    if environment:
        raise ValueError("; ".join(environment))
    authorization = config["authorization"]
    expected_paths = {
        "structural_producer": "benchmarks/run_r4a_contextual_screen.py",
        "structural_record": "benchmarks/reference_results/r4a_contextual_screen.json",
        "structural_checker": "benchmarks/check_r4a_contextual_screen.py",
    }
    for key, expected in expected_paths.items():
        if authorization.get(key) != expected:
            raise ValueError(f"R4a {key} drifted from {expected!r}")
    if authorization.get("exactly_one_later_structural_execution") is not True:
        raise ValueError("R4a does not authorize exactly one structural execution")
    if authorization.get("sampled_execution_authorized_by_this_config") is not False:
        raise ValueError("the structural producer may not consume sampled authority")
    return config


def _frozen_model(config: dict):
    specs = {row["key"]: row for row in load_mapping_config()["systems"]}
    key = config["system"]["system"]
    if key not in specs:
        raise ValueError(f"no mapping-axis model specification for {key!r}")
    model, construction = _build_model(specs[key])
    if model.n != config["system"]["n_qubits"]:
        raise ValueError("the frozen BeH2 qubit count drifted")
    if config["system"]["mapping"] != "jw":
        raise ValueError("R4a is preregistered only on the Jordan--Wigner bank")
    return model, construction


def _bases(model, config: dict) -> tuple[list[Generator], list[Generator], tuple[int, ...]]:
    occupied = occupied_spin_orbitals(model)
    candidates = determinant_excitations(
        model.n, occupied, max_rank=2, conserve_sz=True
    )
    full = [identity_generator(model.n), *candidates]
    by_label = {row.label: row for row in full}
    if len(by_label) != len(full):
        raise AssertionError("the complete determinant family contains duplicate labels")
    labels = config["system"]["acase_bank"]["basis_labels"]
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(f"frozen A-CASE labels are absent from the family: {missing}")
    acase = [by_label[label] for label in labels]
    if len(acase) != config["system"]["acase_bank"]["basis_size"]:
        raise ValueError("the frozen A-CASE bank size drifted")
    return full, acase, occupied


def _deduplicate_projected(
    source: Sequence[Generator], images: Sequence, *, tol: float
) -> tuple[list[Generator], list[int], list[dict]]:
    """Remove zero and scalar-proportional images in source-basis order."""
    if len(source) != len(images):
        raise ValueError("source and projected generator counts differ")
    kept: list[Generator] = []
    annihilated: list[int] = []
    duplicates: list[dict] = []
    seen: dict[tuple, tuple[int, str]] = {}
    for index, (generator, image) in enumerate(zip(source, images)):
        if image.is_zero(tol):
            annihilated.append(index)
            continue
        key = scalar_free_key(image)
        if key is None:
            annihilated.append(index)
            continue
        if key in seen:
            retained_index, retained_label = seen[key]
            duplicates.append(
                {
                    "candidate_index": index,
                    "candidate_label": generator.label,
                    "retained_index": retained_index,
                    "retained_label": retained_label,
                }
            )
            continue
        seen[key] = (index, generator.label)
        kept.append(Generator(generator.label, image))
    if not kept or kept[0].label != "I":
        raise AssertionError("identity must survive as the first projected basis direction")
    return kept, annihilated, duplicates


def _arm_record(
    name: str,
    source: Sequence[Generator],
    images: Sequence,
    *,
    reference,
    hamiltonian,
    exact_energy: float,
    active_qubits: int,
    removed_fraction: float,
    gates: dict,
    tol: float,
) -> dict:
    kept, annihilated, duplicate_rows = _deduplicate_projected(
        source, images, tol=tol
    )
    bank = MatrixElementBank(reference, hamiltonian, kept)
    solved = bank.solve()
    energy = float(solved.ground_energy)
    signed_error = (energy - exact_energy) * 1e3
    bias = abs(signed_error)
    codes = sorted(code for code in bank.word_set() if code != 0)
    if not codes and hamiltonian.nnz():
        raise AssertionError("a solved nontrivial arm produced an empty word universe")
    block_sizes = [int(value) for value in gates["block_sizes"]]
    settings = {
        str(block_size): len(
            block_commuting_partition(active_qubits, codes, block_size)
        )
        for block_size in block_sizes
    }
    clears_bias = bias <= float(gates["admissible_bias_millihartree"])
    clears_words = len(codes) <= int(gates["word_universe_ceiling"])
    duplicate_indices = [row["candidate_index"] for row in duplicate_rows]
    return {
        "name": name,
        "basis_size_before_deduplication": len(source),
        "basis_size_after_deduplication": len(kept),
        "basis_labels_before_deduplication": [row.label for row in source],
        "basis_labels_after_deduplication": [row.label for row in kept],
        "candidate_index_convention": (
            "zero-based index in basis_labels_before_deduplication; identity is index 0"
        ),
        "annihilated_candidate_indices": annihilated,
        "annihilated_candidate_labels": [source[index].label for index in annihilated],
        "duplicate_projected_candidate_indices": duplicate_indices,
        "duplicate_projected_candidates": duplicate_rows,
        "projected_deduplication_rule": (
            "clifford_qc.subspace.generator_core.scalar_free_key in frozen source order"
        ),
        "active_qubits": active_qubits,
        "ground_energy": energy,
        "signed_error_millihartree": signed_error,
        "bias_millihartree": bias,
        "retained_overlap_rank": int(solved.effective_rank),
        "hamiltonian_removed_hs_fraction": float(removed_fraction),
        "word_universe": len(codes),
        "word_universe_convention": "non_identity_words_only",
        "settings_by_block_size": settings,
        "clears_bias_margin": clears_bias,
        "clears_word_universe_ceiling": clears_words,
        "admissible": clears_bias and clears_words,
    }


def _control_arm(
    name: str,
    basis: Sequence[Generator],
    *,
    reference,
    hamiltonian,
    exact_energy: float,
    gates: dict,
    tol: float,
) -> dict:
    return _arm_record(
        name,
        basis,
        [row.mv for row in basis],
        reference=reference,
        hamiltonian=hamiltonian,
        exact_energy=exact_energy,
        active_qubits=hamiltonian.n,
        removed_fraction=0.0,
        gates=gates,
        tol=tol,
    )


def _contextual_arm(
    name: str,
    basis: Sequence[Generator],
    contextual,
    *,
    exact_energy: float,
    gates: dict,
    tol: float,
) -> dict:
    problem = contextual.problem
    return _arm_record(
        name,
        basis,
        problem.generators,
        reference=problem.reference,
        hamiltonian=problem.hamiltonian,
        exact_energy=exact_energy,
        active_qubits=problem.n,
        removed_fraction=contextual.hamiltonian_removed_hs_fraction,
        gates=gates,
        tol=tol,
    )


def build_record(config: dict | None = None) -> dict:
    """Build the frozen structural record; this is the result-producing call."""
    config = load_config() if config is None else config
    environment = _environment_problems(config)
    if environment:
        raise ValueError("; ".join(environment))
    model, construction = _frozen_model(config)
    full_basis, acase_basis, occupied = _bases(model, config)
    reference = ExactMVBackend().state(model.reference, ())
    exact_backend = SectorStatevectorBackend(
        model.n,
        int(model.metadata["n_electrons"]),
        float(model.metadata["sz"]),
    )
    exact_energy = float(
        exact_backend.ground_state(model.hamiltonian, k=1, method="dense")[0][0]
    )
    if abs(exact_energy - construction["external_sector_energy"]) > 1e-10:
        raise AssertionError("dense sector energy disagrees with FCIDUMP provenance")

    screen = config["structural_screen"]
    gates = {
        "accuracy_target_millihartree": float(
            screen["accuracy_target_millihartree"]
        ),
        "margin_factor": float(screen["margin_factor"]),
        "admissible_bias_millihartree": float(
            screen["admissible_bias_millihartree"]
        ),
        "word_universe_ceiling": int(screen["word_universe_ceiling"]),
        "word_universe_convention": screen["word_universe_convention"],
        "block_sizes": [int(value) for value in screen["block_sizes"]],
        "grouping_rule": screen["grouping_rule"],
        "hamiltonian_removed_hs_fraction_denominator": screen[
            "hamiltonian_removed_hs_fraction_denominator"
        ],
    }
    if not math.isclose(
        gates["admissible_bias_millihartree"],
        gates["accuracy_target_millihartree"] / gates["margin_factor"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("the structural bias gate is not target / margin_factor")
    tol = float(config["contextual_restriction"]["tolerance"])

    controls = {
        "full_qse": _control_arm(
            "full_qse",
            full_basis,
            reference=reference,
            hamiltonian=model.hamiltonian,
            exact_energy=exact_energy,
            gates=gates,
            tol=tol,
        ),
        "acase": _control_arm(
            "acase",
            acase_basis,
            reference=reference,
            hamiltonian=model.hamiltonian,
            exact_energy=exact_energy,
            gates=gates,
            tol=tol,
        ),
    }

    rungs = []
    for rung in config["contextual_restriction"]["fixed_qubit_ladder"]:
        plan = compile_contextual_restriction(
            model.hamiltonian,
            reference,
            int(rung),
            tol=tol,
            label=f"r4a-contextual-{rung}",
        )
        full_contextual = project_contextual_problem(
            plan,
            hamiltonian=model.hamiltonian,
            reference=reference,
            generators=full_basis,
            tol=tol,
        )
        acase_contextual = project_contextual_problem(
            plan,
            hamiltonian=model.hamiltonian,
            reference=reference,
            generators=acase_basis,
            tol=tol,
        )
        if not full_contextual.problem.hamiltonian.is_close(
            acase_contextual.problem.hamiltonian, tol
        ) or not full_contextual.problem.reference.is_close(
            acase_contextual.problem.reference, tol
        ):
            raise AssertionError("the two contextual arms did not share one projection")
        cs_qse = _contextual_arm(
            "cs_qse",
            full_basis,
            full_contextual,
            exact_energy=exact_energy,
            gates=gates,
            tol=tol,
        )
        cs_acase = _contextual_arm(
            "cs_acase",
            acase_basis,
            acase_contextual,
            exact_energy=exact_energy,
            gates=gates,
            tol=tol,
        )
        arms = [
            copy.deepcopy(controls["full_qse"]),
            cs_qse,
            copy.deepcopy(controls["acase"]),
            cs_acase,
        ]
        admissible = all(row["admissible"] for row in arms)
        rungs.append(
            {
                "fixed_qubits": int(rung),
                "contextual_plan": plan.as_dict(),
                "arms": arms,
                "failed_bias_margin": [
                    row["name"] for row in arms if not row["clears_bias_margin"]
                ],
                "failed_word_universe_ceiling": [
                    row["name"]
                    for row in arms
                    if not row["clears_word_universe_ceiling"]
                ],
                "all_four_arms_admissible": admissible,
            }
        )

    passing = [row["fixed_qubits"] for row in rungs if row["all_four_arms_admissible"]]
    selected = max(passing) if passing else None
    if selected is None:
        reason = (
            "no contextual rung admits all four arms; R4a stops without sampling"
        )
        next_step = "R4a_stops_without_sampling"
    else:
        reason = (
            f"rung {selected} is the largest rung admitting all four arms; only a "
            "separate later commit may authorize sampled execution"
        )
        next_step = "eligible_for_separate_sampled_preregistration"

    record = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "status": "structural_screen_complete_no_samples",
        "evidence_tier": "structural",
        "preregistration": {
            "path": str(CONFIG.relative_to(ROOT)),
            "sha256": _sha256(CONFIG),
            "git_blob_sha1": _git_blob_sha1(CONFIG),
            "merged_commit": PREREGISTRATION_COMMIT,
        },
        "execution_authority": {
            "source": "authorization.exactly_one_later_structural_execution",
            "structural_execution_consumed": True,
            "sampled_execution_performed": False,
            "sampled_execution_authorized_by_this_record": False,
        },
        "system": {
            "system": config["system"]["system"],
            "label": config["system"]["label"],
            "mapping": config["system"]["mapping"],
            "n_qubits": model.n,
            "occupied_spin_orbitals": list(occupied),
            "n_electrons": int(model.metadata["n_electrons"]),
            "sz": float(model.metadata["sz"]),
            "exact_sector_energy": exact_energy,
            "construction": construction,
            "complete_basis_size": len(full_basis),
            "complete_basis_labels": [row.label for row in full_basis],
            "frozen_acase_basis_labels": [row.label for row in acase_basis],
        },
        "gates": gates,
        "arm_order": [row["name"] for row in config["comparison"]["arms"]],
        "contextual_rungs": rungs,
        "structural_gate": {
            "passing_contextual_rungs": passing,
            "selected_contextual_rung": selected,
            "all_four_arms_admitted_at_selected_rung": selected is not None,
            "reason": reason,
            "next_step": next_step,
            "sampled_execution_authorized": False,
        },
        "claim_boundary": (
            "This is the single preregistered BeH2/JW structural admission screen. "
            "It reports deterministic basis, bias, Hamiltonian-removal, word-universe, "
            "and setting-count fields only. It draws no samples, prices no C(epsilon), "
            "reports no cost ratio or Delta, and does not classify QR5."
        ),
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
    gate = record["structural_gate"]
    for rung in record["contextual_rungs"]:
        status = "PASS" if rung["all_four_arms_admissible"] else "fail"
        arms = ", ".join(
            f"{row['name']}: bias={row['bias_millihartree']:.6f} mHa, "
            f"W={row['word_universe']}"
            for row in rung["arms"]
        )
        print(f"{status:>4} rung={rung['fixed_qubits']}: {arms}")
    print(f"selected contextual rung: {gate['selected_contextual_rung']}")
    print(f"sampled execution authorized: {gate['sampled_execution_authorized']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
