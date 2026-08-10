"""Checkpointed orchestration for the preregistered PRD-CASE paper suite.

Dry-run is the default.  The expensive exact-simulation tier begins only with
``--execute`` and writes one stamped record per (system, M) task, outside the
source tree by default.  The finite-shot tier is deliberately not dispatched
here.  The matched A-CASE stage is also preregistered but separate: absence of
that baseline is explicit in every record and must block paper-level analysis,
rather than being mistaken for a favorable comparison.  Exact statevector,
matched-baseline and finite-shot evidence remain separately labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "prd_case_paper_suite.json"
DEFAULT_RESULTS = Path("/tmp/clifford_qc_prd_case_suite")
SCHEMA = "clifford_qc.prd_case_paper_suite.v1"
CHECKPOINT_SCHEMA = "clifford_qc.prd_case_suite_checkpoint.v1"
_MODEL_CACHE = {}


def _canonical_bytes(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def config_sha256(document: dict) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def load_manifest(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(document)
    return document


def _strictly_increasing_positive_ints(values, field: str, *, minimum=1):
    if not values or any(type(value) is not int or value < minimum
                         for value in values):
        raise ValueError(f"{field} must contain positive integers >= {minimum}")
    if values != sorted(set(values)):
        raise ValueError(f"{field} must be strictly increasing and unique")


def validate_manifest(document: dict) -> None:
    problems: list[str] = []
    if document.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if document.get("preregistered") is not True:
        problems.append("preregistered must be true")
    if document.get("results_committed_separately") is not True:
        problems.append("results_committed_separately must be true")

    systems = document.get("systems") or []
    ids = [entry.get("id") for entry in systems]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        problems.append("system ids must be present and unique")
    primary = [entry for entry in systems
               if entry.get("analysis_role") == "primary"]
    validation = [entry for entry in systems
                  if entry.get("analysis_role") == "construction_validation"]
    if len(primary) != document.get("physical_instance_count"):
        problems.append("physical_instance_count must equal primary systems")
    if len(validation) != document.get("validation_instance_count"):
        problems.append("validation_instance_count must equal validation systems")
    if any(entry.get("count_as_distinct") is not True for entry in primary):
        problems.append("every primary system must count as distinct")
    if any(entry.get("count_as_distinct") is not False for entry in validation):
        problems.append("construction validations must not count as distinct")

    families = {entry.get("family") for entry in primary}
    if not {"molecule", "hubbard"}.issubset(families):
        problems.append("primary suite must contain molecular and Hubbard families")
    for entry in systems:
        if not isinstance(entry.get("builder"), dict) or not entry["builder"].get("kind"):
            problems.append(f"{entry.get('id')}: builder.kind is required")
        if entry.get("default_run") not in (True, False):
            problems.append(f"{entry.get('id')}: default_run must be boolean")

    exact = document.get("exact_simulation") or {}
    try:
        _strictly_increasing_positive_ints(exact.get("m_budgets"),
                                           "exact_simulation.m_budgets", minimum=2)
        _strictly_increasing_positive_ints(exact.get("packet_k"),
                                           "exact_simulation.packet_k")
    except ValueError as error:
        problems.append(str(error))
    shifts = exact.get("shift_grid") or []
    if not shifts or len(shifts) != len(set(shifts)) or any(
            type(value) not in (int, float) for value in shifts):
        problems.append("shift_grid must contain unique numeric values")
    if exact.get("word_budget_ratio", 0) <= 0:
        problems.append("word_budget_ratio must be positive")
    if exact.get("absolute_packet_word_abort", 0) < exact.get(
            "grouping_word_limit", 0):
        problems.append("absolute_packet_word_abort must cover grouping_word_limit")
    extension = exact.get("accuracy_only_packet_extension") or {}
    if extension.get("implemented_by_this_driver") is not False:
        problems.append("large-sector K extension must remain a separate stage")

    method_ids = {entry.get("id") for entry in document.get("methods") or []}
    required_methods = {
        "orthonormalized_power_krylov", "orthogonal_residual", "davidson",
        "packet_davidson", "matched_selected_ci", "acase_determinant",
        "acase_word",
    }
    if not required_methods.issubset(method_ids):
        problems.append("the mandatory matched-method matrix is incomplete")
    stage_ids = {entry.get("id") for entry in document.get("execution_stages") or []}
    if not {"preconditioned_expansion", "matched_acase",
            "large_sector_packet_accuracy", "finite_shot"}.issubset(stage_ids):
        problems.append("the required evidence execution stages are incomplete")
    for method in document.get("methods") or []:
        if method.get("execution_stage") not in stage_ids:
            problems.append(f"{method.get('id')}: execution_stage is unknown")

    finite = document.get("finite_shot_tier") or {}
    if finite.get("implemented_by_this_driver") is not False:
        problems.append("finite-shot evidence must remain separate from this driver")
    if len(finite.get("seeds") or []) < 20:
        problems.append("finite-shot tier requires at least 20 seeds")
    unknown_finite = set(finite.get("systems") or []) - set(ids)
    if unknown_finite:
        problems.append(f"finite-shot systems are unknown: {sorted(unknown_finite)}")

    boundaries = " ".join(document.get("claim_boundaries") or []).lower()
    if "no quantum-advantage" not in boundaries:
        problems.append("the exact tier must explicitly reject a quantum-advantage claim")

    if problems:
        raise ValueError("invalid PRD-CASE paper-suite manifest:\n  "
                         + "\n  ".join(problems))


def build_tasks(document: dict, *, system_ids=(), include_exploratory=False):
    selected = set(system_ids)
    global_m = document["exact_simulation"]["m_budgets"]
    tasks = []
    for system in document["systems"]:
        enabled = bool(system["default_run"])
        if include_exploratory and system["analysis_role"] == "exploratory_scaling":
            enabled = True
        if not enabled or (selected and system["id"] not in selected):
            continue
        for M in system.get("m_budgets", global_m):
            tasks.append({
                "task_id": f"{system['id']}__M{M}",
                "system": system["id"],
                "M": int(M),
                "analysis_role": system["analysis_role"],
                "builder": system["builder"],
                "expected": system.get("expected", {}),
                "validation_target": system.get("validation_target"),
                "energy_tolerance": system.get("energy_tolerance"),
            })
    return tasks


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _new_checkpoint(manifest: dict, tasks: list[dict]) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_sha256": config_sha256(manifest),
        "created_utc_unix": time.time(),
        "tasks": {task["task_id"]: {"status": "pending"} for task in tasks},
    }


def load_checkpoint(path: Path, manifest: dict, tasks: list[dict]) -> dict:
    expected_hash = config_sha256(manifest)
    if not path.exists():
        return _new_checkpoint(manifest, tasks)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible")
    if checkpoint.get("config_sha256") != expected_hash:
        raise ValueError("checkpoint belongs to a different manifest; refusing resume")
    live = {task["task_id"] for task in tasks}
    recorded = checkpoint.setdefault("tasks", {})
    for task_id in live:
        recorded.setdefault(task_id, {"status": "pending"})
    return checkpoint


def _h2o_geometry(scale: float):
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", tuple(scale * value for value in (0.2774, 0.8929, 0.2544))),
        ("H", tuple(scale * value for value in (0.6068, -0.2383, -0.7169))),
    ]


def _build_system_uncached(system_id: str, builder: dict):
    kind = builder["kind"]
    if kind == "hubbard":
        from clifford_qc.models.lattice import hubbard
        shape = tuple(int(value) for value in builder["shape"])
        model = hubbard(shape, t=float(builder["t"]), U=float(builder["U"]))
        return model, {"source": "native_hubbard", **builder}
    if kind == "h4_chain":
        from benchmarks import run_phase10_hybrid as phase10
        from clifford_qc.models.chemistry import h4_chain
        phase10._pyscf_memory_probe_fallback()
        spacing = float(builder["spacing_angstrom"])
        return h4_chain(spacing=spacing), {
            "source": "pyscf_sto3g", **builder,
        }
    if kind == "beh2_frozen_core":
        from benchmarks import run_phase10_hybrid as phase10
        from clifford_qc.models.chemistry import beh2_frozen_core
        phase10._pyscf_memory_probe_fallback()
        bond = float(builder["bond_length_angstrom"])
        return beh2_frozen_core(bond_length=bond), {
            "source": "pyscf_sto3g", **builder,
        }
    if kind == "h2o_qsci_scaled":
        from benchmarks import run_phase10_hybrid as phase10
        from clifford_qc.models.chemistry import molecule_model
        phase10._pyscf_memory_probe_fallback()
        scale = float(builder["bond_scale"])
        model = molecule_model(
            _h2o_geometry(scale), name=f"h2o_qsci_bond_scale_{scale:g}",
            occupied_indices=[0, 1], active_indices=[2, 3, 4, 5, 6])
        return model, {"source": "pyscf_sto3g", **builder,
                       "geometry_angstrom": _h2o_geometry(scale)}
    if kind == "phase10_registry":
        from benchmarks import run_phase10_hybrid as phase10
        return phase10.build_system(builder["source_key"])
    raise ValueError(f"{system_id}: unknown builder kind {kind!r}")


def build_system_from_spec(system_id: str, builder: dict):
    key = (system_id, json.dumps(builder, sort_keys=True, separators=(",", ":")))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_system_uncached(system_id, builder)
    return _MODEL_CACHE[key]


@contextmanager
def _registered_for_one_run(pe, system_id: str, builder: dict,
                            absolute_word_abort: int):
    original_builder = pe.phase10.build_system
    original_systems = pe.PRIMARY_SYSTEMS
    original_pricer = pe.price_packet_basis

    def guarded_pricer(*args, abort_above=None, **kwargs):
        ceiling = int(absolute_word_abort)
        effective = ceiling if abort_above is None else min(ceiling, int(abort_above))
        return original_pricer(*args, abort_above=effective, **kwargs)

    pe.phase10.build_system = lambda name: build_system_from_spec(name, builder)
    pe.PRIMARY_SYSTEMS = (system_id,)
    pe.price_packet_basis = guarded_pricer
    try:
        yield
    finally:
        pe.phase10.build_system = original_builder
        pe.PRIMARY_SYSTEMS = original_systems
        pe.price_packet_basis = original_pricer


def execute_task(task: dict, manifest: dict) -> dict:
    from benchmarks import run_preconditioned_expansion as pe
    from clifford_qc.reproducibility import stamp_record

    exact = manifest["exact_simulation"]
    if int(exact["grouping_word_limit"]) != int(pe.DEFAULT_GROUPING_WORD_LIMIT):
        raise ValueError("manifest grouping_word_limit differs from the driver; "
                         "refusing a silently ignored resource setting")
    with _registered_for_one_run(
            pe, task["system"], task["builder"],
            absolute_word_abort=int(exact["absolute_packet_word_abort"])):
        record = pe.run_system(
            task["system"], seed=int(exact["seed"]),
            total_directions=int(task["M"]),
            shift_grid=tuple(float(value) for value in exact["shift_grid"]),
            packet_K=tuple(int(value) for value in exact["packet_k"]),
            word_budget_ratio=float(exact["word_budget_ratio"]),
            reference_blocks=tuple(int(value) for value in exact["reference_blocks"]),
            regression_tolerance=float(exact["regression_energy_tolerance"]),
            span_tolerance=float(exact["regression_subspace_tolerance"]),
        )
    expected = task.get("expected") or {}
    mismatches = {}
    for key in ("n_qubits", "sector_dimension"):
        if key in expected and int(record[key]) != int(expected[key]):
            mismatches[key] = {"expected": expected[key], "observed": record[key]}
    if mismatches:
        raise AssertionError(f"{task['task_id']}: construction mismatch {mismatches}")
    produced = [method["id"] for method in manifest["methods"]
                if method["execution_stage"] == "preconditioned_expansion"]
    pending = [method["id"] for method in manifest["methods"]
               if method["execution_stage"] != "preconditioned_expansion"]
    return stamp_record({
        "schema": "clifford_qc.prd_case_suite_task.v1",
        "config_sha256": config_sha256(manifest),
        "task": task,
        "execution_stage": "preconditioned_expansion",
        "produced_methods": produced,
        "pending_methods": pending,
        "paper_analysis_complete": False,
        "absolute_packet_word_abort": int(exact["absolute_packet_word_abort"]),
        "claim_boundaries": manifest["claim_boundaries"],
        "record": record,
    })


def cross_check_validation_records(tasks: list[dict], results_dir: Path) -> dict:
    checks = []
    by_key = {(task["system"], task["M"]): task for task in tasks}
    for task in tasks:
        target = task.get("validation_target")
        if not target:
            continue
        source_path = results_dir / f"{task['task_id']}.json"
        target_task = by_key.get((target, task["M"]))
        if target_task is None:
            checks.append({"validation": task["system"], "target": target,
                           "M": task["M"], "status": "target_not_scheduled"})
            continue
        target_path = results_dir / f"{target_task['task_id']}.json"
        if not source_path.exists() or not target_path.exists():
            checks.append({"validation": task["system"], "target": target,
                           "M": task["M"], "status": "record_missing"})
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        expected = json.loads(target_path.read_text(encoding="utf-8"))
        delta = float(source["record"]["exact_energy"]
                      - expected["record"]["exact_energy"])
        tolerance = float(task["energy_tolerance"])
        status = "passed" if abs(delta) <= tolerance else "failed"
        checks.append({"validation": task["system"], "target": target,
                       "M": task["M"], "energy_delta": delta,
                       "tolerance": tolerance, "status": status})
        if status == "failed":
            raise AssertionError(
                f"{task['system']} and {target} disagree by {delta} Ha")
    return {"schema": "clifford_qc.prd_case_suite_cross_checks.v1",
            "checks": checks}


def _parse_systems(value: str):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--systems", default="",
                        help="comma-separated manifest ids; empty means defaults")
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="run tasks; omitted means dry-run only")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    system_ids = _parse_systems(args.systems)
    known = {entry["id"] for entry in manifest["systems"]}
    unknown = sorted(set(system_ids) - known)
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    tasks = build_tasks(manifest, system_ids=system_ids,
                        include_exploratory=args.include_exploratory)
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise SystemExit("--max-tasks must be positive")
        tasks = tasks[:args.max_tasks]

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "config_sha256": config_sha256(manifest),
        "task_count": len(tasks),
        "physical_instance_count": manifest["physical_instance_count"],
        "validation_instance_count": manifest["validation_instance_count"],
        "implemented_stage": "preconditioned_expansion",
        "paper_analysis_complete": False,
        "pending_stages": ["matched_acase", "large_sector_packet_accuracy",
                           "finite_shot"],
        "tasks": [{"task_id": task["task_id"], "role": task["analysis_role"]}
                  for task in tasks],
    }, indent=2))
    if not args.execute:
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, manifest, tasks)
    _atomic_json(checkpoint_path, checkpoint)
    failures = 0
    for task in tasks:
        state = checkpoint["tasks"][task["task_id"]]
        if state.get("status") == "completed":
            continue
        state.update({"status": "running", "started_utc_unix": time.time()})
        _atomic_json(checkpoint_path, checkpoint)
        output = args.results_dir / f"{task['task_id']}.json"
        try:
            record = execute_task(task, manifest)
            _atomic_json(output, record)
            state.update({"status": "completed", "output": output.name,
                          "finished_utc_unix": time.time()})
        except Exception as error:  # checkpoint the scientific failure verbatim
            failures += 1
            state.update({"status": "failed", "error": repr(error),
                          "finished_utc_unix": time.time()})
            _atomic_json(checkpoint_path, checkpoint)
            if args.fail_fast:
                raise
        _atomic_json(checkpoint_path, checkpoint)
    if failures:
        raise SystemExit(f"{failures} task(s) failed; see {checkpoint_path}")
    _atomic_json(args.results_dir / "cross_checks.json",
                 cross_check_validation_records(tasks, args.results_dir))


if __name__ == "__main__":
    main()
