"""Cross-layer orchestration workflows.

This module owns bridges between algorithms and projected-subspace kernels so
neither numerical layer depends on the other's workflow implementation.
"""

from __future__ import annotations

from .backends.exact_mv import ExactMVBackend


def adapt_warm_start(model, pool, *, max_operators: int = 4, **kwargs):
    """Run exact ADAPT-VQE and return ``(rho, AdaptResult)`` for A-CASE.

    A-CASE treats the optimized ADAPT state as its single reference, so the
    subspace grows around a correlated state rather than a bare determinant.
    """
    # Imported at call time so importing either numerical package does not
    # eagerly initialize the other through this orchestration bridge.
    from .algorithms.adapt import ansatz_program, run_adapt

    result = run_adapt(model, pool, max_operators=max_operators, **kwargs)
    by_label = {op.label: op for op in pool}
    # Preserve the selected ansatz order; filtering the pool by membership
    # would silently reorder noncommuting rotors.
    chosen = [by_label[label] for label in result.labels]
    program = ansatz_program(model, chosen)
    return ExactMVBackend().state(program, result.parameters), result

__all__ = ["WorkflowResult", "adapt_warm_start", "export_workflow_evidence",
           "run_workflow", "verify_workflow_evidence"]

# The unified interface below orchestrates the existing providers; it does not
# alter their numerical methods or provide an additional certification theorem.


def _workflow_json(value):
    """Convert numerical resource metadata to strict, portable JSON values."""
    import math
    import numpy as np

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        # Preserve the fact that a diagnostic is nonfinite, without emitting
        # invalid JSON or silently converting it to a measured zero.
        if not math.isfinite(number):
            return {"nonfinite": str(number)}
        return number
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": _workflow_json(value.real), "imag": _workflow_json(value.imag)}
    if isinstance(value, np.ndarray):
        return _workflow_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_workflow_json(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("workflow metadata keys must be strings")
        return {k: _workflow_json(v) for k, v in value.items()}
    raise TypeError(f"unsupported workflow metadata: {type(value).__name__}")


def _workflow_bytes(value):
    import json
    return (json.dumps(_workflow_json(value), sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


class WorkflowResult:
    """Immutable serialized result; ``as_dict`` returns a fresh copy."""
    __slots__ = ("_payload",)

    def __init__(self, record):
        self._payload = _workflow_bytes(record)

    def as_dict(self):
        import json
        return json.loads(self._payload)


def _workflow_stage(status="not_measured", *, seconds=None, scope):
    return {"status": status, "wall_seconds": seconds, "scope": scope}


def _caller_stage(report, label):
    import math
    if report is None:
        return _workflow_stage(scope=f"{label} outside wrapper; not independently measured")
    if not isinstance(report, dict) or set(report) != {"wall_seconds", "scope"}:
        raise ValueError(f"{label} cost must contain wall_seconds and scope")
    seconds, scope = report["wall_seconds"], report["scope"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{label} wall_seconds must be finite and nonnegative")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"{label} cost scope must be nonempty")
    return _workflow_stage("caller_reported", seconds=float(seconds), scope=scope)


def _workflow_source_digest():
    """Fingerprint installed package source bytes, not the full experiment."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _validate_projected_solve_options(options, *, max_modes):
    """Reject invalid, data-independent options before any sampling occurs.

    Per-mode lengths are bounded by the generator count; their exact required
    length depends on the reconstructed overlap and is checked after sampling.
    """
    import numpy as np

    for name, value in options.items():
        vector_allowed = name in {"overlap_noise_floor", "overlap_ridge"}
        array = np.asarray(value)
        if (array.dtype.kind not in "iuf" or array.ndim > int(vector_allowed)):
            shape = "a real scalar or 1-D sequence" if vector_allowed else "a real scalar"
            raise ValueError(f"{name} must be {shape}")
        if array.ndim == 1 and not 0 < array.size <= max_modes:
            raise ValueError(f"{name} must have between 1 and {max_modes} entries")
        if name == "max_condition":
            # +inf disables the condition-number cutoff, as in solve_projected.
            if np.isnan(array).any() or np.any(array <= 0):
                raise ValueError("max_condition must be positive and not NaN")
        elif not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{name} must be nonnegative and finite")


def run_workflow(provider, *, path, problem, unit, solve_options=None,
                 shots=None, seed=None, delta=0.05, sector_config=None,
                 preparation_cost=None, validation_cost=None, provenance=None):
    """Run one existing provider under a common result and cost contract.

    ``provider`` is already constructed: upstream model/reference/provider
    preparation is outside the measured timings. ``problem`` and ``unit`` are
    caller-supplied labels, not inferred physical metadata. For sampled runs,
    ``shots`` is the number *per setting* (or a count vector), with a fixed seed.
    This returns entrywise S/H confidence radii, not an energy confidence interval.
    Sector action solving needs SciPy. Unsupported path/provider combinations are
    rejected before any sampling or solve.
    """
    import platform
    import sys
    import time
    import numpy as np
    from .backends.sector_reference import SectorActionProvider
    from .subspace.providers import ExactProjectedProvider
    from .subspace.sampled import SampledProjectedProvider
    from .subspace.sector_solver import SectorSolveConfig, solve_sector
    from .sampled_reference import shot_vector, nonnegative_integer

    providers = {"exact_projection": ExactProjectedProvider,
                 "sampled_projection": SampledProjectedProvider,
                 "sector_action": SectorActionProvider}
    if path not in providers:
        raise ValueError(f"unsupported workflow path: {path!r}")
    if not isinstance(provider, providers[path]):
        raise TypeError(f"{path} requires {providers[path].__name__}")
    if not isinstance(problem, str) or not problem.strip() or not isinstance(unit, str) or not unit.strip():
        raise ValueError("problem and energy unit must be nonempty strings")
    options = {} if solve_options is None else dict(solve_options)
    allowed_options = {"tau_s", "rel_tau", "max_condition", "overlap_noise_floor",
                       "overlap_ridge", "norm_floor"}
    if set(options) - allowed_options:
        raise ValueError("unsupported projected solve options: " + ", ".join(sorted(set(options) - allowed_options)))
    if path != "sampled_projection" and (shots is not None or seed is not None):
        raise ValueError("shots and seed apply only to sampled projection")
    if path != "sector_action" and sector_config is not None:
        raise ValueError("sector_config applies only to sector actions")
    if path == "sector_action" and options:
        raise ValueError("sector actions use sector_config, not solve_options")
    if path != "sector_action":
        _validate_projected_solve_options(options, max_modes=len(provider.bank))
    if sector_config is not None and not isinstance(sector_config, SectorSolveConfig):
        raise TypeError("sector_config must be a SectorSolveConfig")
    if path == "sampled_projection":
        if shots is None or seed is None:
            raise ValueError("sampled projection requires shots and seed")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not 0 < delta < 1:
            raise ValueError("delta must lie in (0,1)")
        nonnegative_integer(seed, "seed")
        counts = shot_vector(provider.plan.settings, shots)
        if any(count < 1 for count in counts):
            raise ValueError("sampled projection requires shots for every setting")
    # Validate caller metadata before running costly operations.
    caller = _workflow_json({"preparation_cost": preparation_cost,
                             "validation_cost": validation_cost, "provenance": provenance})
    stages = {
        "preparation": _caller_stage(preparation_cost, "preparation"),
        "sampling": _workflow_stage("not_applicable" if path != "sampled_projection" else "not_measured",
                                    scope="reference sampling inside wrapper"),
        "solve": _workflow_stage(scope="provider solve and reconstruction inside wrapper"),
        "validation": _caller_stage(validation_cost, "validation"),
        "export": _workflow_stage(scope="outside run; archive publication is not included"),
    }
    start = time.perf_counter()
    uncertainty = {"status": "not_available", "energy_interval": None}
    details = {}
    failed = False
    if path == "exact_projection":
        solved = provider.solve(**options)
        status = "projected_solve_completed"
        residual = {"status": "not_computed", "value": None}
        rank, condition = solved.effective_rank, solved.condition_number
        resources, energies = solved.resources, solved.energies
        assumptions = ["Exact reference expectations within the declared projected span.",
                       "A projected solve does not establish the full-space ground state."]
        tier = "exact_numerical_simulation"
    elif path == "sampled_projection":
        sampling_start = time.perf_counter()
        observations = provider.reference.sample(provider.plan.settings, counts, seed=seed)
        stages["sampling"] = _workflow_stage("measured", seconds=time.perf_counter()-sampling_start,
                                             scope="sampling call only; excludes prior provider compilation")
        solve_start = time.perf_counter()
        residual = {"status": "not_computed", "value": None}
        try:
            measured = provider.reconstruct(observations, delta=delta)
            uncertainty = {"status": "entrywise_only", "energy_interval": None,
                           "entry_radii": measured.entry_radii, "confidence_delta": delta,
                           "evidence": measured.evidence,
                           "scope": "simultaneous entrywise S/H radii under the provider's iid bounded-shot assumptions; not a Ritz-energy interval"}
            solved = measured.problem.solve(**options)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Preserve already-spent shots and histograms if numerical assembly
            # or the projected solve fails; no energy is promoted.
            failed = True
            status = "sampled_projection_failed"
            rank, condition, energies = None, None, ()
            resources = provider.bank.resources()
            error = {"type": type(exc).__name__, "message": str(exc)}
        else:
            status = "sampled_projected_solve_completed"
            rank, condition = solved.effective_rank, solved.condition_number
            resources, energies = solved.resources, solved.energies
        details = {"settings_identity": observations.settings_identity,
                   "counts": observations.counts,
                   "groups": [{"setting_key": g.setting_key, "support": g.support,
                               "shots": g.shots, "histogram": dict(g.hist),
                               "word_codes": g.word_codes,
                               "readouts": [[code, sign, positions] for code, (sign, positions)
                                            in sorted(g.readouts.items())]}
                              for g in observations.batch.groups],
                   "scope": "observed histograms and signed readouts; circuits and preparation must be supplied separately for replay"}
        if failed:
            details["error"] = error
        assumptions = ["Sampling assumptions are required and are not verified by metadata.",
                       "The sampled Ritz energy is not a ground-state certificate or confidence interval."]
        tier = "finite_sample_empirical"
    else:
        solved = solve_sector(provider, config=sector_config)
        status = solved.status
        energies, resources = solved.energies, solved.resources
        rank, condition = None, None
        residual = {"status": "reported" if solved.residuals else "unavailable",
                    "roots": [v.as_dict() for v in solved.residuals]}
        details = {"attempts": solved.attempts, "spin": solved.spin}
        assumptions = ["Explicit classical sector actions and declared spin sector.",
                       "Numerical convergence does not identify a ground root."]
        tier = "exact_numerical_simulation"
    elapsed = time.perf_counter() - (solve_start if path == "sampled_projection" else start)
    stages["solve"] = _workflow_stage("failed" if failed else "measured", seconds=elapsed,
                                      scope="reconstruction and projected solve" if path == "sampled_projection"
                                      else "provider solve; excludes prior provider construction")
    measurement = ({"shots": observations.batch.hardware_shots,
                    "circuits": observations.batch.circuits,
                    "observation_identity": observations.identity,
                    "seed": int(seed), "shots_scope": "this fixed-endpoint batch only"}
                   if path == "sampled_projection" else
                   {"shots": None, "circuits": None, "shots_scope": "not applicable to this exact classical path"})
    record = {
        "schema": "clifford_qc.workflow_result.v1", "path": path,
        "problem": {"label": problem, "energy_unit": unit,
                    "reference_identity": provider.reference_identity,
                    "hamiltonian_identity": provider.hamiltonian_identity,
                    "sector_identity": getattr(provider, "sector_identity", None)},
        "outcome": {"status": status, "energies": energies,
                    "retained_rank": rank, "condition_number": condition,
                    "residual": residual, "uncertainty": uncertainty,
                    "ground_state_status": "not_established", "method_details": details},
        "request": {"solve_options": options,
                    "sampling_delta": delta if path == "sampled_projection" else None},
        "evidence_schema": "clifford_qc.evidence.v1",
        "evidence": {"energies": {"tier": tier, "scope": "declared workflow and inputs; " + assumptions[0]}},
        "assumptions": assumptions,
        "cost_ledger": {"schema": "clifford_qc.workflow_cost.v1", "stages": stages,
                        "measurement": measurement, "provider_resources": resources,
                        "provider_resources_scope": "provider-defined resource snapshot; counters can include prior use unless the provider explicitly reports per-solve deltas",
                        "process_peak_rss_bytes": None,
                        "logical_gate_counts": None, "logical_depth": None,
                        "device_runtime_seconds": None,
                        "preparation_caller_report": caller["preparation_cost"],
                        "validation_caller_report": caller["validation_cost"],
                        "end_to_end_total": None,
                        "total_scope": "unavailable: preparation, external validation, export and process RSS are not measured by this wrapper"},
        "provenance": {"caller_supplied": caller["provenance"], "python": sys.version,
                       "numpy": np.__version__, "platform": platform.platform(),
                       "package_source_sha256": _workflow_source_digest(),
                       "scope": "installed package Python bytes at result creation plus runtime and caller metadata; not a full dependency/input closure or independent source verification"},
    }
    from .evidence import validate_evidence
    validate_evidence(record)
    return WorkflowResult(record)


def export_workflow_evidence(result, destination, *, attachments=None, max_bytes=64 * 2**20):
    """Publish a portable, checksummed ZIP at a caller-controlled durable path.

    The caller must select storage with the desired retention and access policy.
    Existing files are never overwritten. Attachments are explicit name/path pairs;
    no source completeness, independent checking, or authenticity is implied.
    """
    import hashlib
    import os
    from pathlib import Path
    import re
    import tempfile
    import zipfile

    if not isinstance(result, WorkflowResult):
        raise TypeError("result must be a WorkflowResult")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    destination = Path(destination)
    payloads = {"result.json": result._payload}
    remaining = max_bytes - len(result._payload)
    if remaining < 0:
        raise ValueError("export exceeds byte budget")
    for name, source in (attachments or {}).items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError("attachment names must be simple safe filenames")
        path = Path(source)
        if not path.is_file() or path.is_symlink():
            raise ValueError("attachments must be regular non-symlink files")
        if path.stat().st_size > remaining:
            raise ValueError("attachment exceeds remaining export byte budget")
        # A file may grow after stat; never read more than one byte beyond
        # the remaining budget, and reject that byte before retaining it.
        with path.open("rb") as stream:
            data = stream.read(remaining + 1)
        if len(data) > remaining:
            raise ValueError("attachment exceeds remaining export byte budget")
        payloads["attachments/" + name] = data
        remaining -= len(data)
    manifest = {"schema": "clifford_qc.workflow_archive.v1",
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                          for name, data in sorted(payloads.items())},
                "scope": "byte integrity only; caller controls retention, provenance and authenticity"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".workflow-", suffix=".zip", dir=destination.parent)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in sorted({**payloads, "manifest.json": _workflow_bytes(manifest)}.items()):
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
            stream.flush()
            os.fsync(stream.fileno())
        verify_workflow_evidence(temp, max_bytes=max_bytes)
        os.link(temp, destination)  # atomic no-overwrite publication on same filesystem
    finally:
        os.unlink(temp)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "size_bytes": destination.stat().st_size}


def verify_workflow_evidence(path, *, max_bytes=64 * 2**20):
    """Verify archive membership and digests without extracting its contents."""
    import hashlib
    import json
    import math
    import zipfile

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    def finite(value):
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or "manifest.json" not in names or "result.json" not in names:
            raise ValueError("invalid archive membership")
        if sum(entry.file_size for entry in entries if entry.filename != "manifest.json") > max_bytes or archive.getinfo("manifest.json").file_size > 1024 * 1024:
            raise ValueError("archive exceeds byte budget")
        manifest = json.loads(archive.read("manifest.json"), object_pairs_hook=unique,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        finite(manifest)
        if manifest.get("schema") != "clifford_qc.workflow_archive.v1" or not isinstance(manifest.get("files"), dict):
            raise ValueError("invalid archive manifest")
        if set(names) != {"manifest.json", *manifest["files"]}:
            raise ValueError("archive membership differs from manifest")
        if "result.json" not in manifest["files"]:
            raise ValueError("result missing from manifest")
        for name, ref in manifest["files"].items():
            if name != "result.json" and (not name.startswith("attachments/") or "/" in name[12:] or "\\" in name or name[12:] in ("", ".", "..")):
                raise ValueError("invalid archive member name")
            data = archive.read(name)
            if ref != {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}:
                raise ValueError(f"archive digest or size mismatch: {name}")
        result = json.loads(archive.read("result.json"), object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        finite(result)
        if result.get("schema") != "clifford_qc.workflow_result.v1":
            raise ValueError("invalid workflow result schema")
    return manifest
