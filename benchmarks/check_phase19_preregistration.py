"""Validate Phase 19's outcome-free, fixed-reference energy declaration.

Checks immutable inputs, the complete protocol, rebuilt Hamiltonian/cover/
rotation/readout/resource manifests and (when a record exists) strict commit
order. It performs no measurement, state expectation or device-cost search.

    python benchmarks/check_phase19_preregistration.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - script execution
    sys.path.insert(0, str(ROOT))

from benchmarks import phase19_structure as structure

CONFIG = ROOT / structure.CONFIG_PATH
EXPECTED_CONFIG_SHA256 = "01e8b91fff879ed228c7264cabec3d7e49adf6e9e6da200a4132872b5ef0a9c7"
RESULT_KEYS = frozenset({
    "result", "results", "winner", "verdict", "decision", "sample", "samples",
    "hist", "histogram", "observed_radius", "empirical_variance", "predicted_variance",
    "setting_variances", "estimate", "estimate_error", "exact_energy",
    "reference_energy", "energy_error", "covariance_ratio", "certified_endpoint",
    "certified_total_shots", "cost_us", "runtime_us", "C_time_epsilon_us",
})


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _walk(value, path="root", *, result_free=True):
    if isinstance(value, dict):
        for key, child in value.items():
            if result_free and key in RESULT_KEYS:
                raise ValueError(f"result field in preregistration: {path}.{key}")
            _walk(child, f"{path}.{key}", result_free=result_free)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _walk(child, f"{path}[{i}]", result_free=result_free)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite number at {path}")


def read_json(path: Path, *, result_free=True):
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_reject_constant)
    _walk(data, result_free=result_free)
    return data


def load_config(path: Path = CONFIG) -> dict:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("preregistration must be an object")
    if structure.canonical_digest(config) != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen Phase 19 configuration drifted (keys, types or values)")
    return config


def _compare(expected, actual, path="bank") -> list[str]:
    """Closed recursive comparison: only float leaves receive tolerances."""
    if type(expected) is not type(actual):
        return [f"{path}: type differs"]
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [f"{path}: key set differs"]
        return [p for key in expected for p in _compare(expected[key], actual[key], f"{path}.{key}")]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length differs"]
        return [p for i, (a, b) in enumerate(zip(expected, actual))
                for p in _compare(a, b, f"{path}[{i}]")]
    if isinstance(expected, float):
        if math.isfinite(actual) and math.isclose(expected, actual,
                rel_tol=structure.FLOAT_TOLERANCE["relative"],
                abs_tol=structure.FLOAT_TOLERANCE["absolute"]):
            return []
    elif expected == actual:
        return []
    return [f"{path}: value differs"]


def _binding_problems(binding: dict, root: Path) -> list[str]:
    relative = binding["path"]
    path = root / relative
    if not path.is_file():
        return [f"missing frozen file: {relative}"]
    actual = structure.file_binding(relative, root)
    return [] if actual == binding else [f"frozen file digest drifted: {relative}"]


def historical_config_seeds(root: Path = ROOT) -> set[int]:
    """Read seed namespaces from older configs, without opening result records."""
    seeds = set()

    def collect(value):
        if isinstance(value, int) and not isinstance(value, bool):
            seeds.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "seed" in key.lower():
                    collect(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in (root / "benchmarks/configs").rglob("*.json"):
        if path == root / structure.CONFIG_PATH or "phase19_banks" in path.parts:
            continue
        walk(read_json(path, result_free=False))
    return seeds


def static_problems(config: dict, root: Path = ROOT) -> list[str]:
    """Check the frozen design and rebuild only structural bank quantities."""
    try:
        _walk(config)
        if structure.canonical_digest(config) != EXPECTED_CONFIG_SHA256:
            return ["frozen Phase 19 configuration drifted (keys, types or values)"]
    except ValueError as exc:
        return [str(exc)]
    problems = []
    roots = {config["protocol"]["seeds"][name] for name in ("headline", "covariance_audit")}
    if roots & historical_config_seeds(root):
        problems.append("Phase 19 seed namespace aliases an older config")
    for row in config["lineage"]["files"] + config["bank_manifests"]:
        # Bank descriptors include an id beside their binding.
        binding = {key: row[key] for key in ("path", "sha256", "git_blob_sha1")}
        problems.extend(_binding_problems(binding, root))
    if problems:
        return problems
    for row in config["bank_manifests"]:
        declared = read_json(root / row["path"])
        rebuilt = structure.build_bank(row["id"], root)
        problems.extend(_compare(declared, rebuilt, row["id"]))
        # All scores/schedules are structural; no random seed is consumed here.
        for arm in declared["arms"]:
            scores = [setting["max_abs_value"] for setting in arm["settings"]]
            for budget in config["protocol"]["effective_shot_endpoints"]:
                shots = structure.coefficient_range_schedule(scores, budget)
                if sum(shots) != budget or any(n < 2 for n in shots):
                    problems.append(f"{row['id']}/{arm['name']}: invalid fixed schedule")
    return problems


def temporal_problems(config: dict, root: Path = ROOT) -> list[str]:
    """A future record must name an execution strictly between declaration and record.

    Missing/shallow history is a failure when a record exists, not a silent skip.
    The declaration gate remains usable before and after the future execution.
    """
    record_path = root / structure.RECORD_PATH
    if not record_path.is_file():
        return []

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()

    try:
        if git("rev-parse", "--is-shallow-repository") != "false":
            return ["record chronology requires full git history"]
        record = read_json(record_path, result_free=False)
        execution = record.get("provenance", {}).get("git_commit")
        declaration_merge = record.get("provenance", {}).get("preregistration_merge_commit")
        for name, sha in (("execution", execution), ("preregistration merge", declaration_merge)):
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
                return [f"record {name} must be a full commit SHA"]
        if len(git("rev-list", "--parents", "-n", "1", declaration_merge).split()) < 3:
            return ["preregistration merge must be a merge commit"]
        if record.get("declaration_sha256") != EXPECTED_CONFIG_SHA256:
            return ["record does not bind this declaration"]
        declaration = git("log", "-1", "--format=%H", "HEAD", "--", structure.CONFIG_PATH)
        records = git("log", "--reverse", "--format=%H", "HEAD", "--", structure.RECORD_PATH).splitlines()
        if not declaration or not records:
            return ["declaration/record commit history is missing"]
        first_record = records[0]
        for a, b in ((declaration, declaration_merge), (declaration_merge, execution),
                     (execution, first_record)):
            if a == b:
                return ["declaration, execution and first record must be strictly ordered"]
            git("merge-base", "--is-ancestor", a, b)
        for name, sha in (("merge", declaration_merge), ("execution", execution)):
            frozen = json.loads(git("show", f"{sha}:{structure.CONFIG_PATH}"),
                                object_pairs_hook=_pairs, parse_constant=_reject_constant)
            if structure.canonical_digest(frozen) != EXPECTED_CONFIG_SHA256:
                return [f"{name} commit did not contain the frozen declaration"]
    except (OSError, ValueError, subprocess.CalledProcessError, AttributeError) as exc:
        return [f"record chronology failed: {exc}"]
    return []


def main() -> int:
    try:
        config = load_config()
        problems = static_problems(config) + temporal_problems(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems = [str(exc)]
    if problems:
        print("Phase 19 preregistration failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("OK: frozen Phase 19 inputs, covers, readouts, resources and schedules; no outcomes evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
