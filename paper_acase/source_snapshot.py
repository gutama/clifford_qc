"""Content-only provenance for the retained P2 manuscript table values.

The four inputs that later changed live in ``data/source_snapshot``. Their
original Git blob ids come from the existing table headers. Other inputs
remain at their original paths, with their bytes bound by the same manifest.
No Git history or historical Python execution is needed to check this data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_RELATIVE = "paper_acase/SOURCE_SNAPSHOT.json"
SOURCE_PATHS = frozenset({
    "benchmarks/reference_results/acase_ladder_summary.csv",
    "benchmarks/reference_results/fcidump_h4.json",
    "benchmarks/reference_results/warm_start_h4.json",
    "benchmarks/reference_results/krylov_width.json",
    "benchmarks/reference_results/matched_h4.json",
    "benchmarks/reference_results/finite_shot_optimization.json",
    "benchmarks/results/phase12_paper_b_five_system.json",
    "benchmarks/reference_results/clifford_hierarchy_h4.json",
    "benchmarks/reference_results/clifford_hierarchy_beh2.json",
    "examples/data/wannier_hubbard_dimer.json",
    "paper_acase/data/response_bootstrap.json",
    "paper_acase/data/response_bootstrap_illconditioned.json",
})
TABLE_PATHS = frozenset({
    "tables/clifford_hierarchy.tex", "tables/conditioning_results.tex",
    "tables/dimer_results.tex", "tables/finite_shot.tex",
    "tables/h4_results.tex", "tables/ladder_results.tex",
    "tables/matched_results.tex", "tables/phase12_primary.tex",
    "tables/response_results.tex", "tables/warm_start_results.tex",
})
HISTORICAL_GENERATOR = "paper_acase/data/source_snapshot/make_tables.py.txt"


def _load(root: Path) -> dict:
    path = _contained_path(root, MANIFEST_RELATIVE)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("table snapshot manifest must be an object")
    return manifest


def _relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("snapshot path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"snapshot path escapes the repository: {value}")
    return path


def _contained_path(root: Path, value: object, *, boundary: Path | None = None) -> Path:
    """Reject lexical escapes and symlinks resolving outside the source tree."""
    relative = _relative_path(value)
    path = root / relative
    resolved = path.resolve()
    if (not resolved.is_relative_to(root.resolve())
            or (boundary is not None
                and not resolved.is_relative_to(boundary.resolve()))):
        raise ValueError(f"snapshot path escapes its source directory: {value}")
    return path


def source_path(original: str, root: Path = ROOT) -> Path:
    """Resolve a declared table input without consulting the moving record."""
    if original not in SOURCE_PATHS:
        raise ValueError(f"undeclared table source: {original}")
    sources = _load(root).get("sources")
    if not isinstance(sources, list):
        raise ValueError("table snapshot sources must be a list")
    matches = [entry for entry in sources
               if isinstance(entry, dict) and entry.get("source_path") == original]
    if len(matches) != 1:
        raise ValueError(f"table snapshot must declare exactly one source: {original}")
    return _contained_path(root, matches[0].get("path"))


def table_path(name: str, root: Path = ROOT) -> Path:
    """Resolve a declared table target within the paper source directory."""
    relative = f"tables/{name}"
    if relative not in TABLE_PATHS:
        raise ValueError(f"undeclared snapshot table: {name}")
    return _contained_path(root, f"paper_acase/{relative}",
                           boundary=root / "paper_acase")


def table_body_digest(payload: bytes) -> str:
    """Hash every table byte except the two generated provenance header types."""
    body = b"\n".join(
        line for line in payload.split(b"\n")
        if not line.startswith((b"% source-git-blob-sha:",
                                b"% generator-git-blob-sha:")))
    return hashlib.sha256(body).hexdigest()


def validate_table_body(name: str, payload: bytes, root: Path = ROOT) -> None:
    """Refuse a changed scientific body before a generator writes its target."""
    table_path(name, root)
    tables = _load(root).get("tables")
    if not isinstance(tables, list):
        raise ValueError("table snapshot tables must be a list")
    matches = [item for item in tables
               if isinstance(item, dict) and item.get("path") == f"tables/{name}"]
    if len(matches) != 1:
        raise ValueError(f"table snapshot must declare exactly one body: {name}")
    if table_body_digest(payload) != matches[0].get("body_sha256"):
        raise ValueError(f"retained manuscript table body drifted: tables/{name}")


def _content_problems(root: Path, item: dict, label: str) -> list[str]:
    try:
        relative = _relative_path(item.get("path"))
        path = _contained_path(root, item.get("path"))
    except ValueError as error:
        return [str(error)]
    if not path.is_file():
        return [f"missing {label}: {relative}"]
    payload = path.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload,
                           usedforsecurity=False).hexdigest()
    problems = []
    if git_blob != item.get("git_blob_sha1"):
        problems.append(f"{label} Git blob drifted: {relative}")
    if hashlib.sha256(payload).hexdigest() != item.get("sha256"):
        problems.append(f"{label} SHA-256 drifted: {relative}")
    return problems


def snapshot_problems(root: Path = ROOT, *, check_tables: bool = True) -> list[str]:
    """Verify the declared inputs, inert recipe and retained scientific cells."""
    try:
        manifest = _load(root)
    except (OSError, ValueError) as error:
        return [f"cannot read table snapshot: {error}"]
    problems = []
    if manifest.get("schema") != "clifford_qc.paper_table_snapshot.v1":
        problems.append("table snapshot schema is not v1")
    if manifest.get("arxiv") != "2608.08739":
        problems.append("table snapshot arXiv identifier drifted")
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        problems.append("table snapshot sources must be a list")
        sources = []
    source_names = [item["source_path"] for item in sources
                    if isinstance(item, dict)
                    and isinstance(item.get("source_path"), str)]
    if (len(sources) != len(SOURCE_PATHS)
            or len(source_names) != len(set(source_names))
            or set(source_names) != SOURCE_PATHS):
        problems.append("table snapshot must declare each of its 12 sources exactly once")
    for item in sources:
        if not isinstance(item, dict):
            problems.append("table snapshot source must be an object")
            continue
        problems.extend(_content_problems(root, item, "table snapshot source"))

    generator = manifest.get("historical_generator")
    if not isinstance(generator, dict):
        problems.append("table snapshot is missing its historical generator")
    else:
        if (generator.get("path") != HISTORICAL_GENERATOR
                or generator.get("source_path") != "paper_acase/make_tables.py"
                or generator.get("execution") != "inert_provenance_only"):
            problems.append("table snapshot historical-generator boundary drifted")
        problems.extend(_content_problems(root, generator, "historical generator"))

    tables = manifest.get("tables", [])
    if not isinstance(tables, list):
        problems.append("table snapshot tables must be a list")
        tables = []
    table_names = [item["path"] for item in tables
                   if isinstance(item, dict) and isinstance(item.get("path"), str)]
    if (len(tables) != len(TABLE_PATHS)
            or len(table_names) != len(set(table_names))
            or set(table_names) != TABLE_PATHS):
        problems.append("table snapshot must declare each of its 10 table bodies exactly once")
    if check_tables:
        for item in tables:
            if not isinstance(item, dict):
                problems.append("table snapshot table must be an object")
                continue
            try:
                relative = _relative_path(item.get("path"))
                path = _contained_path(root, f"paper_acase/{relative}",
                                       boundary=root / "paper_acase")
            except ValueError as error:
                problems.append(str(error))
                continue
            if not path.is_file():
                problems.append(f"missing snapshot table: {relative}")
            elif table_body_digest(path.read_bytes()) != item.get("body_sha256"):
                problems.append(f"retained manuscript table body drifted: {relative}")
    return problems


def require_snapshot(*, check_tables: bool = True) -> None:
    problems = snapshot_problems(check_tables=check_tables)
    if problems:
        raise ValueError("table snapshot verification failed:\n" + "\n".join(problems))


def main() -> int:
    problems = snapshot_problems()
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    print("OK: 12 table inputs, inert historical generator and 10 retained table bodies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
