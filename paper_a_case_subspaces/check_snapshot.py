"""Verify the restored source boundary of arXiv:2608.00560.

Text sources must remain byte-identical to commit ``67ea0dd``.  The four PDF
figures are generated outputs: their canonical historical blob ids are kept in
``SOURCE_SNAPSHOT.json``, while this checkout may regenerate them with a newer
Matplotlib/PDF metadata version.  Their scientific inputs and the manuscript's
staleness checks remain pinned by ``check_manuscript.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_SNAPSHOT.json"
EXPECTED_FILES = 20
EXPECTED_TEXT_SOURCES = 16
EXPECTED_GENERATED_FIGURES = 4
CONTROL_FILES = {"SOURCE_SNAPSHOT.json", "check_snapshot.py"}
LATEX_BUILD_FILES = {
    "manuscript.aux",
    "manuscript.bbl",
    "manuscript.blg",
    "manuscript.log",
    "manuscript.out",
    "manuscript.pdf",
}


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def snapshot_problems(
    here: Path = HERE, manifest_path: Path = MANIFEST
) -> tuple[list[str], int, list[tuple[str, bool]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("schema") != "clifford_qc.paper_source_snapshot.v1":
        problems.append("snapshot schema is not v1")
    if manifest.get("arxiv") != "2608.00560":
        problems.append("snapshot arXiv identifier drifted")
    if manifest.get("source_commit") != "67ea0dd97c68737b2c5709a0ce12714869d7cbd0":
        problems.append("snapshot source commit drifted")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        problems.append("snapshot files must be a list")
        entries = []
    if len(entries) != EXPECTED_FILES:
        problems.append(
            f"snapshot manifest has {len(entries)} entries, expected {EXPECTED_FILES}"
        )

    listed: set[str] = set()
    generated = []
    matched_text = 0
    for item in entries:
        if not isinstance(item, dict):
            problems.append("snapshot file entry is not an object")
            continue
        relative = item.get("path")
        role = item.get("role")
        expected_sha = item.get("git_blob_sha1")
        if not isinstance(relative, str) or not relative:
            problems.append("snapshot file entry has no path")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            problems.append(f"snapshot path escapes its directory: {relative}")
            continue
        if relative in listed:
            problems.append(f"duplicate snapshot path: {relative}")
            continue
        listed.add(relative)
        if role not in ("published_text_source", "generated_figure"):
            problems.append(f"unknown snapshot role for {relative}: {role!r}")
            continue
        if not isinstance(expected_sha, str) or len(expected_sha) != 40:
            problems.append(f"invalid Git blob id for {relative}")
            continue
        path = here / relative
        if not path.exists():
            problems.append(f"missing snapshot file: {relative}")
            continue
        actual = _git_blob_sha1(path)
        if role == "published_text_source":
            if actual != expected_sha:
                problems.append(
                    f"published text source drifted: {relative} "
                    f"({actual} != {expected_sha})"
                )
            else:
                matched_text += 1
        else:
            generated.append((relative, actual == expected_sha))

    if sum(item.get("role") == "published_text_source" for item in entries
           if isinstance(item, dict)) != EXPECTED_TEXT_SOURCES:
        problems.append(
            f"snapshot manifest must list {EXPECTED_TEXT_SOURCES} text sources"
        )
    if sum(item.get("role") == "generated_figure" for item in entries
           if isinstance(item, dict)) != EXPECTED_GENERATED_FIGURES:
        problems.append(
            f"snapshot manifest must list {EXPECTED_GENERATED_FIGURES} figures"
        )

    discovered = set()
    for path in here.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(here)
        if (
            relative.as_posix() in CONTROL_FILES
            or relative.name in LATEX_BUILD_FILES
            or "__pycache__" in relative.parts
            or relative.suffix == ".pyc"
        ):
            continue
        discovered.add(relative.as_posix())
    for relative in sorted(discovered - listed):
        problems.append(f"unlisted snapshot file: {relative}")
    return problems, matched_text, generated


def main() -> int:
    problems, matched_text, generated = snapshot_problems()

    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    canonical = sum(match for _, match in generated)
    print(
        f"OK: {matched_text} published text sources match 67ea0dd; "
        f"{len(generated)} generated figures present "
        f"({canonical} byte-identical historical PDFs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
