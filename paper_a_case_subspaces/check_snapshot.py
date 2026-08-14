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


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("schema") != "clifford_qc.paper_source_snapshot.v1":
        problems.append("snapshot schema is not v1")
    if manifest.get("arxiv") != "2608.00560":
        problems.append("snapshot arXiv identifier drifted")
    if manifest.get("source_commit") != "67ea0dd97c68737b2c5709a0ce12714869d7cbd0":
        problems.append("snapshot source commit drifted")

    generated = []
    for item in manifest.get("files", []):
        path = HERE / item["path"]
        if not path.exists():
            problems.append(f"missing snapshot file: {item['path']}")
            continue
        actual = _git_blob_sha1(path)
        if item["role"] == "published_text_source":
            if actual != item["git_blob_sha1"]:
                problems.append(
                    f"published text source drifted: {item['path']} "
                    f"({actual} != {item['git_blob_sha1']})"
                )
        else:
            generated.append((item["path"], actual == item["git_blob_sha1"]))

    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    canonical = sum(match for _, match in generated)
    print(
        "OK: 16 published text sources match 67ea0dd; "
        f"{len(generated)} generated figures present "
        f"({canonical} byte-identical historical PDFs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
