"""Structural and evidence-language checks for the standalone A-CASE paper.

Checks:
  1. every LaTeX environment is balanced;
  2. every \\ref / \\eqref resolves to a \\label, and no label is duplicated;
  3. every \\cite key exists in references.bib, and no bib entry is unused;
  4. every \\input target exists (i.e. make_tables.py has been run);
  5. every table row has the column count its tabular preamble declares,
     counting the generated fragments and \\multicolumn spans;
  6. no numeric table cell is typed into the manuscript instead of generated;
  7. every referenced figure asset exists and is newer than the record it is
     generated from.

Checks 5-7 are the ones that catch drift rather than typos, and they are the
reason this file is not just a phrase linter: check 5 is what caught Table III
declaring ten columns for nine-cell rows, which LaTeX renders as a silent empty
column rather than an error.

Exits nonzero on any failure so it can gate a release.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"
FIGURE_MANIFEST = HERE / "paper_assets" / "manifest.json"

# which committed record each figure is plotted from; kept in step with
# make_figures.py so a regenerated record forces a regenerated figure
FIGURE_SOURCES = {
    "validation_ladder.pdf": (
        ROOT / "benchmarks" / "reference_results" / "acase_ladder_summary.csv",
    ),
    "response_bootstrap.pdf": (HERE / "data" / "response_bootstrap.json",),
    "conditioning_bands.pdf": (
        HERE / "data" / "response_bootstrap.json",
        HERE / "data" / "response_bootstrap_illconditioned.json",
    ),
    # pipeline.pdf is a schematic: it plots no record.
}


def _cells(line: str) -> int:
    """Column count of a table row, with ``\\multicolumn{k}`` spanning k."""
    total = 0
    for cell in line.rstrip("\\").split("&"):
        span = re.search(r"\\multicolumn\{(\d+)\}", cell)
        total += int(span.group(1)) if span else 1
    return total


def _row_sources(body: str):
    """The tabular body plus the contents of any fragment it \\inputs."""
    yield re.sub(r"\\input\{[^}]+\}", "", body)
    for match in re.finditer(r"\\input\{([^}]+)\}", body):
        target = HERE / match.group(1)
        if not target.suffix:
            target = target.with_suffix(".tex")
        if target.exists():
            yield target.read_text()


def _source_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    text = TEX.read_text()
    problems: list[str] = []

    opens = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", text))
    closes = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
    for env in set(opens) | set(closes):
        if opens[env] != closes[env]:
            problems.append(
                f"unbalanced environment {env}: "
                f"{opens[env]} begin vs {closes[env]} end")

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    label_set = set(labels)
    for label, count in collections.Counter(labels).items():
        if count > 1:
            problems.append(f"duplicate label: {label}")
    for match in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", text):
        if match.group(1) not in label_set:
            problems.append(f"dangling reference: {match.group(1)}")

    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    cited: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        for key in (x.strip() for x in match.group(1).split(",")):
            if key:
                cited.add(key)
            if key and key not in keys:
                problems.append(f"missing bib key: {key}")
    # An entry in the .bib that nothing cites is usually a baseline or a method
    # that lost its citation in an edit, not a harmless leftover: bibtex drops
    # it silently and the reader sees an uncredited method.
    for key in sorted(keys - cited):
        problems.append(f"bib entry is never cited: {key}")

    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        target = HERE / match.group(1)
        if not target.exists() and not target.with_suffix(".tex").exists():
            problems.append(f"missing input: {match.group(1)} "
                            "(run paper_acase/make_tables.py)")

    for match in re.finditer(
            r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", text, re.S):
        # T is the manuscript's fixed-width, wrapping text column.
        ncol = len(re.sub(r"[^lcrT]", "", match.group(1)))
        for fragment in _row_sources(match.group(2)):
            for line in fragment.splitlines():
                line = line.strip()
                if not line.endswith(r"\\") or line.startswith("%"):
                    continue
                got = _cells(line)
                if got != ncol:
                    problems.append(
                        f"table row has {got} cells, preamble declares "
                        f"{ncol}: {line[:60]}")

    # Sec. VII states that every table value is read from a committed record.
    # A hand-typed label column is fine; a hand-typed number is the thing that
    # goes stale. Headers may legitimately carry digits, and \colrule is what
    # separates them from the body in a ruledtabular.
    for match in re.finditer(
            r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", text, re.S):
        inline = re.sub(r"\\input\{[^}]+\}", "", match.group(2))
        if r"\colrule" in inline:
            inline = inline.split(r"\colrule", 1)[1]
        for line in inline.splitlines():
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            for cell in line.rstrip("\\").split("&")[1:]:
                bare = re.sub(r"\\[A-Za-z]+|[{}$\\^_~,]", " ", cell)
                if re.search(r"\d", bare):
                    problems.append(
                        "numeric table cell is typed into the manuscript "
                        "rather than generated into tables/ "
                        f"(run paper_acase/make_tables.py): {line[:60]}")
                    break

    for match in re.finditer(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        asset = HERE / match.group(1)
        if not asset.exists():
            problems.append(f"missing figure: {match.group(1)}")
            continue
        for record in FIGURE_SOURCES.get(asset.name, ()):
            if record.exists() and asset.stat().st_mtime < record.stat().st_mtime:
                problems.append(
                    f"{asset.name} is older than {record.name} "
                    "(run paper_acase/make_figures.py)")

    if not FIGURE_MANIFEST.exists():
        problems.append("missing figure manifest "
                        "(run paper_acase/make_figures.py)")
    else:
        manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
        expected_generator = _source_digest(HERE / "make_figures.py")
        if manifest.get("generator", {}).get("sha256") != expected_generator:
            problems.append("figure manifest has a stale generator digest "
                            "(run paper_acase/make_figures.py)")
        expected_sources = {
            "pipeline.pdf": (),
            "validation_ladder.pdf": FIGURE_SOURCES["validation_ladder.pdf"],
            "response_bootstrap.pdf": FIGURE_SOURCES["response_bootstrap.pdf"],
            "conditioning_bands.pdf": FIGURE_SOURCES["conditioning_bands.pdf"],
        }
        figures = manifest.get("figures", {})
        if set(figures) != set(expected_sources):
            problems.append("figure manifest names do not match the manuscript")
        for name, paths in expected_sources.items():
            recorded = figures.get(name, {}).get("sources", {})
            expected = {
                str(path.resolve().relative_to(ROOT)): _source_digest(path)
                for path in paths
            }
            if recorded != expected:
                problems.append(
                    f"{name} manifest has stale source digests "
                    "(run paper_acase/make_figures.py)")

    required = [
        "heuristic",
        "conditional on the surviving replicas",
        "no quantum advantage",
        "not a DFT",
        "Adaptive Clifford-Algebra Subspace Eigensolver",
        "ADAPT-GCIM",
        "not a total-resource advantage claim",
        "support count certified to describe",
    ]
    lowered = text.lower()
    for phrase in required:
        if phrase.lower() not in lowered:
            problems.append(f"missing evidence/scope phrase: {phrase}")
    forbidden = ["certified response interval", "quantum speedup is",
                 "outperforms krylov"]
    for phrase in forbidden:
        if phrase in lowered:
            problems.append(f"forbidden overclaim phrase: {phrase}")

    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{len(problems)} problem(s)" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
