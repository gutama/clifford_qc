"""Structural and evidence-language checks for the standalone DA-CASE paper.

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
TABLE_GENERATOR = HERE / "make_tables.py"

# Re-derived from make_tables.py rather than restated, so the two cannot drift
# into disagreeing about which record backs which table.
sys.path.insert(0, str(HERE))
from make_tables import TABLE_SOURCES  # noqa: E402

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


def _brace_group(source: str, start: int) -> tuple[str, int]:
    """Return a balanced braced group and the offset just after it."""
    if start >= len(source) or source[start] != "{":
        raise ValueError("expected a braced group")
    depth = 0
    for offset in range(start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:offset], offset + 1
    raise ValueError("unterminated braced group")


def _tabulars(source: str):
    """Yield tabular preambles and bodies, including braced column widths."""
    marker = r"\begin{tabular}"
    closing = r"\end{tabular}"
    cursor = 0
    while (begin := source.find(marker, cursor)) >= 0:
        preamble_start = begin + len(marker)
        while (preamble_start < len(source)
               and source[preamble_start].isspace()):
            preamble_start += 1
        preamble, body_start = _brace_group(source, preamble_start)
        end = source.find(closing, body_start)
        if end < 0:
            raise ValueError("tabular environment has no closing marker")
        yield preamble, source[body_start:end]
        cursor = end + len(closing)


def _column_count(preamble: str) -> int:
    """Count columns while ignoring intercolumn declarations."""
    count = 0
    offset = 0
    while offset < len(preamble):
        token = preamble[offset]
        if token in "lcrX":
            count += 1
            offset += 1
        elif token in "pmb":
            count += 1
            offset += 1
            while offset < len(preamble) and preamble[offset].isspace():
                offset += 1
            _, offset = _brace_group(preamble, offset)
        elif token in "@><!":
            offset += 1
            while offset < len(preamble) and preamble[offset].isspace():
                offset += 1
            _, offset = _brace_group(preamble, offset)
        elif token == "*":
            repeats, offset = _brace_group(preamble, offset + 1)
            repeated, offset = _brace_group(preamble, offset)
            count += int(repeats) * _column_count(repeated)
        else:
            offset += 1
    return count


def _source_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_blob_sha(path: Path) -> str:
    """Return the byte-exact SHA-1 object id used by Git blob objects."""
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


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

    # Every source-bound table, not just the Phase 12 one: a record regenerated
    # without rerunning the generator leaves a table that still typesets, and
    # only the binding catches it.  Tables the manuscript does not input are
    # skipped, so trimming the paper does not strand a check on a dead file.
    for name, sources in sorted(TABLE_SOURCES.items()):
        table = HERE / "tables" / name
        if not table.exists():
            continue
        if f"tables/{name}" not in text and f"tables/{Path(name).stem}" not in text:
            continue
        expected = [f"% source-git-blob-sha: {_git_blob_sha(path)}"
                    for path in sources]
        expected.append(
            f"% generator-git-blob-sha: {_git_blob_sha(TABLE_GENERATOR)}")
        headers = table.read_text(encoding="utf-8").splitlines()[:len(expected)]
        if headers != expected:
            problems.append(
                f"{name} is stale against its result record or table "
                "generator (run paper_acase/make_tables.py)")

    if re.search(r"\\usepackage(?:\[[^]]*\])?\{array\}", text):
        problems.append(
            "array package is incompatible with REVTeX 4.2f under the "
            "arXiv TeX Live 2025 stack; use standard tabular columns")

    for preamble, body in _tabulars(text):
        ncol = _column_count(preamble)
        for fragment in _row_sources(body):
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
    for _, body in _tabulars(text):
        inline = re.sub(r"\\input\{[^}]+\}", "", body)
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

    # Figure freshness is decided by source digests rather than mtimes.  Only
    # figures referenced by the current manuscript are release dependencies:
    # a simplified manuscript must not be held hostage by stale assets it no
    # longer includes.
    referenced_figures: list[tuple[str, Path]] = []
    for match in re.finditer(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        relative = match.group(1)
        asset = HERE / relative
        referenced_figures.append((Path(relative).name, asset))
        if not asset.exists():
            problems.append(f"missing figure: {relative}")

    if referenced_figures:
        if not FIGURE_MANIFEST.exists():
            problems.append("missing figure manifest "
                            "(run paper_acase/make_figures.py)")
        else:
            manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
            expected_generator = _source_digest(HERE / "make_figures.py")
            if manifest.get("generator", {}).get("sha256") != expected_generator:
                problems.append("figure manifest has a stale generator digest "
                                "(run paper_acase/make_figures.py)")
            figures = manifest.get("figures", {})
            for name, _ in referenced_figures:
                if name not in figures:
                    problems.append(f"{name} is absent from the figure manifest")
                    continue
                paths = FIGURE_SOURCES.get(name, ())
                recorded = figures[name].get("sources", {})
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
        "Dyadic Adaptive Clifford-Algebra Subspace Eigensolver",
        "ADAPT-GCIM",
        "not a total-resource advantage claim",
        "support count certified to describe",
        "private development repository",
        "editors and referees",
        "tagged archival release",
    ]
    # Source line breaks are not semantic in LaTeX, so a phrase that happens to
    # wrap must neither satisfy a forbidden check nor fail a required one.
    lowered = re.sub(r"\s+", " ", text.lower())
    for phrase in required:
        if re.sub(r"\s+", " ", phrase.lower()) not in lowered:
            problems.append(f"missing evidence/scope phrase: {phrase}")
    if re.search(r"\\date\s*\{[^{}]*\\today[^{}]*\}", text, re.S):
        problems.append("manuscript date must be fixed for archival rebuilds")
    forbidden = ["certified response interval", "quantum speedup is",
                 "outperforms krylov", "cliffordqc2026",
                 r"in the public \texttt{clifford\_qc} repository"]
    for phrase in forbidden:
        if re.sub(r"\s+", " ", phrase.lower()) in lowered:
            problems.append(f"forbidden overclaim phrase: {phrase}")

    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{len(problems)} problem(s)" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
