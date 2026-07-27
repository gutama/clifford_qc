"""Structural consistency checks for the manuscript.

Guards the class of defect that produced the stale H4 numbers and the
mislabelled figure legend: values that drift out of sync because nothing
mechanically ties the .tex to the committed records. Run it before every
submission, alongside ``make_figures.py`` and ``make_tables.py``:

    python paper/make_figures.py && python paper/make_tables.py
    python paper/check_manuscript.py

Checks:
  1. every LaTeX environment is balanced;
  2. every \\ref / \\eqref resolves to a \\label, and no label is duplicated;
  3. every \\cite key exists in references.bib;
  4. every \\input target exists (i.e. make_tables.py has been run);
  5. every table row has the column count its tabular preamble declares;
  6. every referenced figure asset exists and is newer than the record it
     is generated from.

Exits nonzero on any failure so it can gate a release.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"
DATA = ROOT / "benchmarks" / "reference_results"

# which committed record each figure is plotted from; kept in step with
# make_figures.py so a regenerated record forces a regenerated figure
FIGURE_SOURCES = {
    "calibration.pdf": ("calibration.jsonl",),
    "selection_quality.pdf": ("spin_headline_n4.jsonl",),
    "grouping.pdf": ("spin_headline_n4.jsonl",),
    "allocation.pdf": ("spin_headline_n4.jsonl",),
    "chemistry.pdf": ("chemistry.jsonl",),
}


def main() -> int:
    s = TEX.read_text()
    problems: list[str] = []

    # 1. environment balance
    opens = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
    closes = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
    for k in set(opens) | set(closes):
        if opens[k] != closes[k]:
            problems.append(f"unbalanced environment {k}: "
                            f"{opens[k]} begin vs {closes[k]} end")

    # 2. labels and references
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    label_set = set(labels)
    for k, v in collections.Counter(labels).items():
        if v > 1:
            problems.append(f"duplicate label: {k}")
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", s):
        if m.group(1) not in label_set:
            problems.append(f"dangling reference: {m.group(1)}")

    # 3. bibliography keys
    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    for m in re.finditer(r"\\cite\{([^}]+)\}", s):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k and k not in keys:
                problems.append(f"missing bib key: {k}")

    # 4. generated table fragments
    for m in re.finditer(r"\\input\{([^}]+)\}", s):
        target = HERE / m.group(1)
        if not target.exists() and not target.with_suffix(".tex").exists():
            problems.append(f"missing \\input target: {m.group(1)} "
                            "(run paper/make_tables.py)")

    # 5. table column counts
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                         s, re.S):
        ncol = len(re.sub(r"[^lcr]", "", m.group(1)))
        for frag in _row_sources(m.group(2)):
            for line in frag.splitlines():
                line = line.strip()
                if not line.endswith(r"\\") or line.startswith("%"):
                    continue
                got = line.count("&") + 1
                if got != ncol:
                    problems.append(f"table row has {got} cells, preamble "
                                    f"declares {ncol}: {line[:60]}")

    # 5b. no numeric table body is typed into the manuscript
    #
    # Sec. VII states that every table value is read from a committed record.
    # Two tables were not: the H4 summary, whose rerun-agreement rows silently
    # contradicted the record shipped beside them, and the infinite-shot
    # ranking, which happened to still be right. Nothing detected either,
    # because the artifact checks look at the fragments and the fragments were
    # not where those numbers lived. A hand-typed *label* column is fine; a
    # hand-typed number is the thing that goes stale.
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                         s, re.S):
        inline = re.sub(r"\\input\{[^}]+\}", "", m.group(2))
        # Only the body. A header may legitimately carry digits -- percentile
        # superscripts such as \epsilon_E^{50} are labels, not measurements --
        # and \colrule is what separates the two in a ruledtabular.
        if r"\colrule" in inline:
            inline = inline.split(r"\colrule", 1)[1]
        for line in inline.splitlines():
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            cells = line.rstrip("\\").split("&")[1:]      # skip the row label
            for cell in cells:
                # strip LaTeX that legitimately carries digits in a header
                bare = re.sub(r"\\[A-Za-z]+|[{}$\\^_~,]", " ", cell)
                if re.search(r"\d", bare):
                    problems.append(
                        "numeric table cell is typed into the manuscript "
                        "rather than generated into tables/ "
                        f"(run paper/make_tables.py): {line[:60]}")
                    break

    # 6. figure assets exist and are not older than the records they plot
    for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", s):
        asset = HERE / m.group(1)
        if not asset.exists():
            problems.append(f"missing figure asset: {m.group(1)}")
            continue
        for record in FIGURE_SOURCES.get(asset.name, ()):
            src = DATA / record
            if src.exists() and asset.stat().st_mtime < src.stat().st_mtime:
                problems.append(f"{asset.name} is older than {record} "
                                "(run paper/make_figures.py)")

    for p in problems:
        print(f"FAIL {p}")
    print(f"{len(problems)} problem(s)" if problems else "OK")
    return 1 if problems else 0


def _row_sources(body: str):
    """The tabular body plus the contents of any fragment it \\inputs."""
    yield re.sub(r"\\input\{[^}]+\}", "", body)
    for m in re.finditer(r"\\input\{([^}]+)\}", body):
        target = HERE / m.group(1)
        if not target.suffix:
            target = target.with_suffix(".tex")
        if target.exists():
            yield target.read_text()


if __name__ == "__main__":
    sys.exit(main())
