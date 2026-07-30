"""Structural and evidence-language checks for the standalone A-CASE paper."""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"


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
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        for key in (x.strip() for x in match.group(1).split(",")):
            if key and key not in keys:
                problems.append(f"missing bib key: {key}")

    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        target = HERE / match.group(1)
        if not target.exists() and not target.with_suffix(".tex").exists():
            problems.append(f"missing input: {match.group(1)}")

    for match in re.finditer(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        if not (HERE / match.group(1)).exists():
            problems.append(f"missing figure: {match.group(1)}")

    required = [
        "heuristic",
        "conditional on the surviving replicas",
        "no quantum advantage",
        "not a DFT",
        "Adaptive Clifford-Algebra Subspace Eigensolver",
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

