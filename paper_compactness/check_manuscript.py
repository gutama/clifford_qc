"""Structural and evidence checks for the compactness manuscript.

The paper claims that no number in it is typed by hand, and that every pattern
its prose states in words still holds in the records.  This gate enforces both:

  1.  every LaTeX environment is balanced;
  2.  every ``\\ref``/``\\eqref`` resolves to a ``\\label``, and no label repeats;
  3.  every ``\\cite`` key exists in references.bib, and no entry is uncited;
  4.  every ``\\input`` target exists and every generated fragment is registered;
  5.  every table row has the column count its preamble declares;
  6.  no numeric cell is typed into the manuscript instead of generated;
  7.  no numeral appears in the body prose outside a short allowlist of
      notation, no generated count is spelled out beside the noun it counts,
      and every ``\\pk`` macro used is defined and every one defined is used;
  8.  every generated fragment is byte-identical to what the generator produces
      now.  The generator asserts the prose's stated patterns against the
      records, so a record that stops supporting a sentence fails here too;
  9.  the start-span diagnostic binds the record it was computed from;
 10.  each figure's manifest entry matches its generator, its input digests and
      the committed asset;
 11.  the statements that bound the paper's claims are still present.

``--self-test`` injects one fault of each kind the gate exists to catch -- a
numeral typed into a paragraph, a value edited into a fragment, a count spelled
out, a dropped scope statement -- and fails unless every one is reported.

Exits nonzero on any failure so it can gate a release.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"
FIGURE_MANIFEST = HERE / "paper_assets" / "manifest.json"
FIGURE_GENERATOR = HERE / "make_figures.py"

sys.path.insert(0, str(HERE))
import make_tables  # noqa: E402
from make_tables import BUDGET, START, TABLE_SOURCES, git_blob_sha  # noqa: E402
from make_figures import FIGURE_SOURCES  # noqa: E402

# Notation the body may carry digits for.  Each is a name or an algebraic form,
# never a measured quantity; every other digit in the prose fails the scan.
CONCEPTUAL_NUMERALS = (
    r"\$_\{?\d+\}?\$",                 # chemical subscripts: BeH$_2$, H$_2$O
    r"STO-3G",                         # the basis set's name
    r"\^2(?!\d)",                      # a square: S^2, K^2, g^2
    r"\\tfrac\{1\}\{2\}",              # the two-level lowering's prefactor
    r"\+4\\lvert c\\rvert",            # and its discriminant
    r"revtex4-2|apsrev4-2",
)

# Counts the generator produces, spelled out as words, would go stale without
# failing the numeral scan.  A number word within two words of one of these
# nouns is rejected; "three yardsticks" or "two-by-two" is description.
NUMBER_WORDS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "fifty", "hundred", "dozen",
)
COUNTED_NOUNS = ("cells", "cell", "inputs", "payloads", "trajectories",
                 "repetitions", "repeats", "campaigns", "audits", "steps")
SPELLED_COUNT = re.compile(
    rf"\b({'|'.join(NUMBER_WORDS)})\b((?:[\s~]+[A-Za-z-]+){{0,2}}?)"
    rf"[\s~]+({'|'.join(COUNTED_NOUNS)})\b",
    re.IGNORECASE)

# Statements the argument depends on.  Losing one in an edit would leave the
# abstract claiming a scope the body no longer states.
REQUIRED_PHRASES = (
    "declared before implementation",
    "packets add no column compactness",
    "excited singlet",
    "not an optimized variant",
    "bitwise identical",
    "We do not test that reading",
    "cost model",
    "exact in arithmetic",
    "not that it is the lowest one",
    "no hardware or quantum-advantage claim",
    "not blind data",
)


def _cells(line: str) -> int:
    total = 0
    for cell in line.rstrip("\\").split("&"):
        span = re.search(r"\\multicolumn\{(\d+)\}", cell)
        total += int(span.group(1)) if span else 1
    return total


def _brace_group(source: str, start: int) -> tuple[str, int]:
    if start >= len(source) or source[start] != "{":
        raise ValueError("expected a braced group")
    depth, offset = 0, start
    while offset < len(source):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:offset], offset + 1
        offset += 1
    raise ValueError("unbalanced braced group")


def _tabulars(source: str):
    marker, closing = r"\begin{tabular}", r"\end{tabular}"
    cursor = 0
    while (begin := source.find(marker, cursor)) >= 0:
        preamble_start = begin + len(marker)
        while preamble_start < len(source) and source[preamble_start].isspace():
            preamble_start += 1
        preamble, body_start = _brace_group(source, preamble_start)
        end = source.find(closing, body_start)
        if end < 0:
            raise ValueError("tabular environment has no closing marker")
        yield preamble, source[body_start:end]
        cursor = end + len(closing)


def _column_count(preamble: str) -> int:
    count, offset = 0, 0
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


def _row_sources(body: str, tables: Path):
    yield re.sub(r"\\input\{[^}]+\}", "", body)
    for match in re.finditer(r"\\input\{([^}]+)\}", body):
        target = tables.parent / match.group(1)
        if target.parent.name == "tables":
            target = tables / target.name
        if not target.suffix:
            target = target.with_suffix(".tex")
        if target.exists():
            yield target.read_text(encoding="utf-8")


def _source_digest(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_conceptual_numerals(text: str) -> str:
    for pattern in CONCEPTUAL_NUMERALS:
        text = re.sub(pattern, " ", text)
    return text


def _prose(text: str) -> str:
    """The body, with everything that legitimately carries digits removed."""
    body = text.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\bibliography{", 1)[0]
    body = re.sub(r"(?m)^\s*%.*$", "", body)
    body = re.sub(r"(?<!\\)%.*", "", body)
    for command in ("cite", "label", "ref", "eqref", "input", "includegraphics",
                    "bibliographystyle", "date", "url"):
        body = re.sub(rf"\\{command}(?:\[[^\]]*\])?\{{[^}}]*\}}", " ", body)
    body = re.sub(r"\\begin\{tabular\}\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", body)
    body = re.sub(r"\\multicolumn\{\d+\}", " ", body)
    return _strip_conceptual_numerals(body)


def prose_numerals(text: str) -> list[tuple[str, str]]:
    prose = _prose(text)
    found = []
    for match in re.finditer(r"\d+", prose):
        context = prose[max(0, match.start() - 40):match.end() + 20]
        found.append((match.group(0), " ".join(context.split())))
    return found


def spelled_counts(text: str) -> list[str]:
    prose = " ".join(_prose(text).split())
    return [match.group(0) for match in SPELLED_COUNT.finditer(prose)]


def missing_phrases(text: str) -> list[str]:
    # Whitespace is collapsed on both sides: LaTeX source rewraps freely, and a
    # phrase a reflow splits across two lines is still in the paper.
    body = " ".join(_prose(text).split())
    return [phrase for phrase in REQUIRED_PHRASES if " ".join(phrase.split()) not in body]


def stale_fragments(committed: Path, generated: Path) -> list[str]:
    """Fragments that differ by a byte from what the generator produces now."""
    problems = []
    names = {path.name for path in generated.iterdir()}
    for missing in sorted(set(TABLE_SOURCES) - names):
        problems.append(f"{missing} is registered but the generator no longer emits it")
    for regenerated in sorted(generated.iterdir()):
        target = committed / regenerated.name
        if not target.exists():
            problems.append(f"{regenerated.name} is generated but not committed")
        elif target.read_bytes() != regenerated.read_bytes():
            problems.append(f"{regenerated.name} does not match what the generator produces now -- "
                            "a hand edit, a stale record or a stale generator run "
                            "(run paper_compactness/make_tables.py)")
    return problems


def check(text: str, tables: Path) -> list[str]:
    problems: list[str] = []

    opens = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", text))
    closes = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
    for env in set(opens) | set(closes):
        if opens[env] != closes[env]:
            problems.append(f"unbalanced environment {env}: {opens[env]} begin vs {closes[env]} end")

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    for label, count in collections.Counter(labels).items():
        if count > 1:
            problems.append(f"duplicate label: {label}")
    for match in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", text):
        if match.group(1) not in set(labels):
            problems.append(f"dangling reference: {match.group(1)}")

    keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    cited: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        for key in (item.strip() for item in match.group(1).split(",")):
            if key:
                cited.add(key)
                if key not in keys:
                    problems.append(f"missing bib key: {key}")
    for key in sorted(keys - cited):
        problems.append(f"bib entry is never cited: {key}")

    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        name = Path(match.group(1)).with_suffix(".tex").name
        if Path(match.group(1)).parent.name == "tables":
            if name not in TABLE_SOURCES:
                problems.append(f"unregistered generated table input: {match.group(1)}")
            if not (tables / name).exists():
                problems.append(f"missing input: {match.group(1)} (run paper_compactness/make_tables.py)")
    for fragment in sorted(tables.glob("*.tex")):
        if fragment.name not in TABLE_SOURCES:
            problems.append(f"unregistered committed table fragment: {fragment.name}")

    for preamble, body in _tabulars(text):
        ncol = _column_count(preamble)
        for fragment in _row_sources(body, tables):
            for line in fragment.splitlines():
                line = line.strip()
                if not line.endswith(r"\\") or line.startswith("%"):
                    continue
                if _cells(line) != ncol:
                    problems.append(f"table row has {_cells(line)} cells, preamble declares {ncol}: {line[:60]}")

    for _, body in _tabulars(text):
        inline = re.sub(r"\\input\{[^}]+\}", "", body)
        for line in inline.splitlines():
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            for cell in line.rstrip("\\").split("&"):
                bare = _strip_conceptual_numerals(re.sub(r"\\multicolumn\{\d+\}", " ", cell))
                bare = re.sub(r"\\[A-Za-z]+|[{}$\\^_~,]", " ", bare)
                if re.search(r"\d", bare):
                    problems.append("numeric table cell typed into the manuscript rather than generated: "
                                    f"{line[:60]}")
                    break

    for numeral, context in prose_numerals(text):
        problems.append(f"numeral typed into the body prose rather than generated: {numeral!r} in {context!r}")
    for phrase in spelled_counts(text):
        problems.append(f"count spelled out in the body prose rather than generated: {phrase!r}")

    numbers = tables / "numbers.tex"
    if numbers.exists():
        defined = set(re.findall(r"\\newcommand\{\\(pk\w+)\}", numbers.read_text(encoding="utf-8")))
        used = set(re.findall(r"\\(pk[A-Za-z]+)", text)) - {"pkg"}
        for name in sorted(used - defined):
            problems.append(f"undefined generated macro: \\{name}")
        for name in sorted(defined - used):
            problems.append(f"generated macro is never used: \\{name} (drop it from make_tables.py or use it)")
    else:
        problems.append("missing tables/numbers.tex (run paper_compactness/make_tables.py)")

    for phrase in missing_phrases(text):
        problems.append(f"required scope statement is missing: {phrase!r}")
    return problems


def check_generated(tables: Path) -> list[str]:
    problems: list[str] = []
    if START.exists():
        payload = json.loads(START.read_text(encoding="utf-8"))
        if payload.get("record_git_blob_sha") != git_blob_sha(BUDGET):
            problems.append("data/start_weights.json binds a different budget record "
                            "(run paper_compactness/make_start_weights.py)")
    else:
        problems.append("missing data/start_weights.json (run paper_compactness/make_start_weights.py)")
    with tempfile.TemporaryDirectory() as scratch:
        generated = Path(scratch) / "tables"
        generated.mkdir()
        try:
            make_tables.main(tables=generated)
        except SystemExit as failure:
            problems.append(f"the table generator refuses to run: {failure}")
        else:
            problems += stale_fragments(tables, generated)
    return problems


def check_figures(text: str) -> list[str]:
    problems: list[str] = []
    referenced = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        asset = HERE / match.group(1)
        referenced.append((asset.name, asset))
        if not asset.exists():
            problems.append(f"missing figure: {match.group(1)}")
    if not referenced:
        return problems
    if not FIGURE_MANIFEST.exists():
        return problems + ["missing figure manifest (run paper_compactness/make_figures.py)"]
    manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("generator", {}).get("sha256") != _source_digest(FIGURE_GENERATOR):
        problems.append("figure manifest has a stale generator digest (run paper_compactness/make_figures.py)")
    figures = manifest.get("figures", {})
    for name, asset in referenced:
        if name not in figures or name not in FIGURE_SOURCES:
            problems.append(f"{name} is absent from the figure manifest or undeclared by make_figures.py")
            continue
        expected = {str(path.resolve().relative_to(ROOT)): _source_digest(path) for path in FIGURE_SOURCES[name]}
        if figures[name].get("sources", {}) != expected:
            problems.append(f"{name} manifest has stale source digests (run paper_compactness/make_figures.py)")
        if asset.exists() and figures[name].get("asset_git_blob_sha") != git_blob_sha(asset):
            problems.append(f"{name} does not match its manifest asset digest (run paper_compactness/make_figures.py)")
    return problems


def self_test() -> int:
    """Inject one fault of each kind and require the gate to name every one."""
    text = TEX.read_text(encoding="utf-8")
    tables = HERE / "tables"
    faults = {
        "typed numeral": text.replace(r"\section{Results}", "\\section{Results}\nPackets used 145 columns.", 1),
        "spelled count": text.replace(r"\section{Results}", "\\section{Results}\nAll fifty two cells ran.", 1),
        "dropped scope statement": text.replace("not blind data", "blind data"),
        "undefined macro": text.replace(r"\pkCells{}", r"\pkCellCount{}", 1),
    }
    failures = []
    baseline = set(check(text, tables))
    for name, faulty in faults.items():
        if faulty == text:
            failures.append(f"could not inject the {name} fault")
        elif not set(check(faulty, tables)) - baseline:
            failures.append(f"the gate missed a {name}")
    with tempfile.TemporaryDirectory() as scratch:
        edited = Path(scratch) / "tables"
        edited.mkdir()
        for fragment in tables.glob("*.tex"):
            (edited / fragment.name).write_bytes(fragment.read_bytes())
        numbers = edited / "numbers.tex"
        numbers.write_text(numbers.read_text(encoding="utf-8").replace(r"\ensuremath{52}", r"\ensuremath{53}", 1),
                           encoding="utf-8")
        generated = Path(scratch) / "generated"
        generated.mkdir()
        make_tables.main(tables=generated)
        if not any("numbers.tex" in problem for problem in stale_fragments(edited, generated)):
            failures.append("the gate missed a value edited into a fragment")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print(f"self-test passed: {len(faults) + 1} injected faults reported")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return self_test()
    text = TEX.read_text(encoding="utf-8")
    tables = HERE / "tables"
    problems = check(text, tables) + check_generated(tables) + check_figures(text)
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print("manuscript checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
