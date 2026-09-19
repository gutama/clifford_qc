"""Structural and evidence checks for the architecture manuscript.

The paper claims that no number in it is typed by hand and that its
architecture tables cannot drift from the code they describe.  Both claims are
only worth making if something enforces them, so this gate checks:

  1.  every LaTeX environment is balanced;
  2.  every ``\\ref``/``\\eqref`` resolves to a ``\\label``, and no label repeats;
  3.  every ``\\cite`` key exists in references.bib, and no entry is uncited;
  4.  every ``\\input`` target exists, so the table generator has been run;
  5.  every table row has the column count its preamble declares;
  6.  no numeric cell is typed into the manuscript instead of generated;
  7.  no numeral appears in the body prose outside the short allowlist of
      conceptual notation, and every ``\\cqc`` macro used is defined and
      every one defined is used;
  8.  every generated fragment is byte-identical to what the generator
      produces right now, so a value edited by hand into a fragment fails here
      rather than typesetting;
  9.  each referenced figure exists and its manifest entry still matches the
      digests of its generator, its inputs, and the committed asset;
 10.  the committed source census still equals a census recomputed now, so a
      module without a layer, a gate without a class, or a record whose
      evidence declaration changed fails here rather than typesetting;
 11.  the evidence-language invariants the paper commits to are still present.

Checks 7 and 10 are the ones specific to this manuscript: check 7 is what makes
"every number is generated" a statement rather than an intention, and check 10
keeps the source and gate census tables current, since no experiment rebuilds them.

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
TABLE_GENERATOR = HERE / "make_tables.py"
FIGURE_GENERATOR = HERE / "make_figures.py"

sys.path.insert(0, str(HERE))
import make_tables  # noqa: E402
from make_tables import (  # noqa: E402
    CENSUS, TABLE_SOURCES, gate_census, layer_census, record_census,
)
from make_figures import (  # noqa: E402
    FIGURE_SOURCES, architecture_sources_digest,
)

# Notation the body prose may carry digits for.  Each is a name or an algebraic
# form, never a measured quantity: a block size the text is discussing, a
# chemical subscript, the dimension of the algebra, the document class.  The
# scan below rejects every other digit, so a result typed into a paragraph
# fails regardless of how many digits it contains.
CONCEPTUAL_NUMERALS = (
    r"\$_\{?\d+\}?\$",            # chemical subscripts: H$_4$, H$_{12}$
    r"\$k\s*=\s*(?:\d+|n)\$",      # a named block size: $k=1$, $k=10$
    r"2\^n",                        # dimension of the representation
    r"Cl\}?\(2n",                   # the algebra itself, plain or \mathrm
    r"M\(2\^n",
    r"P/2\)",                       # the rotor half-angle
    r"\\frac\{\\mathrm\{i\}\}\{2\}",   # its exact commutator normalization
    r"E_P'\(0\)",                    # derivative at the initial angle
    r"revtex4-2",
)

# Statements the paper's argument depends on.  Losing one in an edit would
# leave a claim in the abstract that the body no longer supports.
REQUIRED_PHRASES = (
    "illustrative",
    "not a hardware",
    "heuristic",
    "inadmissible",
    "indeterminate",
    "no advantage claim",
    "bias floor",
    "preregistered",
    "is not enforced by a schema",
    # Resource scope: the lifetime bound covers coefficient rows and is not a
    # process-level memory bound.  Losing this sentence would leave the paper
    # quoting a bound the implementation does not make.
    "not a total-process",
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
    marker = r"\begin{tabular}"
    closing = r"\end{tabular}"
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


def _row_sources(body: str):
    yield re.sub(r"\\input\{[^}]+\}", "", body)
    for match in re.finditer(r"\\input\{([^}]+)\}", body):
        target = HERE / match.group(1)
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


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _prose(text: str) -> str:
    """The body text, with everything that legitimately carries digits removed.

    Citation keys, labels, file paths and the document class all contain
    numerals that say nothing about a result, so they are stripped before the
    numeral scan rather than allowlisted one at a time.
    """
    body = text.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\bibliography{", 1)[0]
    body = re.sub(r"(?m)^\s*%.*$", "", body)
    body = re.sub(r"(?<!\\)%.*", "", body)
    for command in ("cite", "label", "ref", "eqref", "input", "includegraphics",
                    "bibliographystyle", "documentclass", "usepackage", "date"):
        body = re.sub(rf"\\{command}(?:\[[^\]]*\])?\{{[^}}]*\}}", " ", body)
    # A tabular preamble is typesetting, not a result: p{0.52\columnwidth} is
    # a column width and must not be read as a hand-typed number.
    body = re.sub(r"\\begin\{tabular\}\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", body)
    return _strip_conceptual_numerals(body)


def prose_numerals(text: str) -> list[tuple[str, str]]:
    """Every numeral in the body prose, with its context.

    The reproducibility section claims every empirical quantity is generated, prose
    included, so the scan is for any numeral at all.  A threshold on digit
    count would let a three-digit setting count or a shot budget through, and
    those are exactly the quantities this manuscript quotes.
    """
    prose = _prose(text)
    found = []
    for match in re.finditer(r"\d+", prose):
        context = prose[max(0, match.start() - 40):match.end() + 20]
        found.append((match.group(0), " ".join(context.split())))
    return found


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    problems: list[str] = []

    opens = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", text))
    closes = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", text))
    for env in set(opens) | set(closes):
        if opens[env] != closes[env]:
            problems.append(f"unbalanced environment {env}: "
                            f"{opens[env]} begin vs {closes[env]} end")

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
            if not key:
                continue
            cited.add(key)
            if key not in keys:
                problems.append(f"missing bib key: {key}")
    for key in sorted(keys - cited):
        problems.append(f"bib entry is never cited: {key}")

    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        target = HERE / match.group(1)
        if not target.exists() and not target.with_suffix(".tex").exists():
            problems.append(f"missing input: {match.group(1)} "
                            "(run paper_architecture/make_tables.py)")
        resolved = target if target.suffix else target.with_suffix(".tex")
        if resolved.parent == HERE / "tables" and resolved.name not in TABLE_SOURCES:
            problems.append(
                f"unregistered generated table input: {match.group(1)}")

    for fragment in sorted((HERE / "tables").glob("*.tex")):
        if fragment.name not in TABLE_SOURCES:
            problems.append(
                f"unregistered committed table fragment: {fragment.name}")

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

    # A hand-typed label column is fine; a hand-typed number is what goes stale.
    # Header rows stay in the scan: conceptual notation such as $k=1$ is
    # removed by the allowlist, while a result typed into a header must fail.
    for _, body in _tabulars(text):
        inline = re.sub(r"\\input\{[^}]+\}", "", body)
        for line in inline.splitlines():
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            for cell in line.rstrip("\\").split("&")[1:]:
                bare = _strip_conceptual_numerals(cell)
                bare = re.sub(r"\\[A-Za-z]+|[{}$\\^_~,]", " ", bare)
                if re.search(r"\d", bare):
                    problems.append(
                        "numeric table cell is typed into the manuscript "
                        "rather than generated into tables/ "
                        f"(run paper_architecture/make_tables.py): {line[:60]}")
                    break

    for numeral, context in prose_numerals(text):
        problems.append(
            "numeral typed into the body prose rather than generated into "
            f"tables/numbers.tex: {numeral!r} in {context!r}")

    numbers = HERE / "tables" / "numbers.tex"
    if not numbers.exists():
        problems.append("missing tables/numbers.tex "
                        "(run paper_architecture/make_tables.py)")
    else:
        defined = set(re.findall(r"\\newcommand\{\\(cqc\w+)\}",
                                 numbers.read_text(encoding="utf-8")))
        used = set(re.findall(r"\\(cqc[A-Za-z][A-Za-z0-9]*)", text))
        for name in sorted(used - defined):
            problems.append(f"undefined generated macro: \\{name}")
        for name in sorted(defined - used):
            problems.append(
                f"generated macro is never used: \\{name} "
                "(drop it from make_tables.py or use it)")

    # Regenerate every fragment into a scratch tree and compare byte for byte.
    # The provenance headers each fragment carries digest the generator and the
    # source records, so they catch a record regenerated without rerunning the
    # generator -- but a value edited by hand into a fragment changes neither,
    # and the header check alone would pass it.  This is the check that makes
    # "no number in this paper is typed by hand" true of the tables as well as
    # of the prose.
    for name in sorted(TABLE_SOURCES):
        if not (HERE / "tables" / name).exists():
            problems.append(f"missing generated table: {name} "
                            "(run paper_architecture/make_tables.py)")
    with tempfile.TemporaryDirectory() as scratch:
        tables = Path(scratch) / "tables"
        data = Path(scratch) / "data"
        tables.mkdir(parents=True)
        data.mkdir(parents=True)
        try:
            make_tables.main(tables=tables, data=data)
        except SystemExit as failure:            # a census refusal, not a crash
            problems.append(f"the table generator refuses to run: {failure}")
        else:
            generated_names = {path.name for path in tables.iterdir()}
            for missing in sorted(set(TABLE_SOURCES) - generated_names):
                problems.append(
                    f"{missing} is registered but the table generator no "
                    "longer emits it (run paper_architecture/make_tables.py)")
            for regenerated in sorted(tables.iterdir()):
                committed = HERE / "tables" / regenerated.name
                if not committed.exists():
                    problems.append(
                        f"{regenerated.name} is generated but not committed "
                        "(run paper_architecture/make_tables.py)")
                elif committed.read_bytes() != regenerated.read_bytes():
                    problems.append(
                        f"{regenerated.name} does not match what the generator "
                        "produces now -- a hand edit, a stale record, or a "
                        "stale generator run "
                        "(run paper_architecture/make_tables.py)")
            generated_census = data / CENSUS.name
            if not generated_census.exists():
                problems.append(
                    "the table generator no longer emits "
                    "data/source_census.json "
                    "(run paper_architecture/make_tables.py)")
            else:
                committed_census = (
                    CENSUS.read_bytes() if CENSUS.exists() else b"")
                if committed_census != generated_census.read_bytes():
                    problems.append(
                        "data/source_census.json does not match a census "
                        "taken now (run paper_architecture/make_tables.py)")

    referenced: list[tuple[str, Path]] = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        relative = match.group(1)
        asset = HERE / relative
        referenced.append((Path(relative).name, asset))
        if not asset.exists():
            problems.append(f"missing figure: {relative}")
    if referenced:
        if not FIGURE_MANIFEST.exists():
            problems.append("missing figure manifest "
                            "(run paper_architecture/make_figures.py)")
        else:
            manifest = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
            if manifest.get("generator", {}).get("sha256") != _source_digest(
                    FIGURE_GENERATOR):
                problems.append("figure manifest has a stale generator digest "
                                "(run paper_architecture/make_figures.py)")
            figures = manifest.get("figures", {})
            for name, asset in referenced:
                if name not in figures:
                    problems.append(f"{name} is absent from the figure manifest")
                    continue
                if name not in FIGURE_SOURCES:
                    problems.append(
                        f"{name} is not declared by make_figures.py")
                    continue
                expected = {
                    str(path.resolve().relative_to(ROOT)): _source_digest(path)
                    for path in FIGURE_SOURCES[name]
                }
                if figures[name].get("sources", {}) != expected:
                    problems.append(
                        f"{name} manifest has stale source digests "
                        "(run paper_architecture/make_figures.py)")
                if (asset.exists() and
                        figures[name].get("asset_git_blob_sha") !=
                        _git_blob_sha(asset)):
                    problems.append(
                        f"{name} does not match its manifest asset digest "
                        "(run paper_architecture/make_figures.py)")
                if (name == "layer_stack.pdf" and
                        figures[name].get("architecture_sources_sha256") !=
                        architecture_sources_digest()):
                    problems.append(
                        "layer_stack.pdf manifest has a stale package-source "
                        "digest (run paper_architecture/make_figures.py)")

    # The architecture tables describe code, and no experiment rebuilds them,
    # so the census is compared section by section as well: the regeneration
    # above would catch the same drift, but not name which part of the
    # repository moved.
    if not CENSUS.exists():
        problems.append("missing data/source_census.json "
                        "(run paper_architecture/make_tables.py)")
    else:
        committed = json.loads(CENSUS.read_text(encoding="utf-8"))
        try:
            current = {
                "source": layer_census(),
                "gates": gate_census(),
                "records": record_census(),
            }
        except SystemExit as failure:
            problems.append(f"the live census refuses to run: {failure}")
        else:
            for section, value in current.items():
                if committed.get(section) != value:
                    problems.append(
                        f"the committed {section} census no longer matches the "
                        "repository (run paper_architecture/make_tables.py)")

    # Whitespace is collapsed on both sides before the search.  The invariant
    # is the statement, not its line breaks: LaTeX source rewraps freely, and a
    # phrase that a reflow split across two source lines is still present in
    # the paper.  Matching the raw text made the gate report a missing
    # commitment whenever a paragraph was rewrapped, which is a false alarm
    # that teaches an author to stop trusting the check.
    comment_stripped_body = " ".join(_prose(text).split())
    for phrase in REQUIRED_PHRASES:
        if " ".join(phrase.split()) not in comment_stripped_body:
            problems.append(f"required evidence-language phrase is missing: "
                            f"{phrase!r}")

    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print("manuscript checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
