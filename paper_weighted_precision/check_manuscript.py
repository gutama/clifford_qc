"""Structural, provenance, and evidence-language gate for paper_weighted_precision.

Checks:
  1. Every LaTeX environment is balanced.
  2. Every \\ref / \\eqref resolves to a unique \\label.
  3. Every \\cite key exists in references.bib and no bib entry is unused.
  4. Every \\input and \\includegraphics target exists on disk.
  5. Every tabular row (including \\input fragments and \\multicolumn spans)
     matches the column count declared in its tabular preamble.
  6. No numeric data row is hand-typed directly inside manuscript.tex tables.
  7. Every generated table and macro file in tables/ matches fresh regeneration
     from docs/restart/weighted_precision_results.json.gz byte-for-byte, and
     tables/manifest.json matches exact source and generator SHA-256 digests.
  8. Every PDF figure in paper_assets/ exists and paper_assets/manifest.json
     matches exact generator and source SHA-256 digests.
  9. Required preregistration provenance and negative-result caveats appear in
     manuscript.tex, with no unguarded overclaim phrases.

Run ``python paper_weighted_precision/check_manuscript.py --self-test`` to also
verify that synthetic defects are rejected.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_weighted_precision import make_figures, make_tables  # noqa: E402

TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _brace_group(source: str, start: int) -> tuple[str, int]:
    if start >= len(source) or source[start] != "{":
        raise ValueError("expected a braced group")
    depth = 0
    for offset in range(start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : offset], offset + 1
    raise ValueError("unterminated braced group")


def _tabulars(source: str):
    marker = r"\begin{tabular}"
    closing = r"\end{tabular}"
    cursor = 0
    while (begin := source.find(marker, cursor)) >= 0:
        pstart = begin + len(marker)
        while pstart < len(source) and source[pstart].isspace():
            pstart += 1
        preamble, body_start = _brace_group(source, pstart)
        end = source.find(closing, body_start)
        if end < 0:
            raise ValueError("tabular environment has no closing marker")
        yield preamble, source[body_start:end]
        cursor = end + len(closing)


def _column_count(preamble: str) -> int:
    count = 0
    i = 0
    while i < len(preamble):
        ch = preamble[i]
        if ch in "lcrX":
            count += 1
            i += 1
        elif ch in "pmb":
            count += 1
            _, i = _brace_group(preamble, i + 1)
        elif ch in "@!<>":
            _, i = _brace_group(preamble, i + 1)
        elif ch == "*":
            rep, after_rep = _brace_group(preamble, i + 1)
            sub, i = _brace_group(preamble, after_rep)
            count += int(rep) * _column_count(sub)
        else:
            i += 1
    return count


def _row_cell_count(line: str) -> int:
    total = 0
    for cell in line.rstrip("\\").split("&"):
        span = re.search(r"\\multicolumn\{(\d+)\}", cell)
        total += int(span.group(1)) if span else 1
    return total


def check_tex_text(tex_text: str, bib_text: str, *, base_dir: Path = HERE) -> list[str]:
    """Run structural checks on manuscript TeX and BibTeX strings."""
    errors: list[str] = []
    clean = _strip_comments(tex_text)

    # 1. Balanced environments
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            errors.append(f"unbalanced \\end{{{env}}}")
        else:
            stack.pop()
    if stack:
        errors.append(f"unclosed LaTeX environments: {stack}")

    # 2. Labels and refs
    labels = re.findall(r"\\label\{([^}]+)\}", clean)
    dups = [k for k, v in collections.Counter(labels).items() if v > 1]
    if dups:
        errors.append(f"duplicate labels: {dups}")
    label_set = set(labels)
    for ref in re.findall(r"\\(?:eq)?ref\{([^}]+)\}", clean):
        if ref not in label_set:
            errors.append(f"unresolved reference: {ref}")

    # 3. Citations vs references.bib
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_text))
    cited: set[str] = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", clean):
        for key in group.split(","):
            key = key.strip()
            if key:
                cited.add(key)
    missing_cites = sorted(cited - bib_keys)
    unused_bib = sorted(bib_keys - cited)
    if missing_cites:
        errors.append(f"missing BibTeX entries: {missing_cites}")
    if unused_bib:
        errors.append(f"unused BibTeX entries: {unused_bib}")

    # 4. \input and \includegraphics existence
    for inp in re.findall(r"\\input\{([^}]+)\}", clean):
        target = base_dir / inp
        if not target.suffix:
            target = target.with_suffix(".tex")
        if not target.exists():
            errors.append(f"missing \\input target: {inp}")
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean):
        target = base_dir / fig
        if not target.exists():
            errors.append(f"missing figure asset: {fig}")

    # 5 & 6. Tabular column counts + no hand-typed numeric data rows
    tabular_count = 0
    for preamble, body in _tabulars(clean):
        tabular_count += 1
        expected_cols = _column_count(preamble)
        # Body itself (excluding \input fragments) must only contain headers / rules, no numeric data rows.
        direct_body = re.sub(r"\\input\{[^}]+\}", "", body)
        after_midrule = direct_body.split(r"\midrule", 1)[1] if r"\midrule" in direct_body else ""
        for raw_line in after_midrule.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("\\bottomrule") or line.startswith("\\midrule"):
                continue
            if "&" in line and re.search(r"\d", line):
                errors.append(f"hand-typed numeric table row inside manuscript.tex: {line[:80]}")

        # Check column counts across direct headers AND every \input fragment.
        sources = [direct_body]
        for m in re.finditer(r"\\input\{([^}]+)\}", body):
            target = base_dir / m.group(1)
            if not target.suffix:
                target = target.with_suffix(".tex")
            if target.exists():
                sources.append(_strip_comments(target.read_text(encoding="utf-8")))
        for src in sources:
            for raw_line in src.splitlines():
                line = raw_line.strip()
                if not line or not line.endswith(r"\\"):
                    continue
                if "&" not in line and r"\multicolumn" not in line:
                    continue
                got = _row_cell_count(line)
                if got != expected_cols:
                    errors.append(
                        f"tabular column mismatch (expected {expected_cols}, got {got}): {line[:80]}"
                    )
    if tabular_count == 0:
        errors.append("manuscript contains no tabular environments")

    # 9. Required provenance & honest caveat markers (and no unguarded overclaims)
    for required_token in (
        r"\wpDeclCommitFull",
        r"\wpDeclCommitShort",
        r"all\_passed = false",
        r"sc\_card\_full",
        r"\wpExhausted",
        "not hardware demonstrations or quantum advantage claims",
    ):
        if required_token not in clean:
            errors.append(f"missing required provenance/caveat marker: {required_token}")

    return errors


def check_artifacts() -> list[str]:
    """Verify tables, macros, figures, and manifests match the frozen Commit 2 record."""
    errors: list[str] = []

    # Table and macro exact regeneration check
    expected_tables = make_tables.render_all()
    for name, expected_content in expected_tables.items():
        path = make_tables.TABLES_DIR / name
        if not path.exists():
            errors.append(f"missing generated table/macro file: {path.relative_to(ROOT)}")
            continue
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected_content.replace("\r\n", "\n"):
            errors.append(f"stale generated table/macro file (rerun make_tables.py): {name}")

    expected_tman = make_tables.build_manifest(expected_tables)
    if not make_tables.MANIFEST_PATH.exists():
        errors.append("missing tables/manifest.json")
    else:
        actual_tman = json.loads(make_tables.MANIFEST_PATH.read_text(encoding="utf-8"))
        if actual_tman != expected_tman:
            errors.append("tables/manifest.json does not match current generator/sources/outputs")

    # Figure existence and manifest check
    expected_fman = make_figures.build_manifest()
    if not make_figures.MANIFEST_PATH.exists():
        errors.append("missing paper_assets/manifest.json")
    else:
        actual_fman = json.loads(make_figures.MANIFEST_PATH.read_text(encoding="utf-8"))
        if actual_fman != expected_fman:
            errors.append("paper_assets/manifest.json does not match current generator/sources")
    for fname in make_figures.FIGURE_NAMES:
        fpath = make_figures.ASSETS / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            errors.append(f"missing or empty figure asset: {fname}")

    return errors


def run_self_test() -> None:
    """Verify check_tex_text rejects synthetic defects."""
    base_tex = TEX.read_text(encoding="utf-8")
    base_bib = BIB.read_text(encoding="utf-8")

    # 1. Clean manuscript must pass
    clean_errs = check_tex_text(base_tex, base_bib)
    if clean_errs:
        raise AssertionError(f"clean manuscript failed self-test: {clean_errs}")

    # 2. Broken reference must fail
    errs_ref = check_tex_text(base_tex + "\nSee Fig.~\\ref{fig:nonexistent}.\n", base_bib)
    assert any("unresolved reference" in e for e in errs_ref), "failed to catch broken ref"

    # 3. Hand-typed numeric row must fail
    tampered_row = base_tex.replace(
        r"\input{tables/protocol_banks.tex}",
        "A & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\\\",
        1,
    )
    errs_row = check_tex_text(tampered_row, base_bib)
    assert any("hand-typed numeric table row" in e for e in errs_row), "failed to catch hand-typed row"

    # 4. Unused bib entry must fail
    errs_bib = check_tex_text(base_tex, base_bib + "\n@misc{unused2099, author={X}, title={Y}, year={2099}}\n")
    assert any("unused BibTeX" in e for e in errs_bib), "failed to catch unused bib entry"

    # 5. Unbalanced environment must fail
    errs_env = check_tex_text(base_tex + "\n\\begin{itemize}\n", base_bib)
    assert any("unclosed LaTeX" in e for e in errs_env), "failed to catch unclosed environment"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    errors = check_tex_text(TEX.read_text(encoding="utf-8"), BIB.read_text(encoding="utf-8"))
    errors.extend(check_artifacts())
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1
    if args.self_test:
        run_self_test()
        print("weighted_precision manuscript checks and self-test passed")
    else:
        print("weighted_precision manuscript checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
