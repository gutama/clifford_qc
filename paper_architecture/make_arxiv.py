"""Assemble and verify the arXiv submission package for this manuscript.

arXiv does not run BibTeX, it decides between the DVI and PDF toolchains by
scanning the first few lines of the source, and it extracts the upload into a
single working directory rather than the repository the paper was written in.
Each of those is a way a manuscript that builds here fails to build there, and
none of them is visible in a local ``pdflatex`` run.  This script closes all
three by building the submission the way arXiv will:

  1.  it stages only the files the manuscript actually reads -- the source, the
      ``.bbl``, the generated table fragments it inputs and the figures it
      includes -- so nothing unreferenced is published by accident and nothing
      referenced is left behind;
  2.  it rebuilds that staging tree with ``pdflatex`` alone, with no
      ``references.bib`` in reach, which is the failure mode a local build can
      never reproduce because the ``.bbl`` is always already there;
  3.  it compares the page count of that build against the full local build, so
      a submission that compiles but drops its bibliography is still a failure.

It also writes the plain-text abstract for the submission form.  The web form
takes text, not LaTeX, and silently accepts ``\\pkg{}`` as three literal words;
deriving it from the manuscript keeps the posted abstract and the typeset one
from drifting apart.

    python paper_architecture/make_arxiv.py            # build and verify
    python paper_architecture/make_arxiv.py --check    # structure only, no LaTeX

``--check`` runs everywhere the manuscript gate runs: it needs no TeX
installation, so continuous integration can hold the packaging contract even
though it cannot typeset.  Exits nonzero on any problem.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "manuscript.tex"
BIB = HERE / "references.bib"
STEM = TEX.stem
DEFAULT_OUT = HERE / "arxiv"

# arXiv rejects an abstract longer than this many characters on the submission
# form, and truncates nothing: the submission simply does not go through.
ABSTRACT_LIMIT = 1920

# The ordinary upload ceiling.  Every asset here is a small vector figure, so
# this is a guard against a stray raster or a committed build product being
# swept into the package, not a constraint the paper is near.
PACKAGE_LIMIT_BYTES = 50 * 1024 * 1024

# arXiv reads only the opening lines to choose a toolchain, so a \pdfoutput
# further down the preamble does not count.
PDFOUTPUT_WINDOW = 5

# What pdflatex can embed directly.  An EPS figure would need the DVI route and
# would contradict the \pdfoutput=1 this paper sets.
FIGURE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

# Matplotlib's default PDF font type is 3, which embeds glyphs as uninterpreted
# drawing procedures: the text is not searchable, not extractable, and renders
# poorly at small sizes in some viewers.  APS and most physics journals ask for
# Type 1 or TrueType instead, and the figure generator sets ``pdf.fonttype``
# accordingly -- but an asset committed before that setting keeps its Type 3
# fonts, and no digest in the figure manifest can tell, because the asset
# matches its own recorded hash and its generator has not changed since.
TYPE3_MARKER = re.compile(rb"/Subtype\s*/Type3")


def _strip_comments(text: str) -> str:
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    return re.sub(r"(?<!\\)%.*", "", text)


def referenced_inputs(text: str) -> list[str]:
    """Every ``\\input`` target, as the relative path the manuscript writes."""
    found = []
    for match in re.finditer(r"\\input\{([^}]+)\}", _strip_comments(text)):
        target = match.group(1)
        found.append(target if Path(target).suffix else target + ".tex")
    return sorted(dict.fromkeys(found))


def referenced_figures(text: str) -> list[str]:
    """Every ``\\includegraphics`` target, as written."""
    found = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", _strip_comments(text))
    return sorted(dict.fromkeys(found))


def plain_abstract(text: str) -> str:
    """The abstract as the submission form wants it: text, not LaTeX.

    Only the markup this manuscript's abstract actually uses is handled, and
    anything left carrying a backslash is reported rather than silently posted:
    a macro that leaks through here becomes literal characters on the abstract
    page, where nobody rebuilding the paper would ever see it.
    """
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not match:
        raise ValueError("the manuscript has no abstract environment")
    body = _strip_comments(match.group(1))
    body = body.replace(r"\pkg{}", "clifford_qc").replace(r"\pkg", "clifford_qc")
    body = re.sub(r"\\(?:emph|texttt|textit|textbf|text)\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\cite\{[^}]*\}", "", body)
    body = body.replace(r"\_", "_").replace(r"\&", "&").replace(r"\%", "%")
    body = body.replace(r"\-", "").replace("~", " ")
    body = body.replace("---", "\u2014").replace("--", "\u2013")
    return " ".join(body.split())


def _wrap(paragraph: str, width: int = 78) -> str:
    lines, current = [], ""
    for word in paragraph.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(lines)


def structural_problems(text: str) -> list[str]:
    """Everything about the package that can be decided without a TeX run."""
    problems: list[str] = []

    head = text.splitlines()[:PDFOUTPUT_WINDOW]
    if not any(re.match(r"\s*\\pdfoutput\s*=\s*1", line) for line in head):
        problems.append(
            f"\\pdfoutput=1 is not in the first {PDFOUTPUT_WINDOW} lines: arXiv "
            "decides the toolchain from the opening of the file, and these "
            "figures need pdflatex")

    for relative in referenced_inputs(text) + referenced_figures(text):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            problems.append(
                f"{relative} reaches outside the paper directory; arXiv "
                "extracts the upload into one tree and cannot follow it")
            continue
        if not (HERE / path).exists():
            problems.append(f"missing package input: {relative}")

    for relative in referenced_figures(text):
        figure = HERE / relative
        suffix = Path(relative).suffix.lower()
        if suffix and suffix not in FIGURE_SUFFIXES:
            problems.append(
                f"{relative} is not a format pdflatex embeds "
                f"({', '.join(sorted(FIGURE_SUFFIXES))})")
        elif (suffix == ".pdf" and figure.exists()
                and TYPE3_MARKER.search(figure.read_bytes())):
            problems.append(
                f"{relative} embeds Type 3 fonts; regenerate it with "
                "make_figures.py, which sets a TrueType font type")

    if r"\bibliography{" not in _strip_comments(text):
        problems.append(
            "no \\bibliography: this script stages a .bbl because arXiv does "
            "not run BibTeX, and there would be nothing for it to stage")

    for local in sorted(HERE.glob("*.sty")) + sorted(HERE.glob("*.cls")):
        if local.name not in {path.name for path in _staged_paths(text)}:
            problems.append(
                f"{local.name} is a local class or style file and is not "
                "staged; arXiv would build against its own copy or none")

    problems.extend(bibliography_problems())

    try:
        abstract = plain_abstract(text)
    except ValueError as failure:
        problems.append(str(failure))
    else:
        if len(abstract) > ABSTRACT_LIMIT:
            problems.append(
                f"the plain-text abstract is {len(abstract)} characters "
                f"against arXiv's {ABSTRACT_LIMIT}-character form limit")
        leaked = sorted(set(re.findall(r"\\[A-Za-z]+", abstract)))
        if leaked:
            problems.append(
                "LaTeX survives into the plain-text abstract and would be "
                f"posted literally: {', '.join(leaked)}")

    total = sum((HERE / path).stat().st_size
                for path in _staged_paths(text) if (HERE / path).exists())
    if total > PACKAGE_LIMIT_BYTES:
        problems.append(
            f"the staged files total {total} bytes, over arXiv's "
            f"{PACKAGE_LIMIT_BYTES}-byte upload limit")

    return problems


def bibliography_problems() -> list[str]:
    """What BibTeX will refuse, checked without running BibTeX.

    The .bbl is the one shipped file that no local edit produces directly, so a
    bibliography BibTeX cannot parse is a submission with no reference list.
    The trap here is real rather than theoretical: BibTeX starts an entry at an
    at-sign wherever it finds one, including inside a comment, so a sentence
    about an "@article" in a header comment silently truncates the entry that
    follows it.  Nothing in the manuscript gate sees that, because the gate
    matches entry keys and a comment has none.
    """
    problems: list[str] = []
    if not BIB.exists():
        return [f"missing {BIB.name}, so no .bbl can be produced"]
    for number, line in enumerate(
            BIB.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("%") and "@" in stripped:
            problems.append(
                f"{BIB.name} line {number} is a comment containing an "
                "at-sign; BibTeX reads it as the start of an entry and drops "
                "the entry that follows")
    return problems


def _staged_paths(text: str) -> list[Path]:
    """Repository-relative paths the package ships, the .bbl aside.

    The ``.bbl`` is a build product rather than a committed file, so it is
    named at staging time instead of here.
    """
    return [Path(TEX.name)] + [
        Path(relative) for relative in referenced_inputs(text)
    ] + [Path(relative) for relative in referenced_figures(text)]


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def _pages(log: str) -> int | None:
    match = re.search(r"Output written on .*? \((\d+) pages?", log)
    return int(match.group(1)) if match else None


def _typeset(directory: Path, *, bibtex: bool) -> tuple[int | None, str]:
    """Run the manuscript through pdflatex the way the caller asks for.

    Three passes, because REVTeX needs one to write the aux file, one to
    resolve the bibliography and the cross-references, and one to settle the
    float placement those moved.
    """
    log = ""
    for index in range(3):
        done = _run(["pdflatex", "-interaction=nonstopmode",
                     "-halt-on-error", f"{STEM}.tex"], directory)
        log = done.stdout + done.stderr
        if done.returncode != 0:
            raise SystemExit(
                f"pdflatex failed in {directory}:\n" + log[-4000:])
        if bibtex and index == 0:
            done = _run(["bibtex", STEM], directory)
            if done.returncode != 0:
                raise SystemExit(
                    "bibtex failed:\n" + done.stdout + done.stderr)
    return _pages(log), log


def build(out: Path, *, keep_scratch: bool = False) -> int:
    """Stage, archive and verify the submission.  Returns a process status."""
    text = TEX.read_text(encoding="utf-8")
    problems = structural_problems(text)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1

    if shutil.which("pdflatex") is None or shutil.which("bibtex") is None:
        raise SystemExit(
            "pdflatex and bibtex are needed to build the package; run with "
            "--check for the structural checks alone")

    with tempfile.TemporaryDirectory() as scratch:
        reference = Path(scratch) / "reference"
        shutil.copytree(HERE, reference, ignore=shutil.ignore_patterns(
            "arxiv", "__pycache__", "*.pyc"))
        reference_pages, _ = _typeset(reference, bibtex=True)
        bbl = reference / f"{STEM}.bbl"
        if not bbl.exists():
            raise SystemExit(
                "bibtex produced no .bbl, so there is nothing to ship in "
                "place of the bibliography arXiv will not rebuild")

        # Stage exactly what the manuscript reads, plus the .bbl.  Nothing
        # else: references.bib is deliberately absent so the verification
        # build below cannot quietly fall back on it.
        staged = Path(scratch) / "package"
        staged.mkdir()
        shipped: list[Path] = []
        for relative in _staged_paths(text):
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / relative, destination)
            shipped.append(relative)
        shutil.copy2(bbl, staged / bbl.name)
        shipped.append(Path(bbl.name))

        # The verification arXiv itself performs: pdflatex only, three passes,
        # no bibtex and no .bib within reach.
        verify = Path(scratch) / "verify"
        shutil.copytree(staged, verify)
        pages, log = _typeset(verify, bibtex=False)
        if re.search(r"Citation\s+.*undefined", log):
            raise SystemExit(
                "the pdflatex-only build reports undefined citations, so the "
                "staged .bbl is not being read")
        if re.search(r"Reference\s+.*undefined", log):
            raise SystemExit(
                "the pdflatex-only build leaves a cross-reference unresolved")
        if pages != reference_pages:
            raise SystemExit(
                f"the staged package typesets {pages} pages against "
                f"{reference_pages} for the full local build")

        out.mkdir(parents=True, exist_ok=True)
        archive = out / f"{STEM}-arxiv.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for relative in sorted(shipped, key=str):
                tar.add(staged / relative, arcname=str(relative))

        abstract = plain_abstract(text)
        (out / "abstract.txt").write_text(_wrap(abstract) + "\n",
                                          encoding="utf-8")
        manifest = [
            f"{hashlib.sha256((staged / relative).read_bytes()).hexdigest()}  "
            f"{relative}" for relative in sorted(shipped, key=str)
        ]
        (out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n",
                                          encoding="utf-8")
        shutil.copy2(verify / f"{STEM}.pdf", out / f"{STEM}.pdf")
        if keep_scratch:
            shutil.copytree(staged, out / "package", dirs_exist_ok=True)

    print(f"{archive} ({archive.stat().st_size} bytes)")
    print(f"  {len(shipped)} files, {pages} pages from pdflatex alone")
    print(f"  abstract {len(abstract)} of {ABSTRACT_LIMIT} characters")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="run the structural checks only; needs no TeX installation")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="directory for the archive, the abstract and the manifest")
    parser.add_argument(
        "--keep-package", action="store_true",
        help="also leave the unpacked staging tree beside the archive")
    args = parser.parse_args(argv)

    text = TEX.read_text(encoding="utf-8")
    if args.check:
        problems = structural_problems(text)
        for problem in problems:
            print(f"FAIL: {problem}")
        if problems:
            print(f"{len(problems)} problem(s)")
            return 1
        staged = _staged_paths(text)
        print(f"arxiv package checks passed "
              f"({len(staged) + 1} files including the .bbl, "
              f"abstract {len(plain_abstract(text))} of "
              f"{ABSTRACT_LIMIT} characters)")
        return 0
    return build(args.out, keep_scratch=args.keep_package)


if __name__ == "__main__":
    raise SystemExit(main())
