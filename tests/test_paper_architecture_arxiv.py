"""The arXiv packaging contract, at the points a local build cannot reach.

A manuscript that typesets here can still fail on arXiv, because arXiv builds
it differently: no BibTeX run, a toolchain chosen from the opening lines of the
source, and one flat working directory instead of the repository.  The
packaging script encodes those differences; these tests pin the parts of it
that would otherwise only be exercised by an actual submission.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_architecture"
sys.path.insert(0, str(PAPER))

import make_arxiv  # noqa: E402

MANUSCRIPT = (PAPER / "manuscript.tex").read_text(encoding="utf-8")


def _document(preamble: str = "\\pdfoutput=1\n", abstract: str = "Short.",
              body: str = "\\bibliography{references}\n") -> str:
    return (preamble
            + "\\begin{document}\n"
            + f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n"
            + body
            + "\\end{document}\n")


# --- the real manuscript stays shippable -----------------------------------
def test_the_committed_manuscript_packages_cleanly():
    assert make_arxiv.structural_problems(MANUSCRIPT) == []


def test_every_file_the_manuscript_reads_is_staged():
    """The package is the manuscript's own reads, not a directory sweep.

    A file the manuscript inputs but the stager does not copy is a submission
    that fails to compile on arXiv and nowhere else.
    """
    staged = {str(path) for path in make_arxiv._staged_paths(MANUSCRIPT)}
    read = set(make_arxiv.referenced_inputs(MANUSCRIPT))
    read |= set(make_arxiv.referenced_figures(MANUSCRIPT))
    assert read <= staged
    assert "manuscript.tex" in staged


def test_nothing_unreferenced_is_shipped():
    """Staging a whole directory would publish scratch files and drafts."""
    staged = {str(path) for path in make_arxiv._staged_paths(MANUSCRIPT)}
    assert "references.bib" not in staged
    assert not any(name.endswith(".py") for name in staged)
    assert not any(name.endswith(".json") for name in staged)


# --- the toolchain arXiv picks ---------------------------------------------
def test_a_missing_pdfoutput_is_a_failure():
    problems = make_arxiv.structural_problems(_document(preamble=""))
    assert any("pdfoutput" in problem for problem in problems)


def test_pdfoutput_below_the_scanned_window_does_not_count():
    """arXiv reads the opening lines only, so placement is the whole point."""
    buried = "\n" * make_arxiv.PDFOUTPUT_WINDOW + "\\pdfoutput=1\n"
    problems = make_arxiv.structural_problems(_document(preamble=buried))
    assert any("pdfoutput" in problem for problem in problems)


def test_committed_figures_carry_no_type3_fonts():
    """Type 3 is matplotlib's default and what APS asks authors not to send.

    Two schematics were committed before the generator set a font type and kept
    their Type 3 fonts through every later check, because a figure declaring no
    input sources is bound only to its own hash and to a generator digest that
    had not moved.
    """
    assert make_arxiv.structural_problems(MANUSCRIPT) == []
    for figure in make_arxiv.referenced_figures(MANUSCRIPT):
        data = (PAPER / figure).read_bytes()
        assert not make_arxiv.TYPE3_MARKER.search(data), figure


def test_a_type3_figure_is_reported(tmp_path, monkeypatch):
    figure = tmp_path / "paper_assets" / "drawn.pdf"
    figure.parent.mkdir()
    figure.write_bytes(b"%PDF-1.4\n/Subtype /Type3\n")
    monkeypatch.setattr(make_arxiv, "HERE", tmp_path)

    body = "\\includegraphics{paper_assets/drawn.pdf}\n\\bibliography{references}\n"
    problems = make_arxiv.structural_problems(_document(body=body))
    assert any("Type 3" in problem for problem in problems)


def test_an_eps_figure_contradicts_pdfoutput():
    body = ("\\includegraphics{paper_assets/layer_stack.eps}\n"
            "\\bibliography{references}\n")
    problems = make_arxiv.structural_problems(_document(body=body))
    assert any("pdflatex embeds" in problem for problem in problems)


# --- one flat working directory --------------------------------------------
def test_an_input_outside_the_paper_directory_is_rejected():
    body = "\\input{../paper/tables/headline.tex}\n\\bibliography{references}\n"
    problems = make_arxiv.structural_problems(_document(body=body))
    assert any("outside the paper directory" in problem for problem in problems)


def test_a_missing_input_is_named():
    body = "\\input{tables/not_generated.tex}\n\\bibliography{references}\n"
    problems = make_arxiv.structural_problems(_document(body=body))
    assert any("tables/not_generated.tex" in problem for problem in problems)


def test_an_extensionless_input_resolves_to_tex():
    assert make_arxiv.referenced_inputs(
        "\\input{tables/numbers}") == ["tables/numbers.tex"]


def test_a_commented_out_input_is_not_packaged():
    assert make_arxiv.referenced_inputs("% \\input{tables/dropped}") == []


# --- the abstract the submission form receives -----------------------------
def test_the_abstract_is_plain_text():
    abstract = make_arxiv.plain_abstract(MANUSCRIPT)
    assert "clifford_qc" in abstract
    assert "\\" not in abstract
    assert "\n" not in abstract


def test_a_macro_left_in_the_abstract_is_reported():
    """It would be posted literally, where no rebuild of the paper shows it."""
    problems = make_arxiv.structural_problems(
        _document(abstract="A \\ref{tab:layers} reference."))
    assert any("posted literally" in problem for problem in problems)


def test_an_overlong_abstract_is_reported():
    problems = make_arxiv.structural_problems(
        _document(abstract="word " * (make_arxiv.ABSTRACT_LIMIT // 2)))
    assert any("form limit" in problem for problem in problems)


def test_a_manuscript_without_a_bibliography_is_reported():
    problems = make_arxiv.structural_problems(_document(body=""))
    assert any("does not run BibTeX" in problem for problem in problems)


# --- the .bbl is the one shipped file nothing else produces ----------------
def test_the_committed_bibliography_is_parseable():
    assert make_arxiv.bibliography_problems() == []


def test_an_at_sign_in_a_bib_comment_is_reported(tmp_path, monkeypatch):
    """BibTeX starts an entry at an at-sign, comment or not.

    A header comment mentioning an entry type truncated the entry after it and
    cost a build; the manuscript gate cannot see it, because it matches entry
    keys and a comment has none.
    """
    bibliography = tmp_path / "references.bib"
    bibliography.write_text(
        "% theirs carry this as an @article entry\n"
        "@misc{key, title = {A title}}\n", encoding="utf-8")
    monkeypatch.setattr(make_arxiv, "BIB", bibliography)

    problems = make_arxiv.bibliography_problems()
    assert any("at-sign" in problem for problem in problems)


def test_a_plain_bib_comment_is_fine(tmp_path, monkeypatch):
    bibliography = tmp_path / "references.bib"
    bibliography.write_text(
        "% shared entries are copied verbatim\n"
        "@misc{key, title = {A title}}\n", encoding="utf-8")
    monkeypatch.setattr(make_arxiv, "BIB", bibliography)

    assert make_arxiv.bibliography_problems() == []
