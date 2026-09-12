"""The architecture manuscript's gate, at the points where it can go quiet.

A checker that cannot fail protects nothing, and three of these checks were
added because they were absent: the census counted an evidence *role* as a
tier, a changed nested tier moved nothing the census compared, and a value
edited by hand into a generated fragment passed a check that only re-derived
the hashes of that fragment's inputs.  Each test below pins one of those.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_architecture"
sys.path.insert(0, str(PAPER))

import check_manuscript  # noqa: E402
import make_tables  # noqa: E402


# --- an evidence role is not an evidence tier ------------------------------
def test_a_top_level_tier_is_a_tier():
    assert make_tables._declaration_form({"evidence_tier": "exact"}) == "top_level_tier"


def test_a_role_is_not_counted_as_a_tier():
    """`evidence_role` says what a record is for, not what its numbers are.

    Counting it as a tier inflated the coverage the paper reports on itself by
    two records and one distinct tier string.
    """
    assert make_tables._declaration_form(
        {"evidence_role": "scope_decision_only"}) == "role_only"


def test_a_per_quantity_mapping_is_its_own_form():
    assert make_tables._declaration_form(
        {"evidence": {"energies": "exact"}}) == "per_quantity_mapping"


def test_a_nested_tier_is_its_own_form():
    assert make_tables._declaration_form(
        {"protocol": {"evidence_tier": "structural"}}) == "nested_in_subobject"


def test_a_record_with_no_evidence_field_is_unlabelled():
    assert make_tables._declaration_form({"schema": "x"}) == "none"


# --- the census has to notice a declaration changing -----------------------
def test_digest_notices_a_nested_tier_changing():
    before = make_tables._declaration_digest({"protocol": {"evidence_tier": "exact"}})
    after = make_tables._declaration_digest({"protocol": {"evidence_tier": "heuristic"}})
    assert before != after


def test_digest_notices_one_entry_of_a_per_quantity_map_changing():
    before = make_tables._declaration_digest(
        {"evidence": {"energies": "exact", "counts": "exact"}})
    after = make_tables._declaration_digest(
        {"evidence": {"energies": "exact", "counts": "heuristic"}})
    assert before != after


def test_digest_ignores_everything_that_is_not_a_declaration():
    """Otherwise the census would move whenever any record was regenerated."""
    before = make_tables._declaration_digest(
        {"evidence_tier": "exact", "energy": -1.5})
    after = make_tables._declaration_digest(
        {"evidence_tier": "exact", "energy": -2.5})
    assert before == after


# --- the prose scan catches results, not notation --------------------------
def _manuscript(body: str) -> str:
    return "\\begin{document}\n" + body + "\n\\bibliography{references}\n"


def test_prose_scan_rejects_a_three_digit_result():
    """The scan used to start at four digits, which let a setting count pass."""
    found = check_manuscript.prose_numerals(
        _manuscript("The bank compiles to 353 settings."))
    assert [numeral for numeral, _ in found] == ["353"]


def test_prose_scan_rejects_a_two_digit_result():
    found = check_manuscript.prose_numerals(
        _manuscript("It refuses 12 of them."))
    assert [numeral for numeral, _ in found] == ["12"]


def test_prose_scan_allows_conceptual_notation():
    body = ("On the BeH$_2$ bank at $k=8$, in $\\mathrm{Cl}(2n,\\mathbb{C})"
            "\\cong M(2^n,\\mathbb{C})$, with rotors "
            "$\\exp(-\\mathrm{i}\\theta P/2)$ and H$_4$ beside it.")
    assert check_manuscript.prose_numerals(_manuscript(body)) == []


def test_prose_scan_ignores_citation_and_label_digits():
    body = "As shown \\cite{peruzzo2014vqe} in Table~\\ref{tab:layers}."
    assert check_manuscript.prose_numerals(_manuscript(body)) == []


# --- the CI split is read from the workflow, not declared ------------------
def test_ci_gate_sets_separates_dispatch_from_automatic(tmp_path, monkeypatch):
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - run: python benchmarks/check_docs.py\n"
        "  sampled:\n"
        "    if: github.event_name == 'workflow_dispatch'\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - gate: check_protocol_cost\n",
        encoding="utf-8")
    monkeypatch.setattr(make_tables, "CI_WORKFLOW", workflow)
    automatic, dispatch = make_tables.ci_gate_sets()
    assert automatic == {"check_docs"}
    assert dispatch == {"check_protocol_cost"}


# --- the committed fragments are what the generator produces ---------------
def test_committed_fragments_match_a_fresh_generation():
    """A hand edit changes no input hash, so only regeneration catches it."""
    with tempfile.TemporaryDirectory() as scratch:
        tables = Path(scratch) / "tables"
        data = Path(scratch) / "data"
        tables.mkdir(parents=True)
        data.mkdir(parents=True)
        make_tables.main(tables=tables, data=data)
        stale = [regenerated.name for regenerated in sorted(tables.iterdir())
                 if (PAPER / "tables" / regenerated.name).read_bytes()
                 != regenerated.read_bytes()]
        assert stale == []
        committed_census = json.loads((PAPER / "data" / "source_census.json")
                                      .read_text(encoding="utf-8"))
        fresh = json.loads((data / "source_census.json").read_text(encoding="utf-8"))
        assert committed_census == fresh
