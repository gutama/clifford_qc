"""The architecture manuscript's gate, at the points where it can go quiet.

A checker that cannot fail protects nothing.  These tests pin failures already
found in review: roles confused with tiers, nested declarations omitted from a
census, scratch destinations leaking between generator calls, and hand-edited
fragments accepted by a check that only re-derived their input hashes.
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


def test_a_role_with_a_nested_tier_reports_both():
    payload = {
        "evidence_role": "scope_decision_only",
        "protocol": {"evidence_tier": "asymptotic"},
    }
    assert make_tables._declaration_form(payload) == "role_with_nested_tier"


def test_a_per_quantity_mapping_is_its_own_form():
    assert make_tables._declaration_form(
        {"evidence": {"energies": "exact"}}) == "per_quantity_mapping"


def test_a_tier_basis_without_a_tier_is_its_own_form():
    assert make_tables._declaration_form(
        {"evidence_tier_basis": "oracle comparison"}) == "basis_only"


def test_a_record_with_no_evidence_field_is_unlabelled():
    assert make_tables._declaration_form({"schema": "x"}) == "none"


# --- the census has to notice a declaration changing -----------------------
def test_digest_notices_shared_and_producer_specific_evidence_labels():
    for key in (*make_tables.EVIDENCE_KEYS, "search_uncertainty_evidence",
                "reported_evidence_tier"):
        assert make_tables._declaration_digest({key: "oracle"}) != make_tables._declaration_digest({key: "sampled"})


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
    body = ("On BeH$_2$ and H$_{12}$ at $k=8$ and $k=10$, in $\\mathrm{Cl}(2n,\\mathbb{C})"
            "\\cong M(2^n,\\mathbb{C})$, with rotors "
            "$\\exp(-\\mathrm{i}\\theta P/2)$ and H$_4$ beside it.")
    assert check_manuscript.prose_numerals(_manuscript(body)) == []


def test_prose_scan_ignores_citation_and_label_digits():
    body = "As shown \\cite{peruzzo2014vqe} in Table~\\ref{tab:layers}."
    assert check_manuscript.prose_numerals(_manuscript(body)) == []


# --- the CI split is read from the workflow, not declared ------------------
def test_ci_gate_sets_detects_a_same_size_policy_swap(tmp_path, monkeypatch):
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


def test_scratch_generation_restores_default_destinations(tmp_path):
    original = make_tables.TABLES, make_tables.DATA, make_tables.CENSUS
    make_tables.main(tables=tmp_path / "tables", data=tmp_path / "data")
    assert (make_tables.TABLES, make_tables.DATA, make_tables.CENSUS) == original


def test_checker_rejects_a_hand_edited_fragment(tmp_path, monkeypatch, capsys):
    paper = tmp_path / "paper"
    tables = paper / "tables"
    tables.mkdir(parents=True)
    clean_fragment = "generated value: one\n"
    (tables / "numbers.tex").write_text(
        "generated value: two\n", encoding="utf-8")
    census = {"source": {}, "gates": {}, "records": {}}
    census_text = json.dumps(census)
    census_path = paper / "source_census.json"
    census_path.write_text(census_text, encoding="utf-8")
    manuscript = paper / "manuscript.tex"
    manuscript.write_text(
        "\\begin{document}\n"
        + "\n".join(check_manuscript.REQUIRED_PHRASES)
        + "\n\\end{document}\n",
        encoding="utf-8")
    bibliography = paper / "references.bib"
    bibliography.write_text("", encoding="utf-8")

    def generate(*, tables, data):
        (tables / "numbers.tex").write_text(
            clean_fragment, encoding="utf-8")
        (data / census_path.name).write_text(
            census_text, encoding="utf-8")

    monkeypatch.setattr(check_manuscript, "HERE", paper)
    monkeypatch.setattr(check_manuscript, "TEX", manuscript)
    monkeypatch.setattr(check_manuscript, "BIB", bibliography)
    monkeypatch.setattr(check_manuscript, "CENSUS", census_path)
    monkeypatch.setattr(
        check_manuscript, "TABLE_SOURCES", {"numbers.tex": ()})
    monkeypatch.setattr(check_manuscript, "layer_census", lambda: {})
    monkeypatch.setattr(check_manuscript, "gate_census", lambda: {})
    monkeypatch.setattr(check_manuscript, "record_census", lambda: {})
    monkeypatch.setattr(check_manuscript.make_tables, "main", generate)

    assert check_manuscript.main() == 1
    assert "numbers.tex does not match what the generator produces now" in (
        capsys.readouterr().out)
