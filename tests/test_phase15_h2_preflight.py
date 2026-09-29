"""The Phase 15 preflight producer and result checker, run without the preflight.

Nothing here forms H² on a declared bank. The end-to-end tests count two banks
on the 2-site Hubbard dimer, which is outside the declaration, under the frozen
rules. The support counts are held to an independent dense Pauli
decomposition, the checker must re-derive the producer's decision on its own,
and each tampered record must be caught by the check written for it.
"""

from __future__ import annotations

import copy
import itertools
import json

import numpy as np
import pytest

from benchmarks import check_phase15_h2_preflight as checker
from benchmarks import check_phase15_preregistration as gate
from benchmarks import run_phase15_h2_preflight as producer
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.subspace.contracts import as_multivector
from clifford_qc.subspace.generators import identity_generator


@pytest.fixture(scope="module")
def dimer():
    model = hubbard(2)
    raw = _raw_pool(model)
    identity = identity_generator(model.n)
    return {"toy": (model, [identity, *raw]), "toy_small": (model, [identity, raw[0]])}


def _toy_config(dimer, *, anchor=None):
    config = copy.deepcopy(gate.load_config())
    config["banks"]["systems"] = sorted(dimer)
    if anchor is not None:
        config["clauses"]["storage"]["max_coefficient_occurrences"] = anchor
    config["measured_before_freezing"] = {
        name: gate.quantities_for(config, *inputs) for name, inputs in dimer.items()}
    return config


@pytest.fixture(scope="module")
def toy(dimer):
    config = _toy_config(dimer)
    raw = json.dumps(config).encode()
    record = producer.run_preflight(config, config_bytes=raw, inputs=dimer.__getitem__)
    return config, raw, record


def _all_problems(config, raw, record, inputs, *, recount=True):
    problems = (checker.declaration_problems(config, record, raw)
                + checker.completeness_problems(config, record))
    if problems:
        return problems
    problems += checker.freeze_problems(config, record)
    problems += checker.arithmetic_problems(record)
    problems += checker.rule_problems(config, record)
    problems += checker.dense_problems(config, record, inputs=inputs)
    if recount:
        problems += checker.recount_problems(config, record, inputs=inputs)
    return problems


# ------------------------------------------------------------------ end to end

def test_the_toy_run_re_derives_and_recounts_cleanly(toy, dimer):
    config, raw, record = toy
    assert record["schema"] == producer.SCHEMA
    assert record["quantum_advantage_claim"] is False
    for entry in record["banks"].values():
        assert all(entry["deterministic_checks"].values())
        assert entry["decision"]["status"] == "ELIGIBLE"
    assert record["decision"]["verdict"] == "FULL"
    assert _all_problems(config, raw, record, dimer.__getitem__) == []


def test_support_counts_match_an_independent_dense_decomposition(dimer):
    """Pauli coefficients of each dense A_i† H² A_j, with no multivector product."""
    model, selected = dimer["toy"]
    config = _toy_config(dimer)
    counted = producer.count_bank(config, model, selected)
    n = model.n
    h = to_matrix(as_multivector(model.hamiltonian))
    words = {code: to_matrix(MV(n, {code: 1.0})) for code in range(4 ** n)}
    strict = config["pruning"]["strict_tolerance"]
    support, occurrences = set(), 0
    for j, right in enumerate(selected):
        column = h @ h @ to_matrix(right.mv)
        for i in range(j + 1):
            row = to_matrix(selected[i].mv).conj().T @ column
            kept = {code for code, word in words.items()
                    if abs(np.trace(word @ row)) / 2 ** n > strict}
            support |= kept
            occurrences += len(kept)
    assert counted["counts"]["k_word_universe_strict"] == len(support)
    assert counted["counts"]["k_coefficient_occurrences_strict"] == occurrences
    bank = gate.first_moment_bank(model, selected)
    union = support | set(bank.word_set())
    assert counted["digests"]["combined_strict_words_sha256"] == producer._digest(union)


def test_a_bank_over_the_anchor_restricts_the_verdict(dimer):
    config = _toy_config(dimer)
    totals = {name: producer.count_bank(config, *inputs)["counts"][
        "total_coefficient_occurrences"] for name, inputs in dimer.items()}
    anchor = min(totals.values())
    assert anchor < max(totals.values())
    config = _toy_config(dimer, anchor=anchor)
    raw = json.dumps(config).encode()
    record = producer.run_preflight(config, config_bytes=raw, inputs=dimer.__getitem__)
    statuses = record["decision"]["bank_statuses"]
    assert statuses == {"toy": "STORAGE_PROHIBITIVE", "toy_small": "ELIGIBLE"}
    assert record["decision"]["verdict"] == "RESTRICTED"
    assert record["decision"]["consequence"] == config["consequences"]["RESTRICTED"]
    assert _all_problems(config, raw, record, dimer.__getitem__, recount=False) == []


# ------------------------------------------------------------------ the rule

def _counts(sh, combined, combined_strict, total, total_strict):
    return {"sh_word_universe": sh, "combined_word_universe": combined,
            "combined_word_universe_strict": combined_strict,
            "total_coefficient_occurrences": total,
            "total_coefficient_occurrences_strict": total_strict}


def test_producer_gate_and_checker_agree_on_every_clause_outcome():
    config = gate.load_config()
    ratio = config["clauses"]["word"]["max_ratio"]
    anchor = config["clauses"]["storage"]["max_coefficient_occurrences"]
    ok = {"check": True}
    seen = set()
    for word, word_s, storage, storage_s in itertools.product((False, True), repeat=4):
        if (word and not word_s) or (storage and not storage_s):
            continue  # strict pruning only removes terms
        counts = _counts(100, 100 * ratio + (not word), 100 * ratio + (not word_s),
                         anchor + (not storage), anchor + (not storage_s))
        mine = checker.derive_status(config, counts, ok)
        assert mine == producer.evaluate(config, counts, ok)["status"]
        assert mine == gate.status_of((word, storage), (word_s, storage_s))
        seen.add(mine)
    assert seen == checker.STATUSES - {"INVALID"}
    assert producer.evaluate(config, counts, {"check": False})["status"] == "INVALID"


def test_verdicts_agree_between_checker_and_gate():
    for statuses in itertools.product(sorted(checker.STATUSES), repeat=3):
        assert checker.derive_verdict(list(statuses)) == gate.verdict_of(statuses)


# ------------------------------------------------------------------ tampering

def _bank(record, name="toy"):
    return record["banks"][name]


TAMPERS = {
    "a row count": (lambda r: _bank(r)["rows"][0].__setitem__(2, 999), "the rows give"),
    "a verdict": (lambda r: r["decision"].__setitem__("verdict", "RESTRICTED"),
                  "recorded verdict"),
    "a status": (lambda r: _bank(r)["decision"].__setitem__("status", "WORD_PROHIBITIVE"),
                 "recorded status"),
    "an impossible union": (
        lambda r: _bank(r)["counts"].__setitem__("combined_word_universe", 10 ** 6),
        "is not a union"),
    "an inherited claim boundary": (
        lambda r: r.__setitem__("claim_boundary",
                                r["preregistration"]["config_claim_boundary_at_landing"]),
        "inherits the config's claim boundary"),
    "an advantage claim": (
        lambda r: r.__setitem__("quantum_advantage_claim", True), "quantum_advantage_claim"),
    "a frozen number moved": (
        lambda r: _bank(r)["measured_at_execution"].__setitem__("block_pairs", 3),
        "not the frozen"),
    "a pairing": (lambda r: _bank(r)["rows"][1].__setitem__(4, 1e3),
                  "from the dense second moment"),
    "a digest": (lambda r: _bank(r)["digests"].__setitem__("k_words_sha256", "0" * 64),
                 "digest recounts differently"),
    "a dropped row": (lambda r: _bank(r)["rows"].pop(), "exactly once"),
    "a failed check left unenforced": (
        lambda r: _bank(r)["deterministic_checks"].__setitem__("gram_psd", False),
        "not what their values give"),
    "a strict count above the declared": (
        lambda r: _bank(r)["counts"].__setitem__("h2_terms_strict", 10 ** 6),
        "strict h2_terms exceeds"),
    "a misquoted consequence": (
        lambda r: r["decision"].__setitem__("consequence", "build everywhere"),
        "consequence"),
    "a softened clause": (
        lambda r: r["clauses"]["storage"].__setitem__("max_coefficient_occurrences", 10 ** 12),
        "record's clauses is not the config's"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_tamper_is_caught(toy, dimer, name):
    config, raw, record = toy
    broken = copy.deepcopy(record)
    edit, expected = TAMPERS[name]
    edit(broken)
    problems = _all_problems(config, raw, broken, dimer.__getitem__)
    assert any(expected in problem for problem in problems), problems


def test_a_changed_config_breaks_the_digest(toy):
    config, raw, record = toy
    assert any("digest" in problem for problem in checker.declaration_problems(
        config, record, raw + b" "))


# ------------------------------------------------------------------ refusals

def test_the_producer_refuses_a_bank_subset_on_the_committed_path(capsys):
    assert producer.main(["--banks", "beh2"]) == 1
    assert "bank subset" in capsys.readouterr().out


def test_the_producer_refuses_to_overwrite_a_record(tmp_path, monkeypatch, capsys):
    existing = tmp_path / "record.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(producer, "RECORD", existing)
    assert producer.main(["--out", str(existing)]) == 1
    assert "runs once" in capsys.readouterr().out


def test_the_checker_fails_without_a_record(tmp_path, capsys):
    assert checker.main(["--record", str(tmp_path / "absent.json")]) == 1
    assert "missing record" in capsys.readouterr().out
