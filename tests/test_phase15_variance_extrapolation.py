"""The Phase 15 variance-extrapolation producer and checker, run off the declaration.

Nothing here computes a prefix variance on a declared bank. The pipeline runs
on two undeclared toys: LiH CAS(4e,4o) with five generators, inside the
domain, and the Hubbard dimer with three, outside it. The checker must
re-derive every fit, check, status and the verdict on its own, the rule's two
statements must agree everywhere, and each tampered field must be caught.
"""

from __future__ import annotations

import copy
import itertools
import json
import math

import numpy as np
import pytest

from benchmarks import check_phase15_extrapolation_preregistration as gate
from benchmarks import check_phase15_preregistration as preflight_gate
from benchmarks import check_phase15_variance_extrapolation as checker
from benchmarks import run_phase15_variance_extrapolation as producer
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.models import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import SecondMomentBank
from clifford_qc.subspace.generators import identity_generator


def _basis(model, count):
    return model, [identity_generator(model.n), *_raw_pool(model)[:count]]


@pytest.fixture(scope="module")
def toys():
    lih = fcidump_model("benchmarks/data/lih_sto3g_r1.5949_cas4e4o.FCIDUMP")
    return {"lih_toy": _basis(lih, 4), "dimer_toy": _basis(hubbard(2), 2)}


@pytest.fixture(scope="module")
def toy(toys):
    config = copy.deepcopy(gate.load_config())
    validation = {"banks": {}}
    measured = {}
    for name, (model, selected) in toys.items():
        bank = preflight_gate.first_moment_bank(model, selected)
        root = SecondMomentBank(bank).residual(bank.solve(), 0).as_dict()
        validation["banks"][name] = {"roots": [root]}
        measured[name] = gate.quantities_for(name, model, selected, root, None)
    config["banks"]["systems"] = sorted(toys)
    config["measured_before_freezing"] = measured
    config["required_banks"] = sorted(n for n, row in measured.items() if row["in_domain"])
    config["diagnostic_banks"] = sorted(n for n, row in measured.items() if not row["in_domain"])
    raw = json.dumps(config).encode()
    record = producer.run_extrapolation(config, config_bytes=raw, inputs=toys.__getitem__,
                                        validation=validation)
    return config, raw, validation, record


def _all_problems(config, raw, validation, record, inputs, *, recompute=True):
    problems = (checker.declaration_problems(config, record, raw)
                + checker.completeness_problems(config, record))
    if problems:
        return problems
    problems += checker.freeze_problems(config, record)
    problems += checker.rederivation_problems(config, record, validation)
    if recompute:
        problems += checker.recompute_problems(config, record, validation, inputs=inputs)
    return problems


# ------------------------------------------------------------------ end to end

def test_the_toys_split_across_the_domain(toy):
    config, *_ = toy
    assert config["required_banks"] == ["lih_toy"]
    assert config["diagnostic_banks"] == ["dimer_toy"]


def test_the_toy_run_re_derives_and_recomputes_cleanly(toy, toys):
    config, raw, validation, record = toy
    assert record["schema"] == producer.SCHEMA and record["quantum_advantage_claim"] is False
    for entry in record["banks"].values():
        assert all(entry["deterministic_checks"].values())
        assert entry["status"] in gate.BANK_STATUSES
    assert record["banks"]["lih_toy"]["units"] == "hartree"
    assert record["banks"]["dimer_toy"]["units"] == "model"
    assert record["decision"]["verdict"] in gate.VERDICTS
    assert _all_problems(config, raw, validation, record, toys.__getitem__) == []


def test_the_brillouin_window_is_not_extrapolable(toy):
    """Revision 1 on the toy that exposed it: LiH's singles leave the Ritz state
    unchanged, so its window variances differ only by rounding."""
    _, _, _, record = toy
    fit = record["banks"]["lih_toy"]["window"]["fit"]
    assert not fit["extrapolable"] and "only by rounding" in fit["reason"]
    assert record["banks"]["lih_toy"]["status"] == "NOT_EXTRAPOLABLE"


def test_a_subset_without_the_required_bank_is_incomplete(toy, toys):
    config, raw, validation, _ = toy
    partial = producer.run_extrapolation(config, config_bytes=raw, banks=["dimer_toy"],
                                         inputs=toys.__getitem__, validation=validation)
    assert partial["decision"]["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------ the rule, twice

def test_the_two_statements_of_the_fit_agree():
    rng = np.random.default_rng(3)
    for _ in range(200):
        size = int(rng.integers(2, 6))
        variances = sorted(rng.uniform(1e-4, 1e-1, size), reverse=True)
        energies = [-1.0 + rng.normal(0.3, 0.2) * v for v in variances]
        resolved = [bool(rng.random() > 0.1) for _ in range(size)]
        scales = list(rng.choice([1.0, 1e6, 1e11], size))
        mine = gate.extrapolate(variances, energies, resolved=resolved, scales=scales)
        theirs = checker.derive_fit([
            {"variance": v, "energy": e, "resolved": r, "cancellation_scale": s}
            for v, e, r, s in zip(variances, energies, resolved, scales)])
        assert mine["extrapolable"] == theirs["extrapolable"]
        if "intercept" in theirs:
            assert mine["intercept"] == pytest.approx(theirs["intercept"], rel=1e-9, abs=1e-12)


def test_the_two_statements_of_the_ladder_and_verdict_agree():
    config = gate.load_config()
    factor = config["fit"]["improvement_factor"]
    ok = {"check": True}
    for extrapolable, error in itertools.product((True, False), (0.0, 0.4, 0.5, 0.7, 1.0, 1.3, -0.45)):
        fit = {"extrapolable": extrapolable, "extrapolated_error": error}
        assert (checker.derive_status(config, fit, 1.0, ok)
                == gate.status_of(extrapolable, error, 1.0, factor))
    assert checker.derive_status(config, {"extrapolable": True, "extrapolated_error": 0.0},
                                 1.0, {"check": False}) == "INVALID"
    for statuses in itertools.product(gate.BANK_STATUSES + ("INVALID",), repeat=3):
        assert checker.derive_verdict(list(statuses)) == gate.verdict_of(statuses)


# ------------------------------------------------------------------ tampering

def _lih(record):
    return record["banks"]["lih_toy"]


def _dimer(record):
    return record["banks"]["dimer_toy"]


TAMPERS = {
    "a prefix energy": (lambda r: _lih(r)["trajectory"][-1].__setitem__("energy", -9.0),
                        "final_error drifted"),
    "a prefix variance": (lambda r: _lih(r)["trajectory"][-2].__setitem__("variance", 0.5),
                          "window fit refits"),
    "a fitted intercept": (lambda r: r["banks"]["dimer_toy"]["window"]["fit"].__setitem__(
        "intercept", -4.0), "window fit refits"),
    "a status": (lambda r: _lih(r).__setitem__(
        "status", "WORSENS" if _lih(r)["status"] != "WORSENS" else "IMPROVES"),
        "recorded status"),
    "the verdict": (lambda r: r["decision"].__setitem__("verdict", "GO" if r["decision"][
        "verdict"] != "GO" else "NO_GO"), "recorded verdict"),
    "the consequence": (lambda r: r["decision"].__setitem__("consequence", "ship it"),
                        "consequence"),
    "a check flag": (lambda r: _lih(r)["deterministic_checks"].__setitem__(
        "weinstein_interval", False), "recorded deterministic checks"),
    "the final error": (lambda r: _lih(r).__setitem__("final_error", 1.0), "final_error"),
    "the Temple bound": (lambda r: _lih(r)["diagnostics"].__setitem__(
        "temple_lower_bound", 0.0), "Temple"),
    "a role": (lambda r: _lih(r).__setitem__("role", "diagnostic"), "role"),
    "the window": (lambda r: _lih(r)["window"].__setitem__("prefixes", [1, 2, 3]),
                   "window is not"),
    "a frozen number": (lambda r: _lih(r)["measured_at_execution"].__setitem__(
        "exact_gap", 9.0), "differs from the freeze"),
    "an inherited claim boundary": (lambda r: r.__setitem__(
        "claim_boundary", r["preregistration"]["config_claim_boundary_at_landing"]),
        "inherits"),
    "an advantage claim": (lambda r: r.__setitem__("quantum_advantage_claim", True),
                           "quantum_advantage_claim"),
    "a softened factor": (lambda r: r["fit"].__setitem__("improvement_factor", 0.99),
                          "record's fit"),
    "the chemical-accuracy flag": (lambda r: _lih(r)["diagnostics"].__setitem__(
        "reaches_chemical_accuracy_where_final_does_not",
        not _lih(r)["diagnostics"]["reaches_chemical_accuracy_where_final_does_not"]),
        "chemical-accuracy"),
    # Fields the PR #117 review found unchecked.
    "a gain": (lambda r: _dimer(r).__setitem__("gain", 1e6), "gain"),
    "a gain without an extrapolation": (lambda r: _lih(r).__setitem__("gain", 2.0), "gain"),
    "a window residual RMS": (lambda r: _dimer(r)["window"]["fit"].__setitem__(
        "residual_rms", 1e6), "window fit refits"),
    "a diagnostic residual RMS": (lambda r: _dimer(r)["diagnostics"]["window_2"].__setitem__(
        "residual_rms", 1e6), "diagnostic window_2 refits differently"),
    "a fit reason": (lambda r: _lih(r)["window"]["fit"].__setitem__(
        "reason", "the slope is not positive"), "window fit refits"),
    "a residual norm": (lambda r: _lih(r)["trajectory"][0].__setitem__(
        "residual_norm", 2 * _lih(r)["trajectory"][0]["residual_norm"]), "residual_norm"),
    "a resolved flag": (lambda r: _lih(r)["trajectory"][0].__setitem__(
        "resolved", not _lih(r)["trajectory"][0]["resolved"]), "resolved flag"),
    "a nearest-eigenvalue distance": (lambda r: _lih(r)["trajectory"][0].__setitem__(
        "nearest_eigenvalue_distance", 0.1), "nearest-eigenvalue distance"),
    "a Weinstein flag": (lambda r: _lih(r)["trajectory"][0].__setitem__(
        "weinstein_holds", False), "Weinstein flag"),
    "a shrunken cancellation scale": (lambda r: _lih(r)["trajectory"][0].__setitem__(
        "cancellation_scale", 1e-9), "below its own second moment"),
    "a dense difference": (lambda r: _lih(r)["window"]["dense"][0].__setitem__(
        "variance_minus_dense", 1.0), "variance_minus_dense"),
    "a dense prefix": (lambda r: _lih(r)["window"]["dense"][0].__setitem__("prefix", 99),
                       "dense residuals are not the window's"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_each_tamper_is_caught(toy, toys, name):
    config, raw, validation, record = toy
    broken = copy.deepcopy(record)
    edit, expected = TAMPERS[name]
    edit(broken)
    problems = _all_problems(config, raw, validation, broken, toys.__getitem__,
                             recompute=False)
    assert any(expected in problem for problem in problems), problems


def test_a_trajectory_that_does_not_recompute_is_caught(toy, toys):
    config, raw, validation, record = toy
    broken = copy.deepcopy(record)
    broken["banks"]["dimer_toy"]["trajectory"][0]["energy"] += 1e-6
    assert any("recomputes to" in p for p in checker.recompute_problems(
        config, broken, validation, inputs=toys.__getitem__))


def _widen_a_scale(record):
    """The PR #117 review's attack on the toy: a wrong prefix-1 variance, hidden
    behind a cancellation scale widened until the old rebuild tolerance
    ``1e-12 * scale`` covered it, with every derived field made consistent.
    The widened scale also unresolves the row, so the all-prefix diagnostic
    drops it."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    entry = _lih(record)
    row = entry["trajectory"][0]
    row["variance"] += 0.01
    row["cancellation_scale"] = 1e11
    row["residual_norm"] = math.sqrt(row["variance"])
    row["resolved"] = row["variance"] > RESOLUTION * row["cancellation_scale"]
    floor = math.sqrt(RESOLUTION * row["cancellation_scale"])
    row["weinstein_holds"] = (row["nearest_eigenvalue_distance"]
                              <= max(row["residual_norm"], floor) * (1 + 1e-9))
    entry["diagnostics"]["all_resolved"] = producer._all_resolved(entry["trajectory"])


def test_a_widened_scale_cannot_hide_a_wrong_variance_from_the_rebuild(toy, toys):
    config, raw, validation, record = toy
    broken = copy.deepcopy(record)
    _widen_a_scale(broken)
    assert _all_problems(config, raw, validation, broken, toys.__getitem__,
                         recompute=False) == []  # only the rebuild can see it
    problems = checker.recompute_problems(config, broken, validation,
                                          inputs=toys.__getitem__)
    assert any("prefix 1 cancellation_scale, variance" in p for p in problems), problems


def test_units_that_do_not_recompute_are_caught(toy, toys):
    config, raw, validation, record = toy
    broken = copy.deepcopy(record)
    _lih(broken)["units"] = "model"
    assert _all_problems(config, raw, validation, broken, toys.__getitem__,
                         recompute=False) == []
    assert any("units recompute" in p for p in checker.recompute_problems(
        config, broken, validation, inputs=toys.__getitem__))


def test_a_changed_config_breaks_the_digest(toy):
    config, raw, _, record = toy
    assert any("digest" in p for p in checker.declaration_problems(config, record, raw + b" "))


# ------------------------------------------------------------------ the committed record

def test_the_committed_record_re_derives_without_a_rebuild():
    """The five-bank record, re-read without rebuilding a bank."""
    config = gate.load_config()
    record = json.loads(producer.RECORD.read_text(encoding="utf-8"))
    validation = json.loads(gate.VALIDATION.read_text(encoding="utf-8"))
    assert _all_problems(config, gate.CONFIG.read_bytes(), validation, record, None,
                         recompute=False) == []
    assert record["provenance"]["git_dirty"] is False
    assert record["decision"]["verdict"] == "CONDITIONAL"


def test_the_producer_refuses_to_overwrite_the_committed_record(capsys):
    assert producer.main([]) == 1
    assert "runs once" in capsys.readouterr().out


# ------------------------------------------------------------------ refusals

def test_the_producer_refuses_a_subset_on_the_committed_path(capsys):
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
