"""The folded-spectrum validation producer and checker, on an undeclared toy bank."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks import check_folded_spectrum_validation as checker
from benchmarks import run_folded_spectrum_validation as producer
from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace.generators import identity_generator

TOY = "hubbard_2x2"  # a licensed name, bound here to the undeclared 1x3 chain


def _toy_inputs(name):
    assert name == TOY
    model = hubbard(3)
    return model, [identity_generator(model.n), *_raw_pool(model)[:5]]


@pytest.fixture(scope="module")
def toy():
    preflight = json.loads(producer.PREFLIGHT.read_bytes())
    record = producer.run_validation(preflight, banks=[TOY], inputs=_toy_inputs,
                                     preflight_bytes=producer.PREFLIGHT.read_bytes())
    return json.loads(json.dumps(record, allow_nan=False))


def test_the_toy_bank_validates_and_restates(toy):
    entry = toy["banks"][TOY]
    assert entry["status"] == "VALIDATED" and all(entry["checks"].values())
    assert toy["summary"]["all_validated"] is False  # one licensed bank of five
    assert checker.derived_problems(toy) == []
    assert checker.recompute_problems(toy, inputs=_toy_inputs) == []


def test_refinement_and_minimization_tighten_the_toy_ground_interval(toy):
    entry = toy["banks"][TOY]
    reading = entry["reading"]
    assert reading["ground_refined_target_bound"] < reading["ground_ritz_residual_norm"]
    assert reading["minimized_residual_norm"] <= reading["ground_refined_target_bound"] + 1e-12
    assert entry["minimization"]["converged"] and entry["minimization"]["monotone"]


def test_the_record_carries_v1_evidence(toy):
    from clifford_qc.evidence import validate_evidence

    validate_evidence(toy)
    assert toy["quantum_advantage_claim"] is False


@pytest.mark.parametrize("path, value", [
    (("banks", TOY, "roots", 0, "isolated_level"), 99),
    (("banks", TOY, "roots", 1, "target_levels"), []),
    (("banks", TOY, "roots", 0, "refined_not_above_ritz"), False),
    (("banks", TOY, "roots", 0, "ritz_variance"), -1.0),
    (("banks", TOY, "roots", 2, "matches_dense"), False),
    (("banks", TOY, "roots", 0, "weinstein_interval"), [-100.0, 100.0]),
    (("banks", TOY, "minimization", "converged"), False),
    (("banks", TOY, "minimization", "iterations"), 99),
    (("banks", TOY, "checks", "sector_premise"), False),
    (("banks", TOY, "status"), "FAILED"),
    (("banks", TOY, "reading", "isolated_levels"), [0, 1, 2]),
    (("summary", "all_validated"), True),
])
def test_each_forged_field_is_caught(toy, path, value):
    forged = copy.deepcopy(toy)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert checker.derived_problems(forged)


def test_a_self_consistent_forgery_fails_the_rebuild(toy):
    forged = copy.deepcopy(toy)
    row = forged["banks"][TOY]["roots"][0]
    row["folded_value"] *= 0.25
    row["dense_folded_value"] = row["folded_value"]
    bound = (max(row["folded_value"], 0.0) + 1e-12 * row["folded_scale"]) ** 0.5
    row["target_interval"] = [row["target"] - bound, row["target"] + bound]
    levels = forged["banks"][TOY]["sector"]["levels"]
    row["target_levels"] = checker._inside(levels, row["target_interval"])
    forged["banks"][TOY]["reading"]["ground_refined_target_bound"] = row["folded_value"] ** 0.5
    assert any("folded_value" in p for p in
               checker.recompute_problems(forged, inputs=_toy_inputs))


def test_unlicensed_banks_and_subsets_are_refused(tmp_path):
    preflight = json.loads(producer.PREFLIGHT.read_bytes())
    with pytest.raises(ValueError, match="does not license"):
        producer.run_validation(preflight, banks=["lih"], inputs=_toy_inputs)

    class Args:
        out = producer.RECORD
        overwrite = False

    assert producer.refusals(Args(), reduced=True)
    Args.out = tmp_path / "elsewhere.json"
    assert producer.refusals(Args(), reduced=True) == []
