"""R1 device-card, fidelity, coverage, and break-even contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from clifford_qc.measurement.cost import (
    DeviceCard,
    SettingResources,
    break_even_surface,
    cost_schedule,
    estimator_information,
    inflate_shots_for_fidelity,
    setting_duration_us,
    setting_fidelity,
)


CARDS = Path(__file__).parents[1] / "benchmarks" / "configs" / "device_cards"


def test_all_declared_device_cards_load_and_hash_stably():
    paths = sorted(CARDS.glob("*.json"))
    assert [path.stem for path in paths] == [
        "ion-like", "logical-alltoall", "superconducting-like"
    ]
    cards = [DeviceCard.load(path) for path in paths]
    assert len({card.sha256 for card in cards}) == len(cards)
    assert DeviceCard.from_dict(cards[0].to_dict()).sha256 == cards[0].sha256
    assert all("vendor" not in card.calibration_status for card in cards)


def test_device_card_rejects_unknown_and_unphysical_fields():
    card = DeviceCard.load(CARDS / "logical-alltoall.json").to_dict()
    with pytest.raises(ValueError, match="unknown"):
        DeviceCard.from_dict({**card, "secret_default": 1})
    with pytest.raises(ValueError, match="eps_2q"):
        DeviceCard.from_dict({**card, "eps_2q": 1.0})
    with pytest.raises(TypeError, match="routing"):
        DeviceCard.from_dict({**card, "routing": 0})


def test_setting_time_and_fidelity_follow_the_declared_formula():
    card = DeviceCard.load(CARDS / "superconducting-like.json")
    setting = SettingResources(n_1q=3, n_2q=2, d_1q=2, d_2q=1)
    assert setting_duration_us(card, setting) == pytest.approx(
        1.0 + 0.05 * 2 + 0.3 * 2 + 1.0 + 1.0
    )
    assert setting_fidelity(card, setting, 4) == pytest.approx(
        0.999**3 * 0.99**6 * 0.98**4
    )


def test_fidelity_inflation_delivers_at_least_the_requested_effective_shots():
    card = DeviceCard.load(CARDS / "superconducting-like.json")
    settings = [
        SettingResources(1, 0, 1, 0),
        SettingResources(2, 3, 1, 2),
    ]
    raw = inflate_shots_for_fidelity(card, settings, 1000, 4)
    ledger = cost_schedule(
        card, settings, raw, n_qubits=4, evidence_tier="exact", epsilon=0.0016
    )
    assert ledger["effective_shots"] >= 2000
    assert ledger["accuracy"]["status"] == "priced"
    assert ledger["accuracy"]["C_time_epsilon_us"] == ledger["fixed_shot_time_us"]


def test_fixed_shot_cost_does_not_masquerade_as_c_epsilon():
    card = DeviceCard.load(CARDS / "logical-alltoall.json")
    ledger = cost_schedule(
        card, [SettingResources(1, 0, 1, 0)], 100,
        n_qubits=2, evidence_tier="exact",
    )
    assert ledger["fixed_shot_time_us"] == pytest.approx(205.0)
    assert ledger["accuracy"] == {
        "epsilon": None,
        "evidence_tier": "exact",
        "status": "fixed_shot_only",
        "C_time_epsilon_us": None,
    }


def test_inadmissible_setting_abstains_from_accuracy_matched_cost():
    card = replace(
        DeviceCard.load(CARDS / "logical-alltoall.json"),
        eps_2q=0.2,
        fidelity_floor=0.9,
    )
    ledger = cost_schedule(
        card, [SettingResources(0, 4, 0, 3)], 100,
        n_qubits=2, evidence_tier="exact", epsilon=0.0016,
    )
    assert ledger["admissible"] is False
    assert ledger["accuracy"]["status"] == "inadmissible"
    assert ledger["accuracy"]["C_time_epsilon_us"] is None


def test_pooled_estimator_uses_every_compatible_setting_on_the_same_bank():
    compatibility = [
        [True, True, False],
        [False, True, True],
    ]
    assignment = [0, 0, 1]
    assigned = estimator_information(
        compatibility, assignment, [100, 200], estimator="single_assignment",
        fidelities=[1.0, 0.5],
    )
    pooled = estimator_information(
        compatibility, assignment, [100, 200], estimator="pooled",
        fidelities=[1.0, 0.5],
    )
    assert assigned["compatible_settings_per_word"] == {
        "min": 1, "mean": 1.0, "max": 1
    }
    assert pooled["compatible_settings_per_word"] == {
        "min": 1, "mean": 4 / 3, "max": 2
    }
    assert assigned["raw_word_shots"]["total"] == 400
    assert pooled["raw_word_shots"]["total"] == 600
    assert pooled["effective_word_shots"]["total"] == pytest.approx(300)


def test_break_even_surface_reports_admissibility_and_named_card():
    card = DeviceCard.load(CARDS / "logical-alltoall.json")
    qwc = [SettingResources(1, 0, 1, 0)] * 4
    deep = [SettingResources(2, 2, 1, 1)]
    surface = break_even_surface(
        card, qwc, deep, 100, n_qubits=4,
        t_2q_ratios=[0.0, 2.0], eps_2q_values=[0.0, 0.4],
        evidence_tier="exact",
    )
    assert surface["base_device_card"]["name"] == "logical-alltoall"
    assert len(surface["points"]) == 4
    assert {point["winner"] for point in surface["points"]} <= {
        "reference", "candidate", "tie", "neither"
    }


def test_break_even_surface_treats_roundoff_scale_time_difference_as_tie():
    card = replace(
        DeviceCard.load(CARDS / "logical-alltoall.json"),
        t_1q_us=0.1,
        t_readout_us=0.1,
        t_reset_us=0.2,
    )
    reference = [SettingResources(15, 0, 15, 0)]
    candidate = [
        SettingResources(12, 0, 12, 0),
        SettingResources(0, 0, 0, 0),
    ]
    surface = break_even_surface(
        card, reference, candidate, 1, n_qubits=1,
        t_2q_ratios=[0.0], eps_2q_values=[0.0], evidence_tier="exact",
    )
    point = surface["points"][0]
    assert point["reference_time_us"] != point["candidate_time_us"]
    assert point["winner"] == "tie"
