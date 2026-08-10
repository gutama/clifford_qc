"""Unit tests for the preconditioned residual-expansion driver.

These test the *claims the driver is allowed to make*, not just that it runs:
that unpreconditioned expansion really is Lanczos, that the shift is chosen
without the exact ground energy, that budget matching is enforced rather than
asserted in prose, and that the invariant gate actually rejects the failures it
advertises.
"""

import json

import numpy as np
import pytest

from benchmarks import run_preconditioned_expansion as pe


def _toy():
    """A small real symmetric Hamiltonian with a non-degenerate ground state."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(12, 12))
    matrix = 0.5 * (matrix + matrix.T)
    matrix += np.diag(np.arange(12, dtype=float))

    def apply(vector):
        return matrix @ vector

    unit = np.zeros(12, dtype=complex)
    unit[0] = 1.0
    return matrix, apply, unit


def test_orthogonal_residual_spans_the_krylov_space():
    """The theorem the driver refuses to spend a research phase on.

    ``r_m = (H - E_m) p_{m-1}(H)|psi>`` lies in ``K_{m+1}``, so residual
    expansion and power Krylov must reach the same Ritz value -- with the
    orthogonal arm no worse, because it solves the same span at ``S = I``.
    """
    matrix, apply, unit = _toy()
    for M in (3, 5, 7):
        residual = pe.expand(apply, unit, M)
        krylov = pe.power_krylov(apply, unit, M)
        assert residual["M"] == krylov["M"] == M
        # Same span, so the two Ritz values may differ only by what the raw
        # monomial basis loses to its own conditioning.  The bound is two-sided
        # on purpose: in exact arithmetic the orthogonal arm is the sharper
        # value, but the ill-conditioned arm's roundoff can put it a hair
        # lower, and asserting the strict inequality would fail on that.
        bound = max(1e-8, np.finfo(float).eps * krylov["kappa_S"]
                    * max(1.0, abs(krylov["energy"])))
        assert abs(residual["energy"] - krylov["energy"]) <= bound


def test_expansion_holds_the_overlap_matrix_at_identity():
    _, apply, unit = _toy()
    run = pe.expand(apply, unit, 6)
    assert run["orthonormality_defect"] < 1e-12
    assert run["kappa_S"] == pytest.approx(1.0, abs=1e-8)


def test_nested_ritz_energies_are_non_increasing():
    _, apply, unit = _toy()
    for kwargs in ({}, {"diagonal": np.arange(12, dtype=float), "mu": 0.5}):
        curve = pe.expand(apply, unit, 8, **kwargs)["energies"]
        assert all(later <= earlier + 1e-12
                   for earlier, later in zip(curve, curve[1:]))


def test_preconditioning_beats_the_unpreconditioned_arm_at_matched_M():
    """Davidson is the principal method precisely because of this inequality."""
    matrix, apply, unit = _toy()
    exact = float(np.linalg.eigvalsh(matrix)[0])
    diagonal = np.real(np.diag(matrix))
    plain = pe.expand(apply, unit, 5)
    sweep = pe.select_shift(apply, unit, 5, diagonal, pe.DEFAULT_SHIFT_GRID)
    davidson = pe.expand(apply, unit, 5, diagonal=diagonal,
                         mu=sweep["selected_mu"])
    assert abs(davidson["energy"] - exact) < abs(plain["energy"] - exact)


def test_full_K_packet_reproduces_the_uncompressed_direction():
    matrix, apply, unit = _toy()
    diagonal = np.real(np.diag(matrix))
    full = pe.expand(apply, unit, 6, diagonal=diagonal, mu=0.4, packet_K=12)
    plain = pe.expand(apply, unit, 6, diagonal=diagonal, mu=0.4)
    assert full["energy"] == pytest.approx(plain["energy"], abs=1e-12)


def test_a_truncated_run_is_internally_consistent():
    """Truncation changes the trajectory, so no cross-run ordering holds.

    Top-K truncation changes the next Ritz state and therefore every later
    correction, so the truncated and uncompressed final subspaces are not
    nested and a truncated run can legitimately end lower at fixed M. What must
    hold is what follows from each run's *own* nested basis: variationality
    against the exact ground energy, a monotone energy curve, and S = I.
    """
    matrix, apply, unit = _toy()
    exact = float(np.linalg.eigvalsh(matrix)[0])
    diagonal = np.real(np.diag(matrix))
    for K in (2, 4, 8):
        run = pe.expand(apply, unit, 6, diagonal=diagonal, mu=0.4, packet_K=K)
        assert run["energy"] >= exact - 1e-12
        assert all(later <= earlier + 1e-12
                   for earlier, later in zip(run["energies"],
                                             run["energies"][1:]))
        assert run["orthonormality_defect"] < 1e-10


def test_every_realized_packet_is_retained_for_pricing():
    """Pricing needs the whole trajectory, not the first step's packet."""
    _, apply, unit = _toy()
    diagonal = np.real(np.diag(matrix_of := np.eye(12)) * 0 + 1.0)
    run = pe.expand(apply, unit, 5, diagonal=diagonal, mu=0.4, packet_K=3)
    assert len(run["packet_supports"]) == run["M"] - 1
    assert len(run["packet_coefficients"]) == len(run["packet_supports"])
    for support, coefficients in zip(run["packet_supports"],
                                     run["packet_coefficients"]):
        assert len(support) == len(coefficients)


# ------------------------------------------------- packet coefficient alignment


def test_packet_generators_drop_the_reference_from_masks_and_coefficients():
    """Regression: the reference is mid-list, so end-truncation misaligns.

    ``_packet_generators`` drops the reference determinant because it enters as
    the identity. Filtering the generators while truncating the coefficient
    array from the end shifts every coefficient after the reference onto the
    wrong generator -- which is what happened in the first committed Hubbard
    2x2 K=32 packet, where the reference sat at index 22 of 32.
    """
    n = 4
    reference = (0, 1)
    reference_mask = 0b1100                      # qubits 0 and 1 occupied
    masks = [0b1010, reference_mask, 0b0101]     # reference in the middle
    coefficients = [10.0, 99.0, 30.0]

    pairs = pe._packet_generators(reference, masks, n, "pkt", coefficients)

    assert len(pairs) == 2                       # the reference is dropped
    assert [weight for _, weight in pairs] == [10.0, 30.0]
    # The labels carry the original positions, so the survivors are 0 and 2 --
    # never 0 and 1, which is what an end-truncated array would have produced.
    assert [generator.label for generator, _ in pairs] == ["pkt[0]", "pkt[2]"]


def test_packet_generators_reject_a_coefficient_count_mismatch():
    with pytest.raises(ValueError, match="same order"):
        pe._packet_generators((0, 1), [0b1010, 0b0101], 4, "pkt", [1.0])


def test_a_compiled_packet_uses_each_generator_own_coefficient():
    n = 4
    reference = (0, 1)
    masks = [0b1010, 0b1100, 0b0101]             # middle entry is the reference
    pairs = pe._packet_generators(reference, masks, n, "pkt",
                                  [2.0, 99.0, 5.0])
    packet = pe._compile_packet(pairs, "packet")
    separate = pe._compile_packet([(pairs[0][0], 2.0)], "a")
    other = pe._compile_packet([(pairs[1][0], 5.0)], "b")
    assert set(packet.mv.terms) == set(separate.mv.terms) | set(other.mv.terms)


def test_shift_selection_never_consults_the_exact_ground_energy():
    """The criterion is the projected Ritz energy, which is variational.

    The selected shift must be the minimiser of the curve the record retains,
    so the choice can be re-derived from the artifact without rerunning it.
    """
    matrix, apply, unit = _toy()
    diagonal = np.real(np.diag(matrix))
    sweep = pe.select_shift(apply, unit, 5, diagonal, pe.DEFAULT_SHIFT_GRID)

    assert sweep["oracle_free"] is True
    assert "projected" in sweep["selection_criterion"]
    live = [point for point in sweep["curve"]
            if point["projected_energy"] is not None]
    assert sweep["selected_mu"] == min(
        live, key=lambda point: point["projected_energy"])["mu"]
    assert len(sweep["curve"]) == len(pe.DEFAULT_SHIFT_GRID)  # full curve kept


def test_a_singular_denominator_is_reported_not_clipped():
    _, apply, unit = _toy()
    # Put the whole diagonal exactly on the first Ritz energy, so the mu = 0
    # denominator vanishes identically rather than merely getting small.
    seed = pe.expand(apply, unit, 1)
    diagonal = np.full(12, seed["energy"])
    run = pe.expand(apply, unit, 4, diagonal=diagonal, mu=0.0)
    assert run.get("breakdown") == "singular_denominator"


def test_happy_breakdown_stops_growth_and_says_so():
    """A closed span must terminate with a named reason, not a silent stop."""
    matrix = np.diag([1.0, 2.0, 3.0])
    matrix[0, 1] = matrix[1, 0] = 0.3

    def apply(vector):
        return matrix @ vector

    unit = np.array([1.0, 0.0, 0.0], dtype=complex)
    run = pe.expand(apply, unit, 3)
    assert run["M"] == 2                       # the third direction is unreachable
    assert run["stopped_reason"] == "happy_breakdown"


def test_top_k_packet_keeps_the_largest_components():
    vector = np.array([0.1, -0.9, 0.4, 0.05, -0.6], dtype=complex)
    packet, support = pe._top_k_packet(vector, 2)
    assert sorted(support.tolist()) == [1, 4]
    assert packet[0] == 0 and packet[2] == 0 and packet[3] == 0


# --------------------------------------------------------------- cost charging


def test_shift_selection_is_charged_to_the_row_that_spends_it():
    """Audit and block rows selected a shift without paying for it.

    ``select_shift`` was called outside those rows' timing window, so they
    reported ``selection_work = 9`` beside a matvec count covering only the
    final expansion -- a tenth of the true cost, and not comparable with the
    primary Davidson row whose sweep was charged.
    """
    matrix, apply, unit = _toy()
    diagonal = np.real(np.diag(matrix))

    class Counter:
        def __init__(self):
            self.matvecs = 0

        def __call__(self, vector):
            self.matvecs += 1
            return apply(vector)

    counting = Counter()
    row = {}
    sweep = pe._charged_shift_sweep(counting, unit, 5, diagonal,
                                    pe.DEFAULT_SHIFT_GRID, row)

    assert row["mu"] == sweep["selected_mu"]
    assert row["selection_work"] == len(pe.DEFAULT_SHIFT_GRID)
    # The sweep runs one expansion per grid point, so its matvec bill dwarfs
    # the single expansion that follows it.
    assert row["_charged_matvecs"] == counting.matvecs
    assert row["_charged_matvecs"] > 0
    assert row["_charged_seconds"] >= 0.0


def test_a_charged_sweep_is_folded_into_the_row_and_consumed_once():
    """The charge is popped, so a reused row cannot be billed twice."""
    row = {"_charged_matvecs": 100, "_charged_seconds": 1.0,
           "reference_block_size": 1}
    finished = pe._finish_row(
        dict(row), {"energy": -1.0, "M": 3, "energies": [-1.0],
                    "stopped_reason": "budget_exhausted"},
        -1.0, matvecs=7, wall=0.5)
    assert finished["matvecs"] == 107
    assert finished["wall_seconds"] == pytest.approx(1.5)
    assert "_charged_matvecs" not in finished


# ------------------------------------------------------------------ invariants


def _record(**overrides):
    """A minimal record that passes every invariant, for negative testing."""
    def row(method, category="exact_simulation", **extra):
        base = {field: None for field in pe.REQUIRED_FIELDS}
        base.update({
            "method": method, "evidence_category": category, "seed": 0,
            "implementable": False, "preconditioner_category": "none",
            "M": 3, "declared_total_directions": 3,
            "realized_total_directions": 3, "stopped_reason": "budget_exhausted",
            "nested_energies": [-0.5, -0.9, -1.0], "reference_policy": "p",
            "reference_identity": "p:mask=3", "reference_block_size": 1,
            "shift_rule": "none", "energy": -1.0, "energy_error": 0.0,
            "absolute_error": 0.0, "orthonormality_defect": 0.0,
            "kappa_S": 1.0, "selection_work": 0})
        base.update(extra)
        return base

    record = {
        "exact_energy": -1.0,
        "total_directions": 3,
        "spin_ordering": "interleaved",
        "sector_masks": [3, 5, 6],
        "arms": [row("power_krylov"), row("orthonormalized_power_krylov"),
                 row("orthogonal_residual"), row("davidson"),
                 row("matched_selected_ci", "classical_control")],
        "reference_audit": [],
        "reference_blocks": [],
        "oracle_diagnostics": [],
        "shift_sweep": {
            "oracle_free": True,
            "selection_criterion": "min projected Ritz energy (variational)",
            "selected_mu": 0.1,
            "curve": [{"mu": 0.1, "projected_energy": -1.0},
                      {"mu": 0.5, "projected_energy": -0.8}]},
        "krylov_agreement": {
            "orthonormalized_energy": -1.0,
            "orthogonal_residual_energy": -1.0,
            "energy_gap": 0.0,
            "energy_tolerance": pe.REGRESSION_ENERGY_TOLERANCE,
            "orthonormalized_rank": 3,
            "orthogonal_residual_rank": 3,
            "ranks_match": True,
            "subspace_gap": 0.0,
            "subspace_tolerance": pe.REGRESSION_SPAN_TOLERANCE},
        "packet_program": {
            "pricing": [], "runs": [], "packet_masks": {},
            "full_K_control": {"difference": 0.0}},
    }
    record.update(overrides)
    return record


def test_the_invariant_gate_accepts_a_well_formed_record():
    pe.check_invariants(_record())


def test_a_row_below_the_variational_bound_is_rejected():
    record = _record()
    record["arms"][2]["energy"] = -1.5
    with pytest.raises(AssertionError, match="variational bound"):
        pe.check_invariants(record)


def test_an_energy_disagreement_with_the_orthonormal_krylov_basis_is_rejected():
    record = _record()
    record["krylov_agreement"]["energy_gap"] = 1e-6
    with pytest.raises(AssertionError, match="beyond the tolerance"):
        pe.check_invariants(record)


def test_a_span_disagreement_with_the_krylov_basis_is_rejected():
    record = _record()
    record["krylov_agreement"]["subspace_gap"] = 1e-3
    with pytest.raises(AssertionError, match="does not span the Krylov space"):
        pe.check_invariants(record)


def test_the_regression_gate_is_not_sized_by_the_raw_arm_conditioning():
    """The tolerance must not scale with the ill-conditioned arm's kappa(S).

    Sizing it that way made the gate vacuous: at kappa(S) = 7.4e15 on
    hubbard_2x3 the admitted energy error was tens of hartree.
    """
    record = _record()
    record["arms"][0]["kappa_S"] = 7.4e15            # power_krylov
    record["krylov_agreement"]["energy_gap"] = 1e-3  # tiny next to kappa*eps
    with pytest.raises(AssertionError, match="beyond the tolerance"):
        pe.check_invariants(record)


def test_a_residual_arm_spanning_less_than_the_krylov_basis_is_rejected():
    record = _record()
    record["krylov_agreement"].update(
        {"ranks_match": False, "orthogonal_residual_rank": 2,
         "orthonormalized_rank": 3})
    with pytest.raises(AssertionError, match="retained fewer directions"):
        pe.check_invariants(record)


def test_a_rank_deficient_monomial_basis_is_not_held_against_the_residual_arm():
    """Spanning *more* is the monomial basis going rank-deficient first."""
    record = _record()
    record["krylov_agreement"].update(
        {"ranks_match": False, "orthogonal_residual_rank": 7,
         "orthonormalized_rank": 5})
    pe.check_invariants(record)


def test_a_broken_overlap_identity_is_rejected():
    record = _record()
    record["arms"][2]["orthonormality_defect"] = 1e-6
    with pytest.raises(AssertionError, match="orthonormality defect"):
        pe.check_invariants(record)


def test_an_increasing_nested_energy_is_rejected():
    record = _record()
    record["arms"][2]["nested_energies"] = [-1.0, -0.9]
    with pytest.raises(AssertionError, match="increased along nesting"):
        pe.check_invariants(record)


def test_a_full_K_packet_that_misses_the_direction_is_rejected():
    record = _record()
    record["packet_program"]["full_K_control"]["difference"] = 1e-6
    with pytest.raises(AssertionError, match="full-K packet"):
        pe.check_invariants(record)


def test_a_short_budget_without_a_happy_breakdown_is_rejected():
    record = _record()
    block = dict(record["arms"][0])
    block.update({"method": "davidson[classical_block_L2]",
                  "reference_block_size": 2, "M": 1,
                  "realized_total_directions": 2,
                  "stopped_reason": "budget_exhausted"})
    record["reference_blocks"] = [block]
    with pytest.raises(AssertionError, match="without a happy breakdown"):
        pe.check_invariants(record)


def test_a_short_budget_with_a_happy_breakdown_is_accepted():
    record = _record()
    block = dict(record["arms"][0])
    block.update({"method": "davidson[classical_block_L2]",
                  "reference_block_size": 2, "M": 1,
                  "realized_total_directions": 2,
                  "stopped_reason": "happy_breakdown"})
    record["reference_blocks"] = [block]
    pe.check_invariants(record)


def test_a_shift_that_is_not_the_curve_minimiser_is_rejected():
    record = _record()
    record["shift_sweep"]["selected_mu"] = 0.5      # -0.8 is not the minimum
    with pytest.raises(AssertionError, match="curve's own minimiser"):
        pe.check_invariants(record)


def test_an_oracle_row_outside_the_quarantine_is_rejected():
    record = _record()
    record["arms"][2]["evidence_category"] = "oracle_diagnostic"
    with pytest.raises(AssertionError, match="outside the quarantined"):
        pe.check_invariants(record)


def test_a_packet_reaching_outside_the_sector_is_rejected_structurally():
    """The structural certificate is a proof, and it is what the gate reads."""
    record = _record()
    record["packet_program"]["pricing"] = [
        {"K": 4, "structural_sector_certificate": {
            "in_sector": False, "outside_sector": [7]}}]
    with pytest.raises(AssertionError, match="outside the sector"):
        pe.check_invariants(record)


def test_a_failing_numeric_certificate_is_rejected():
    record = _record()
    record["packet_program"]["pricing"] = [
        {"K": 4,
         "structural_sector_certificate": {"in_sector": True,
                                           "outside_sector": []},
         "reference_conditioned_certificate": {"max_sector_leakage": 1e-3}}]
    with pytest.raises(AssertionError, match="numeric reference-conditioned"):
        pe.check_invariants(record)


def test_a_skipped_numeric_certificate_leaves_the_structural_one_in_charge():
    """Above the qubit threshold the projector is unaffordable, not ignored."""
    record = _record()
    record["packet_program"]["pricing"] = [
        {"K": 4,
         "structural_sector_certificate": {"in_sector": True,
                                           "outside_sector": []},
         "reference_conditioned_certificate": {"skipped": "n=12 exceeds ..."}}]
    pe.check_invariants(record)


def test_a_packet_determinant_outside_the_sector_is_rejected():
    record = _record()
    record["packet_program"]["packet_masks"] = {"4": [3, 7]}   # 7 is not a mask
    with pytest.raises(AssertionError, match="outside the sector"):
        pe.check_invariants(record)


def test_a_missing_required_arm_is_rejected():
    record = _record()
    record["arms"] = record["arms"][:2]
    with pytest.raises(AssertionError, match="required arm"):
        pe.check_invariants(record)


# -------------------------------------------------------------- word preflight


def test_the_packet_gate_falls_back_when_the_committed_record_is_absent(
        monkeypatch, tmp_path):
    monkeypatch.setattr(pe, "COMMITTED_LADDER", tmp_path / "absent.json")
    anchor = pe.committed_acase_word_cost("hubbard_2x2")
    assert anchor["available"] is False
    assert "falls back" in anchor["note"]


def test_the_packet_gate_prices_against_one_committed_acase_direction():
    anchor = pe.committed_acase_word_cost("hubbard_2x2")
    if not anchor.get("available"):
        pytest.skip("the committed Phase 12 record is not present")
    assert anchor["acase_M"] > 0
    assert anchor["words_per_direction"] == pytest.approx(
        anchor["acase_W"] / anchor["acase_M"])


def test_an_unknown_system_is_refused():
    with pytest.raises(ValueError, match="unknown primary system"):
        pe.run_system("not_a_system")


def test_a_degenerate_direction_budget_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        pe.run_system("hubbard_2x2", total_directions=1)
