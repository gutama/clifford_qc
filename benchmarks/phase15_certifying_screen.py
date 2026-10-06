"""Q18-S2: the oracle allocation screen on ground-certifying Hubbard 2x2 bases.

Q18-S1 priced the residual norm on the frozen Hubbard 2x2 bank, whose Ritz
root sits 0.86 t above the exact sector ground state with ``sigma = 1.52 t``.
Its Weinstein interval ``[E - sigma, E + sigma]`` contains the six lowest
sector eigenvalues (five distinct levels), so a measured ``sigma`` there
certifies nothing about the ground state. This module asks the question one step earlier than a pilot
experiment would: along the two committed A-CASE trajectories that extend that
bank, is there a basis whose exact interval does certify the ground state, and
there, could *any* shot allocation under the declared QWC partitions measure
``sigma`` for at most ten times the energy's shots?

The answer is read from exact reference variances, so it is an oracle screen.
The decision statistic is the Neyman ratio Q18 already reports: both sides at
their variance-optimal allocation across settings, production shots only. A
practical allocator measured against the energy's own optimal cost cannot
undercut it to first order. Uniform ``R`` and Q18-S1's integer allocation with
a charged, unperformed pilot are reported beside it and enter no clause.

The estimator, its linearization, the groupings and the derivative validator
are Q18's and Q18-S1's, imported rather than restated. The one addition is a
vectorized evaluation of the measured pencil for the Richardson endpoints; the
tests and the frozen-prefix lineage check hold it to Q18's own evaluation.
"""

from __future__ import annotations

import math
import time

from benchmarks import check_phase15_measured_residual_preregistration as gate
from benchmarks import run_phase15_measured_residual_preflight as original
from benchmarks.phase15_residual_successor import integer_oracle_allocation, summarize_derivative

# A Ritz energy below the exact ground energy by more than this is not a
# rounding artifact, and the spectral reference does not describe the state.
SPECTRAL_ROUNDING_ULPS = 100.0
# Two sector eigenvalues closer than this, relative to max(1, |E0|), are one level.
DEGENERACY_TOLERANCE = 1e-9
# The ladder's energy history and a rebuilt prefix solve must agree to this.
HISTORY_TOLERANCE = 1e-9
# The deterministic-check tolerances Q18's producer applies.
VARIANCE_TOLERANCE = original.VARIANCE_TOLERANCE
MEAN_TOLERANCE = original.MEAN_TOLERANCE
STATUSES = ("AFFORDABLE", "PROHIBITIVE", "GROUPING_SENSITIVE")
VERDICTS = ("OPEN", "CLOSED", "GROUPING_SENSITIVE", "UNREACHED", "INVALID")


# ------------------------------------------------------------------ the domain

def spectral_rounding(energy: float, ground: float) -> float:
    return SPECTRAL_ROUNDING_ULPS * math.ulp(1.0) * max(1.0, abs(energy), abs(ground))


def certifying(energy: float, sigma: float, ground: float, first: float) -> bool:
    """Whether ``[E - sigma, E + sigma]`` certifies the sector ground state.

    Weinstein puts an eigenvalue within ``sigma`` of ``E``. With ``E >= E0``
    variationally, ``E + sigma < E1`` leaves ``E0`` as the only level the
    interval can hold, so the measured pair brackets it as ``[E - sigma, E]``.
    The condition implies the energy-only ground dominance the package's
    convergence report reads, because ``sigma^2 >= (E - E0)(E1 - E)`` for any
    sector state with ``E0 <= E <= E1``.
    """
    return energy + sigma < first and energy >= ground - spectral_rounding(energy, ground)


def ground_weight_lower_bound(energy: float, ground: float, first: float) -> float:
    """``p0 >= 1 - (E - E0)/(E1 - E0)``, clipped to ``[0, 1]``: energy only."""
    return max(0.0, min(1.0, 1.0 - max(energy - ground, 0.0) / (first - ground)))


def temple_lower_bound(energy: float, variance: float, first: float) -> float | None:
    """Temple's ``E0 >= E - sigma^2/(E1 - E)`` with the exact ``E1``; oracle."""
    if energy >= first:
        return None
    return energy - variance / (first - energy)


# ------------------------------------------------------------------ the rule

def status_of(ratios: dict, declared: str, alternative: str | None, threshold: float,
              checks: dict) -> str:
    """Q18's ladder read on the Neyman ratios; any failed check is INVALID."""
    if not all(checks.values()) or any(value is None for value in ratios.values()):
        return "INVALID"
    return gate.status_of(ratios[declared],
                          ratios[alternative] if alternative is not None else None,
                          threshold)


def verdict_of(entries) -> str:
    """The decision over every prefix, read only on the certifying domain.

    ``entries`` carry ``in_domain`` and ``status``. A prefix outside the
    domain is priced and reported, but it cannot move the verdict.
    """
    domain = [entry["status"] for entry in entries if entry["in_domain"]]
    if "INVALID" in domain:
        return "INVALID"
    if not domain:
        return "UNREACHED"
    if "AFFORDABLE" in domain:
        return "OPEN"
    if all(status == "PROHIBITIVE" for status in domain):
        return "CLOSED"
    return "GROUPING_SENSITIVE"


# ------------------------------------------------------------------ inputs

def trajectory_generators(model, labels, *, level4: bool):
    """The committed ladder's generators, looked up by label in its own pool."""
    from benchmarks.run_acase_ladder import build_candidates
    from clifford_qc.subspace import identity_generator

    candidates = build_candidates(model, level4=level4)
    by_label = {generator.label: generator for generator in candidates}
    if len(by_label) != len(candidates):
        raise AssertionError("the candidate pool repeats a label")
    labels = list(labels)
    if not labels or labels[0] != "I" or len(set(labels)) != len(labels):
        raise ValueError("trajectory labels must be unique and begin with the identity")
    missing = sorted(set(labels[1:]) - set(by_label))
    if missing:
        raise ValueError(f"labels absent from the candidate pool: {missing}")
    return [identity_generator(model.n), *(by_label[label] for label in labels[1:])], len(
        candidates)


def sector_premises(model, generator_lists) -> dict:
    """What makes the sector spectrum the right reference, and bounds storage.

    Each ``A_j|psi>`` lies in the reference's ``(N, S_z)`` sector, so every
    Ritz state does and the sector spectrum is its reference. That is a
    statement about the reference's images, not the operators: a level-4
    configuration generator moves other determinants out of the sector while
    mapping the reference into it, so operator leakage is reported beside it
    and is not the premise. Every word of the Hamiltonian and the generators
    lies in the two-spin-parity sector, which is closed under products, so no
    second-moment row leaves its ``4^n / 4`` words.
    """
    import numpy as np

    from benchmarks.check_phase15_preregistration import in_spin_parity_sector, sector_ceiling
    from benchmarks.run_phase15_h2_preflight import reference_vector
    from clifford_qc.fermion import total_number_op, total_sz_op
    from clifford_qc.pauli_action import PauliLinearOperator
    from clifford_qc.subspace.symmetry import sector_leakage

    n = model.n
    metadata = model.metadata
    convention = metadata["spin_convention"]
    psi = reference_vector(model)
    number = PauliLinearOperator(total_number_op(n))
    spin = PauliLinearOperator(total_sz_op(n, spin_ordering=convention))
    hamiltonian = model.hamiltonian.to_mv()
    operator_leakage, image_leakage = 0.0, 0.0
    violations = sum(not in_spin_parity_sector(n, code) for code in hamiltonian.terms)
    seen = set()
    for generators in generator_lists:
        for generator in generators:
            if generator.label in seen:
                continue
            seen.add(generator.label)
            rates = sector_leakage(generator, spin_ordering=convention)
            operator_leakage = max(operator_leakage, rates["particle_number"], rates["sz"])
            image = PauliLinearOperator(generator.mv).matvec(psi)
            norm = float(np.linalg.norm(image))
            if norm > 0.0:
                image_leakage = max(
                    image_leakage,
                    float(np.linalg.norm(number.matvec(image) - metadata["n_electrons"] * image))
                    / norm,
                    float(np.linalg.norm(spin.matvec(image) - metadata["sz"] * image)) / norm)
            violations += sum(not in_spin_parity_sector(n, code) for code in generator.mv.terms)
    hamiltonian_leak = sector_leakage(hamiltonian, spin_ordering=convention)
    ceiling = sector_ceiling(n)
    largest = max(len(generators) for generators in generator_lists)
    return {
        "distinct_generators": len(seen),
        "reference_image_sector_leakage": image_leakage,
        "generator_operator_sector_leakage": operator_leakage,
        "hamiltonian_sector_leakage": max(hamiltonian_leak.values()),
        "spin_parity_sector_violations": violations,
        "sector_word_ceiling": ceiling,
        "largest_basis": largest,
        "coefficient_occurrence_bound": 3 * largest * (largest + 1) // 2 * ceiling,
    }


def sector_spectrum(model) -> dict:
    """The exact ``(N, S_z)`` ground and first distinct excited energies."""
    from benchmarks.check_phase15_extrapolation_preregistration import sector_spectrum as lowest

    # Every eigenvalue of the sector, so the first *distinct* level is found
    # even when the ground level is degenerate.
    values, dimension = lowest(model, count=4 ** model.n)
    ground = values[0]
    tolerance = DEGENERACY_TOLERANCE * max(1.0, abs(ground))
    first = next((value for value in values[1:] if value > ground + tolerance), None)
    if first is None:
        raise ValueError("the sector has no level above its ground energy")
    degeneracy = sum(1 for value in values if value <= ground + tolerance)
    return {"sector_dimension": int(dimension), "ground_energy": float(ground),
            "first_excited_energy": float(first), "gap": float(first - ground),
            "ground_degeneracy": degeneracy, "midpoint": float(0.5 * (ground + first))}


# ------------------------------------------------------------------ the pencil

class CompiledRows:
    """The ``(S, H, K)`` rows as index arrays, so a mean vector gives the pencil.

    ``gate.projected_matrices`` sums each row in Python; the Richardson ladder
    evaluates the pencil sixteen times per prefix on rows of up to ``4^n/4``
    words. This is the same linear map, summed by NumPy.
    """

    def __init__(self, rows, size: int, words):
        import numpy as np

        self.size = size
        self.words = sorted(words)
        index = {word: k for k, word in enumerate(self.words)}
        pairs = sorted(rows)
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        self.parts = []
        for part in range(3):
            pair_index, word_index, coefficient = [], [], []
            for p, key in enumerate(pairs):
                for word, value in rows[key][part].items():
                    pair_index.append(p)
                    word_index.append(index[word])
                    coefficient.append(complex(value))
            self.parts.append((np.array(pair_index, dtype=int), np.array(word_index, dtype=int),
                               np.array(coefficient, dtype=complex)))

    def vector(self, means):
        import numpy as np

        return np.array([float(means.get(word, 0.0)) for word in self.words])

    def matrices(self, values):
        import numpy as np

        out = []
        count = len(self.pairs)
        for pair_index, word_index, coefficient in self.parts:
            terms = coefficient * values[word_index]
            entries = (np.bincount(pair_index, weights=terms.real, minlength=count)
                       + 1j * np.bincount(pair_index, weights=terms.imag, minlength=count))
            matrix = np.zeros((self.size, self.size), dtype=complex)
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            matrix[a, b] = entries
            lower = a != b
            matrix[b[lower], a[lower]] = np.conj(entries[lower])
            out.append(matrix)
        return tuple(out)


def derivative_check(compiled: CompiledRows, means, weights, direction, *, rule,
                     cancellation_scale):
    """Q18-S1's fixed Richardson ladder on the compiled pencil.

    Every endpoint solves the perturbed measured pencil and reads
    ``sigma^2 = c'Kc/c'Sc - E^2``, exactly as ``gate.residual_variance`` does,
    and records whether the full block and a nondegenerate ground root survive.
    """
    from clifford_qc.subspace.linalg import solve_projected

    base = compiled.vector(means)
    step_vector = compiled.vector(direction)
    analytic = float(sum(weights.get(word, 0.0) * value for word, value in direction.items()))
    norm = math.sqrt(sum(value * value for word, value in weights.items() if word))
    evaluations = []
    for step in rule["steps"]:
        endpoints, domains = [], []
        for sign in (1.0, -1.0):
            S, H, K = compiled.matrices(base + sign * step * step_vector)
            result = solve_projected(S, H)
            energy = float(result.energies[0])
            c = result.coefficients[:, 0]
            denominator = float((c.conj() @ S @ c).real)
            endpoints.append(float((c.conj() @ K @ c).real) / denominator - energy ** 2)
            gap = (float(result.energies[1]) - energy
                   if len(result.energies) > 1 else math.inf)
            domains.append(bool(result.effective_rank == compiled.size and denominator > 0.0
                                and gap > gate.GAP_FLOOR * max(1.0, abs(energy))))
        plus, minus = endpoints
        evaluations.append({"step": step, "plus": plus, "minus": minus,
                            "numeric": (plus - minus) / (2.0 * step),
                            "domain_valid": all(domains)})
    return summarize_derivative(evaluations, analytic, norm, cancellation_scale, rule)


def allocation_summary(price, variance, rule) -> dict:
    """Q18-S1's integer oracle allocation, totals only.

    Per-setting counts are a ceiling of the recorded setting variances, which
    the checker rederives, so the record keeps the totals and not the counts.
    """
    sigma = math.sqrt(variance)
    standard_error = sigma * rule["standard_error_fraction_of_sigma"]
    sides = {}
    for side, scale in (("energy", 1.0), ("residual", 1.0 / (4.0 * variance))):
        values = [max(v, 0.0) * scale for v in price[f"{side}_setting_variances"]]
        allocation = integer_oracle_allocation(
            values, standard_error, pilot_shots=rule["pilot_shots_per_setting"],
            minimum_shots=rule["minimum_shots_per_setting"])
        allocation.pop("shots_per_setting")
        sides[side] = allocation
    return {"evidence": "oracle_integer_cost_with_charged_unperformed_pilot",
            "standard_error": standard_error, **sides,
            "production_ratio": sides["residual"]["production_shots"]
            / sides["energy"]["production_shots"],
            "total_cost_ratio": sides["residual"]["total_shots"] / sides["energy"]["total_shots"],
            "estimator_licensed": False}


# ------------------------------------------------------------------ one prefix

def evaluate_prefix(config, bank, second, size: int, spectrum: dict, history_energy: float, *,
                    protocols: tuple[str, str | None], say=None) -> dict:
    """Every number of one prefix: its interval, domain, prices, checks, status."""
    from clifford_qc.measurement.functionals import ritz_functional
    from clifford_qc.subspace.second_moment import RESOLUTION

    started = time.perf_counter()
    order = list(range(size))
    result = bank.solve(order)
    residual = second.residual(result, 0)
    energy = float(result.ground_energy)
    variance = float(residual.variance)
    sigma = math.sqrt(max(variance, 0.0))
    scale = float(residual.cancellation_scale)
    ground, first = spectrum["ground_energy"], spectrum["first_excited_energy"]
    energies = [float(value) for value in result.energies]
    gap = energies[1] - energies[0] if len(energies) > 1 else math.inf
    entry = {
        "basis_size": size,
        "ground_energy": energy,
        "ladder_energy": float(history_energy),
        "error": energy - ground,
        "variance": variance,
        "residual_norm": sigma,
        "variance_resolved": bool(residual.resolved),
        "cancellation_scale": scale,
        "variance_resolution": RESOLUTION * scale,
        "weinstein_upper": energy + sigma,
        "ground_weight_lower_bound": ground_weight_lower_bound(energy, ground, first),
        "temple_lower_bound": temple_lower_bound(energy, variance, first),
        "certifying": bool(residual.resolved) and certifying(energy, sigma, ground, first),
        "effective_rank": int(result.effective_rank),
        "subspace_gap": gap,
    }
    entry["ground_dominated"] = entry["ground_weight_lower_bound"] > 0.5
    entry["in_domain"] = entry["certifying"]
    if not residual.resolved:
        # sigma^2 is rounding: the linearization d sigma = d sigma^2 / (2 sigma)
        # has no regime, and R's explicit 1/sigma^2 diverges. Not priced.
        entry.update(role="unpriced", status="UNRESOLVED", protocols={},
                     deterministic_checks={}, seconds=round(time.perf_counter() - started, 2))
        return entry

    rows = gate.block_rows(bank, second, order)
    sh_words = set(bank.word_set(order))
    combined = gate.row_words(rows)
    means = gate.reference_means(bank.reference, combined)
    weights, functional_variance = gate.residual_functional(rows, size, means)
    residual_weights = {int(word): float(value) for word, value in weights.items()}
    measured_residual = {w: v for w, v in residual_weights.items() if w != 0}
    energy_weights = dict(ritz_functional(bank, result.indices, result.ritz_vector(),
                                          result.ground_energy).coefficients)
    reference_mean = float(sum(value * means.get(word, 0.0)
                               for word, value in residual_weights.items()))
    entry["universes"] = {"raw_sh_word_universe": len(sh_words),
                          "raw_combined_word_universe": len(combined),
                          "measured_sh_words": len(sh_words - {0}),
                          "measured_combined_words": len(combined - {0})}
    entry["estimator"] = {
        "functional_variance": float(functional_variance),
        "matched_precision_half": sigma / 4.0,
        "residual_functional_words": len(measured_residual),
        "residual_functional_norm": original._norm(residual_weights),
        "residual_identity_weight": float(residual_weights.get(0, 0.0)),
        "residual_reference_mean": reference_mean,
        "energy_functional_words": len(energy_weights),
        "energy_functional_norm": original._norm(energy_weights),
    }

    rule = config["derivative_validation"]
    compiled = CompiledRows(rows, size, combined)
    directions = gate.check_directions(residual_weights, sh_words,
                                       seed=int(rule["seed"]))
    finite = {key: derivative_check(compiled, means, residual_weights, directions[key],
                                    rule=rule, cancellation_scale=scale)
              for key in rule["directions"]}
    if say:
        say(f"M={size}: sigma={sigma:.6g} certifying={entry['certifying']} derivatives "
            + ", ".join(f"{k}={'pass' if v['passes'] else 'FAIL'}" for k, v in finite.items()))

    priced = {}
    declared, alternative = protocols
    for role, protocol in (("declared", declared), ("alternative", alternative)):
        if protocol is None:
            continue
        price = original.price_protocol(bank.reference, protocol, bank.reference.n, sh_words,
                                        combined, energy_weights, measured_residual,
                                        functional_variance, include_setting_variances=True)
        price["allocation_diagnostic"] = (
            allocation_summary(price, functional_variance, config["allocation_diagnostic"])
            if price["variances_nonnegative"] else None)
        priced[protocol] = {"role": role, **price}
    entry["finite_differences"] = finite
    entry["protocols"] = priced

    checks = {
        "full_rank": int(result.effective_rank) == size,
        "ground_root_nondegenerate": gap > gate.GAP_FLOOR * max(1.0, abs(energy)),
        "energy_reproduces_ladder": abs(energy - history_energy) <= HISTORY_TOLERANCE,
        "energy_consistent_with_spectrum": energy >= ground - spectral_rounding(energy, ground),
        "residual_functional_reproduces_variance": (
            abs(functional_variance - variance) <= VARIANCE_TOLERANCE * (1.0 + scale)),
        "residual_functional_mean_zero": abs(reference_mean) <= MEAN_TOLERANCE * (1.0 + scale),
        "linearization_validated": all(value["passes"] for value in finite.values()),
        "partitions_valid": all(not p["partition_problems"] for p in priced.values()),
        "group_variances_nonnegative": all(p["variances_nonnegative"] for p in priced.values()),
    }
    entry["deterministic_checks"] = checks
    ratios = {protocol: price["neyman_ratio"] for protocol, price in priced.items()}
    entry["status"] = status_of(ratios, declared, alternative,
                                float(config["statistic"]["max_ratio"]), checks)
    entry["role"] = "decision" if entry["in_domain"] else "diagnostic"
    entry["seconds"] = round(time.perf_counter() - started, 2)
    return entry


def evaluate_trajectory(config, model, generators, history, spectrum, *, protocols,
                        prefixes, progress=None) -> dict:
    """Every declared prefix of one committed trajectory, on one shared bank."""
    from clifford_qc.backends import ExactMVBackend
    from clifford_qc.subspace import SecondMomentBank
    from clifford_qc.subspace.elements import MatrixElementBank

    if model.metadata.get("spin_convention") != "interleaved":
        raise ValueError("the sector ceiling assumes the interleaved spin convention")
    # check_phase15_preregistration.first_moment_bank's construction, without
    # forming the whole block: each prefix builds only its own pairs.
    rho = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho, model.hamiltonian, generators)
    second = SecondMomentBank(bank)
    out = {}
    for size in prefixes:
        out[str(size)] = evaluate_prefix(
            config, bank, second, size, spectrum, history[size - 1], protocols=protocols,
            say=progress)
    return out
