"""Numerical validation and explicitly oracle allocation for Q18's successor.

These helpers do not alter Q18's frozen rule or record. Richardson estimates
validate the same derivative; allocation uses exact setting variances and
charges, but does not perform, a pilot. Neither supplies finite-shot coverage.
"""

from __future__ import annotations

import math

from benchmarks import check_phase15_measured_residual_preregistration as original


def summarize_derivative(evaluations, analytic, norm, cancellation_scale, rule):
    """Read the fixed step ladder without selecting a step after seeing errors.

    D(h) has a roundoff floor epsilon*scale/h. For R(h)=(4D(h/2)-D(h))/3,
    the propagated floor is (4 floor(h/2)+floor(h))/3. Both final Richardson
    estimates must match the analytic derivative, and must agree with each
    other. The earliest pair is diagnostic. All endpoints must retain the
    declared block and its nondegenerate ground root.
    """
    from clifford_qc.subspace.second_moment import RESOLUTION

    relative = float(rule["relative_tolerance"]) * norm
    estimates = []
    for coarse, fine in zip(evaluations, evaluations[1:]):
        value = (4.0 * fine["numeric"] - coarse["numeric"]) / 3.0
        floor = RESOLUTION * cancellation_scale * (
            4.0 / fine["step"] + 1.0 / coarse["step"]) / 3.0
        tolerance = relative + floor
        estimates.append({"coarse_step": coarse["step"], "fine_step": fine["step"],
                          "numeric": value, "rounding_floor": floor,
                          "tolerance": tolerance,
                          "passes": abs(value - analytic) <= tolerance})
    required = estimates[-2:]
    stability_tolerance = relative + sum(r["rounding_floor"] for r in required)
    stable = abs(required[1]["numeric"] - required[0]["numeric"]) <= stability_tolerance
    domain_ok = all(row["domain_valid"] for row in evaluations)
    finite = all(math.isfinite(row[key]) for row in evaluations
                 for key in ("numeric", "plus", "minus"))
    final = estimates[-1]
    return {
        "evaluations": evaluations, "richardson": estimates,
        "analytic": analytic, "numeric": final["numeric"],
        "tolerance": final["tolerance"], "stability_tolerance": stability_tolerance,
        "stable": stable, "domain_valid": domain_ok,
        "passes": bool(finite and domain_ok and stable and all(r["passes"] for r in required)),
    }


def derivative_check(rows, size, means, weights, direction, *, rule,
                     cancellation_scale, **_legacy):
    """Evaluate every prespecified endpoint of the nonlinear measured pencil."""
    from clifford_qc.subspace.linalg import solve_projected

    analytic = float(sum(weights.get(word, 0.0) * value
                         for word, value in direction.items()))
    norm = math.sqrt(sum(value * value for word, value in weights.items() if word))
    evaluations = []
    for step in rule["steps"]:
        endpoints, domains = [], []
        for sign in (1.0, -1.0):
            perturbed = {word: means.get(word, 0.0) + sign * step * direction.get(word, 0.0)
                         for word in set(means) | set(direction)}
            S, H, K = original.projected_matrices(rows, size, perturbed)
            result = solve_projected(S, H)
            energy = float(result.energies[0])
            c = result.coefficients[:, 0]
            denominator = float((c.conj() @ S @ c).real)
            endpoints.append(float((c.conj() @ K @ c).real) / denominator - energy ** 2)
            gap = (float(result.energies[1]) - energy
                   if len(result.energies) > 1 else math.inf)
            domains.append(bool(result.effective_rank == size and denominator > 0.0
                                and gap > original.GAP_FLOOR * max(1.0, abs(energy))))
        plus, minus = endpoints
        evaluations.append({"step": step, "plus": plus, "minus": minus,
                            "numeric": (plus - minus) / (2.0 * step),
                            "domain_valid": all(domains)})
    return summarize_derivative(evaluations, analytic, norm, cancellation_scale, rule)


def integer_oracle_allocation(variances, standard_error, *, pilot_shots, minimum_shots):
    """Ceil the Neyman allocation, with positive per-setting floors.

    Pilot outcomes are excluded from the production estimate. Their cost is
    charged on both sides, although these weights come from exact variances,
    not estimated pilot variances. This is an oracle diagnostic, not a pilot
    experiment or a realizable allocation claim. Negative variances must be
    checked by the caller before rounding-level values are clipped to zero.
    """
    if not math.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("standard_error must be finite and positive")
    for name, value in (("pilot_shots", pilot_shots), ("minimum_shots", minimum_shots)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = [float(v) for v in variances]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("variances must be finite and nonnegative")
    roots = [math.sqrt(v) for v in values]
    total_root = math.fsum(roots)
    target = standard_error ** 2
    counts = [max(minimum_shots, math.ceil(root * total_root / target)) for root in roots]
    achieved = math.fsum(v / shots for v, shots in zip(values, counts))
    production = sum(counts)
    pilot = pilot_shots * len(values)
    return {"shots_per_setting": counts, "production_shots": production,
            "pilot_shots": pilot, "total_shots": production + pilot,
            "achieved_variance": achieved, "target_variance": target,
            "target_met": achieved <= target * (1.0 + 1e-12)}


def allocation_diagnostic(price, variance, rule):
    """Integer cost screen at the same standard error for energy and sigma."""
    sigma = math.sqrt(variance)
    standard_error = sigma * rule["standard_error_fraction_of_sigma"]
    energy = integer_oracle_allocation(
        [max(v, 0.0) for v in price["energy_setting_variances"]], standard_error,
        pilot_shots=rule["pilot_shots_per_setting"], minimum_shots=rule["minimum_shots_per_setting"])
    residual = integer_oracle_allocation(
        [max(v, 0.0) / (4.0 * variance) for v in price["residual_setting_variances"]],
        standard_error, pilot_shots=rule["pilot_shots_per_setting"],
        minimum_shots=rule["minimum_shots_per_setting"])
    return {"evidence": "oracle_integer_cost_with_charged_unperformed_pilot",
            "standard_error": standard_error, "energy": energy, "residual": residual,
            "total_cost_ratio": residual["total_shots"] / energy["total_shots"],
            "estimator_licensed": False}
