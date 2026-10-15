"""Q18-S4: the measured residual's cost under fully commuting grouping.

Q18-S1 priced the ground Ritz root's residual norm against its energy under
single-assignment QWC with uniform shots, and read NONE on all five licensed
banks. Q18-S4 asks the same question under the other grouping endpoint the
package ships: fully commuting settings, measured by a Clifford
diagonalizer. The estimator, its linearization, the derivative validator, the
statistic and the tenfold bar are Q18-S1's; only the partition changes, and
with it both sides' setting counts and one-shot variances.

Two fully commuting partitions are declared, as Q18 declared two QWC ones, so
that no status rests on one heuristic:

* ``fc_greedy``, the shipped colouring: ``block_commuting_partition`` at block
  size ``n``, largest conflict degree first, ties by code;
* ``fc_isotropic_cover``, ``isotropic_cover`` at block size ``n``: settings grown
  as isotropic subspaces that cover the most unassigned words.

Both partition the measured universe listed in ascending code order, with the
identity excluded, and both build ``W x W`` conflict masks. That is why
H2O CAS(8e,6o), at 868,149 measured words, is outside this successor.

A fully commuting setting reads ``sum_g f_w P_w`` as one observable, so its
one-shot variance is ``Var_psi`` of that sum, exactly as for a QWC setting.
The mapping-axis ``_functional_variance`` computes it from the operator
square; :func:`statevector_moments` recomputes it from matrix-free action on
the reference statevector, a separate path every setting must agree with.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

from benchmarks import check_phase15_measured_residual_preregistration as q18
from benchmarks import check_phase15_preregistration as preflight_gate
from benchmarks.phase15_residual_successor import derivative_check

PROTOCOLS = ("fc_greedy", "fc_isotropic_cover")
# Per-setting agreement between the operator-square and statevector variances,
# relative to max(1, <O^2>, <O>^2), the guard _functional_variance applies.
STATEVECTOR_TOLERANCE = 1e-9
# The executed functionals must be Q18-S1's: same words, same norms, same sigma^2.
PREDECESSOR_RELATIVE = 1e-9
VARIANCE_TOLERANCE = 1e-9
MEAN_TOLERANCE = 1e-9


def fc_partition(words, protocol: str, n: int) -> list[list[int]]:
    """The declared fully commuting partition of a universe, as lists of codes.

    Words are listed in ascending code order with the identity removed, so
    each partition is a function of the word set alone.
    """
    from clifford_qc.measurement.block_commuting import block_commuting_partition
    from clifford_qc.measurement.regrouping import isotropic_cover

    codes = sorted(int(code) for code in words if code != 0)
    if not codes:
        return []
    if protocol == "fc_greedy":
        parts = block_commuting_partition(n, codes, n)
    elif protocol == "fc_isotropic_cover":
        parts = isotropic_cover(n, codes, n)
    else:
        raise ValueError(f"unsupported grouping protocol {protocol!r}")
    return [sorted(codes[index] for index in group) for group in parts]


def partition_digest(groups) -> str:
    """SHA-256 of a partition, canonical in setting order and member order."""
    canonical = sorted(sorted(int(code) for code in group) for group in groups)
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def anticommuting_pairs(n: int, codes) -> int:
    """How many pairs in ``codes`` anticommute, from their per-qubit letters.

    Letters are packed two bits per qubit, ``I`` as zero. Two letters
    anticommute when both are non-identity and they differ, and two words
    anticommute when an odd number of qubits do.
    """
    import numpy as np

    codes = np.asarray([int(code) for code in codes], dtype=np.int64)
    if codes.size < 2:
        return 0
    odd = np.zeros((codes.size, codes.size), dtype=bool)
    for qubit in range(n):
        letters = (codes >> (2 * qubit)) & 3
        odd ^= ((letters[:, None] != 0) & (letters[None, :] != 0)
                & (letters[:, None] != letters[None, :]))
    return int(np.triu(odd, 1).sum())


def partition_problems(groups, universe, n: int) -> list[str]:
    """Exactly-once coverage of the measured universe by commuting settings."""
    problems = []
    assigned = [code for group in groups for code in group]
    if len(assigned) != len(set(assigned)):
        problems.append("a word is assigned to more than one setting")
    if set(assigned) != {int(code) for code in universe} - {0}:
        problems.append("the settings do not cover exactly the measured universe")
    if any(anticommuting_pairs(n, group) for group in groups):
        problems.append("a setting is not fully commuting")
    return problems


def statevector_moments(psi, n: int, group, weights) -> tuple[float, float]:
    """``(<O^2>, <O>)`` for ``O = sum_g f_w P_w``, by matrix-free action on ``psi``."""
    import numpy as np

    from clifford_qc.multivector import MV
    from clifford_qc.pauli_action import PauliLinearOperator

    live = {code: float(weights[code]) for code in group
            if code in weights and weights[code] != 0.0}
    if not live:
        return 0.0, 0.0
    image = PauliLinearOperator(MV(n, live)).matvec(psi)
    return float(np.vdot(image, image).real), float(np.vdot(psi, image).real)


def price_protocol(reference, psi, protocol: str, n: int, sh_words, combined_words,
                   energy_weights, residual_weights, variance: float) -> dict:
    """Both partitions under one protocol, their one-shot variances and ``R``.

    Each setting's variance comes from the operator square and is checked,
    setting by setting, against the statevector's ``<O^2> - <O>^2``.
    """
    sides = {}
    nonnegative, agreement, worst = True, True, 0.0
    for side, words, weights in (("energy", sh_words, energy_weights),
                                 ("residual", combined_words, residual_weights)):
        groups = fc_partition(words, protocol, n)
        problems = partition_problems(groups, words, n)
        as_words = [[_word(n, code) for code in group] for group in groups]
        try:
            package = q18.group_variances(reference, as_words, weights)
        except AssertionError:
            package, nonnegative = [], False
        for group, value in zip(groups, package):
            second, mean = statevector_moments(psi, n, group, weights)
            guard = max(1.0, abs(second), mean * mean)
            deviation = abs(value - (second - mean * mean)) / guard
            worst = max(worst, deviation)
            agreement &= deviation <= STATEVECTOR_TOLERANCE
        sides[side] = {"groups": groups, "problems": problems, "variances": package}
    energy, residual = sides["energy"], sides["residual"]
    energy_variance = float(sum(energy["variances"]))
    residual_variance = float(sum(residual["variances"]))
    energy_neyman = float(sum(math.sqrt(max(v, 0.0)) for v in energy["variances"]))
    residual_neyman = float(sum(math.sqrt(max(v, 0.0)) for v in residual["variances"]))
    priced = nonnegative and energy_variance > 0.0 and variance > 0.0
    ratio = (q18.cost_ratio(residual_settings=len(residual["groups"]),
                            residual_variance=residual_variance,
                            energy_settings=len(energy["groups"]),
                            energy_variance=energy_variance, variance=variance)
             if priced else None)
    neyman = (residual_neyman ** 2 / (4.0 * variance * energy_neyman ** 2)
              if priced and energy_neyman > 0.0 else None)
    return {
        "energy_settings": len(energy["groups"]),
        "energy_partitioned_words": sum(len(group) for group in energy["groups"]),
        "energy_largest_setting": max((len(g) for g in energy["groups"]), default=0),
        "energy_partition_sha256": partition_digest(energy["groups"]),
        "energy_variance_one_shot": energy_variance,
        "energy_neyman_sum": energy_neyman,
        "residual_settings": len(residual["groups"]),
        "residual_partitioned_words": sum(len(group) for group in residual["groups"]),
        "residual_largest_setting": max((len(g) for g in residual["groups"]), default=0),
        "residual_partition_sha256": partition_digest(residual["groups"]),
        "residual_variance_one_shot": residual_variance,
        "residual_neyman_sum": residual_neyman,
        "cost_ratio": ratio,
        "neyman_ratio": neyman,
        "partition_problems": energy["problems"] + residual["problems"],
        "variances_nonnegative": nonnegative,
        "statevector_agreement": bool(agreement and nonnegative),
        "statevector_worst_relative_deviation": worst,
        "energy_setting_variances": [float(v) for v in energy["variances"]],
        "residual_setting_variances": [float(v) for v in residual["variances"]],
    }


def _word(n: int, code: int):
    from clifford_qc.ir import PauliWord

    return PauliWord(n, code)


def _norm(weights) -> float:
    return math.sqrt(sum(value * value for word, value in weights.items() if word != 0))


def predecessor_anchor(estimator: dict, predecessor: dict) -> dict:
    """The executed functionals against Q18-S1's record of the same bank."""
    theirs = predecessor["estimator"]
    rows = {}
    for key in ("residual_functional_words", "energy_functional_words"):
        rows[key] = {"executed": estimator[key], "q18s1": theirs[key],
                     "agrees": estimator[key] == theirs[key]}
    for key in ("variance", "residual_functional_norm", "energy_functional_norm",
                "ground_energy"):
        rows[key] = {"executed": estimator[key], "q18s1": theirs[key],
                     "agrees": math.isclose(estimator[key], theirs[key],
                                            rel_tol=PREDECESSOR_RELATIVE, abs_tol=1e-12)}
    return rows


def evaluate_bank(config: dict, name: str, model, selected, frozen: dict,
                  validation_root: dict, predecessor: dict, *, progress=None,
                  internals: dict | None = None) -> dict:
    """The rows, functionals, checks, fully commuting prices and status of one bank.

    ``internals``, when given, receives the reference statevector, both
    universes and both functionals' weights, so a checker can recompute every
    setting variance its own way. The record never carries them.
    """
    from benchmarks.run_phase15_h2_preflight import _digest, reference_vector
    from clifford_qc.measurement.functionals import ritz_functional
    from clifford_qc.subspace import SecondMomentBank

    def say(line):
        if progress is not None:
            progress(f"  {name}: {line}")

    started = time.perf_counter()
    grouping = config["grouping"]
    rule = config["derivative_validation"]
    directions_rule = config["linearization"]["finite_difference"]
    threshold = float(config["statistic"]["max_ratio"])
    size = len(selected)

    bank = preflight_gate.first_moment_bank(model, selected)
    second = SecondMomentBank(bank)
    rows = q18.block_rows(bank, second, list(range(size)))
    sh_words = set(bank.word_set())
    combined = q18.row_words(rows)
    say(f"rows built, raw |U_SH|={len(sh_words)} raw |U|={len(combined)}")

    means = q18.reference_means(bank.reference, combined)
    weights, variance = q18.residual_functional(rows, size, means)
    residual_weights = {int(word): float(value) for word, value in weights.items()}
    result = bank.solve()
    energy = ritz_functional(bank, result.indices, result.ritz_vector(), result.ground_energy)
    energy_weights = {int(word): float(value) for word, value in energy.coefficients.items()}
    measured_residual = {w: v for w, v in residual_weights.items() if w != 0}
    scale = float(validation_root["cancellation_scale"])
    reference_mean = float(sum(value * means.get(word, 0.0)
                               for word, value in residual_weights.items()))
    residual_norm = math.sqrt(max(variance, 0.0))
    estimator = {
        "ground_energy": float(result.ground_energy),
        "variance": float(variance),
        "residual_norm": residual_norm,
        "matched_precision_half": residual_norm / 4.0,
        "cancellation_scale": scale,
        "residual_functional_words": len(measured_residual),
        "residual_functional_norm": _norm(residual_weights),
        "residual_identity_weight": float(residual_weights.get(0, 0.0)),
        "residual_reference_mean": reference_mean,
        "energy_functional_words": len(energy_weights),
        "energy_functional_norm": _norm(energy_weights),
    }
    anchor = predecessor_anchor(estimator, predecessor)
    say(f"functionals built, sigma^2={variance:.6g}")

    directions = q18.check_directions(residual_weights, sh_words,
                                      seed=int(directions_rule["seed"]))
    finite = {key: derivative_check(rows, size, means, residual_weights, directions[key],
                                    rule=rule, cancellation_scale=scale)
              for key in directions_rule["directions"]}
    say("derivatives " + ", ".join(
        f"{key}={'pass' if value['passes'] else 'FAIL'}" for key, value in finite.items()))

    psi = reference_vector(model)
    if internals is not None:
        internals.update(psi=psi, sh_words=sh_words, combined=combined,
                         energy_weights=energy_weights, residual_weights=measured_residual)
    protocols = {}
    declared = grouping["declared_protocol_by_system"][name]
    alternative = grouping["alternative_protocol_by_system"][name]
    for role, protocol in (("declared", declared), ("alternative", alternative)):
        priced = price_protocol(bank.reference, psi, protocol, model.n, sh_words, combined,
                                energy_weights, measured_residual, variance)
        protocols[protocol] = {"role": role, **priced}
        say(f"{protocol}: G_SH={priced['energy_settings']} G_U={priced['residual_settings']} "
            f"R={priced['cost_ratio']}")

    checks = {
        "combined_universe_matches_preflight": (
            len(combined) == frozen["raw_combined_word_universe"]
            and _digest(combined) == frozen["raw_combined_words_sha256"]
            and 0 in combined),
        "energy_settings_reproduce_freeze": all(
            protocols[p]["energy_settings"] == frozen["fc_energy_settings"][p]
            for p in protocols),
        "functionals_reproduce_q18s1": all(row["agrees"] for row in anchor.values()),
        "partitions_valid": all(not p["partition_problems"] for p in protocols.values()),
        "residual_functional_reproduces_variance": (
            abs(variance - float(validation_root["variance"]))
            <= VARIANCE_TOLERANCE * (1.0 + scale)),
        "residual_functional_mean_zero": bool(
            abs(reference_mean) <= MEAN_TOLERANCE * (1.0 + scale)),
        "linearization_matches_finite_differences": all(
            value["passes"] for value in finite.values()),
        "group_variances_nonnegative": all(
            p["variances_nonnegative"] for p in protocols.values()),
        "group_variances_match_statevector": all(
            p["statevector_agreement"] for p in protocols.values()),
    }
    ratios = {p: protocols[p]["cost_ratio"] for p in protocols}
    if all(checks.values()) and all(value is not None for value in ratios.values()):
        status = q18.status_of(ratios[declared], ratios[alternative], threshold)
    else:
        status = "INVALID"
    return {
        "n_qubits": model.n,
        "basis_size": size,
        "universes": {
            "raw_sh_word_universe": len(sh_words),
            "raw_combined_word_universe": len(combined),
            "measured_sh_words": len(sh_words - {0}),
            "measured_combined_words": len(combined - {0}),
            "raw_combined_words_sha256": _digest(combined),
        },
        "estimator": estimator,
        "predecessor_anchor": anchor,
        "finite_differences": finite,
        "protocols": protocols,
        "deterministic_checks": checks,
        "status": status,
        "seconds": round(time.perf_counter() - started, 2),
    }
