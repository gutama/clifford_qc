"""Opt-in estimators for measured projected pencils.

An A-CASE generator need not be unitary, so its overlap diagonal need not be
one. :func:`repair_overlap` projects the *normalized* live block onto the
correlation matrices, then restores the measured diagonal. This minimizes a
normalized Frobenius distance, not the unscaled distance and not a
covariance-weighted distance. :func:`solve_rotated` thresholds a fixed-angle
rotated pencil. :func:`solve_constrained_ratio` never inverts ``S``: it
minimizes the Rayleigh ratio under a propagated standard-error budget built
from the full covariance of the measured entries. None of these supplies a
variational bound or an energy confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .linalg import (DEFAULT_MAX_CONDITION, DEFAULT_NORM_FLOOR, DEFAULT_TAU_S,
                     SubspaceResult, fix_phase, normalize_overlap, solve_projected)


def _rank_one_projection(C: np.ndarray, vector: np.ndarray, tolerance: float
                         ) -> np.ndarray | None:
    """Accept a rank-one candidate only with a primal/dual KKT certificate.

    For X=zz*, |z_i|=1, stationarity is X-C+diag(y)-Z=0,
    Z>=0 and Zz=0. Set y_i=Re(((C-X)z)_i / z_i) and test these
    conditions. This avoids slow alternating projections for large, aligned
    off-diagonal errors without clipping data or dropping positive live rows.
    """
    if np.any(np.abs(vector) <= np.finfo(float).eps):
        return None
    z = vector / np.abs(vector)
    X = np.outer(z, z.conj())
    residual = C - X
    multipliers = (residual @ z) * z.conj()
    slack = np.diag(multipliers.real) - residual
    limit = max(tolerance, 50.0 * np.finfo(float).eps * np.linalg.norm(C, ord=2))
    if (np.max(np.abs(multipliers.imag)) > limit or
            np.linalg.eigvalsh(slack)[0] < -limit or
            np.linalg.norm(slack @ z, ord=np.inf) > limit):
        return None
    return X


def repair_overlap(overlap: np.ndarray, *, norm_floor: float = DEFAULT_NORM_FLOOR,
                   tolerance: float = 1e-10, max_iterations: int | np.integer = 1000
                   ) -> tuple[np.ndarray, dict]:
    """Nearest-correlation repair of the solver's live block.

    Higham's alternating projections with Dykstra's correction solve the
    Hermitian nearest-correlation problem on the normalized live block.
    A final PSD projection and diagonal congruence remove the tolerance-sized
    feasibility defect; the distance optimum is therefore numerical, to the
    requested iteration tolerance. No positive eigenvalue floor is added.

    Rows whose norm is at most ``norm_floor`` are decoupled and retain their
    diagonal, including negative measurements; these are exactly the rows the
    canonical solver drops. Only the live block is constrained to be PSD.
    Nonconvergence raises rather than returning an unconverged repair.

    Reference: N. J. Higham, IMA J. Numer. Anal. 22 (2002), 329-343,
    doi:10.1093/imanum/22.3.329. The complex-Hermitian extension uses the same
    orthogonal projections in the real Frobenius inner-product space.
    """
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) \
            or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)
    S = np.asarray(overlap, dtype=complex)
    frame = normalize_overlap(S, norm_floor=norm_floor)
    diagonal, live = frame.diagonal, frame.indices
    scale = frame.norms[live]
    C = frame.matrix
    C = 0.5 * (C + C.conj().T)
    np.fill_diagonal(C, 1.0)
    input_values, input_vectors = np.linalg.eigh(C)
    input_min = float(input_values[0])
    Y = C.copy()
    iterations = 0
    certificate = (_rank_one_projection(C, input_vectors[:, -1], tolerance)
                   if input_min < 0.0 else None)
    if certificate is not None:
        Y = certificate
        np.fill_diagonal(Y, 1.0)
    elif input_min < 0.0:
        correction = np.zeros_like(C)
        X = C.copy()
        for iterations in range(1, max_iterations + 1):
            old_X, old_Y = X, Y
            R = Y - correction
            values, vectors = np.linalg.eigh(R)
            X = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
            correction = X - R
            Y = X.copy()
            np.fill_diagonal(Y, 1.0)
            size = max(float(np.linalg.norm(Y, "fro")), 1.0)
            change = max(float(np.linalg.norm(X - old_X, "fro")),
                         float(np.linalg.norm(Y - old_Y, "fro")),
                         float(np.linalg.norm(Y - X, "fro"))) / size
            if change <= tolerance and np.linalg.eigvalsh(Y)[0] >= -tolerance:
                break
        else:
            raise np.linalg.LinAlgError(
                f"nearest-correlation repair did not converge in {max_iterations} iterations")
        # Make the returned matrix feasible to roundoff, with no ridge.
        values, vectors = np.linalg.eigh(Y)
        Y = (vectors * np.maximum(values, 0.0)) @ vectors.conj().T
        unit_scale = np.sqrt(Y.diagonal().real)
        Y = (Y / unit_scale[:, None]) / unit_scale[None, :]
        Y = 0.5 * (Y + Y.conj().T)
        np.fill_diagonal(Y, 1.0)

    repaired = np.diag(diagonal).astype(complex)
    repaired[np.ix_(live, live)] = (Y * scale[:, None]) * scale[None, :]
    np.fill_diagonal(repaired, diagonal)
    return repaired, {
        "method": ("normalized_rank_one_kkt" if certificate is not None
                   else "normalized_higham_dykstra"),
        "rank_one_kkt_certified": certificate is not None,
        "evidence": "heuristic",
        "iterations": iterations,
        "tolerance": float(tolerance),
        "live_generators": int(live.size),
        "decoupled_generators": int(S.shape[0] - live.size),
        "nonpositive_diagonal_indices": tuple(int(i) for i in np.flatnonzero(diagonal <= 0)),
        "live_indices": tuple(int(i) for i in live),
        "normalized_min_eigenvalue_before": input_min,
        "normalized_min_eigenvalue_after": float(np.linalg.eigvalsh(Y)[0]),
        "normalized_correction_fro": float(np.linalg.norm(Y - C, "fro")),
        "correction_fro": float(np.linalg.norm(repaired - S, "fro")),
    }


DEFAULT_RATIO_DENOMINATOR_CV = 0.1
DEFAULT_RATIO_TOLERANCE = 1e-8
# Line-search stops are accepted only below this first-order KKT residual.
_RATIO_KKT_TOLERANCE = 1e-4
_RATIO_ACTIVE_TOLERANCE = 1e-6
DEFAULT_RATIO_MAX_ITERATIONS = 500
# Positivity margin for c^* S c on unit-norm normalized coordinates.
_RATIO_DENOMINATOR_FLOOR = 1e-12


def hermitian_parameter_layout(size: int) -> tuple[tuple[int, int, str], ...]:
    """Canonical real coordinates of a ``size x size`` Hermitian matrix.

    Upper triangle in row-major order. A diagonal entry contributes its real
    part; an off-diagonal entry contributes its real part, then its imaginary
    part. There are ``size**2`` coordinates. Covariances passed to
    :func:`solve_constrained_ratio` use this order for ``S`` then ``H``.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
        raise ValueError("size must be a positive integer")
    out = []
    for a in range(int(size)):
        for b in range(a, int(size)):
            out.append((a, b, "real"))
            if a < b:
                out.append((a, b, "imag"))
    return tuple(out)


def hermitian_parameters(matrix: np.ndarray) -> np.ndarray:
    """A Hermitian matrix's coordinates in :func:`hermitian_parameter_layout` order."""
    M = np.asarray(matrix, dtype=complex)
    layout = hermitian_parameter_layout(M.shape[0])
    return np.array([M[a, b].real if part == "real" else M[a, b].imag
                     for a, b, part in layout], dtype=float)


class _HermitianLayout:
    """Vectorized maps between coefficients, coordinates and Hermitian matrices."""

    def __init__(self, size: int):
        layout = hermitian_parameter_layout(size)
        self.size = size
        self.rows = np.array([a for a, _, _ in layout])
        self.cols = np.array([b for _, b, _ in layout])
        self.diagonal = self.rows == self.cols
        self.imaginary = np.array([part == "imag" for _, _, part in layout])

    def features(self, c: np.ndarray) -> np.ndarray:
        """``phi(c)`` with ``phi(c) . parameters(M) = c^* M c``."""
        w = np.conjugate(c[self.rows]) * c[self.cols]
        return np.where(self.diagonal, w.real,
                        np.where(self.imaginary, -2.0 * w.imag, 2.0 * w.real))

    def matrix(self, weights: np.ndarray) -> np.ndarray:
        """``sum_k weights_k B_k``, so ``c^* matrix(t) c = phi(c) . t``."""
        upper = np.zeros((self.size, self.size), dtype=complex)
        values = np.where(self.imaginary, 1j * weights, weights)
        np.add.at(upper, (self.rows, self.cols), values)
        diagonal = np.diag(upper.diagonal().real)
        strict = np.triu(upper, 1)
        return diagonal + strict + strict.conj().T


def _validated_covariance(covariance: np.ndarray, size: int) -> np.ndarray:
    count = size * size
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2 * count, 2 * count) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"covariance must be a finite {2 * count} x {2 * count} matrix "
                         "over the S then H Hermitian coordinates")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    roundoff = 100.0 * np.finfo(float).eps * scale * matrix.shape[0]
    if np.max(np.abs(matrix - matrix.T)) > roundoff:
        raise ValueError("covariance must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    if np.linalg.eigvalsh(matrix)[0] < -roundoff:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def ratio_statistics(S: np.ndarray, H: np.ndarray, covariance: np.ndarray,
                     coefficients: np.ndarray) -> dict[str, float]:
    """First-order statistics of ``E(c) = c^* H c / c^* S c`` at fixed ``c``.

    With numerator ``N`` and denominator ``D``, both linear in the measured
    coordinates, ``Var E ~ (Var N - 2 E Cov(N, D) + E^2 Var D) / D^2``. This is
    the ratio-estimator variance of Fischer et al. (arXiv:2510.25640, Eq. 5),
    written through the full covariance of every measured entry. It conditions
    on ``c``; when ``c`` was chosen from the same data, the selection is not in it.
    """
    S, H = np.asarray(S, dtype=complex), np.asarray(H, dtype=complex)
    size = S.shape[0]
    layout = _HermitianLayout(size)
    cov = _validated_covariance(covariance, size)
    c = np.asarray(coefficients, dtype=complex)
    if c.shape != (size,) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be a finite vector matching the matrices")
    phi = layout.features(c)
    count = size * size
    denominator = float(phi @ hermitian_parameters(S))
    numerator = float(phi @ hermitian_parameters(H))
    if not denominator > 0.0:
        raise ValueError("c^* S c must be positive for the ratio estimator")
    energy = numerator / denominator
    var_d = float(phi @ cov[:count, :count] @ phi)
    var_n = float(phi @ cov[count:, count:] @ phi)
    cov_nd = float(phi @ cov[:count, count:] @ phi)
    variance = max(0.0, var_n - 2.0 * energy * cov_nd + energy * energy * var_d)
    return {"energy": energy, "denominator": denominator,
            "standard_error": math.sqrt(variance / denominator ** 2),
            "denominator_standard_error": math.sqrt(max(0.0, var_d)),
            "denominator_cv": math.sqrt(max(0.0, var_d)) / denominator}


def solve_constrained_ratio(S: np.ndarray, H: np.ndarray, covariance: np.ndarray,
                            labels: Sequence[str] | None = None, *, error_budget: float,
                            max_denominator_cv: float = DEFAULT_RATIO_DENOMINATOR_CV,
                            tolerance: float = DEFAULT_RATIO_TOLERANCE,
                            max_iterations: int | np.integer = DEFAULT_RATIO_MAX_ITERATIONS,
                            norm_floor: float = DEFAULT_NORM_FLOOR,
                            resources: dict | None = None) -> SubspaceResult:
    """Minimize ``c^* H c / c^* S c`` subject to a propagated standard-error budget.

    This is the constrained subspace estimator of Fischer et al.
    (arXiv:2510.25640, Eq. 3): instead of inverting a noisy ``S``, choose the
    coefficients whose ratio-estimator standard error (see
    :func:`ratio_statistics`) is at most ``error_budget``, in the energy units
    of ``H``. ``covariance`` is the covariance of the measured ``S`` and ``H``
    coordinates, in :func:`hermitian_parameter_layout` order, ``S`` first.

    The declared starts include the first live generator, which for an
    identity first generator is the direct reference-energy estimate (the
    paper's ``c = (1, 0, ...)``), and the lowest root of the canonical
    thresholded solve. Four balanced phase mixtures with the component of
    that root orthogonal to the reference escape pure stationary starts.
    Mixtures are omitted when that component is numerically zero.
    The lowest feasible accepted endpoint is returned; a failed run is
    recorded under ``runs``, and only a total failure raises. If the
    reference itself is feasible, the result is never above it.
    Coefficients are complex, the live set and scaling follow the canonical
    solver, and dead rows are exactly zero. This bounded deterministic
    multistart search does not certify a global minimum.

    Two gates are additions of this implementation, not of the paper:
    ``c^* S c`` must be positive, and its coefficient of variation must be at
    most ``max_denominator_cv``, because the first-order ratio variance is
    only meaningful while the denominator is well resolved. The budget should
    be fixed independently of the data; the paper's rule, a multiple of the
    measured reference energy, is data-dependent and is not implemented.

    SLSQP (SciPy, the ``research`` extra) solves the program with analytic
    gradients under a unit-norm, fixed-phase gauge. ``tolerance`` is SLSQP's
    absolute objective precision in units of ``error_budget``. A line-search
    stop is accepted only at a feasible endpoint whose first-order KKT
    residual is certified small; other nonconvergence, or an endpoint
    violating the budget or gates, raises. No fallback hides a failure.

    Minimizing a noisy ratio is biased downward: the search exploits shot
    noise up to roughly the budget. The energy is variational only within the
    budget, and the propagated standard error conditions on the selected
    coefficients. Use the grouped bootstrap, which reruns the search, for
    intervals.
    """
    from ..capabilities import require

    require("sparse_linalg", feature="the constrained-ratio pencil estimator")
    from scipy.optimize import minimize, nnls

    for name, value in (("error_budget", error_budget),
                        ("max_denominator_cv", max_denominator_cv),
                        ("tolerance", tolerance)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
                or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be a positive finite real scalar")
    if not tolerance < 1.0:
        raise ValueError("tolerance must be in (0, 1)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) \
            or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    budget, kappa = float(error_budget), float(max_denominator_cv)
    S, H = np.asarray(S, dtype=complex), np.asarray(H, dtype=complex)
    frame = normalize_overlap(S, norm_floor=norm_floor)
    if H.shape != S.shape or not np.all(np.isfinite(H)):
        raise ValueError("Hamiltonian must be finite and match the overlap shape")
    if not np.allclose(H, H.conj().T, atol=1e-10, rtol=0.0):
        raise ValueError("Hamiltonian must be Hermitian")
    size = S.shape[0]
    labels = tuple(labels) if labels is not None else tuple(f"A{i}" for i in range(size))
    if len(labels) != size:
        raise ValueError("labels and matrix size disagree")
    cov = _validated_covariance(covariance, size)
    count = size * size
    cov_ss, cov_hh = cov[:count, :count], cov[count:, count:]
    cov_sh = 0.5 * (cov[:count, count:] + cov[count:, :count])
    layout = _HermitianLayout(size)
    theta_s, theta_h = hermitian_parameters(S), hermitian_parameters(H)
    live = frame.indices
    scale = frame.scaling[live]
    width = live.size

    def coefficients_of(y: np.ndarray) -> np.ndarray:
        c = np.zeros(size, dtype=complex)
        c[live] = y / scale
        return c

    def unpack(x: np.ndarray) -> np.ndarray:
        imaginary = np.concatenate(([0.0], x[width:]))
        return x[:width] + 1j * imaginary

    def real_gradient(wirtinger: np.ndarray) -> np.ndarray:
        # f real, df = 2 Re(g^* dc); c_live = y / scale.
        g = 2.0 * wirtinger[live] / scale
        return np.concatenate((g.real, g.imag[1:]))

    def evaluate(x: np.ndarray) -> dict:
        c = coefficients_of(unpack(x))
        phi = layout.features(c)
        D, N = float(phi @ theta_s), float(phi @ theta_h)
        E = N / D if D != 0.0 else math.inf
        a_ss, a_hh, a_sh = cov_ss @ phi, cov_hh @ phi, cov_sh @ phi
        var_d = float(phi @ a_ss)
        F = float(phi @ a_hh) - 2.0 * E * float(phi @ a_sh) + E * E * var_d
        gD = real_gradient(S @ c)
        gE = real_gradient((H @ c - E * (S @ c)) / D) if D != 0.0 else np.zeros_like(gD)
        dF_dE = -2.0 * float(phi @ a_sh) + 2.0 * E * var_d
        gF = real_gradient(2.0 * (layout.matrix(a_hh - 2.0 * E * a_sh + E * E * a_ss) @ c)) \
            + dF_dE * gE
        gV = real_gradient(2.0 * (layout.matrix(a_ss) @ c))
        return {"c": c, "D": D, "E": E, "F": F, "var_d": var_d,
                "gD": gD, "gE": gE, "gF": gF, "gV": gV}

    def statistics(x: np.ndarray) -> dict:
        point = evaluate(x)
        D = point["D"]
        if not D > 0.0:
            return {**point, "standard_error": math.inf, "denominator_cv": math.inf,
                    "feasible": False}
        sigma = math.sqrt(max(0.0, point["F"])) / D
        cv = math.sqrt(max(0.0, point["var_d"])) / D
        slack = 1.0 + 1e-6
        feasible = (D >= _RATIO_DENOMINATOR_FLOOR and sigma <= budget * slack
                    and cv <= kappa * slack)
        return {**point, "standard_error": sigma, "denominator_cv": cv, "feasible": feasible}

    def gauge(c: np.ndarray) -> np.ndarray:
        """Unit-norm normalized coordinates with a real first component."""
        y = c[live] * scale
        if abs(y[0]) > 0.0:
            y = y * (np.conjugate(y[0]) / abs(y[0]))
        y = y / np.linalg.norm(y)
        return np.concatenate((y.real, y.imag[1:]))

    x0 = np.zeros(2 * width - 1)
    x0[0] = 1.0
    start = statistics(x0)
    origin = start["E"] if start["D"] > 0.0 else 0.0
    starts: list[tuple[str, np.ndarray | None]] = [("reference", x0)]
    canonical_error: str | None = None
    if width > 1:
        # Both pure directions can be stationary: a feasible reference can
        # be a maximum while the canonical ground root violates the budget.
        # Balanced mixtures give SLSQP a nonstationary path between them.
        try:
            canonical = solve_projected(S, H, norm_floor=norm_floor)
            canonical_start = gauge(canonical.ritz_vector(0))
            starts.append(("canonical_ground_root", canonical_start))
            direction = unpack(canonical_start)
            direction[0] = 0.0
            length = float(np.linalg.norm(direction))
            if length > 100.0 * np.finfo(float).eps:
                direction /= length
                for suffix, phase in (("real_plus", 1.0), ("real_minus", -1.0),
                                      ("imag_plus", 1j), ("imag_minus", -1j)):
                    y = phase * direction
                    y[0] = 1.0
                    y /= math.sqrt(2.0)
                    starts.append((f"reference_canonical_mix_{suffix}",
                                   np.concatenate((y.real, y.imag[1:]))))
        except (ValueError, np.linalg.LinAlgError) as error:
            starts.append(("canonical_ground_root", None))
            canonical_error = str(error)
    runs: list[dict] = []
    candidates: list[tuple[dict, dict]] = []
    if width == 1:
        runs.append({"start": "reference", "stopping": "single_live_generator",
                     "iterations": 0, "evaluations": 0, "kkt_residual": None,
                     "energy": float(start["E"]), "feasible": bool(start["feasible"])})
        if start["feasible"]:
            candidates.append((start, runs[-1]))
    else:
        cache: dict = {}

        def point_at(x):
            key = x.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = evaluate(np.asarray(x, dtype=float))
            return cache[key]

        constraints = (
            {"type": "eq", "fun": lambda x: float(x @ x) - 1.0, "jac": lambda x: 2.0 * x},
            {"type": "ineq",
             "fun": lambda x: point_at(x)["D"] ** 2 - point_at(x)["F"] / budget ** 2,
             "jac": lambda x: 2.0 * point_at(x)["D"] * point_at(x)["gD"]
             - point_at(x)["gF"] / budget ** 2},
            {"type": "ineq",
             "fun": lambda x: point_at(x)["D"] ** 2 - point_at(x)["var_d"] / kappa ** 2,
             "jac": lambda x: 2.0 * point_at(x)["D"] * point_at(x)["gD"]
             - point_at(x)["gV"] / kappa ** 2},
            {"type": "ineq", "fun": lambda x: point_at(x)["D"] - _RATIO_DENOMINATOR_FLOOR,
             "jac": lambda x: point_at(x)["gD"]},
        )

        def objective(x):
            point = point_at(x)
            if not math.isfinite(point["E"]):
                return 1e300, np.zeros_like(x)
            return (point["E"] - origin) / budget, point["gE"] / budget

        def kkt_residual(x) -> float:
            # The objective gradient must be a nonnegative combination of the
            # active constraint gradients (the equality enters with both signs).
            gradient = objective(x)[1]
            rows = []
            for constraint in constraints:
                jacobian = constraint["jac"](x)
                if constraint["type"] == "eq":
                    rows.extend((jacobian, -jacobian))
                elif constraint["fun"](x) <= _RATIO_ACTIVE_TOLERANCE:
                    rows.append(jacobian)
            norm = float(np.linalg.norm(gradient))
            return float(nnls(np.array(rows).T, gradient)[1]) / norm if norm else 0.0

        for name, x_start in starts:
            if x_start is None:
                runs.append({"start": name, "stopping": "unavailable",
                             "error": canonical_error})
                continue
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                solution = minimize(objective, x_start, jac=True, method="SLSQP",
                                    constraints=constraints,
                                    options={"maxiter": int(max_iterations),
                                             "ftol": float(tolerance)})
            x_end = solution.x / math.sqrt(float(solution.x @ solution.x))
            point = statistics(x_end)
            run = {"start": name, "iterations": int(solution.nit),
                   "evaluations": int(solution.nfev), "message": str(solution.message),
                   "stopping": "converged", "kkt_residual": None}
            if not solution.success:
                # Status 8 is SLSQP's line-search stop, which at the precision
                # limit happens at converged points; accept it only at a
                # feasible, certified KKT point.
                residual = (kkt_residual(x_end)
                            if solution.status == 8 and point["feasible"] else None)
                if residual is None or residual > _RATIO_KKT_TOLERANCE:
                    run["stopping"] = "failed"
                    runs.append(run)
                    continue
                run.update({"stopping": "line_search_kkt_certified", "kkt_residual": residual})
            run.update({"energy": float(point["E"]) if point["D"] > 0.0 else None,
                        "feasible": bool(point["feasible"])})
            runs.append(run)
            if point["feasible"]:
                candidates.append((point, run))
    if not candidates:
        if all(run["stopping"] in ("failed", "unavailable") for run in runs):
            failed = next(run for run in runs if run["stopping"] == "failed")
            raise np.linalg.LinAlgError(
                f"constrained ratio did not converge: {failed['message']}")
        raise ValueError("constrained ratio found no coefficients within the error budget "
                         "and denominator gate")
    found, chosen = min(candidates, key=lambda entry: entry[0]["E"])
    selected = chosen["start"]
    optimizer_energy = float(found["E"])
    if start["feasible"] and start["E"] <= found["E"]:
        found, selected = start, "reference_start"
    c = found["c"] / math.sqrt(found["D"])
    c = fix_phase(c)
    values = np.linalg.eigvalsh(frame.matrix)
    largest = float(values[-1])
    condition = largest / float(values[0]) if values[0] > 0 else math.inf
    out = dict(resources or {})
    out.update({
        "basis_size": size,
        "live_generators": int(width),
        "dropped_generators": tuple(label for label, keep in zip(labels, frame.live)
                                    if not keep),
        "overlap_eigenvalue_max": largest,
        "overlap_eigenvalue_min": float(values[0]),
        "overlap_negative_modes": int((values < -1e-12 * max(largest, 1.0)).sum()),
        "constrained_ratio": {
            "method": "slsqp_first_order_ratio_variance",
            "start_policy": "reference_canonical_phase_mixtures_v1",
            "variance_model": "full_grouped_covariance_delta_method",
            "evidence": "heuristic",
            "error_budget": budget,
            "max_denominator_cv": kappa,
            "tolerance": float(tolerance),
            "max_iterations": int(max_iterations),
            "standard_error": float(found["standard_error"]),
            "denominator_cv": float(found["denominator_cv"]),
            "budget_active": bool(found["standard_error"] >= budget * (1.0 - 1e-6)),
            "denominator_gate_active": bool(found["denominator_cv"] >= kappa * (1.0 - 1e-6)),
            "selected": selected,
            "start_index": int(live[0]),
            "start_energy": float(start["E"]) if start["D"] > 0.0 else None,
            "start_standard_error": (float(start["standard_error"])
                                     if math.isfinite(start["standard_error"]) else None),
            "start_feasible": bool(start["feasible"]),
            "optimizer_energy": optimizer_energy,
            "stopping": chosen["stopping"],
            "kkt_residual": chosen["kkt_residual"],
            "runs": tuple(runs),
        },
    })
    return SubspaceResult(
        energies=(float(found["E"]),),
        coefficients=c[:, None],
        basis_labels=labels,
        overlap_eigenvalues=tuple(float(v) for v in values[::-1]),
        condition_number=condition,
        effective_rank=int(width),
        resources=out,
    )


def rotation_parameters(*, angle: float, energy_shift: float = 0.0,
                        energy_scale: float = 1.0, pole_tolerance: float = 1e-10
                        ) -> dict[str, float]:
    """Validate a fixed rotation policy; angles are radians, scale is positive.

    This policy contains no data-dependent angle selector. Choose the angle
    independently of evaluation data, or label any angle sweep exploratory.
    """
    values = {"angle": angle, "energy_shift": energy_shift,
              "energy_scale": energy_scale, "pole_tolerance": pole_tolerance}
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite real scalar")
        try:
            values[key] = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{key} must be a finite real scalar") from error
        if not math.isfinite(values[key]):
            raise ValueError(f"{key} must be a finite real scalar")
    if not -math.pi / 2 < values["angle"] < math.pi / 2:
        raise ValueError("angle must be in (-pi/2, pi/2)")
    if values["energy_scale"] <= 0:
        raise ValueError("energy_scale must be positive")
    if not 0 < values["pole_tolerance"] < 1:
        raise ValueError("pole_tolerance must be in (0, 1)")
    return values


def solve_rotated(S: np.ndarray, H: np.ndarray, labels: Sequence[str] | None = None, *, angle: float,
                  energy_shift: float = 0.0, energy_scale: float = 1.0,
                  pole_tolerance: float = 1e-10, tau_s: float = DEFAULT_TAU_S,
                  rel_tau: float = 0.0, max_condition: float = DEFAULT_MAX_CONDITION,
                  norm_floor: float = DEFAULT_NORM_FLOOR,
                  resources: dict | None = None) -> SubspaceResult:
    """Fixed-angle thresholding followed by the inverse eigenvalue map.

    On the original live generators, set h=(H-shift*S)/scale, then solve
    (h*cos(angle)-S*sin(angle), S*cos(angle)+h*sin(angle)). The canonical
    solver normalizes the rotated metric's diagonal before thresholding.
    Return physical energies sorted ascending and coefficients normalized
    in the supplied, unrotated S (the repaired S when repair was requested).

    This conservative first policy requires the rotated live metric to be
    PSD to roundoff, with positive diagonal. It rejects poles and nonpositive
    physical norms instead of dropping roots or crossing eigenvalue branches.
    Shot-derived floors and ridges are deliberately unsupported. Truncation
    changes the retained span; it need not preserve the unrotated Ritz values.
    Zero angle delegates exactly to the original numerical solver.

    See De Vriendt et al., arXiv:2602.11985, for rotated thresholding. This
    implementation does not implement their data-selected angle heuristic.
    """
    policy = rotation_parameters(angle=angle, energy_shift=energy_shift,
                                 energy_scale=energy_scale, pole_tolerance=pole_tolerance)
    S, H = np.asarray(S, dtype=complex), np.asarray(H, dtype=complex)
    frame = normalize_overlap(S, norm_floor=norm_floor)
    if H.shape != S.shape or not np.all(np.isfinite(H)):
        raise ValueError("Hamiltonian must be finite and match the overlap shape")
    if not np.allclose(H, H.conj().T, atol=1e-10, rtol=0.0):
        raise ValueError("Hamiltonian must be Hermitian")
    labels = tuple(labels) if labels is not None else tuple(f"A{i}" for i in range(len(S)))
    if len(labels) != len(S):
        raise ValueError("labels and matrix size disagree")
    if policy["angle"] == 0.0:
        result = solve_projected(S, H, labels, tau_s=tau_s, rel_tau=rel_tau,
                                 max_condition=max_condition, norm_floor=norm_floor,
                                 resources=resources)
        return replace(result, resources={**result.resources, "pencil_rotation": {
            **policy, "angle_policy": "fixed", "evidence": "heuristic",
            "zero_angle_control": True}})

    live = frame.indices
    S_live = frame.matrix
    H_live = ((H / frame.scaling[:, None]) / frame.scaling[None, :])[np.ix_(live, live)]
    h = (H_live - policy["energy_shift"] * S_live) / policy["energy_scale"]
    cosine, sine = math.cos(policy["angle"]), math.sin(policy["angle"])
    rotated_S, rotated_H = cosine * S_live + sine * h, cosine * h - sine * S_live
    if not np.all(np.isfinite(rotated_S)) or not np.all(np.isfinite(rotated_H)):
        raise ValueError("rotated pencil must be finite")
    metric_values = np.linalg.eigvalsh(rotated_S)
    roundoff = 100 * np.finfo(float).eps * max(float(np.max(np.abs(metric_values))), 1.0)
    if metric_values[0] < -roundoff:
        raise ValueError("rotated overlap is indefinite on the original live generators")
    if np.any(rotated_S.diagonal().real <= norm_floor ** 2):
        raise ValueError("rotated overlap has a nonpositive or dead live diagonal")
    result = solve_projected(rotated_S, rotated_H, [labels[i] for i in live],
                             tau_s=tau_s, rel_tau=rel_tau, max_condition=max_condition,
                             norm_floor=norm_floor, resources=resources)
    mu = np.asarray(result.energies)
    denominator = cosine - sine * mu
    pole_scale = np.maximum(1.0, np.abs(cosine) + np.abs(sine * mu))
    if np.any(denominator <= policy["pole_tolerance"] * pole_scale):
        raise ValueError("inverse rotation reaches a pole or nonpositive physical-norm branch")
    energies = policy["energy_shift"] + policy["energy_scale"] * (
        cosine * mu + sine) / denominator
    if not np.all(np.isfinite(energies)):
        raise ValueError("inverse rotation produced nonfinite physical energies")
    vectors = result.coefficients.copy()
    norms = np.einsum("ik,ik->k", vectors.conj(), S_live @ vectors).real
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("rotated Ritz vector has a nonpositive physical norm")
    vectors /= np.sqrt(norms)[None, :]
    coefficients = np.zeros((len(S), len(mu)), dtype=complex)
    coefficients[live, :] = vectors / frame.scaling[live, None]
    order = np.argsort(energies, kind="stable")
    coefficients = np.column_stack([fix_phase(coefficients[:, i]) for i in order])
    out = dict(result.resources)
    out.update({"basis_size": len(S), "live_generators": int(live.size),
                "pencil_rotation": {**policy, "angle_policy": "fixed", "evidence": "heuristic",
                    "zero_angle_control": False,
                    "rotated_metric_min_eigenvalue": float(metric_values[0]),
                    "rotated_energies": tuple(float(v) for v in mu[order]),
                    "inverse_denominators": tuple(float(v) for v in denominator[order]),
                    "original_live_indices": tuple(int(i) for i in live)},
                "dropped_generators": tuple(label for label, keep in zip(labels, frame.live)
                                            if not keep)})
    return replace(result, energies=tuple(float(e) for e in energies[order]),
                   coefficients=coefficients, basis_labels=labels, resources=out)
