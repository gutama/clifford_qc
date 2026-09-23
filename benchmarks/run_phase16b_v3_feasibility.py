"""Execute the third frozen Phase 16B decision experiment: the incumbent priced
with measurement grouping.

Inputs come from `configs/phase16b_v3_feasibility.json`, gated by
`check_phase16b_v3_preregistration.py`. Nothing here chooses a threshold, a
grid, a partition, or a rule; this module evaluates them.

What changes from v2 is the cost model, in two places.

**Settings, not components.** v2 charged a total budget across an arm's
independent real components. v3 charges it across an arm's *settings* under the
active grouping scheme, from the per-arm model frozen in the declaration. That
is the whole experiment: the incumbent's 7 926 words collapse into 913 QWC
settings or 65 fully-commuting ones, while `rt_unitary` gets no grouping at all
because each (lag, part) is its own controlled-evolution circuit.

**Covariance, not a bound.** v2 gave every Pauli word its own independent
Gaussian. Words sharing a setting are read from the same shots, so that is an
unlabelled approximation once grouping is on. Here the readouts are simulated.

The simulation is exact rather than asymptotic, and cheap, because the reference
is a computational basis state. For a commuting group of words measured
jointly, write each word's symplectic x-part (its X/Y support). Pick a maximal
subset of the group's words whose x-parts are linearly independent over GF(2);
their per-shot outcome signs are `k` independent fair coins, because every
nonempty product of them still has a nonzero x-part and hence zero expectation
on a basis state. Every other word's x-part is a GF(2) combination `a` of those,
so its per-shot sign is

    s_P  =  c_P * prod_{i in a} s_{b_i},        c_P = <b| P prod_{i in a} P_{b_i} |b>,

with `c_P = +-1` exactly, because the operator in it is Z-type. That identity is
per shot, not per expectation, so the `n`-shot sample mean of every word in the
group follows by Walsh-transforming one Multinomial(n, uniform on 2^k) draw of
the group's outcome counts. Two words correlate perfectly when their x-parts
agree and not at all when they differ -- the true within-group covariance of
this reference, carried exactly at O(2^k) cost per group with k <= n_qubits,
independent of the shot count. A budget of 1e20 shots is drawn as honestly as
one of 1e4.

`verify_group_construction` checks that structure against the operator algebra
for every group before any sampling: within each x-class, `c_P c_Q` must equal
the exact `<PQ>`, which is the pairwise covariance identity in closed form. On
QWC groups, where the setting *is* a per-qubit product basis, the construction
is additionally checked shot for shot against literal bitstring readouts.

Two readings of the declaration are recorded rather than assumed, because the
frozen text admits more than one:

* `rt_trotter`'s "full Hermitian pencil priced as `d`" is taken as `m(m+1)/2`
  distinct pencil entries at `G_H` settings each. `rt_trotter` is a diagnostic
  and cannot promote, so the reading does not reach the verdict.
* Under the `ungrouped` scheme the per-arm setting model still applies, so
  `rt_hermitian` pays `m * |H|` settings for `d` where v2 charged `d(k)` as a
  single estimand. The continuity clause in the config expects `ungrouped` to
  reproduce v2's ordering and calls any disagreement a producer defect. That
  reading is too narrow -- the declared setting model reprices the candidate as
  well as the incumbent -- but it is not wrong either: the first run of this
  module disagreed under `ungrouped` because `whiten` and `solve_unitary` had
  been rewritten from memory instead of carried over, and rt_unitary's
  *zero-noise* error was 0.67 Hartree. Both now come from v2 unchanged, and
  `tests/test_phase16b_v3_readouts.py` pins them there. The record reports the
  comparison; what a residual disagreement means is read from it, not
  predicted here.

    python benchmarks/run_phase16b_v3_feasibility.py
    python benchmarks/run_phase16b_v3_feasibility.py --replicas 20 --out draft.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.reproducibility import stamp_record

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs" / "phase16b_v3_feasibility.json"
DEFAULT_OUT = ROOT / "reference_results" / "phase16b_v3_feasibility.json"
SCHEMA = "clifford_qc.phase16b_v3_feasibility.v1"
PENCIL_IDENTITY_TOL = 1e-10
BUDGET_SWEEP = 2
MAX_SHOTS = 1 << 62


# ------------------------------------------------------- Pauli bookkeeping

def xz_parts(n, code):
    """Symplectic (x, z) bit masks of a packed Pauli word code."""
    x = z = 0
    for q in range(n):
        letter = (code >> (2 * q)) & 3
        if letter in (1, 2):
            x |= 1 << q
        if letter in (2, 3):
            z |= 1 << q
    return x, z


def basis_state_expectation(n, code, bits):
    """<b|P|b> for a computational basis state, which is 0 unless P is Z-type."""
    x, z = xz_parts(n, code)
    if x:
        return 0.0
    return -1.0 if bin(z & bits).count("1") % 2 else 1.0


def reference_bits(inst):
    """The reference's bitstring, read off Z expectations rather than an index.

    Taking it from the amplitude's position would bake in an ordering
    convention; <Z_q> is the same number whatever the convention is.
    """
    n, psi = inst["n_qubits"], inst["psi"]
    bits = 0
    for q in range(n):
        code = 3 << (2 * q)
        value = float(np.vdot(psi, to_matrix(MV(n, {code: 1.0})) @ psi).real)
        if abs(abs(value) - 1.0) > 1e-9:
            raise RuntimeError(f"{inst['name']}: <Z_{q}> = {value:.6f}, so the reference "
                               "is not a computational basis state")
        if value < 0:
            bits |= 1 << q
    return bits


def word_product(n, codes):
    """Phase and code of a product of Pauli words, via the package's own algebra."""
    product = MV(n, {0: 1.0 + 0j})
    for code in codes:
        product = product * MV(n, {code: 1.0 + 0j})
    (code, coefficient), = product.terms.items()
    return complex(coefficient), int(code)


# ------------------------------------------------------------- partitions

def partition_words(n, codes, scheme):
    """Measurement settings for a word list under one declared scheme."""
    from clifford_qc.ir import PauliWord

    traceless = [code for code in codes if code != 0]
    if scheme == "ungrouped":
        return [[code] for code in traceless]
    words = [PauliWord(n, code) for code in traceless]
    if scheme == "qwc":
        from clifford_qc.measurement.grouping import qwc_groups
        groups = qwc_groups(words)
    elif scheme == "fully_commuting":
        from clifford_qc.measurement.block_commuting import block_commuting_groups
        groups = block_commuting_groups(words, n)
    else:
        raise ValueError(f"unknown grouping scheme {scheme!r}")
    return [[int(word.code) for word in group] for group in groups]


class ReadoutPlan:
    """Simulated group readouts for one word list under one partition.

    Holds, per word, the group it is read in, the GF(2) coordinates `a` of its
    x-part in that group's coin basis, and the deterministic sign `c`. Sampling
    is then one Multinomial draw and one Walsh transform per group, whatever the
    shot count.
    """

    def __init__(self, n, codes, groups, bits):
        self.n = n
        self.codes = [code for code in codes if code != 0]
        self.groups = groups
        self.settings = len(groups)
        self.bits = bits
        self.slot = {code: index for index, code in enumerate(self.codes)}

        self.rank = []
        word_group = np.zeros(len(self.codes), dtype=np.int64)
        word_index = np.zeros(len(self.codes), dtype=np.int64)
        word_sign = np.zeros(len(self.codes), dtype=float)

        for position, group in enumerate(groups):
            rows, pivots, basis_words, coordinates = [], [], [], []
            for code in group:
                x, _ = xz_parts(n, code)
                value, mask = x, 0
                for slot, (pivot, row) in enumerate(zip(pivots, rows)):
                    if (value >> pivot) & 1:
                        value ^= row
                        mask ^= coordinates[slot]
                if value:
                    new_slot = len(basis_words)
                    basis_words.append(code)
                    pivots.append(value.bit_length() - 1)
                    rows.append(value)
                    coordinates.append(mask ^ (1 << new_slot))
                    mask = 1 << new_slot
                index = self.slot[code]
                word_group[index] = position
                word_index[index] = mask
                phase, product_code = word_product(
                    n, [code] + [basis_words[i] for i in range(len(basis_words))
                                 if (mask >> i) & 1])
                sign = complex(phase) * basis_state_expectation(n, product_code, bits)
                if abs(sign.imag) > 1e-12 or abs(abs(sign.real) - 1.0) > 1e-12:
                    raise RuntimeError(
                        f"group {position}: the deterministic sign of word {code} came out "
                        f"{sign!r}, which is not +-1, so the words in this setting do not "
                        "commute the way the scheme claims")
                word_sign[index] = float(sign.real)
            self.rank.append(len(basis_words))

        self.word_group = word_group
        self.word_index = word_index
        self.word_sign = word_sign
        self.exact = np.array(
            [basis_state_expectation(n, code, bits) for code in self.codes])
        self._buckets = {}
        for position, k in enumerate(self.rank):
            self._buckets.setdefault(k, []).append(position)
        self._bucket_arrays = {}
        for k, positions in self._buckets.items():
            order = {position: slot for slot, position in enumerate(positions)}
            mask = np.array([order.get(int(g), -1) for g in word_group])
            selected = np.nonzero([int(g) in order for g in word_group])[0]
            self._bucket_arrays[k] = (np.array(positions, dtype=np.int64),
                                      selected, mask[selected])

    @property
    def cells(self):
        return int(sum(1 << k for k in self.rank))

    @staticmethod
    def shot_chunks(shots):
        """Split a shot count into parts one int64 multinomial can carry.

        Multinomial(n, p) is the sum of independent Multinomial(n_j, p) whose
        n_j sum to n, and the Walsh transform is linear, so transforming the
        parts and adding is an identity rather than an approximation. The split
        is what keeps the counts and the transform's partial sums inside int64
        at the 1e20-shot end of the declared budget grid.
        """
        parts = max(1, -(-shots // MAX_SHOTS))
        base, extra = divmod(shots, parts)
        return [base + (1 if index < extra else 0) for index in range(parts)]

    def sample(self, shots, replicas, rng):
        """Sample-mean estimate of every word, from `shots` readouts per setting."""
        if shots < 1:
            raise ValueError("a setting cannot be read fewer than once")
        chunks = self.shot_chunks(int(shots))
        out = np.empty((replicas, len(self.codes)), dtype=float)
        for k, (_, selected, positions) in self._bucket_arrays.items():
            width = 1 << k
            pvals = np.full(width, 1.0 / width)
            transformed = np.zeros((replicas, len(self._buckets[k]), width))
            for part in chunks:
                counts = rng.multinomial(int(part), pvals,
                                         size=(replicas, len(self._buckets[k])))
                transformed += walsh(counts)
            chi = transformed / float(shots)
            # The constant character's sample mean is exactly one by
            # construction; float division of the count total is not.
            chi[..., 0] = 1.0
            out[:, selected] = (self.word_sign[selected]
                                * chi[:, positions, self.word_index[selected]])
        return out


def walsh(counts):
    """Unnormalized Walsh-Hadamard transform along the last axis, in int64.

    Integer arithmetic on purpose. The transform of a large count vector is a
    difference of terms near `shots / 2**k` whose result is of order
    `sqrt(shots)`, and in float64 that cancellation would eat most of the
    answer at the budgets this experiment prices.
    """
    out = np.array(counts, dtype=np.int64, copy=True)
    width = out.shape[-1]
    step = 1
    while step < width:
        reshaped = out.reshape(out.shape[:-1] + (width // (2 * step), 2, step))
        low = reshaped[..., 0, :].copy()
        high = reshaped[..., 1, :].copy()
        reshaped[..., 0, :] = low + high
        reshaped[..., 1, :] = low - high
        step *= 2
    return out


def verify_group_construction(plan, sample_shots, rng, literal_groups=0):
    """Check the character construction against the operator algebra, and shots.

    Three checks, two closed form and one by simulation.

    Closed form, over every group: within an x-class the product of two words is
    Z-type, so `c_P c_Q` must equal the exact `<PQ>`, and across classes the
    predicted covariance is zero for free because `<PQ>` is zero there; and every
    word's predicted expectation, `c_P` when its x-part vanishes and zero
    otherwise, must equal the independently computed `<P>`.

    By simulation, where a setting is a per-qubit product basis: literal
    bitstrings are drawn and the per-shot sign the construction predicts must
    match the drawn one shot for shot, not on average. That check is the
    independent one -- it re-derives every sign from outcomes rather than from
    the same algebra -- so it is run over every group of a QWC or ungrouped
    partition rather than a sample. A fully commuting setting is a stabilizer
    basis with no per-qubit form, so it has no literal counterpart; what carries
    it is that the signs there come from the same code path this check covers
    exhaustively on the product-basis partitions of the same instance.
    """
    n, bits = plan.n, plan.bits
    pairs = mismatches = 0
    for position, group in enumerate(plan.groups):
        classes = {}
        for code in group:
            classes.setdefault(int(plan.word_index[plan.slot[code]]), []).append(code)
        for members in classes.values():
            for first, second in zip(members, members[1:]):
                phase, product_code = word_product(n, [first, second])
                exact = complex(phase) * basis_state_expectation(n, product_code, bits)
                predicted = (plan.word_sign[plan.slot[first]]
                             * plan.word_sign[plan.slot[second]])
                pairs += 1
                if abs(complex(exact) - predicted) > 1e-9:
                    mismatches += 1
    deterministic = 0
    for index, code in enumerate(plan.codes):
        predicted = plan.word_sign[index] if plan.word_index[index] == 0 else 0.0
        if abs(predicted - plan.exact[index]) > 1e-12:
            mismatches += 1
        deterministic += int(plan.word_index[index] == 0)
    report = {"covariance_pairs_checked": pairs, "covariance_mismatches": mismatches,
              "expectation_identities_checked": len(plan.codes),
              "deterministic_words": deterministic}
    if mismatches:
        raise RuntimeError(f"{mismatches} of {pairs} within-class covariance identities "
                           "failed; the readout construction does not match the algebra")

    if literal_groups:
        checked = shots_checked = 0
        for position in range(min(literal_groups, len(plan.groups))):
            group = plan.groups[position]
            lanes = {}
            for code in group:
                for q in range(n):
                    letter = (code >> (2 * q)) & 3
                    if letter:
                        if lanes.setdefault(q, letter) != letter:
                            raise RuntimeError("a literal product-basis readout was asked "
                                               "for on a group that is not QWC")
            coins = sorted(q for q, letter in lanes.items() if letter in (1, 2))
            draws = rng.integers(0, 2, size=(sample_shots, len(coins)))
            outcome = np.zeros((sample_shots, n), dtype=np.int64)
            for slot, q in enumerate(coins):
                outcome[:, q] = draws[:, slot]
            for q, letter in lanes.items():
                if letter == 3:
                    outcome[:, q] = (bits >> q) & 1
            for code in group:
                index = plan.slot[code]
                support = [q for q in range(n) if (code >> (2 * q)) & 3]
                literal = 1 - 2 * (outcome[:, support].sum(axis=1) % 2)
                a = int(plan.word_index[index])
                predicted = np.full(sample_shots, plan.word_sign[index])
                for slot in range(plan.rank[position]):
                    if (a >> slot) & 1:
                        basis_code = _basis_code(plan, position, slot)
                        support_b = [q for q in range(n) if (basis_code >> (2 * q)) & 3]
                        predicted = predicted * (
                            1 - 2 * (outcome[:, support_b].sum(axis=1) % 2))
                if not np.array_equal(literal, predicted):
                    raise RuntimeError(f"group {position}: the character construction and a "
                                       "literal product-basis readout disagree shot for shot")
                checked += 1
                shots_checked += sample_shots
        report["literal_groups_checked"] = int(min(literal_groups, len(plan.groups)))
        report["literal_words_checked"] = checked
        report["literal_shots_checked"] = shots_checked
    return report


def _basis_code(plan, position, slot):
    """The group's `slot`-th coin word: the one that introduced that direction."""
    for code in plan.groups[position]:
        if int(plan.word_index[plan.slot[code]]) == (1 << slot):
            return code
    raise RuntimeError(f"group {position} has no word for coin direction {slot}")


# --------------------------------------------------------------- estimands

class Estimands:
    """Independent scalar estimands as arrays, with their variance weights."""

    def __init__(self, values, weights, complexity):
        self.values = np.asarray(values, dtype=complex)
        self.weights = np.asarray(weights, dtype=float)
        self.complexity = np.asarray(complexity, dtype=bool)

    @property
    def real_components(self):
        return int(self.complexity.sum() * 2 + (~self.complexity).sum())

    def draw(self, rng):
        """Standard normals, one pair per estimand; imaginary parts unused if real."""
        return rng.standard_normal((self.values.size, 2))

    def perturb(self, variates, eps_scale):
        """`eps_scale` is a scalar multiplier applied to each estimand's weight."""
        eps = self.weights * eps_scale
        out = self.values + eps * variates[:, 0]
        out = out + 1j * eps * variates[:, 1] * self.complexity
        return np.where(self.complexity, out, out.real + 0j)


def build_instance(name, spec):
    from clifford_qc.models import chemistry
    builder = spec["builder"].rsplit(".", 1)[-1]
    model = getattr(chemistry, builder)(**spec["args"])
    mv = model.hamiltonian.to_mv()
    H = 0.5 * (to_matrix(mv) + to_matrix(mv).conj().T)
    values, vectors = np.linalg.eigh(H)

    from clifford_qc.backends import ExactMVBackend
    rho = ExactMVBackend().state(model.reference, ())
    density = to_matrix(rho)
    weights, states = np.linalg.eigh(density)
    if abs(weights[-1] - 1.0) > 1e-9:
        raise RuntimeError(f"{name}: reference density is not a pure state")
    psi = states[:, -1] / np.linalg.norm(states[:, -1])

    lower, upper = float(values[0]), float(values[-1])
    dt = math.pi / (upper - lower)
    lam = float(sum(abs(v) for code, v in mv.terms.items() if code != 0))
    inst = {
        "name": name, "model": model, "mv": mv, "H": H, "psi": psi, "rho": rho,
        "values": values, "vectors": vectors, "exact": float(values[0]),
        "dt": dt, "energy_shift": float((lower + upper) / 2),
        "lower": lower, "upper": upper,
        "branch_ok": bool((upper - lower) * dt < 2 * math.pi),
        "enclosure_contains": bool(lower <= values[0] + 1e-9 and upper >= values[-1] - 1e-9),
        "lambda_one": lam, "n_qubits": model.hamiltonian.n,
        "dimension": int(H.shape[0]),
        "support": distinct_energy_support(values, vectors, psi),
        "ground_weight": float(abs(vectors[:, 0].conj() @ psi) ** 2),
    }
    inst["bits"] = reference_bits(inst)
    amplitudes = int(np.count_nonzero(np.abs(psi) > 1e-12))
    inst["basis_state_reference"] = bool(amplitudes == 1)
    inst["reference_amplitudes"] = amplitudes
    return inst


def distinct_energy_support(values, vectors, psi, floor=1e-6):
    weights = np.abs(vectors.conj().T @ psi) ** 2
    grouped, anchor = [], None
    for value, weight in zip(values, weights):
        if anchor is None or not np.isclose(value, anchor, atol=1e-10, rtol=1e-12):
            grouped.append(float(weight))
            anchor = value
        else:
            grouped[-1] += float(weight)
    return int(sum(weight > floor for weight in grouped))


def whiten(S, cutoff_relative=0.0, floor_absolute=0.0, ridge=0.0):
    norms = np.sqrt(np.clip(np.real(np.diag(S)), 0.0, None))
    live = norms > 1e-300
    if not live.any():
        return None, 0
    scale = np.where(live, norms, 1.0)
    bar = (S / scale[:, None]) / scale[None, :]
    index = np.flatnonzero(live)
    bar = bar[np.ix_(index, index)]
    bar = 0.5 * (bar + bar.conj().T)
    values, vectors = np.linalg.eigh(bar)
    if ridge:
        values = values + ridge
    largest = max(float(values[-1]), 1e-300)
    keep = values > max(cutoff_relative * largest, floor_absolute, 1e-300)
    if not keep.any():
        return None, 0
    transform = np.zeros((S.shape[0], int(keep.sum())), dtype=complex)
    transform[index, :] = vectors[:, keep] / np.sqrt(values[keep])
    return transform / scale[:, None], int(keep.sum())


def solve_hermitian(S, Hm, **policy):
    X, rank = whiten(S, **policy)
    if X is None:
        return float("nan"), 0
    reduced = X.conj().T @ Hm @ X
    reduced = 0.5 * (reduced + reduced.conj().T)
    if not np.all(np.isfinite(reduced)):
        return float("nan"), rank
    return float(np.linalg.eigvalsh(reduced)[0]), rank


def solve_unitary(c, m, dt, energy_shift, **policy):
    shifted = {k: value * np.exp(1j * energy_shift * k * dt) for k, value in c.items()}
    S0 = np.array([[shifted[j - i] for j in range(m)] for i in range(m)])
    S1 = np.array([[shifted[j - i + 1] for j in range(m)] for i in range(m)])
    S0 = 0.5 * (S0 + S0.conj().T)
    X, rank = whiten(S0, **policy)
    if X is None:
        return float("nan"), 0
    lam = np.linalg.eigvals(X.conj().T @ S1 @ X)
    if not np.all(np.isfinite(lam)) or np.any(np.abs(lam) <= 1e-12):
        return float("nan"), rank
    return float(energy_shift + np.min(-np.angle(lam) / dt)), rank


# ------------------------------------------------------------------- arms

def exact_lags(inst, max_lag):
    c = {0: complex(1.0)}
    d = {}
    for k in range(max_lag + 1):
        evolved = expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
        c[k] = complex(np.vdot(inst["psi"], evolved))
        d[k] = complex(np.vdot(inst["psi"], inst["H"] @ evolved))
        if k:
            c[-k] = c[k].conjugate()
            d[-k] = d[k].conjugate()
    d[0] = complex(d[0].real)
    return c, d


def trotter2_step_matrix(inst, microsteps):
    """Dense image of ``gates.trotter2_unitary(terms, dt, microsteps)``.

    Identical object, different arithmetic. Building it in the ``MV`` basis
    costs about five minutes per 8-qubit instance, because a Trotterized
    propagator's Pauli support fills in; the same product of
    ``rotor(P, theta) = cos(theta/2) I - i sin(theta/2) P`` factors in dense
    256x256 arithmetic takes under a second. `verify_trotter_step` checks the
    two against each other on a small instance so the substitution is a
    measured identity rather than an assumption.
    """
    n = inst["mv"].n
    dimension = 1 << n
    dt = inst["dt"] / microsteps
    terms = [(float(np.real(v)), code) for code, v in inst["mv"].terms.items() if code != 0]
    matrices = {code: to_matrix(MV(n, {code: 1.0})) for _, code in terms}
    identity = np.eye(dimension, dtype=complex)

    def rotor_matrix(coefficient, code):
        theta = coefficient * dt
        return (math.cos(theta / 2) * identity
                - 1j * math.sin(theta / 2) * matrices[code])

    half = identity
    for coefficient, code in terms:
        half = half @ rotor_matrix(coefficient, code)
    step = half
    for coefficient, code in reversed(terms):
        step = step @ rotor_matrix(coefficient, code)
    # Sequential, not matrix_power: the reference multiplies the step in one
    # by one, and binary squaring accumulates round-off differently. Matching
    # its order is what keeps the identity check at round-off instead of 1e-9.
    total = identity
    for _ in range(microsteps):
        total = total @ step
    return total


def verify_trotter_step(inst, microsteps, floor=1e-12, slack=10.0):
    """Check the dense step against the MV construction it stands in for.

    The two cannot be required to agree more tightly than the reference agrees
    with itself. On `h4_chain` the MV path's own unitarity defect is about
    9e-10 after 184 terms x 2 x 8 microsteps of sparse products, while the
    dense path's is about 6e-14 -- so the gap between them is the reference's
    accumulated round-off, and the dense construction is the more accurate of
    the two. The admissible gap is therefore scaled to the reference's measured
    defect rather than fixed, and both defects go into the record so the
    substitution is a reported measurement, not a silent swap.
    """
    from clifford_qc.gates import trotter2_unitary
    terms = [(float(np.real(v)), MV(inst["mv"].n, {code: 1.0}))
             for code, v in inst["mv"].terms.items() if code != 0]
    reference = to_matrix(trotter2_unitary(terms, inst["dt"], microsteps))
    dense = trotter2_step_matrix(inst, microsteps)
    identity = np.eye(reference.shape[0], dtype=complex)
    reference_defect = float(np.max(np.abs(reference.conj().T @ reference - identity)))
    dense_defect = float(np.max(np.abs(dense.conj().T @ dense - identity)))
    residual = float(np.max(np.abs(reference - dense)))
    tolerance = max(floor, slack * reference_defect)
    report = {"residual": residual, "tolerance": tolerance,
              "reference_unitarity_defect": reference_defect,
              "dense_unitarity_defect": dense_defect,
              "dense_is_more_accurate": bool(dense_defect < reference_defect)}
    if residual > tolerance:
        raise RuntimeError(f"dense Trotter step disagrees with trotter2_unitary by "
                           f"{residual:.3e}, beyond the reference's own {tolerance:.3e}")
    return report


_TROTTER_VERIFIED: dict[tuple[int, int], dict] = {}


def verified_construction(inst, microsteps):
    key = (inst["mv"].n, microsteps)
    if key not in _TROTTER_VERIFIED:
        report = verify_trotter_step(inst, microsteps)
        report["verified_on_instance"] = inst["name"]
        report["shared_by_qubit_count"] = key[0]
        _TROTTER_VERIFIED[key] = report
    return _TROTTER_VERIFIED[key]


def arm_rt_hermitian(inst, m, cache, g_h):
    c, d = cache["lags"]
    lam = inst["lambda_one"]
    values = [c[k] for k in range(1, m)] + [d[0]] + [d[k] for k in range(1, m)]
    weights = [1.0] * (m - 1) + [lam] * m
    complexity = [True] * (m - 1) + [False] + [True] * (m - 1)
    est = Estimands(values, weights, complexity)

    def assemble(noisy):
        cc = {0: complex(1.0)}
        dd = {0: complex(noisy[m - 1].real)}
        for k in range(1, m):
            cc[k] = noisy[k - 1]; cc[-k] = cc[k].conjugate()
            dd[k] = noisy[m - 1 + k]; dd[-k] = dd[k].conjugate()
        S = np.array([[cc[j - i] for j in range(m)] for i in range(m)])
        Hm = np.array([[dd[j - i] for j in range(m)] for i in range(m)])
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"kind": "gaussian", "estimands": est, "assemble": assemble,
            "c_slots": list(range(m - 1)),
            "settings": 2 * (m - 1) + m * g_h,
            "setting_breakdown": {"c": 2 * (m - 1), "d": m * g_h, "G_H": g_h}}


def arm_rt_unitary(inst, m, cache, g_h):
    c, _ = cache["lags"]
    est = Estimands([c[k] for k in range(1, m + 1)], [1.0] * m, [True] * m)

    def assemble(noisy):
        cc = {0: complex(1.0)}
        for k in range(1, m + 1):
            cc[k] = noisy[k - 1]; cc[-k] = cc[k].conjugate()
        return "unitary", cc

    return {"kind": "gaussian", "estimands": est, "assemble": assemble,
            "c_slots": list(range(m)), "settings": 2 * m,
            "setting_breakdown": {"c": 2 * m, "grouping": "none by declaration"}}


def arm_rt_trotter(inst, m, cache, g_h):
    step = cache.get("trotter_step")
    if step is None:
        step = trotter2_step_matrix(inst, cache["microsteps"])
        cache["trotter_step"] = step
        cache["trotter_verification"] = verified_construction(inst, cache["microsteps"])
    columns, current = [], inst["psi"].copy()
    for _ in range(m):
        columns.append(current.copy())
        current = step @ current
    V = np.column_stack(columns)
    lam = inst["lambda_one"]
    S_exact = V.conj().T @ V
    H_exact = V.conj().T @ (inst["H"] @ V)
    fidelity = float(abs(np.vdot(expm(-1j * inst["H"] * ((m - 1) * inst["dt"])) @ inst["psi"],
                                 V[:, m - 1])) ** 2)
    values = [S_exact[0, k] for k in range(1, m)]
    weights = [1.0] * (m - 1)
    complexity = [True] * (m - 1)
    pairs = []
    for i in range(m):
        values.append(complex(H_exact[i, i].real)); weights.append(lam)
        complexity.append(False); pairs.append((i, i))
        for j in range(i + 1, m):
            values.append(complex(H_exact[i, j])); weights.append(lam)
            complexity.append(True); pairs.append((i, j))
    est = Estimands(values, weights, complexity)

    def assemble(noisy):
        ss = {0: complex(1.0)}
        for k in range(1, m):
            ss[k] = noisy[k - 1]; ss[-k] = ss[k].conjugate()
        S = np.array([[ss[j - i] for j in range(m)] for i in range(m)])
        Hm = np.zeros((m, m), dtype=complex)
        for offset, (i, j) in enumerate(pairs):
            value = noisy[m - 1 + offset]
            if i == j:
                Hm[i, i] = value.real
            else:
                Hm[i, j] = value; Hm[j, i] = np.conj(value)
        return "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    entries = m * (m + 1) // 2
    return {"kind": "gaussian", "estimands": est, "assemble": assemble,
            "propagation_fidelity": fidelity,
            "trotter_step_verification": cache.get("trotter_verification"),
            "settings": 2 * (m - 1) + entries * g_h,
            "setting_breakdown": {"S": 2 * (m - 1), "pencil_entries": entries,
                                  "G_H": g_h,
                                  "reading": "the full Hermitian pencil priced as d"}}


def arm_power_krylov(inst, m, cache, scheme):
    """Moments assembled from grouped word readouts rather than charged whole."""
    n = inst["n_qubits"]
    powers = cache.setdefault("power_terms", {})
    need = 2 * m - 1
    if len(powers) < need:
        current = cache.get("power_running")
        if current is None:
            current = MV(n, {0: 1.0})
        start = len(powers) + 1
        for p in range(start, need + 1):
            current = current * inst["mv"]
            powers[p] = {code: complex(value) for code, value in current.terms.items()}
        cache["power_running"] = current

    union = sorted({code for p in range(1, need + 1) for code in powers[p]})
    plan = readout_plan(inst, union, scheme, cache)
    columns = {code: slot for slot, code in enumerate(plan.codes)}
    coefficients = np.zeros((need, len(plan.codes)))
    constant = np.zeros(need)
    # H is Hermitian, so every H^p has real Pauli coefficients exactly. Repeated
    # MV products of a 184-term operator do not: by p = 15 the support has
    # filled in and the imaginary parts carry round-off of order 1e-9 against
    # coefficients of order 1e13. The residual is therefore measured against the
    # operator's own scale and reported, rather than tested against an absolute
    # constant that says more about p than about the arithmetic.
    hermiticity = 0.0
    for p in range(1, need + 1):
        scale = max((abs(value) for value in powers[p].values()), default=1.0)
        for code, value in powers[p].items():
            hermiticity = max(hermiticity, abs(value.imag) / max(scale, 1e-300))
            if code == 0:
                constant[p - 1] += value.real
            else:
                coefficients[p - 1, columns[code]] += value.real
    if hermiticity > 1e-12:
        raise RuntimeError(f"H^p carries imaginary Pauli coefficients at relative size "
                           f"{hermiticity:.3e}; the moments are not real")

    # The moments are assembled from word expectations because that is how a
    # grouped readout would produce them, and that sum cancels coefficients of
    # order 1e13 down to a result of order 1e10. The dense propagator gives the
    # same numbers by a route with no such cancellation, so the gap between them
    # is this arm's conditioning, measured rather than assumed away.
    exact_moments = coefficients @ plan.exact + constant
    dense_moments, current = [], np.eye(inst["H"].shape[0], dtype=complex)
    for _ in range(need):
        current = current @ inst["H"]
        dense_moments.append(float(np.vdot(inst["psi"], current @ inst["psi"]).real))
    dense_moments = np.array(dense_moments)
    moment_residual = float(np.max(np.abs(exact_moments - dense_moments)
                                   / np.maximum(np.abs(dense_moments), 1.0)))

    def assemble_batch(word_matrix):
        moments = coefficients @ word_matrix + constant[:, None]
        for replica in range(moments.shape[1]):
            mu = {0: 1.0}
            for p in range(1, need + 1):
                mu[p] = float(moments[p - 1, replica])
            S = np.array([[mu[i + j] for j in range(m)] for i in range(m)], dtype=complex)
            Hm = np.array([[mu[i + j + 1] for j in range(m)] for i in range(m)], dtype=complex)
            yield "hermitian", (S, Hm)

    return {"kind": "readout", "plan": plan, "assemble_batch": assemble_batch,
            "settings": plan.settings, "word_count": len(plan.codes),
            "moment_conditioning": {"hermiticity_residual": hermiticity,
                                    "word_sum_vs_dense": moment_residual},
            "setting_breakdown": {"groups": plan.settings,
                                  "words": len(plan.codes),
                                  "readout_cells": plan.cells}}


def arm_acase(inst, cap, cache, scheme):
    """Frozen A-CASE basis; the pencil is one matmul against coefficient matrices."""
    from clifford_qc.models.chemistry import excitation_multivectors
    from clifford_qc.subspace import run_acase
    from clifford_qc.subspace.generators import Generator

    pool = cache.get("pool")
    if pool is None:
        electrons = int(inst["model"].metadata["n_electrons"])
        pool = [Generator(label, ps.to_mv()) for label, ps
                in excitation_multivectors(inst["n_qubits"], electrons)]
        cache["pool"] = pool

    built = cache.setdefault("acase_bank", {})
    if cap not in built:
        result = run_acase(inst["rho"], inst["model"].hamiltonian, pool, max_size=cap,
                           exact_ground_energy=None, target_error=None)
        built[cap] = result
    result = built[cap]
    bank, indices = result.result.bank, result.result.indices
    size = len(indices)
    codes = sorted(bank.word_set(indices))
    plan = readout_plan(inst, codes, scheme, cache)
    position = {code: slot for slot, code in enumerate(plan.codes)}

    key = ("acase_coefficients", cap)
    if key not in cache:
        coeff_s = np.zeros((size * size, len(plan.codes) + 1), dtype=complex)
        coeff_h = np.zeros((size * size, len(plan.codes) + 1), dtype=complex)
        identity_column = len(plan.codes)
        def column(code):
            if code == 0:
                return identity_column
            if code not in position:
                raise RuntimeError(f"pencil term {code} is outside the bank's own word "
                                   "universe, so it would be priced as the identity")
            return position[code]

        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                row = a * size + b
                for code, value in bank.iter_operator_terms("overlap", i, j):
                    coeff_s[row, column(code)] += value
                for code, value in bank.iter_operator_terms("element", i, j):
                    coeff_h[row, column(code)] += value
        cache[key] = (coeff_s, coeff_h)
    coeff_s, coeff_h = cache[key]

    def assemble_batch(word_matrix):
        full = np.vstack([word_matrix, np.ones((1, word_matrix.shape[1]))])
        S_all = coeff_s @ full
        H_all = coeff_h @ full
        for replica in range(full.shape[1]):
            S = S_all[:, replica].reshape(size, size)
            Hm = H_all[:, replica].reshape(size, size)
            yield "hermitian", (0.5 * (S + S.conj().T), 0.5 * (Hm + Hm.conj().T))

    return {"kind": "readout", "plan": plan, "assemble_batch": assemble_batch,
            "settings": plan.settings, "basis_size": size,
            "word_count": len(codes),
            "exact_energy": float(result.result.ground_energy),
            "setting_breakdown": {"groups": plan.settings, "words": len(codes),
                                  "readout_cells": plan.cells}}


def readout_plan(inst, codes, scheme, cache, seed_root=0):
    """Build (and verify, once) the readout plan for one word list and scheme."""
    key = ("plan", scheme, hash(tuple(codes)))
    if key not in cache:
        groups = partition_words(inst["n_qubits"], codes, scheme)
        plan = ReadoutPlan(inst["n_qubits"], codes, groups, inst["bits"])
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed_root, len(codes), plan.settings, 7717]))
        # A QWC setting is a per-qubit product basis, so the construction can be
        # checked against literal bitstrings there. A fully commuting setting is
        # a stabilizer basis and has no per-qubit form; the closed-form pairwise
        # identity covers every group in either case.
        literal = len(groups) if scheme in ("qwc", "ungrouped") else 0
        plan.verification = verify_group_construction(plan, 256, rng, literal)
        plan.verification["scheme"] = scheme
        plan.verification["literal_readouts_available"] = bool(literal)
        cache[key] = plan
    return cache[key]


# ---------------------------------------------------------------- sweeps

def shared_variates(entropy, replicas, max_lag):
    """Paired c-variates for the real-time arms: one draw per replica.

    The first version of this built a fresh generator from the same
    ``SeedSequence`` inside a per-replica comprehension, so all ``replicas``
    entries were the identical array and every real-time cell recorded one
    realization two hundred times. The tell was in the record and went unread:
    median and p90 were bit-identical in every winning cell.

    One generator, advanced once, drawn as a single ``(replicas, lags, 2)``
    block. Pairing across arms is preserved because both real-time arms index
    into the same block for the lags they share; it is the pairing that has to
    be common between arms, not the draw that has to be common between
    replicas.
    """
    if replicas < 1:
        raise ValueError("replicas must be positive")
    rng = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    block = rng.standard_normal((replicas, max_lag + 1, 2))
    if replicas > 1 and np.all(block == block[0]):
        raise RuntimeError("shared variates are identical across replicas")
    return block


def solve(kind, data, inst, m, policy):
    if kind == "unitary":
        return solve_unitary(data, m, inst["dt"], inst["energy_shift"], **policy)
    return solve_hermitian(*data, **policy)


def summarize(energies, exact):
    values = np.asarray(energies, dtype=float)
    errors = np.where(np.isfinite(values), np.abs(values - exact), np.inf)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.9, method="higher")),
        "failures": int((~np.isfinite(values)).sum()),
        "replicas": int(errors.size),
    }


def policies(config):
    ridge = float(config["regularizers"]["ridge"]["coefficient"])
    return {
        "hard_truncation": lambda eps: {"cutoff_relative": max(eps, 1e-13)},
        "per_mode_floor": lambda eps: {"floor_absolute": max(eps, 1e-13)},
        "ridge": lambda eps: {"cutoff_relative": 1e-13, "ridge": ridge},
    }


def zero_noise(arm, inst, m, policy):
    if arm["kind"] == "readout":
        kind, data = next(arm["assemble_batch"](arm["plan"].exact[:, None]))
    else:
        kind, data = arm["assemble"](arm["estimands"].values)
    return solve(kind, data, inst, m, policy)


def run_cell(arm, inst, m, budget, policy_fn, replicas, rng, shared=None):
    """One (arm, basis size, budget, regularizer) cell under the active scheme."""
    settings = int(arm["settings"])
    energies, ranks = [], []
    if arm["kind"] == "readout":
        shots = max(1, int(round(budget / settings)))
        samples = arm["plan"].sample(shots, replicas, rng)
        eps = 1.0 / math.sqrt(shots)
        for kind, data in arm["assemble_batch"](samples.T):
            energy, rank = solve(kind, data, inst, m, policy_fn(eps))
            energies.append(energy)
            ranks.append(rank)
    else:
        shots = budget / settings
        eps = math.sqrt(settings / budget)
        est = arm["estimands"]
        for replica in range(replicas):
            variates = est.draw(rng)
            if shared is not None and "c_slots" in arm:
                slots = arm["c_slots"]
                variates[: len(slots)] = shared[replica][: len(slots)]
            kind, data = arm["assemble"](est.perturb(variates, eps))
            energy, rank = solve(kind, data, inst, m, policy_fn(eps))
            energies.append(energy)
            ranks.append(rank)
    stats = summarize(energies, inst["exact"])
    stats["median_rank"] = float(np.median(ranks))
    stats["eps_bounded"] = float(eps)
    stats["settings"] = settings
    stats["shots_per_setting"] = float(shots)
    return stats


def shots_to_target(arm_entry, reg_name, target):
    table = arm_entry.get("budget", {}).get(reg_name, {})
    best = None
    for size_key, budgets in table.items():
        for budget_key, stats in budgets.items():
            if stats["median"] <= target:
                budget = float(budget_key)
                if best is None or budget < best["budget"]:
                    best = {"budget": budget, "basis_size": int(size_key),
                            "eps_bounded": stats["eps_bounded"],
                            "settings": stats["settings"],
                            "median": stats["median"], "p90": stats["p90"],
                            "failures": stats["failures"]}
    return best


def instance_status(entry, config, target):
    if entry.get("status") == "INVALID":
        return "INVALID", {"reason": "a deterministic check failed"}
    non_ridge = list(config["non_ridge_regularizers"])
    all_regs = list(config["regularizers"])
    incumbent = config["incumbent_arm"]
    arms = entry["arms"]

    found = {}
    for reg in all_regs:
        row = shots_to_target(arms.get(incumbent, {}), reg, target)
        if row:
            found[reg] = row
    incumbent_best = min(found.values(), key=lambda r: r["budget"]) if found else None

    candidates = []
    for arm_name in config["exact_propagation_arms"]:
        for reg in all_regs:
            row = shots_to_target(arms.get(arm_name, {}), reg, target)
            if not row:
                continue
            ratio = (incumbent_best["budget"] / row["budget"]) if incumbent_best else None
            candidates.append({
                "arm": arm_name, "regularizer": reg, "is_ridge": reg not in non_ridge,
                "rt_budget": row["budget"], "rt_basis_size": row["basis_size"],
                "rt_settings": row["settings"], "c_noise": row["eps_bounded"],
                "incumbent_budget": incumbent_best["budget"] if incumbent_best else None,
                "incumbent_settings": incumbent_best["settings"] if incumbent_best else None,
                "ratio": ratio,
                "qualifying": bool(ratio is not None and ratio >= 10.0),
            })

    qualifying = [row for row in candidates if row["qualifying"]]
    passes = [row for row in qualifying if not row["is_ridge"] and row["c_noise"] >= 1e-5]
    marginals = [row for row in qualifying if row["c_noise"] >= 1e-6]
    evidence = {"incumbent_shots_to_target": incumbent_best,
                "incumbent_censored": incumbent_best is None,
                "candidates": candidates}
    if passes:
        return "PASS", {**evidence, "basis": passes}
    if marginals:
        return "MARGINAL", {**evidence, "basis": marginals}
    if incumbent_best is None or not candidates:
        return "UNDETERMINED", {**evidence,
                                "reason": "a required cost comparison is censored on the tested grid"}
    return "FAIL", {**evidence,
                    "reason": "all comparisons complete; no qualifying candidate at c noise >= 1e-6"}


def combine(statuses, config):
    values = [statuses.get(name) for name in config["required_decision_instances"]]
    if any(value is None for value in values):
        return "INCOMPLETE"
    if "INVALID" in values:
        return "INVALID"
    if all(value == "PASS" for value in values):
        return "GO"
    if all(value == "FAIL" for value in values):
        return "NO_GO"
    return "CONDITIONAL"


VERDICT_SEVERITY = {"GO": 0, "CONDITIONAL": 1, "NO_GO": 2, "INCOMPLETE": 3, "INVALID": 4}


def least_favourable(verdicts):
    """The declared rule: a GO must survive the incumbent's best grouping."""
    return max(verdicts.items(), key=lambda kv: (VERDICT_SEVERITY[kv[1]], kv[0]))


def hamiltonian_group_count(inst, scheme, cache):
    key = ("G_H", scheme)
    if key not in cache:
        codes = [code for code in inst["mv"].terms if code != 0]
        cache[key] = len(partition_words(inst["n_qubits"], codes, scheme))
    return cache[key]


def continuity_against_v2(record, config):
    """Compare the ungrouped scheme against the committed v2 ordering.

    The config calls a disagreement here a defect in this producer. That is the
    right instinct and it has already earned its keep: the first run of this
    module disagreed under `ungrouped`, and the cause was a defect -- two
    solvers rewritten from memory rather than carried over from v2, one of which
    put rt_unitary's zero-noise error at 0.67 Hartree. The clause is still too
    narrow, because the per-arm setting model in Section 3 applies under every
    scheme and charges `rt_hermitian` `m * |H|` settings for `d` where v2 charged
    `d(k)` once, so some repricing of the candidate is expected. Both readings
    are recorded. Which one a disagreement supports is decided by looking, not
    by this docstring, and nothing here changes a verdict.
    """
    path = ROOT / "reference_results" / "phase16b_v2_feasibility.json"
    if not path.exists():
        return {"available": False, "reason": "no committed v2 record to compare against"}
    v2 = json.loads(path.read_text(encoding="utf-8"))
    out = {"available": True, "v2_config_digest": v2.get("config_digest"), "instances": {}}
    agree = True
    for name in config["required_decision_instances"]:
        v2_entry = v2.get("instances", {}).get(name, {})
        v3_entry = (record["instances"].get(name, {})
                    .get("schemes", {}).get("ungrouped", {}))
        v2_arms = sorted({row["arm"] for row
                          in v2_entry.get("status_evidence", {}).get("candidates", [])
                          if row.get("qualifying")})
        v3_arms = sorted({row["arm"] for row
                          in v3_entry.get("status_evidence", {}).get("candidates", [])
                          if row.get("qualifying")})
        same = v2_arms == v3_arms
        agree &= same
        out["instances"][name] = {
            "v2_qualifying_arms": v2_arms, "v3_ungrouped_qualifying_arms": v3_arms,
            "agree": bool(same),
            "v2_status": v2_entry.get("status"), "v3_ungrouped_status": v3_entry.get("status"),
        }
    out["orderings_agree"] = bool(agree)
    out["readings"] = {
        "producer_defect": (
            "The config's own clause. It caught one: the first run of this producer "
            "disagreed under ungrouped because whiten and solve_unitary had been "
            "rewritten instead of carried over from v2, leaving rt_unitary's zero-noise "
            "error at 0.67 Ha. Both are now v2's unchanged and pinned by regressions."),
        "declared_setting_model": (
            "Under ungrouped the per-arm setting model still applies, so rt_hermitian "
            "pays m * |H| settings for d where v2 charged d(k) once. Repricing of the "
            "candidate under ungrouped is therefore expected and is not a defect."),
        "note": "Which reading a residual disagreement supports is read from this record.",
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--schemes", nargs="*", default=None)
    args = parser.parse_args()

    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    target = float(config["target"]["value"])
    replicas = args.replicas or int(config["noise_model"]["replicas"])
    seed_root = int(config["noise_model"]["seed_root"])
    budgets = [10.0 ** d for d in config["cost_model"]["budget_grid_decades"]]
    grids = config["basis_sizes_by_arm"]
    caps = list(config["acase_pool"]["growth"]["caps"])
    policy_map = policies(config)
    incumbent = config["incumbent_arm"]
    required = set(config["required_decision_instances"])
    schemes = args.schemes or list(config["grouping_schemes"])
    primary = [s for s in config["decision_rule"]["primary_schemes"] if s in schemes]

    names = args.instances or list(config["instances"])
    record = {"schema": SCHEMA, "config_digest": digest,
              "config_path": str(CONFIG.relative_to(ROOT.parent)),
              "evidence": config["noise_model"]["evidence_label"],
              "quantum_advantage_claim": False,
              "cost_contract": config["cost_model"]["contract"],
              "covariance_treatment": config["covariance_treatment"]["method"],
              "replicas": replicas, "target": target, "schemes": schemes,
              "instances": {}, "decision": {}}
    started = time.time()

    for position, name in enumerate(names):
        spec = config["instances"][name]
        try:
            inst = build_instance(name, spec)
        except ImportError as exc:
            record["instances"][name] = {"skipped": f"optional dependency missing: {exc}"}
            print(f"SKIP {name}: {exc}")
            continue
        cache = {"microsteps": int(config["arms"]["rt_trotter"]["microsteps_per_dt"])}
        max_lag = max(max(grids["rt_unitary"]), max(grids["rt_hermitian"]))
        cache["lags"] = exact_lags(inst, max_lag)

        entry = {
            "n_qubits": inst["n_qubits"], "dimension": inst["dimension"],
            "exact_ground_energy": inst["exact"], "dt": inst["dt"],
            "energy_shift": inst["energy_shift"], "lambda_one": inst["lambda_one"],
            "distinct_energy_support": inst["support"],
            "ground_space_weight": inst["ground_weight"],
            "enclosure": [inst["lower"], inst["upper"]],
            "role": spec["role"], "schemes": {},
            "reference_precondition": {
                "basis_state": inst["basis_state_reference"],
                "nonzero_amplitudes": inst["reference_amplitudes"],
                "bitstring": format(inst["bits"], f"0{inst['n_qubits']}b")[::-1],
            },
            "checks": {"branch_condition": inst["branch_ok"],
                       "enclosure_contains_spectrum": inst["enclosure_contains"]},
        }
        print(f"\n=== {name}: {inst['n_qubits']}q dim={inst['dimension']} "
              f"support={inst['support']} dt={inst['dt']:.5f} "
              f"Lambda={inst['lambda_one']:.3f} ===")

        residual = 0.0
        c, d = cache["lags"]
        for m in grids["rt_hermitian"]:
            S = np.array([[c[j - i] for j in range(m)] for i in range(m)])
            Hm = np.array([[d[j - i] for j in range(m)] for i in range(m)])
            V = np.column_stack([expm(-1j * inst["H"] * (k * inst["dt"])) @ inst["psi"]
                                 for k in range(m)])
            residual = max(residual, float(np.max(np.abs(S - V.conj().T @ V))),
                           float(np.max(np.abs(Hm - V.conj().T @ (inst["H"] @ V)))
                                 / max(inst["lambda_one"], 1.0)))
        entry["checks"]["pencil_identity"] = residual
        entry["checks"]["pencil_identity_ok"] = bool(residual <= PENCIL_IDENTITY_TOL)
        if not inst["basis_state_reference"] and name in required:
            entry["checks"]["basis_state_reference_ok"] = False
        if not all((inst["branch_ok"], inst["enclosure_contains"],
                    entry["checks"]["pencil_identity_ok"],
                    inst["basis_state_reference"] or name not in required)):
            entry["status"] = "INVALID"
            record["instances"][name] = entry
            print("  INVALID: a deterministic check failed")
            continue

        for scheme_index, scheme in enumerate(schemes):
            g_h = hamiltonian_group_count(inst, scheme, cache)
            scheme_entry = {"hamiltonian_settings": g_h, "arms": {}, "verifications": {}}
            print(f"  -- scheme {scheme}: G_H = {g_h}")

            builders = {
                "rt_hermitian": (grids["rt_hermitian"],
                                 lambda i, m, c_, g=g_h: arm_rt_hermitian(i, m, c_, g)),
                "rt_unitary": (grids["rt_unitary"],
                               lambda i, m, c_, g=g_h: arm_rt_unitary(i, m, c_, g)),
                "rt_trotter": (grids["rt_trotter"],
                               lambda i, m, c_, g=g_h: arm_rt_trotter(i, m, c_, g)),
                "power_krylov_control": (grids["power_krylov_control"],
                                         lambda i, m, c_, s=scheme: arm_power_krylov(i, m, c_, s)),
                incumbent: (caps, lambda i, m, c_, s=scheme: arm_acase(i, m, c_, s)),
            }

            for arm_name, (sizes, builder) in builders.items():
                arm_entry = {"budget": {}, "sizes": {}}
                for m in sizes:
                    built = builder(inst, m, cache)
                    zero, zero_rank = zero_noise(built, inst, m,
                                                 policy_map["hard_truncation"](0.0))
                    info = {"settings": int(built["settings"]),
                            "setting_breakdown": built["setting_breakdown"],
                            "zero_noise_error": (abs(zero - inst["exact"])
                                                 if np.isfinite(zero) else None),
                            "zero_noise_rank": zero_rank}
                    for extra in ("basis_size", "word_count", "propagation_fidelity",
                                  "trotter_step_verification", "moment_conditioning"):
                        if extra in built:
                            info[extra] = built[extra]
                    if built["kind"] == "readout":
                        scheme_entry["verifications"][f"{arm_name}@{m}"] = \
                            built["plan"].verification
                    arm_entry["sizes"][str(m)] = info

                    for cell, budget in enumerate(budgets):
                        for reg_index, (reg_name, policy_fn) in enumerate(policy_map.items()):
                            rng = np.random.default_rng(np.random.SeedSequence(
                                [seed_root, position, scheme_index, BUDGET_SWEEP,
                                 cell, m, reg_index]))
                            shared = (shared_variates(
                                [seed_root, position, scheme_index, BUDGET_SWEEP,
                                 cell, m, reg_index, 99], replicas, max_lag)
                                if "c_slots" in built else None)
                            stats = run_cell(built, inst, m, budget, policy_fn,
                                             replicas, rng, shared)
                            bucket = (arm_entry["budget"].setdefault(reg_name, {})
                                      .setdefault(str(m), {}))
                            bucket[f"1e{int(round(math.log10(budget)))}"] = stats
                scheme_entry["arms"][arm_name] = arm_entry
                print(f"     {arm_name}: done")

            entry["schemes"][scheme] = scheme_entry

        reference_scheme = schemes[0]
        arms0 = entry["schemes"][reference_scheme]["arms"]
        if name in required:
            inc = min((v["zero_noise_error"] for v in arms0[incumbent]["sizes"].values()
                       if v["zero_noise_error"] is not None), default=float("inf"))
            cand = min((v["zero_noise_error"]
                        for arm_name in config["exact_propagation_arms"]
                        for v in arms0[arm_name]["sizes"].values()
                        if v["zero_noise_error"] is not None), default=float("inf"))
            entry["admission"] = {"incumbent_zero_noise_error": inc,
                                  "candidate_zero_noise_error": cand,
                                  "read_from_scheme": reference_scheme,
                                  "admits": bool(inc <= target and cand <= target)}
        record["instances"][name] = entry

    per_scheme = {}
    for scheme in schemes:
        statuses = {}
        for name, entry in record["instances"].items():
            if "skipped" in entry:
                continue
            if entry.get("status") == "INVALID":
                statuses[name] = "INVALID"
                continue
            scheme_entry = entry["schemes"][scheme]
            status, evidence = instance_status(scheme_entry, config, target)
            scheme_entry["status"] = status
            scheme_entry["status_evidence"] = evidence
            statuses[name] = status
        per_scheme[scheme] = {"instance_statuses": statuses,
                              "verdict": combine(statuses, config)}
        print(f"\n[{scheme}] " + "  ".join(f"{k}={v}" for k, v in statuses.items())
              + f"  -> {per_scheme[scheme]['verdict']}")

    primary_verdicts = {s: per_scheme[s]["verdict"] for s in primary}
    if primary_verdicts:
        worst_scheme, worst = least_favourable(primary_verdicts)
    else:
        worst_scheme, worst = None, "INCOMPLETE"
    record["decision"] = {
        "per_scheme": per_scheme,
        "primary_schemes": primary,
        "primary_verdicts": primary_verdicts,
        "least_favourable_scheme": worst_scheme,
        "verdict": worst,
        "required_instances": config["required_decision_instances"],
        "rule": "frozen in the config; see decision_rule and verdict_scheme_rule",
    }
    record["continuity_check"] = continuity_against_v2(record, config)
    print(f"\nVERDICT: {worst} (least favourable primary scheme: {worst_scheme})")

    record["elapsed_seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamp_record(record), indent=1, sort_keys=True),
                        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
