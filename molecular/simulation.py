"""One molecular benchmark calculation; invoked in an isolated worker process."""
from __future__ import annotations

import gc
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.observables import double_occupancy, magnetization, total_spin_squared
from clifford_qc.prepared import prepare_fcidump
from clifford_qc.pipeline import solve_prepared
from clifford_qc.subspace import (determinant_excitations, identity_generator,
    lehmann_spectrum, occupied_spin_orbitals, solve_subspace, static_susceptibility)
from clifford_qc.subspace.adaptive import ACASEConfig
from .chemistry import prepare_chemistry

def determinant_ci_energy(hamiltonian, model, n, occupied, n_electrons, sz,
                          max_rank: int = 2, target_s2: float = 0.0,
                          s2_tol: float = 1e-6) -> tuple[float, int, float]:
    """Exact ground energy of the rank-``max_rank`` determinant space.

    An independent CISD reference, and a lesson about what "the CISD energy"
    means. An earlier version of this function took ``eigvalsh(block)[0]`` and
    concluded from the result that PySCF had failed by 55.2 mHa at H2O 2.5x.
    PySCF had not failed. The determinant block spans every spin state at this
    ``S_z``, and at that geometry its lowest root is a *quintet*
    (``<S^2> = 6``), 55.2 mHa below the lowest singlet -- which is exactly the
    energy PySCF returned. CISD from a closed-shell RHF reference means the
    singlet, and PySCF was giving it.

    So the root is chosen by multiplicity and only then by energy. Comparing a
    quintet against CCSD and RHF, which target the singlet, is not a comparison
    at all -- and reading the gap between them as an external-solver bug is how
    a sign error becomes an accusation.

    Restricting the Hamiltonian to determinants within ``max_rank`` excitations
    of the reference and diagonalizing that block is otherwise exact and needs
    no iteration to converge: 141x141 for water in STO-3G.

    Reuses SectorOperator.restrict for both H and S^2. Only the selected
    determinant block is dense; neither full-register sparse matrix is built.
    This remains a small-system reference calculation.
    """
    backend = SectorStatevectorBackend(n, n_electrons, sz)
    basis = backend.basis
    reference = 0
    for q in occupied:
        reference |= 1 << (n - 1 - q)
    rank = np.array([bin(int(b) ^ reference).count("1") // 2 for b in basis])
    kept = np.flatnonzero(rank <= max_rank)
    block = backend.operator(hamiltonian, precompute=False).restrict(kept)
    block = (block + block.conj().T) / 2.0
    # Select by multiplicity, not by energy. The determinant block spans every
    # spin state at this S_z, and at stretched geometries the lowest root is
    # not the singlet: at H2O 2.5x it is a quintet 55.2 mHa below the lowest
    # singlet. Taking eigvalsh()[0] there produced a "CISD" energy for a state
    # of different multiplicity than CCSD and RHF target, and then read the
    # gap as a PySCF failure. PySCF was returning the singlet, correctly.
    spin_block = backend.operator(
        total_spin_squared(model).to_mv(), precompute=False).restrict(kept)
    spin_block = (spin_block + spin_block.conj().T) / 2.0
    # Resolve spin before energy: degenerate H eigenvectors can be arbitrary
    # singlet/triplet mixtures, so filtering their <S^2> can miss valid roots.
    s2, spin_vectors = np.linalg.eigh(spin_block)
    match = np.flatnonzero(np.abs(s2 - target_s2) < s2_tol)
    if match.size == 0:
        raise RuntimeError(
            f"no root of the rank-{max_rank} determinant block has "
            f"<S^2> = {target_s2}; the lowest few are {s2[:4]}")
    spin_vectors = spin_vectors[:, match]
    h_spin = spin_vectors.conj().T @ block @ spin_vectors
    leakage = block @ spin_vectors - spin_vectors @ h_spin
    if np.linalg.norm(leakage) > 1e-10 * max(1.0, np.linalg.norm(block)):
        raise ValueError("target spin subspace is not invariant under the determinant Hamiltonian")
    energies, vectors = np.linalg.eigh((h_spin + h_spin.conj().T) / 2.0)
    ground = spin_vectors @ vectors[:, 0]
    ground_s2 = float(np.vdot(ground, spin_block @ ground).real)
    return float(energies[0]), int(kept.size), ground_s2


class PeakRSS:
    """Sample resident set size in a daemon thread while a block runs.

    ``PLAN.md`` §6 lists bank build time and peak memory beside
    basis size as resource metrics, on the grounds that a 20-dimensional basis
    is not compact if its projected entries cost gigabytes. The adaptive arm
    gets those from the bank for free; the complete-SD arm goes through
    ``solve_subspace`` with no bank, so there is nothing to ask.

    Sampling ``/proc/self/statm`` costs one read per interval and is what
    actually rose to ~8.8 GB on the H2O SD solve. ``tracemalloc`` would measure
    Python allocations more precisely but roughly doubles the runtime of a leg
    that already takes half an hour, and would still miss allocations numpy
    makes outside the Python allocator -- so the cheaper measurement of the
    quantity that matters wins.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _rss() -> int:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self._rss())
            except (OSError, ValueError, IndexError):
                return

    def __enter__(self) -> "PeakRSS":
        # Fail here rather than sampling zero. A zero baseline propagates into
        # the record as ``*_peak_rss_bytes: 0``, which check_molecular.py then
        # rejects as "the sampler never read a resident set size" -- a
        # confusing failure hours later, at the end of a run whose real problem
        # was that /proc was unavailable at the start.
        try:
            self.baseline = self.peak = self._rss()
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError(
                "cannot read /proc/self/statm, so peak RSS cannot be sampled; "
                "the §6 resource metrics would be recorded as zero and "
                "rejected by benchmarks/check_molecular.py"
            ) from exc
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, self._rss())
        except (OSError, ValueError, IndexError):
            pass

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.baseline)


def simulate(key: str, spec: dict, out_dir: Path, *, max_candidates=0,
             max_additions=None, target_error_mha=1.5936, storage="object",
             policy="retain_all", frontier_pairs=32, complete_sd="auto",
             reference_method="dense", refresh_chemistry=False) -> dict:
    """Return a comparative record; record/summary persistence belongs to run.py."""
    t0 = time.time()
    print(f"\n---> Processing {spec['name']} ({key.upper()})...")

    chemistry, fcidump_path, chemistry_hit = prepare_chemistry(
        key, spec, out_dir, refresh=refresh_chemistry)
    e_rhf, e_cisd, e_ccsd = (chemistry[name] for name in ("e_rhf", "e_cisd", "e_ccsd"))
    rhf_converged = chemistry["rhf_converged"]
    cisd_converged = chemistry["cisd_converged"]
    ccsd_converged = chemistry["ccsd_converged"]

    # 3. Ingest FCIDUMP with clifford_qc
    prepared, prepared_path, preparation_hit = prepare_fcidump(
        fcidump_path, out_dir / "cache" / "prepared")
    model = prepared.model()
    model.metadata["kind"] = "fermionic_lattice"
    model.metadata["sites"] = model.n // 2
    model.metadata["n_orbitals"] = 1
    n_electrons = model.metadata["n_electrons"]
    sz = model.metadata["sz"]
    occupied = occupied_spin_orbitals(model)
    all_candidates = determinant_excitations(model.n, occupied, max_rank=2)
    candidates = (
        all_candidates[:max_candidates]
        if max_candidates and max_candidates > 0
        else all_candidates
    )

    # The number of *distinct* Pauli words in H, after the word-code keyed
    # collection the MV product does on construction. This is not the
    # A-CASE ``word_universe``, which is the union over element operators
    # ``A_i'A_j`` and ``A_i'HA_j`` and is legitimately far larger because
    # those are triple products. Reporting one under the other's name was
    # what made the resource column look ~100x the literature value.
    hamiltonian_pauli_terms = len(model.hamiltonian.to_mv().terms)

    # Sector dimension: C(norb, n_alpha) * C(norb, n_beta).
    n_spatial = model.metadata["n_spatial_orbitals"]
    n_alpha = (n_electrons + int(round(2 * sz))) // 2
    n_beta = n_electrons - n_alpha
    sector_dimension = math.comb(n_spatial, n_alpha) * math.comb(n_spatial, n_beta)

    print(f"  Qubits: {model.n}, Electrons: {n_electrons}, Sz: {sz:+g}")
    print(f"  Distinct Pauli words in H: {hamiltonian_pauli_terms}")
    print(f"  Sector dimension (N={n_electrons}, Sz={sz:+g}): {sector_dimension}")
    print(f"  Candidate SD Excitations: {len(candidates)} (total available: {len(all_candidates)})")

    preparation_seconds = time.time() - t0
    t_reference_start = time.time()

    # 4. Sector-Exact FCI Reference via SectorStatevectorBackend
    exact_energies, _ = SectorStatevectorBackend(
        model.n, n_electrons=n_electrons, sz=sz
    ).ground_state(model.hamiltonian, k=1, method=reference_method)
    e_exact = float(exact_energies[0])
    print(f"  Exact Sector FCI E0: {e_exact:+.9f} Ha")

    # Our own CISD reference. PySCF's is cross-checked against it rather
    # than trusted; see determinant_ci_energy for why.
    e_cisd_det, n_sd_determinants, cisd_det_s2 = determinant_ci_energy(
        model.hamiltonian, model, model.n, occupied, n_electrons, sz,
        max_rank=2)
    cisd_agrees = abs(e_cisd_det - e_cisd) < 1e-6
    print(f"  Determinant CISD E0: {e_cisd_det:+.9f} Ha "
          f"({n_sd_determinants} determinants, "
          f"error {(e_cisd_det - e_exact) * 1000:+.4f} mHa)")
    if not cisd_agrees:
        print(f"  WARNING: PySCF CISD disagrees by "
              f"{(e_cisd - e_cisd_det) * 1000:+.4f} mHa at matched "
              f"multiplicity (<S^2> = {cisd_det_s2:.4f})")

    reference_seconds = time.time() - t_reference_start

    # The caller's explicit budget overrides the catalog's historical default.
    max_size = (max_additions if max_additions is not None
                else spec.get("max_additions", len(candidates)))
    target_error = target_error_mha / 1000.0 if target_error_mha > 0 else None

    t_adaptive_start = time.time()
    with PeakRSS() as adaptive_rss:
        adaptive, solver_record = solve_prepared(
            prepared,
            config=ACASEConfig(max_size=max_size, leakage_tol=1e-10,
                               exact_ground_energy=e_exact, target_error=target_error),
            max_candidates=max_candidates, storage=storage, policy=policy,
            frontier_pairs=frontier_pairs)
    rho0 = adaptive.bank.reference
    t_adaptive_elapsed = time.time() - t_adaptive_start
    e_acase = float(adaptive.result.ground_energy)
    err_ha = e_acase - e_exact
    err_mha = err_ha * 1000.0
    print(f"  Adaptive A-CASE E0: {e_acase:+.9f} Ha (Error: {err_mha:+.4f} mHa, Selected M={len(adaptive.labels)})")
    print(f"  Adaptive Stop Reason: {adaptive.stopped_reason}")

    run_complete_sd = (spec.get("complete_sd", True) if complete_sd == "auto"
                       else complete_sd == "on")
    sd_size = len(all_candidates) + 1
    sd_spans_full_sector = sd_size >= sector_dimension
    e_sd = err_sd_mha = t_sd_elapsed = sd_rss = None

    # 7. Response Analysis & Observables (on adaptive result)
    #
    # Every number below is read off ``adaptive.result`` -- the same solve
    # that produced ``e_acase`` above. Splicing observables from one run
    # beside energies from another produced a record in which the reported
    # Ritz vectors could not lower the reference energy by anything like
    # the reported amount, so this coupling is the point rather than an
    # implementation detail.
    spectrum = adaptive.result

    # An M-dimensional subspace cannot have effective rank above M. The
    # committed record briefly claimed rank 11 in a 6-dimensional space,
    # which is how the hand-edit was caught; assert it at the source.
    m_adaptive = len(adaptive.labels)
    if int(spectrum.effective_rank) > m_adaptive:
        raise AssertionError(
            f"{key}: effective_rank {spectrum.effective_rank} exceeds "
            f"basis size {m_adaptive}")

    double_occ = float(spectrum.expectation(double_occupancy(model)))
    spin2 = float(spectrum.expectation(total_spin_squared(model)))
    # Reference values that make <d> legible: the uncorrelated closed-shell
    # fraction. A correlated state sits strictly below it, so <d> equal to
    # this to many digits means no correlation was recovered.
    double_occ_hf = (n_electrons / 2) / n_spatial

    # Lehmann spectrum for the site-0 S_z response.
    mz_obs = magnetization(model, 0, axis="z")
    lehmann_lines = lehmann_spectrum(spectrum, mz_obs, min_weight=1e-12)
    chi0 = float(static_susceptibility(lehmann_lines))
    # This is a projected response. S_z conservation does not enforce S^2,
    # and a small selected space need not resolve the relevant excited states.
    significant_lines = [ln for ln in lehmann_lines if abs(float(ln.weight)) > 1e-8]
    response_diagnostic = (
        "no significant projected weight: the selected Ritz space has not "
        "been validated for magnetic response; chi0 is not a certified "
        "physical susceptibility (S_z conservation does not enforce total spin)"
        if not significant_lines else
        f"{len(significant_lines)} lines above weight 1e-8"
    )

    t_elapsed = time.time() - t0

    # Coefficients of ground state
    ritz_coeffs = {}
    for label, coeff in zip(adaptive.labels, spectrum.ritz_vector(0)):
        if abs(coeff) > 1e-6:
            ritz_coeffs[label] = {"real": float(coeff.real), "imag": float(coeff.imag)}

    # Format Lehmann response lines
    lehmann_data = [
        {
            "final_state": int(line.final_state),
            "excitation_energy_ha": float(line.excitation_energy),
            "weight": float(line.weight),
        }
        for line in lehmann_lines
    ]

    record = {
        "key": key,
        "name": spec["name"],
        "regime": spec.get("regime", "equilibrium"),
        "qubits": model.n,
        "n_electrons": n_electrons,
        "sz": sz,
        "sector_dimension": sector_dimension,
        "rhf_converged": rhf_converged,
        "cisd_converged": cisd_converged,
        "ccsd_converged": ccsd_converged,
        "e_rhf": e_rhf,
        "e_cisd": e_cisd,
        "e_ccsd": e_ccsd,
        "e_exact_fci": e_exact,
        "error_cisd_mha": (e_cisd - e_exact) * 1000.0,
        "e_cisd_determinant": e_cisd_det,
        "error_cisd_determinant_mha": (e_cisd_det - e_exact) * 1000.0,
        "sd_determinant_count": n_sd_determinants,
        "cisd_determinant_s2": cisd_det_s2,
        "cisd_pyscf_agrees": bool(cisd_agrees),
        "error_ccsd_mha": (e_ccsd - e_exact) * 1000.0,
        "e_acase_adaptive": e_acase,
        "error_adaptive_mha": err_mha,
        "complete_sd_run": run_complete_sd,
        "e_sd_complete": e_sd,
        "error_sd_mha": err_sd_mha,
        "adaptive_chemical_accuracy": bool(abs(err_mha) <= 1.5936),
        "cisd_chemical_accuracy":
            bool(abs((e_cisd - e_exact) * 1000.0) <= 1.5936),
        "ccsd_chemical_accuracy":
            bool(abs((e_ccsd - e_exact) * 1000.0) <= 1.5936),
        # A negative CCSD error is the point of the stretched rows: coupled
        # cluster is not variational, so at strong static correlation it can
        # and does land below the exact energy. A subspace method cannot.
        "ccsd_below_exact": bool(e_ccsd < e_exact),
        "sd_chemical_accuracy":
            None if err_sd_mha is None else bool(abs(err_sd_mha) <= 1.5936),
        "sd_spans_full_sector": bool(sd_spans_full_sector),
        # The adaptive stop is fed the exact energy, so "M at chemical
        # accuracy" is an oracle-assisted diagnostic and not a cost the
        # method could reproduce without knowing the answer.
        "adaptive_stop_reason": adaptive.stopped_reason,
        "adaptive_oracle_stop_used": bool(target_error is not None),
        "adaptive_target_error_mha": target_error_mha if target_error else None,
        "subspace_size_m": m_adaptive,
        "complete_sd_candidates": len(all_candidates),
        "complete_sd_basis_size": sd_size,
        "effective_rank": int(spectrum.effective_rank),
        "condition_number": float(spectrum.condition_number),
        "hamiltonian_pauli_terms": hamiltonian_pauli_terms,
        "element_word_universe": int(adaptive.resources.get("word_universe", 0)),
        # §6 resource accounting. The adaptive arm's figures are computed by
        # the bank and were previously discarded; only the SD arm's peak RSS
        # needed new instrumentation, because solve_subspace holds no bank
        # to ask.
        "adaptive_cached_operator_bytes":
            int(adaptive.resources.get("cached_operator_bytes", 0)),
        "adaptive_assemble_seconds":
            float(adaptive.resources.get("assemble_seconds", 0.0)),
        "adaptive_growth_seconds":
            float(adaptive.resources.get("growth_seconds", 0.0)),
        "adaptive_operator_products":
            int(adaptive.resources.get("operator_products", 0)),
        "adaptive_pairs_built": int(adaptive.resources.get("pairs_built", 0)),
        "adaptive_element_cache_hits":
            int(adaptive.resources.get("element_cache_hits", 0)),
        "adaptive_max_generator_support":
            int(adaptive.resources.get("max_generator_support", 0)),
        "adaptive_max_hamiltonian_element_support":
            int(adaptive.resources.get("max_hamiltonian_element_support", 0)),
        "adaptive_seconds": t_adaptive_elapsed,
        "adaptive_peak_rss_bytes": adaptive_rss.peak,
        "adaptive_peak_rss_delta_bytes": adaptive_rss.delta,
        "sd_peak_rss_bytes": None if sd_rss is None else sd_rss.peak,
        "sd_peak_rss_delta_bytes": None if sd_rss is None else sd_rss.delta,
        "double_occupancy": double_occ,
        "double_occupancy_uncorrelated": double_occ_hf,
        "total_spin_squared": spin2,
        "static_susceptibility_chi0": chi0,
        "response_diagnostic": response_diagnostic,
        "elapsed_seconds": t_elapsed,
        "sd_subspace_seconds": t_sd_elapsed,
        "ground_state_coefficients": ritz_coeffs,
        "lehmann_response": lehmann_data,
    }

    record.update({
        "chemistry_cache_hit": chemistry_hit,
        "preparation_cache_hit": preparation_hit,
        "problem_fingerprint": prepared.fingerprint,
        "solver_record": solver_record,
        "reference_method": reference_method,
        "preparation_seconds": preparation_seconds,
        "reference_seconds": reference_seconds,
    })
    del adaptive, spectrum, rho0
    # Projected response operators can hold their own coefficient rows. All are
    # local to the completed adaptive result; collect before allocating the SD arm.
    gc.collect()
    if run_complete_sd:
        rho0 = ExactMVBackend().state(model.reference, ())
        t_sd_start = time.time()
        with PeakRSS() as sd_rss:
            sd_spectrum = solve_subspace(
                rho0, model.hamiltonian,
                [identity_generator(model.n), *all_candidates]
            )
        t_sd_elapsed = time.time() - t_sd_start
        print(f"  SD Peak RSS: {sd_rss.peak / 2**30:.2f} GiB "
              f"(delta {sd_rss.delta / 2**30:.2f} GiB)")
        e_sd = float(sd_spectrum.ground_energy)
        err_sd_mha = (e_sd - e_exact) * 1000.0
        # When the SD basis already spans the whole sector, "exact to
        # machine precision" is arithmetic, not accuracy: there is nothing
        # left to miss. HF/STO-3G is the case here (35 candidates +
        # identity = 36 = the sector). Recording the flag stops that row
        # being read as a result.
        print(f"  Complete SD Subspace E0: {e_sd:+.9f} Ha "
              f"(Error: {err_sd_mha:+.4f} mHa, M={sd_size})")
        print(f"  SD Subspace Wall-Clock Time: {t_sd_elapsed:.1f}s "
              f"({t_sd_elapsed/60:.1f} min)")
        if sd_spans_full_sector:
            print(f"  NOTE: SD basis ({sd_size}) spans the full sector "
                  f"({sector_dimension}); exactness here is tautological.")
    else:
        e_sd = err_sd_mha = None
        t_sd_elapsed = None
        sd_rss = None
        print("  Complete SD Subspace: skipped; spin-checked determinant CISD "
              f"error {(e_cisd_det - e_exact) * 1000.0:+.4f} mHa")

    record.update({
        "e_sd_complete": e_sd,
        "error_sd_mha": err_sd_mha,
        "sd_chemical_accuracy": None if err_sd_mha is None else abs(err_sd_mha) <= 1.5936,
        "sd_subspace_seconds": t_sd_elapsed,
        "sd_peak_rss_bytes": None if sd_rss is None else sd_rss.peak,
        "sd_peak_rss_delta_bytes": None if sd_rss is None else sd_rss.delta,
        "elapsed_seconds": time.time() - t0,
    })
    return record
