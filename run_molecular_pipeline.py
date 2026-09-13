"""Complete molecular simulation pipeline for LiH, BeH2, HF, and H2O using PySCF FCIDUMP and clifford_qc A-CASE."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
from pyscf import cc, ci, gto, scf, tools

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.fcidump import fcidump_model
from clifford_qc.sparse import to_sparse
from clifford_qc.models.observables import (double_occupancy, magnetization,
                                             total_spin_squared)
from clifford_qc.subspace import (MatrixElementBank, determinant_excitations,
                                   identity_generator, lehmann_spectrum,
                                   occupied_spin_orbitals, run_acase,
                                   solve_subspace, static_susceptibility)


MOLECULES = {
    "lih": {
        "name": "Lithium Hydride (LiH)",
        "atom": "Li 0 0 0; H 0 0 1.595",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "beh2": {
        "name": "Beryllium Hydride (BeH2)",
        "atom": "Be 0 0 0; H 0 0 1.326; H 0 0 -1.326",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "hf": {
        "name": "Hydrogen Fluoride (HF)",
        "atom": "F 0 0 0; H 0 0 0.917",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    "h2o": {
        "name": "Water (H2O)",
        "atom": "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
    },
    # ---------------------------------------------------------------- stretched
    #
    # The equilibrium set above cannot answer whether A-CASE has a niche: all
    # four are closed-shell, near-equilibrium, single-reference molecules in a
    # minimal basis, which is exactly where coupled cluster is near-exact. CCSD
    # beats A-CASE on all four, as it should.
    #
    # These are geometries where CCSD breaks. Each was screened against PySCF
    # FCI before being added, and the screening decided the set:
    #
    #   * H2O at 2x goes *non-variational* -- CCSD lands 9.7 mHa BELOW the
    #     exact energy. That is the textbook static-correlation failure and the
    #     sharpest available contrast with a variational subspace method.
    #   * BeH2 at 2x misses chemical accuracy by 3.7x (+5.861 mHa). Its stretch
    #     is non-monotonic -- 2.5x is easier again (+0.753) -- so 2x is the hard
    #     point, not the far one.
    #   * LiH at 3x is a deliberate *control*: a stretched geometry where CCSD
    #     still wins comfortably (+0.132 mHa). A single sigma bond in a minimal
    #     basis stays single-reference, and a claim that stretching alone
    #     favours A-CASE should have to survive this row.
    #
    # Two candidates were screened out rather than quietly included:
    #
    #   * HF at any stretch. With 10 electrons in 6 orbitals there are two
    #     holes, so singles and doubles exhaust the excitation manifold and CCSD
    #     is exact for the *system* at every geometry (-0.000 mHa at 2x). That
    #     is combinatorics, not chemistry, and no stretch can change it.
    #   * H2O at 3x. PySCF CISD comes out 0.368 mHa *below* its FCI reference
    #     there, which is impossible for a variational method -- the reference
    #     is not the ground state. Near-degenerate spectra defeat Davidson the
    #     same way sparse.py documents for ARPACK. An unreliable oracle makes
    #     every error on that row meaningless.
    #   * LiH at 4x, where RHF does not converge at all.
    #   * H2O at 2.5x, removed after it had been run. CCSD is dramatically
    #     non-variational there (-41.1 mHa), which is why it was attractive,
    #     but A-CASE converges to a state with <S^2> = 6 -- a quintet. The
    #     lowest root of the singles-and-doubles block at that geometry is a
    #     quintet 55.2 mHa below the lowest singlet, and neither A-CASE nor a
    #     plain determinant diagonalization constrains spin, so both land on
    #     it. Its energy therefore describes a different state than CCSD and
    #     RHF do, and the row was comparing multiplicities rather than methods.
    #     Recorded here rather than deleted quietly: the same trap waits at any
    #     geometry where a high-spin state drops below the singlet, and
    #     <S^2> is what reveals it.
    #
    # ``complete_sd`` is off for these. That arm reproduces PySCF CISD exactly
    # -- verified to between 7e-14 and 1.1e-11 on all four equilibrium
    # molecules -- and costs half an hour and gigabytes at 14 qubits, so paying
    # it again to re-derive a number PySCF gives in milliseconds buys nothing.
    # ``max_subspace`` bounds the adaptive arm: the oracle stop cannot fire when
    # the target is out of reach, so without a cap a stretched run grows to the
    # entire candidate pool.
    "lih_stretched": {
        "name": "Lithium Hydride, 3x stretched (LiH, 4.785 A) [control]",
        "atom": "Li 0 0 0; H 0 0 4.785",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "complete_sd": False,
        "max_subspace": 30,
        "regime": "stretched",
    },
    "beh2_stretched": {
        "name": "Beryllium Hydride, 2x symmetric stretch (BeH2, 2.652 A)",
        "atom": "Be 0 0 0; H 0 0 2.652; H 0 0 -2.652",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "complete_sd": False,
        "max_subspace": 30,
        "regime": "stretched",
    },
    "h2o_stretched": {
        "name": "Water, 2x symmetric stretch (H2O, 1.9150 A)",
        "atom": "O 0 0 0.2346; H 0 1.5144 -0.9384; H 0 -1.5144 -0.9384",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "complete_sd": False,
        # 30 was killed by the OOM reaper, and the ~18 GiB this comment used to
        # attribute that to was wrong. It scaled BeH2's M=31 peak by the
        # Hamiltonian word ratio (1086/666), but this molecule's own M=21 run
        # already carries its larger word count, so multiplying by the ratio
        # counts it twice. Scaling this row's own 6.53 GiB linearly in M predicts
        # 9.64 GiB, and benchmarks/run_packed_h2o_feasibility.py measures 9.69 on
        # a clean 15 GiB process -- so M=31 fits, and the kill had another cause.
        # The likely one is recorded in PLAN.md section 5: a sequential run held
        # BeH2's 11.2 GiB bank as this molecule's starting point, and 11.2 + 9.7
        # does not fit. That is the defect the explicit delete and gc below fixed.
        # The cap stays at 20 because raising it is a scope decision about what
        # the committed row reports, not a memory question any more.
        "max_subspace": 20,
        "regime": "stretched",
    },
}


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

    Uses the full ``2^n`` sparse Hamiltonian, so it is a small-system tool: it
    is the oracle for the molecules in this pipeline, not a general CI solver.
    """
    basis = SectorStatevectorBackend(n, n_electrons, sz).basis
    reference = 0
    for q in occupied:
        reference |= 1 << (n - 1 - q)
    rank = np.array([bin(int(b) ^ reference).count("1") // 2 for b in basis])
    kept = basis[rank <= max_rank]
    block = to_sparse(hamiltonian).tocsr()[np.ix_(kept, kept)].toarray()
    block = (block + block.conj().T) / 2.0
    energies, vectors = np.linalg.eigh(block)

    # Select by multiplicity, not by energy. The determinant block spans every
    # spin state at this S_z, and at stretched geometries the lowest root is
    # not the singlet: at H2O 2.5x it is a quintet 55.2 mHa below the lowest
    # singlet. Taking eigvalsh()[0] there produced a "CISD" energy for a state
    # of different multiplicity than CCSD and RHF target, and then read the
    # gap as a PySCF failure. PySCF was returning the singlet, correctly.
    spin = to_sparse(total_spin_squared(model).to_mv())
    spin_block = spin.tocsr()[np.ix_(kept, kept)].toarray()
    spin_block = (spin_block + spin_block.conj().T) / 2.0
    s2 = np.real(np.einsum("ij,jk,ki->i", vectors.conj().T, spin_block, vectors))
    match = np.flatnonzero(np.abs(s2 - target_s2) < s2_tol)
    if match.size == 0:
        raise RuntimeError(
            f"no root of the rank-{max_rank} determinant block has "
            f"<S^2> = {target_s2}; the lowest few are {s2[:4]}")
    index = int(match[0])
    return float(energies[index]), int(kept.size), float(s2[index])


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


def run_pipeline(
    max_candidates: int | None = 0,
    max_subspace: int | None = 0,
    target_error_mha: float = 1.5936,
    molecules: list[str] | None = None,
) -> None:
    out_dir = Path("molecular_results")
    out_dir.mkdir(exist_ok=True)

    selected = {k: v for k, v in MOLECULES.items()
                if molecules is None or k in molecules}
    # A partial run merges into the existing summary. Replacing it would delete
    # the rows the run did not compute, which is worse than the drift risk the
    # merge introduces -- and the whole-file rewrite is what makes a full run
    # authoritative again.
    summary_records = {}
    summary_path = out_dir / "results_summary.json"
    if molecules is not None:
        # Seed from the per-molecule files rather than from the summary. They
        # are the source of truth: the summary is derived, and a run killed
        # before it finished will have written per-molecule records that never
        # reached it. Seeding from the summary would silently discard them.
        for path in sorted(out_dir.glob("*_results.json")):
            try:
                seeded = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "key" in seeded:
                summary_records[seeded["key"]] = seeded
        print(f"Merging {len(selected)} molecule(s) into {len(summary_records)} "
              f"existing record(s) recovered from molecular_results/")

    print("==================================================================")
    print("STARTING MOLECULAR SIMULATION PIPELINE (PySCF -> FCIDUMP -> A-CASE)")
    print("==================================================================")

    for key, spec in selected.items():
        t0 = time.time()
        print(f"\n---> Processing {spec['name']} ({key.upper()})...")

        # 1. PySCF RHF calculation
        mol = gto.M(
            atom=spec["atom"],
            basis=spec["basis"],
            charge=spec["charge"],
            spin=spec["spin"],
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.max_cycle = 300
        mf.run()
        e_rhf = float(mf.e_tot)
        # At stretched geometries the restricted reference can fail to
        # converge, and an unconverged reference makes every correlated number
        # built on it meaningless rather than merely inaccurate. Record it
        # instead of discovering it later as an anomalous error.
        rhf_converged = bool(mf.converged)
        print(f"  PySCF RHF Energy: {e_rhf:+.9f} Ha (converged: {rhf_converged})")
        if not rhf_converged:
            print("  WARNING: RHF did not converge; correlated baselines on "
                  "this reference are not trustworthy")

        # 1b. Classical correlated baselines in the same orbital space.
        # Without these the A-CASE column has nothing to be judged against:
        # CISD is the classical method the complete singles/doubles subspace
        # reproduces by construction, and CCSD is the one a chemist would
        # actually run at this cost.
        cisd = ci.CISD(mf)
        cisd.max_cycle = 300
        cisd.run()
        e_cisd = float(cisd.e_tot)
        ccsd = cc.CCSD(mf)
        ccsd.max_cycle = 300
        ccsd.run()
        e_ccsd = float(ccsd.e_tot)
        ccsd_converged = bool(getattr(ccsd, "converged", False))
        cisd_converged = bool(getattr(cisd, "converged", False))
        print(f"  PySCF CISD Energy: {e_cisd:+.9f} Ha (converged: {cisd_converged})")
        print(f"  PySCF CCSD Energy: {e_ccsd:+.9f} Ha (converged: {ccsd_converged})")

        # 2. Dump FCIDUMP
        fcidump_path = out_dir / f"{key}.fcidump"
        tools.fcidump.from_scf(mf, str(fcidump_path))
        print(f"  Exported FCIDUMP: {fcidump_path}")

        # 3. Ingest FCIDUMP with clifford_qc
        model = fcidump_model(fcidump_path)
        model.metadata["kind"] = "fermionic_lattice"
        model.metadata["sites"] = model.n // 2
        model.metadata["n_orbitals"] = 1
        n_electrons = model.metadata["n_electrons"]
        sz = model.metadata["sz"]
        occupied = occupied_spin_orbitals(model)
        rho0 = ExactMVBackend().state(model.reference, ())
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

        # 4. Sector-Exact FCI Reference via SectorStatevectorBackend
        exact_energies, _ = SectorStatevectorBackend(
            model.n, n_electrons=n_electrons, sz=sz
        ).ground_state(model.hamiltonian, k=4, method="dense")
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

        # 5. Adaptive A-CASE Subspace Eigensolver (Optimal Path Selection)
        # A per-molecule cap overrides the global one. Stretched geometries need
        # it: the oracle stop cannot fire when chemical accuracy is out of
        # reach, so growth would otherwise consume the entire candidate pool and
        # cost as much as the complete-SD arm it is supposed to be cheaper than.
        spec_max = spec.get("max_subspace")
        max_size = (
            spec_max if spec_max
            else (max_subspace if max_subspace and max_subspace > 0
                  else len(candidates))
        )
        target_error = target_error_mha / 1000.0 if target_error_mha > 0 else None

        t_adaptive_start = time.time()
        with PeakRSS() as adaptive_rss:
            adaptive = run_acase(
                rho0,
                model.hamiltonian,
                candidates,
                max_size=max_size,
                leakage_tol=1e-10,
                exact_ground_energy=e_exact,
                target_error=target_error,
            )
        t_adaptive_elapsed = time.time() - t_adaptive_start
        e_acase = float(adaptive.result.ground_energy)
        err_ha = e_acase - e_exact
        err_mha = err_ha * 1000.0
        print(f"  Adaptive A-CASE E0: {e_acase:+.9f} Ha (Error: {err_mha:+.4f} mHa, Optimal M={len(adaptive.labels)})")
        print(f"  Adaptive Stop Reason: {adaptive.stopped_reason}")

        # 6. Complete SD Coordinate Subspace Eigensolver (Chemical Accuracy Check)
        # Optional per molecule. The arm reproduces PySCF CISD exactly, so on
        # geometries added only to test CCSD it re-derives at gigabyte scale a
        # number PySCF supplies in milliseconds.
        run_complete_sd = bool(spec.get("complete_sd", True))
        sd_size = len(all_candidates) + 1
        sd_spans_full_sector = sd_size >= sector_dimension
        if run_complete_sd:
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
            print("  Complete SD Subspace: skipped (reproduces PySCF CISD "
                  f"exactly; CISD error {(e_cisd - e_exact) * 1000.0:+.4f} mHa)")

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
        # S_z couples the singlet ground state to triplets. The adaptive basis
        # is grown from a closed-shell determinant through number- and
        # S_z-conserving singlet excitations, so it contains no triplet for the
        # operator to reach and every weight comes out at the numerical floor.
        # chi0 is therefore zero as a property of the *basis*, not of the
        # molecule, and must not be reported as a measured susceptibility.
        significant_lines = [ln for ln in lehmann_lines if abs(float(ln.weight)) > 1e-8]
        response_diagnostic = (
            "no significant weight: the S_z operator couples the singlet "
            "reference to triplet states that the singlet-preserving "
            "excitation basis does not span, so chi0 is a basis artifact "
            "rather than a physical response"
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

        # Write per-molecule result JSON
        with open(out_dir / f"{key}_results.json", "w") as f:
            json.dump(record, f, indent=2)

        summary_records[key] = record

        # Write the summary after every molecule, not once at the end. A run
        # killed partway through used to leave per-molecule records that the
        # summary never learned about -- which is exactly what an OOM kill on
        # the third of four molecules produced.
        with open(summary_path, "w") as f:
            json.dump(summary_records, f, indent=2)

        # Drop this molecule's bank before starting the next one. The banks are
        # the dominant allocation and they are not needed once the record is
        # written; without this the process carries every previous molecule's
        # element operators forward, which is how BeH2's 11.2 GiB became H2O's
        # starting point and got the run killed on a 15 GiB machine. It also
        # makes each arm's RSS baseline mean something.
        if run_complete_sd:
            del sd_spectrum
        del adaptive, spectrum, rho0, model
        gc.collect()

    # Master summary JSON
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    print("\n==================================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {out_dir.resolve()}")
    print("==================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run molecular benchmark pipeline")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Max candidate excitations to evaluate (0 for full candidate pool)",
    )
    parser.add_argument(
        "--max-subspace",
        type=int,
        default=0,
        help="Max adaptive subspace basis size M (0 for unconstrained optimal growth)",
    )
    parser.add_argument(
        "--target-error-mha",
        type=float,
        default=1.5936,
        help="Target energy error threshold in mHa for early stopping (default 1.5936 mHa)",
    )
    parser.add_argument(
        "--molecules",
        default="",
        help="Comma-separated subset to run, e.g. 'h2o_stretched,beh2_stretched' "
             "(default: all). A partial run MERGES into the existing summary "
             "rather than replacing it, so iterating on one geometry does not "
             "discard the others -- but it also means the summary can then hold "
             "rows from different runs, which benchmarks/check_molecular.py "
             f"cannot detect. Available: {', '.join(MOLECULES)}",
    )
    args = parser.parse_args()
    selected = [m.strip() for m in args.molecules.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MOLECULES]
    if unknown:
        parser.error(f"unknown molecule(s): {', '.join(unknown)}. "
                     f"Available: {', '.join(MOLECULES)}")
    run_pipeline(
        max_candidates=args.max_candidates,
        max_subspace=args.max_subspace,
        target_error_mha=args.target_error_mha,
        molecules=selected or None,
    )


if __name__ == "__main__":
    main()

