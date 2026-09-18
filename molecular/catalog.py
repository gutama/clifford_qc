"""Molecule geometries and historical per-molecule experiment budgets."""

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
    # ``max_additions`` bounds the adaptive arm: the oracle stop cannot fire when
    # the target is out of reach, so without a cap a stretched run grows to the
    # entire candidate pool.
    "lih_stretched": {
        "name": "Lithium Hydride, 3x stretched (LiH, 4.785 A) [control]",
        "atom": "Li 0 0 0; H 0 0 4.785",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "complete_sd": False,
        "max_additions": 30,
        "regime": "stretched",
    },
    "beh2_stretched": {
        "name": "Beryllium Hydride, 2x symmetric stretch (BeH2, 2.652 A)",
        "atom": "Be 0 0 0; H 0 0 2.652; H 0 0 -2.652",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 0,
        "complete_sd": False,
        "max_additions": 30,
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
        # does not fit. Fresh per-molecule worker processes now prevent that overlap.
        # The cap stays at 20 because raising it is a scope decision about what
        # the committed row reports, not a memory question any more.
        "max_additions": 20,
        "regime": "stretched",
    },
}
