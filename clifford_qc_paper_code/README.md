# clifford_qc — paper-accompanying code

Operator-centric VQE/ADAPT in Cl(2n, C). Companion code for
"Operator-Centric Variational Quantum Eigensolvers in Complex Clifford Algebra."

Canonical / maintained project:  https://github.com/gutama/clifford_qc

## Files
- ga_qc_operator_improved.py   — algebra kernel (sparse Pauli-word multivectors,
                                  JW generators, gates, channels, matrix backend).
- vqe_tfim_rotor_better.py     — fixed-depth HVA solver (adjoint gradient, L-BFGS-B).
- adapt_vqe_noisy_better_v2.py — exact + finite-shot ADAPT (cumulative escalation,
                                  statistical gates, 30-seed robustness sweep).
- adapt_vqe_tfim.py            — exact ADAPT reference (commutator-gradient pool).
- review_noisy_better.py       — adversarial review battery.
- paper_assets/                — generated figures.

## Run
Each module self-tests:
    python ga_qc_operator_improved.py
    python vqe_tfim_rotor_better.py
    python adapt_vqe_noisy_better_v2.py
