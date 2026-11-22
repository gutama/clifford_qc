# Variance-weighted measurement planning and rank-aware adaptive precision

Standalone short methods manuscript reporting the preregistered two-part study
combining variance-weighted $k$-commuting measurement planning, paid-pilot
Ritz-weighted replanning, GALIC-style two-qubit noise gating, and the
density-free rank-aware `sampled_precision` controller in `clifford_qc`.

## Provenance and Scope

- **Result-free declaration (Commit 1):**
  `docs/restart/weighted_precision_declaration.json`, frozen at commit
  `73e48cc6f9fce7d0d0a568f5edf88ae441816423`.
- **Evaluation record (Commit 2):**
  `docs/restart/weighted_precision_results.json.gz` (1,408 acquisitions,
  3,144,593,750 simulated physical shots, 0 failures, 4 opt-in width-guard
  exhaustions), independently verified in CI by
  `benchmarks/check_weighted_precision.py --verify-history` under Python
  `3.12.14`, NumPy `2.5.2`, SciPy `1.18.0`, and single-threaded BLAS.
- Every numeric table cell (`tables/*.tex`), inline prose macro
  (`tables/numbers.tex`), and vector figure (`paper_assets/*.pdf`) is generated
  directly from those frozen artifacts with zero hand-typed table values.
  The figure manifest also binds the rendered PDF bytes. The Part A ledger
  includes both 68% and 95% delta and bootstrap-spread intervals; the Part B
  ledger reports physical-shot cost alongside bias and RMSE.

## Regenerating and Checking

```bash
python paper_weighted_precision/make_tables.py
python paper_weighted_precision/make_figures.py
python paper_weighted_precision/check_manuscript.py --self-test
```

To compile the manuscript, install REVTeX 4.2 (including `apsrev4-2.bst`)
with your TeX distribution, then run:

```bash
cd paper_weighted_precision
pdflatex -halt-on-error -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -halt-on-error -interaction=nonstopmode manuscript.tex
pdflatex -halt-on-error -interaction=nonstopmode manuscript.tex
```
