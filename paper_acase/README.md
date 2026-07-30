# Paper B — Adaptive operator-generated subspaces

Standalone manuscript built from the A-CASE effective-Hamiltonian, FCIDUMP,
active-space benchmark, and finite-shot response work merged in pull requests
#23 and #24.

The paper does not depend on the ADAPT-VQE manuscript in `../paper/`.  It
defines the method, measurement model, benchmark contract, and evidence labels
again, while citing the operator-centric Clifford-algebra paper for the shared
software representation.

## Files

- `manuscript.tex` — standalone REVTeX 4.2 manuscript.
- `references.bib` — paper-specific bibliography.
- `run_response_record.py` — regenerates the frozen grouped-bootstrap response
  record from the merged implementation.
- `data/response_bootstrap.json` — deterministic paper record, including the
  seeds, shot budget, replica accounting, intervals, and plotted spectrum.
- `make_tables.py` — regenerates all numerical LaTeX table fragments from
  committed benchmark records.
- `make_figures.py` — regenerates the pipeline, validation-ladder, and response
  figures.
- `check_manuscript.py` — checks labels, references, citations, inputs, assets,
  and evidence-language invariants.

## Reproduce

From the repository root:

```bash
python paper_acase/run_response_record.py
python paper_acase/make_tables.py
python paper_acase/make_figures.py
python paper_acase/check_manuscript.py

cd paper_acase
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The rendered `manuscript.pdf` and other LaTeX build products are deliberately
ignored.  They should be regenerated locally or by a release/CI workflow, not
committed to the source tree.

The response record uses 25 QWC groups, 8,000 shots per group, and 200
bootstrap replicas with fixed seeds.  Its percentile intervals are heuristic
and conditional on the replicas that preserve the thresholded rank and root
identity.  They are not finite-sample confidence certificates.

## Data-to-paper map

| Manuscript element | Source |
|---|---|
| Fig. 1 | method contract in `clifford_qc/subspace/` and `clifford_qc/models/` |
| Fig. 2, Table III | `../benchmarks/reference_results/acase_ladder_summary.csv` |
| Fig. 3, Table IV | `data/response_bootstrap.json` |
| Table I | `../examples/data/wannier_hubbard_dimer.json` and closed-form dimer identities |
| Table II | `../benchmarks/reference_results/fcidump_h4.json` |
