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
- `run_response_record.py` — regenerates both grouped-bootstrap response
  records from the merged implementation.
- `check_response_records.py` — compares a fresh run with the committed
  records, requiring exact seeds, replica accounting, and metadata while
  tolerating only last-bit floating-point variation.
- `data/response_bootstrap.json` — deterministic paper record, including the
  seeds, shot budget, replica accounting, intervals, and plotted spectrum.
- `data/response_bootstrap_illconditioned.json` — the same run with the
  determinant generators replaced by Hamiltonian powers. The system,
  observable, word set, grouping, shots, basis size, and seeds are held fixed;
  this isolates the effect of using a differently conditioned representation
  of the same complete sector.
- `make_tables.py` — regenerates all numerical LaTeX table fragments from
  committed benchmark records.
- `make_figures.py` — regenerates the pipeline, validation-ladder, response,
  and conditioning figures.
- `check_manuscript.py` — checks labels, references, citations, inputs, table
  column counts, hand-typed numeric cells, figure staleness, and
  evidence-language invariants.

Two records live under `../benchmarks/reference_results/` because they are
benchmarks rather than paper artifacts:

- `warm_start_h4.json` — A-CASE at the paper's nine-vector budget with the
  reference state varied from the Hartree-Fock determinant to ADAPT-VQE states
  (`benchmarks/run_warm_start.py`). The v2 record also reports the additional
  ADAPT pool-gradient, optimizer, and state-preparation counts; it does not
  claim a physical shot estimate for the exact-simulation stage.
- `krylov_width.json` — the Krylov arm's measurement width, computed through
  the `H^k` collapse of the element universe rather than the quadratic route
  the ladder cannot afford (`benchmarks/run_krylov_width.py`). The v2 record
  certifies that the reported coefficient cutoff preserves the unpruned
  pencil's rank, Ritz energy, conditioning, and normalized matrix entries.

## Reproduce

From the repository root:

```bash
python paper_acase/run_response_record.py
python paper_acase/check_response_records.py
python benchmarks/run_warm_start.py
python benchmarks/run_krylov_width.py
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
| Pipeline figure | method contract in `clifford_qc/subspace/` and `clifford_qc/models/` |
| Method-relation table | primary references in `references.bib`; no numerical claims |
| Validation-ladder figure and table | `../benchmarks/reference_results/acase_ladder_summary.csv` plus certified `krylov_width.json` for the comparator width column |
| Response figure and table | `data/response_bootstrap.json` |
| Conditioning figure and table | `data/response_bootstrap.json` and `data/response_bootstrap_illconditioned.json` |
| Dimer table | `../examples/data/wannier_hubbard_dimer.json` and closed-form dimer identities |
| FCIDUMP H4 table | `../benchmarks/reference_results/fcidump_h4.json` |
| Warm-start resource table | `../benchmarks/reference_results/warm_start_h4.json` |
