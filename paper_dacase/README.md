# Paper B — DA-CASE

Standalone manuscript built from the DA-CASE effective-Hamiltonian, FCIDUMP,
active-space benchmark, finite-shot response, QSCI/packet ensemble, and
matched-budget Phase 12 work.  It incorporates the seed-level and comparator
corrections merged through pull requests #47, #48, and #49.

The paper argues an **architectural** thesis, not a scoreboard one. Fixing a
single reference and reconstructing every overlap, Hamiltonian, and observable
element from Pauli expectations on it has several measurable consequences: the
basis uses one reference context at any size, compatible-setting counts rather
than raw word counts determine the shot-level preparation schedule, a dyadic
block-commuting hierarchy exposes the setting-versus-logical-CX trade, the
measurement width is tunable through generator resolution, and one cached bank
serves energy, projected observables, and Lehmann response. Energy accuracy at matched budget is
explicitly *not* the claim — Sec. "What the architecture does not buy" states
where the method loses, including the inertness of operator dressing against a
sample-independent selected-CI control.

The paper does not depend on the ADAPT-VQE manuscript in `../paper/`.  It
defines the method, measurement model, benchmark contract, and evidence labels
again, while citing the operator-centric Clifford-algebra paper for the shared
software representation.

**Naming boundary.** DA-CASE means *Dyadic Adaptive Clifford-Algebra Subspace
Eigensolver*: the existing adaptive subspace engine followed by a configurable
dyadic Clifford measurement stage.  Its $k=1$ endpoint is exactly QWC and is
kept in the matched ledgers for comparability; $k>1$ changes only measurement
compatibility and block-local Clifford synthesis.  Source APIs, filenames, and
stored benchmark arm labels retain `acase` / `A-CASE` for provenance and
backward compatibility; paper table generation maps those stored labels to
`DA-CASE` without mutating the records.

## Versions

| Version | arXiv | Source | Status |
|---|---|---|---|
| v1 | [2608.08739v1](https://arxiv.org/abs/2608.08739v1), 9 Aug 2026 | tag `dacase-arxiv-v1.0.0` | posted |
| v2 | 2608.08739v2 | this tree, revised 14 Nov 2026 | prepared as a replacement |

v2 is a replacement of 2608.08739, not a new submission. It moves the
manuscript from the August records to the committed benchmark records as they
stood on 14 November 2026. No claim, abstract sentence, method or table layout
changes. What changed, for the replacement's comments field and for anyone
diffing the two versions:

- **Table II, dyadic hierarchy.** Logical CX counts, depths and break-even
  ratios are regenerated from the current records, which need fewer logical
  CX gates for the same settings. At the fully commuting H4 endpoint the sweep
  takes `1886` CX gates rather than `2006`, with mean and maximum two-qubit
  depths `24.16` and `35` rather than `25.72` and `42`; the BeH2 `k = 2`
  break-even ratio is `2.786` rather than `1.857`. Setting counts, `W/G` and
  preparation counts are unchanged, including the `913` to `64` reduction.
- **Table III, finite-shot diagnostic.** Regenerated from the record rebuilt
  under Python 3.12 and NumPy 2.5.2. One uniform per-mode replica changes its
  rank decision (`72/200` rather than `71/200`), and no cell moves by more than
  `0.7` mHa. The `68.9%` variance reduction is unchanged.
- **Prose.** Sec. IV C and Sec. IV D quote the values above: the three H4
  endpoint numbers, the fixed-cutoff uniform median (`4.49` rather than `4.48`
  mHa), the rank-recovery count, and the covariance-aware per-mode median
  (`8.69` rather than `8.73` mHa).
- **One correction that is not a data change.** The exact-Ritz first-order
  variance reduction is `19.9%`. v1 printed `20.0%`; both revisions' records
  give `19.94%`.
- **Table I, matched contract.** Rebound to the current record, which differs
  from the August one in its environment stamp and in the last digits of
  floating-point values; no cell changes.

## Development and review access

The framework and this paper package are currently maintained in a private
development repository while the operator, measurement, subspace, and backend
interfaces are stabilized. Reproducibility is preserved through a frozen
review snapshot rather than through a claim that the moving repository is
public. Editors and referees can receive that access-controlled snapshot from
the authors on request; it must identify the exact source revision and include
the source, frozen inputs, raw records, generators, tests, and environment
instructions used by the manuscript.

A tagged archival release with a persistent identifier is intended after the
framework foundation is stable enough for external reuse. Rendered manuscript
PDFs remain CI/release artifacts and are not committed as source.

## Files

- `manuscript.tex` — standalone REVTeX 4.2 manuscript.
- `references.bib` — paper-specific bibliography.
- `run_response_record.py` — regenerates both grouped-bootstrap response
  records from the merged implementation.
- `check_response_records.py` — compares a fresh run with the committed
  records, requiring exact seeds, replica accounting, and metadata while
  tolerating only last-bit floating-point variation. It reports in four tiers
  — `contract`, `point`, `census`, `intervals` — and names the earliest one
  that failed, because every percentile band is conditional on the replica
  census and a flat diff turns one upstream disagreement into hundreds of
  downstream ones. On a census disagreement it diffs the per-replica
  fingerprint and names the individual replicas that moved.
- `data/response_bootstrap.json` — deterministic paper record (schema `v2`),
  including the seeds, shot budget, replica accounting, intervals, plotted
  spectrum, and an execution-provenance stamp naming the revision, dependency
  set, and BLAS/LAPACK build that produced it.
  `bootstrap.replica_census` adds one row per replica in draw order: its
  outcome, effective rank, smallest normalized overlap eigenvalue, and the
  `rank_decision_margin` / `overlap_threshold` / `controlling_mode` triple
  that records the retention decision itself. Modes are retained against
  per-mode cutoffs, so the margin — not the eigenvalue's sign — is what
  decides the gate. The census is the fingerprint that makes a disagreement
  between two environments diagnosable replica by replica rather than only as
  a pair of totals.
- `data/response_bootstrap_illconditioned.json` — the same run with the
  determinant generators replaced by Hamiltonian powers. The system,
  observable, word set, grouping, shots, basis size, and seeds are held fixed;
  this isolates the effect of using a differently conditioned representation
  of the same complete sector.
- `make_tables.py` — regenerates all numerical LaTeX table fragments from
  the retained manuscript's source snapshot.
- `SOURCE_SNAPSHOT.json` and `source_snapshot.py` — bind all twelve table
  inputs, the v1 table-generator bytes and all ten scientific table bodies,
  and name the manuscript revision they belong to. The four inputs whose
  benchmark records have been regenerated since the preprint are packaged
  under `data/source_snapshot/`, as the committed records stood at v2; the
  v1 generator is inert `.txt` provenance data. `previous_revision` names
  the v1 inputs and table bodies that v2 replaced. Checking this contract
  needs no Git history.
- `make_figures.py` — regenerates the pipeline, validation-ladder, response,
  and conditioning figures. `paper_assets/manifest.json` binds each figure to
  the exact generator and line-ending-normalized input digests without
  requiring cross-platform PDF byte identity.
- `check_manuscript.py` — checks labels, references, citations, inputs, table
  column counts, hand-typed numeric cells, figure staleness, and
  evidence-language invariants.
- `../benchmarks/reference_results/clifford_hierarchy_h4.json` and
  `clifford_hierarchy_beh2.json` — the exact logical block-commuting
  measurement hierarchy for two eight-qubit DA-CASE banks, including the
  Z-only tableau invariant, setting counts, logical CX counts, two-qubit
  depths, and the per-level break-even CX/preparation cost ratio. The H4 bank
  is reconstructed from the matched contract's retained labels; the BeH2 bank
  is an independent DA-CASE run on its own frozen FCIDUMP.
- `../benchmarks/run_clifford_hierarchy.py` and
  `../benchmarks/check_clifford_hierarchy.py` — regenerate and gate those
  records (requires the optional `stim` extra).
- `../benchmarks/make_beh2_fcidump.py` — regenerates the committed BeH2
  CAS(4e,4o) FCIDUMP and its provenance sidecar (requires the `chemistry`
  extra); the emitted file is the immutable benchmark input.

Two records live under `../benchmarks/reference_results/` because they are
benchmarks rather than paper artifacts:

- `warm_start_h4.json` — DA-CASE at the paper's nine-vector budget with the
  reference state varied from the Hartree-Fock determinant to ADAPT-VQE states
  (`benchmarks/run_warm_start.py`). The v2 record also reports the additional
  ADAPT pool-gradient, optimizer, and state-preparation counts; it does not
  claim a physical shot estimate for the exact-simulation stage. Its
  exact-discrete/tolerant-float reproduction gate is
  `benchmarks/check_warm_start.py`.
- `krylov_width.json` — the Krylov arm's measurement width, computed through
  the `H^k` collapse of the element universe rather than the quadratic route
  the ladder cannot afford (`benchmarks/run_krylov_width.py`). The v2 record
  certifies that the reported coefficient cutoff preserves the unpruned
  pencil's rank, Ritz energy, conditioning, and normalized matrix entries.
- `matched_h4.json` — every arm on one H4 contract with the sector, reference,
  pool, budget, and stopping rule held fixed
  (`benchmarks/run_matched_h4.py`). It includes exact ADAPT-GCIM at the
  near-size match (`k=4`, `M=8`) and iteration match (`k=8`, `M=16`), with
  the published fixed `theta=pi/4`, cumulative-surrogate selector, and
  `M=2k` basis rule. Its off-diagonal Hamiltonian/overlap pair counts remain
  separate from DA-CASE's single-reference word universe. The record also
  distinguishes state-evaluation contexts from physical preparations, stores
  QWC groups for the large DA-CASE banks, and sums groups over each changing
  ADAPT selection state. The two DA-CASE arms differ only in generator
  resolution; their retained subspaces are identical and their widths are not.

## Reproduce

The tables are bound to the inputs of v2. At that revision the packaged
inputs are byte-identical to the committed benchmark records, so the
manuscript agrees with what the record gates reproduce. The binding is
deliberate rather than live: running the benchmark producers below, or a
later migration of those records, does not migrate this manuscript.
`make_tables.py` reads the bound inputs and refuses inputs or generated
scientific cells that differ from `SOURCE_SNAPSHOT.json`. It stages and
validates the complete set before replacing any table, so a failed
regeneration preserves the existing files. Snapshot paths must resolve within
the source tree, including through symlinks. This manifest identifies the
retained repository sources; it does not independently identify the uploaded
arXiv artifact. Moving the manuscript to newer records is an explicit
reviewed revision of that boundary: new packaged inputs, new table-body
digests, the prose that quotes them and a new revision number together,
rather than provenance headers refreshed against new values.

From the repository root:

```bash
python paper_dacase/run_response_record.py
python paper_dacase/check_response_records.py
python benchmarks/run_warm_start.py
python benchmarks/check_warm_start.py
python benchmarks/run_krylov_width.py
python benchmarks/run_matched_h4.py
python benchmarks/check_matched_h4.py
pip install -e '.[stim]'
python benchmarks/run_clifford_hierarchy.py --system h4
python benchmarks/run_clifford_hierarchy.py --system beh2
python benchmarks/check_clifford_hierarchy.py
python paper_dacase/make_tables.py
python paper_dacase/make_figures.py
python paper_dacase/source_snapshot.py
python paper_dacase/check_manuscript.py

cd paper_dacase
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

The paths below identify each evidence family. `SOURCE_SNAPSHOT.json` names
the exact input bytes used for the retained tables, including the packaged
copies of four moving benchmark records.

| Manuscript element | Source |
|---|---|
| Pipeline figure | method contract in `clifford_qc/subspace/` and `clifford_qc/models/` |
| Method-relation table | primary references in `references.bib`; no numerical claims |
| Validation-ladder table | `../benchmarks/reference_results/acase_ladder_summary.csv` plus certified `krylov_width.json` for the comparator width column |
| Matched-budget limits prose (Sec. "does not buy") | `../benchmarks/results/phase12_paper_b_five_system.json` for the five-system `M=7` ladder; `../benchmarks/results/m7_seed_replication.jsonl` for the eighty-draw dressing-inertness result |
| Response figure and table | `data/response_bootstrap.json` |
| Conditioning figure and table | `data/response_bootstrap.json` and `data/response_bootstrap_illconditioned.json` |
| Dimer table | `../examples/data/wannier_hubbard_dimer.json` and closed-form dimer identities |
| FCIDUMP H4 table | `../benchmarks/reference_results/fcidump_h4.json` |
| Warm-start resource table | `../benchmarks/reference_results/warm_start_h4.json` |
| Matched DA-CASE/ADAPT-VQE/ADAPT-GCIM table | `../benchmarks/reference_results/matched_h4.json` |
| Dyadic Clifford measurement hierarchy | `../benchmarks/reference_results/clifford_hierarchy_h4.json` and `clifford_hierarchy_beh2.json` |
| Phase 12 matched-budget primary table | `../benchmarks/results/phase12_paper_b_five_system.json` |
| Phase 11 packet seed-cluster inference | `../benchmarks/results/packet_seed_ensemble*.jsonl` plus molecular provenance sidecar and generated Markdown summary |
| Phase 12 M=7 seed replication | `../benchmarks/results/m7_seed_replication.jsonl` plus generated Markdown summary |
