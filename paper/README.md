# Paper A — Confidence-certified, measurement-efficient ADAPT-VQE

Manuscript source for the second `clifford_qc` paper, building on the
operator-centric Clifford formulation of the companion manuscript. It
presents the measurement layer: the global commutator bank, shared-word
cumulative caching and QWC grouping, simultaneous confidence-certified
selection, and the shot-allocation policies, together with the 100-seed
spin study, the molecular proxy-vs-gradient boundary, and the
stabilizer-seeding negative result.

## Files

- `manuscript.tex` — REVTeX 4.2 (`aps,pra`) source.
- `references.bib` — bibliography (companion references plus this paper's).
- `make_figures.py` — regenerates `paper_assets/*.pdf` from the committed
  benchmark JSONL under `../benchmarks/reference_results/`.
- `paper_assets/` — figure PDFs (checked in so the manuscript builds
  without rerunning experiments).

## Build

```bash
python paper/make_figures.py          # regenerate figures from committed data
cd paper
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

Every plotted and tabulated number is read from
`../benchmarks/reference_results/*.jsonl`; see `../REPRODUCING.md` to
regenerate that data from scratch.

## Data → manuscript map

| Manuscript element | Source |
|---|---|
| Fig. 1, Table I (selection quality, circuits, ambiguity) | `spin_headline_n4.jsonl` (100 seeds) |
| Fig. 2 (QWC grouping) | `spin_headline_n4.jsonl` |
| Fig. 3 (allocation) | `spin_headline_n4.jsonl` |
| $n=6$ scaling / XXZ plateau | `spin_n6.jsonl` (20 seeds) |
| Fig. 4, Table II (chemistry) | `chemistry.jsonl` |
| Stabilizer negative result (appendix) | `stabilizer_seeding.jsonl` |
| Fig. calibration, Table calib | `calibration.jsonl` |
| Table baselines | `baselines.jsonl` |
| Infinite-shot ranking + geometry | `chemistry_repair.jsonl` |
