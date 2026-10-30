# How Compact Is a Compact Subspace?

*Columns, Hamiltonian actions and determinant support in a preregistered test
of residual packets.*

This manuscript follows the [architecture paper](../paper_architecture/README.md)
and reports one line of recent development from a single angle, rather than
surveying every change since that paper. The angle is that "compact subspace"
depends on the yardstick. The same trajectories rank three spin-adapted growth
rules differently by:

- retained columns,
- Hamiltonian actions spent choosing them, and
- determinant support of the span.

The three rules are greedy selected CI, residual packets and an unrestricted
Davidson correction.

The subject is the declared column-budget comparison
([declaration](../benchmarks/PACKET_BUDGET_CONVERGENCE_DECLARATION.md),
[report](../benchmarks/results/packet_budget_convergence.md)). Under its
declared column rule, packets add no compactness over selected CI. Davidson
leads in columns and actions wherever it reaches the ground state, and never in
support. On the primary lattice it stops on an excited singlet. A ledger of the
separately declared engineering campaigns on the packet arm shows that exact
speed-ups changed time, memory or work but never the column verdict.

Everything is exact classical computation on explicit sector vectors of at most
4,900 determinants. The paper measures no matrix element on a device, compares
against no optimized selected CI, and makes no scaling or quantum-advantage
claim. Its discussion names the cost model under which packets could still
matter, one that prices bounded column support, and explains why these
simulations cannot test it.

## Files

- `manuscript.tex` — REVTeX 4.2 (`aps,pra`) source.
- `references.bib` — bibliography. Entries shared with the architecture paper
  are copied verbatim from `paper_architecture/references.bib`.
- `make_tables.py` — regenerates `tables/*.tex`, including `tables/numbers.tex`,
  which defines every quantity the prose quotes. It also asserts each pattern
  the prose states in words, for example which inputs read mixed, that Davidson
  stopped on an excited root, and that no rule leads on all three yardsticks.
  It refuses to generate if a record no longer shows one.
- `make_figures.py` — regenerates `paper_assets/*.pdf` and their manifest.
  Matplotlib is imported at draw time, so the gate runs without a plotting
  stack.
- `make_start_weights.py` — writes `data/start_weights.json`, the ground-state
  weight of the shared start span on each input. It uses the independent
  checker's own functions and binds the Git blob of the record it read.
- `check_manuscript.py` — the gate (below). `--self-test` injects one fault of
  each kind and fails unless every one is reported.

## Build

```bash
python paper_compactness/make_start_weights.py   # only when the budget record changes (~12 s)
python paper_compactness/make_tables.py          # tables/
python paper_compactness/make_figures.py         # paper_assets/*.pdf (needs matplotlib)
python paper_compactness/check_manuscript.py
cd paper_compactness
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

The figure manifest digests `make_tables.py`, because the trajectory figure
decodes the record through its loader. After any change to the table
generator, rerun `make_figures.py`.

## Data → manuscript map

Records without a path prefix live under `benchmarks/reference_results/`.

| Manuscript element | Source |
|---|---|
| Inputs table, start weights | `packet_budget_convergence.json`, `data/start_weights.json` |
| Columns/actions, support, leaders and readouts tables; Fig. 2 | `packet_budget_convergence.json` (content-addressed trajectories; readouts via `benchmarks/summarize_packet_budget_convergence.py`) |
| Motivating water and 64-column numbers | `packet_work_reuse.json`, `packet_work_large.json` |
| Engineering ledger | `packet_work_large.json`, `sparse_packet_work.json` (speed clause from `benchmarks/SPARSE_PACKET_WORK_DECLARATION.md`), `exact_projection_cache.json`, `frame_coordinate_projection.json` |
| Fig. 1 | Schematic; plots no record |

## What the gate enforces

`check_manuscript.py` fails, rather than typesetting, when:

1. a generated fragment differs by a byte from what the generator produces now
   (fragments are regenerated into a scratch tree and compared). Because the
   generator asserts the prose's stated patterns, this also catches a record
   that stops supporting a sentence;
2. a numeral appears in the body prose or in a table cell typed into the
   manuscript, outside a short allowlist of notation, or a generated count is
   spelled out beside the noun it counts;
3. a `\pk…` macro is used without being defined, or defined without being used;
4. `data/start_weights.json` binds a different budget record;
5. a figure's manifest no longer matches its generator, input digests or asset;
6. a citation, label, reference, input or table column count is broken;
7. one of the scope statements the abstract depends on has been edited out.

The gate runs in CI beside the architecture manuscript's gate.
