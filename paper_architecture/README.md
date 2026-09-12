# Paper C — Making a simulation framework falsifiable

Manuscript source for the architecture paper: `clifford_qc` described by the
constraints that stop it over-claiming, and the record of what those
constraints produced.

The thesis is architectural, not a scoreboard. Two commitments carry it — one
sparse Pauli-word operator type with no conversion boundary between circuit and
operator descriptions, and an evidence label attached to every number from the
measurement layer through to the committed records — and the paper's evidence is
what the second commitment returned when it was applied: a certified shot
reduction that one device card refuses to price, an accuracy-matched optimum
that belongs to the card rather than the Hamiltonian, a bank that is never
priced because its own bias exceeds the target, two preregistered screens that
returned no result, a reformulation retired against its own falsifier, and a
coverage census showing where the contract is not yet enforced.

There is no advantage claim in this paper, and every device cost in it is
logical accounting under an explicitly illustrative device card.

## Files

- `manuscript.tex` — REVTeX 4.2 (`aps,pra`) source.
- `references.bib` — bibliography. Entries shared with the companion
  manuscripts are copied verbatim from theirs.
- `make_tables.py` — regenerates `tables/*.tex` and `data/source_census.json`.
- `make_figures.py` — regenerates `paper_assets/*.pdf` and their manifest.
  Matplotlib is imported at call time, so the manuscript gate runs in an
  environment without a plotting stack.
- `check_manuscript.py` — the gate: structure, citations, table column counts,
  generated-number coverage, artifact freshness, and the source census.
- `data/source_census.json` — the committed census of the package, the
  benchmark gates, and the evidence declarations of the record set.

## Build

```bash
python paper_architecture/make_tables.py     # tables/ and data/source_census.json
python paper_architecture/make_figures.py    # paper_assets/*.pdf (needs matplotlib)
python paper_architecture/check_manuscript.py
cd paper_architecture
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`check_manuscript.py` runs in CI; the two generators do not, because a figure
rebuild needs a plotting stack and a table rebuild is what the gate is checking
against.

## Data → manuscript map

Every table and every quantity in the body prose is generated. Nothing is typed
by hand, and `check_manuscript.py` rejects a bare decimal or a four-digit
integer anywhere in the body text.

| Manuscript element | Source |
|---|---|
| Table I (layers), Fig. 1(a) | `data/source_census.json`, recomputed from `clifford_qc/` |
| Table II (evidence coverage) | `data/source_census.json`, recomputed from `benchmarks/reference_results/` |
| Table III (gate classes) | `data/source_census.json`, recomputed from `benchmarks/` |
| Table IV, Fig. 3 (block-commuting hierarchy) | `clifford_hierarchy_h4_v2.json`, `clifford_hierarchy_beh2_v2.json` |
| Table V (QWC vs fully commuting) | `phase14b_qwc_vs_fc.json` |
| Table VI, Fig. 4 (accuracy-matched cost) | `protocol_cost.json` |
| Table VII (mapping vs instance) | `mapping_axis.json`, `r3c_lih_full_cost.json`, `r3d_qr3_refinement.json` |
| Table VIII (contextual screen) | `r4a_contextual_screen.json` |
| Table IX (programme ledger) | `PHASE_STATUS.json` |
| Sec. VI E (preconditioner) | `g1_structural_preconditioner.json` |
| `tables/numbers.tex` (all prose quantities) | all of the above |

Records without a path prefix live under `benchmarks/reference_results/`. Of
these, only the block-commuting hierarchy overlaps a companion manuscript, and
this paper reads its `_v2` record; `paper_acase/` reads the `_v1` record of the
same study. Everything else — the preregistered QWC-versus-fully-commuting
comparison, the protocol-cost programme, the mapping axis, the second-instance
pricing and its refinement, the contextual screen and the structural
preconditioner — appears in no other manuscript.

## What the gate enforces

`check_manuscript.py` fails, rather than typesetting, when:

1. a generated table is stale against the record or the generator that produced
   it (blob hashes are embedded in each fragment and re-derived);
2. a figure's manifest no longer matches its generator or its input digests —
   digests rather than bytes, because vector output is not reproducible across
   plotting and font builds while the digest of its inputs is;
3. a number is typed into a table cell or into the body prose instead of being
   generated into `tables/`;
4. a generated macro is used without being defined, or defined without being
   used;
5. the committed source census no longer equals a census recomputed now — a
   module added without a layer, a benchmark gate added without a declared
   class, or a record whose evidence declaration changed;
6. a citation, label, reference, input or table column count is broken;
7. one of the paper's evidence-language invariants has been edited out.

Points 3 and 5 are the ones specific to this paper. Point 3 is what makes
"every number is generated" checkable rather than aspirational, and point 5 is
what keeps the architecture tables honest, since no experiment rebuilds them.

Fault-injection is the way to confirm the gate still works: rename a module
without assigning it a layer, add a `benchmarks/check_*.py` without a class, or
type a decimal into a paragraph, and the checker names it.
