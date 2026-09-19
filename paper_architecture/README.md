# clifford_qc: A Python Toolkit for Quantum Simulation

This paper introduces the operator-centric representation the package is built
on, then the tasks it supports: model construction, operator and circuit
experiments, eigensolver comparisons, measurement reuse, and repeated molecular
solves. It then explains the interfaces, numerical assumptions, memory policies,
and benchmark evidence.

The intended audience is researchers considering the package for their own
experiments. The operator-centric section states the representation and works a
small transverse-field Ising example through it, showing where measurement reuse
and the odd-parity pool restriction come from; its formal development belongs to
the companion preprint and is cited, not repeated. The feature table
distinguishes implemented capabilities from performance evidence. The workflow
section explains virtual subspaces and the prepare/solve/validate pipeline, and
walks one molecular run through the three commands end to end. The distribution
section describes the current alpha status and planned public availability
without claiming an existing package-index release.

The paper preserves the quantitative benchmark record. Device costs use
illustrative assumptions; the fixed-functional shot certificate is not a
certificate of total ground-state error. Evidence-schema coverage remains
incomplete, and the streaming coefficient-row bound is not a process-memory
bound.

For package installation and runnable examples, start with the root
[README](../README.md). The [architecture guide](../ARCHITECTURE.md) describes
implementation boundaries, and the [pipeline guide](../molecular/PIPELINE.md) documents
repeated molecular calculations.

## Files

- `manuscript.tex` — REVTeX 4.2 (`aps,pra`) source.
- `references.bib` — bibliography. Entries shared with the companion
  manuscripts are copied verbatim from theirs — including the three
  Clifford-realization references the operator-centric section cites — with one
  documented exception:
  the companion preprint is entered as `@misc` with an `eprint` field, because
  the `@article` form the others use makes `apsrev4-2` print the identifier
  twice. `paper_a_case_subspaces/` is a frozen published snapshot and is not
  edited to match.
- `make_tables.py` — regenerates `tables/*.tex` and `data/source_census.json`.
- `make_figures.py` — regenerates `paper_assets/*.pdf` and their manifest.
  Matplotlib is imported at call time, so the manuscript gate runs in an
  environment without a plotting stack.
- `check_manuscript.py` — the gate: structure, citations, table column counts,
  generated-number coverage, artifact freshness, and the source census.
- `make_arxiv.py` — assembles the arXiv submission package and rebuilds it the
  way arXiv will, with `pdflatex` alone and no `references.bib` in reach. Its
  `--check` mode needs no TeX and also guards the bibliography against a
  comment BibTeX would read as an entry.
- `ARXIV.md` — the submission itself: form fields, categories, what the archive
  holds and what it deliberately leaves out, and the decisions the script
  cannot make.
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

`check_manuscript.py` runs in CI and invokes `make_tables.py` into a scratch
tree so the committed fragments can be compared byte for byte.  The figure
generator does not run in CI because it needs a plotting stack; figure drift is
checked through the committed input manifest instead.

For the submission package:

```bash
python paper_architecture/make_arxiv.py --check   # structural, no TeX needed
python paper_architecture/make_arxiv.py           # stage, archive, verify
```

The structural half runs in CI beside the manuscript gate.  The full build
stages only the files the manuscript reads, then rebuilds that staging tree
with `pdflatex` alone — no BibTeX, no `references.bib` — because that is what
arXiv does, and a submission that silently drops its bibliography compiles
perfectly well locally.  See [ARXIV.md](ARXIV.md).

## Data → manuscript map

Every quantitative table and measured quantity in the body prose is generated.
The two qualitative tables — the capability map and the pipeline-stage table —
are maintained against the source interfaces.
`check_manuscript.py` rejects every numeral in the body text
outside a short allowlist of conceptual notation.

| Manuscript element | Source |
|---|---|
| Operator-centric section and its worked example | Package interfaces (`clifford_qc/pauli_kernel.py`, `clifford_qc/algorithms/pools.py`) and the companion preprint; the example is algebra, not a measurement |
| Capabilities table and workflow | Package interfaces, `pyproject.toml`, and `molecular/PIPELINE.md` |
| Molecular pipeline table and walkthrough | `clifford_qc/prepared.py`, `clifford_qc/pipeline.py`, and `molecular/PIPELINE.md` |
| Fig. 1 (end-to-end architecture) | Static schematic generated by `make_figures.py`; no experimental input |
| Layers table and layer figure | `data/source_census.json`, recomputed from `clifford_qc/` |
| Evidence coverage table | `data/source_census.json`, recomputed from `benchmarks/reference_results/` |
| Gate classes table | `data/source_census.json`, recomputed from `benchmarks/` |
| Block-commuting hierarchy table and figure | `clifford_hierarchy_h4_v2.json`, `clifford_hierarchy_beh2_v2.json` |
| QWC vs fully commuting table | `phase14b_qwc_vs_fc.json` |
| Accuracy-matched cost table and figure | `protocol_cost.json` |
| Mapping vs instance table | `mapping_axis.json`, `r3c_lih_full_cost.json`, `r3d_qr3_refinement.json` |
| Contextual screen table | `r4a_contextual_screen.json` |
| Pool-filtering subsection | `g1_structural_preconditioner.json` |
| `tables/numbers.tex` (all prose quantities) | all of the above |

The coefficient-storage subsection quotes no measured quantity: the lifetime policy is
described from the interface it exposes, and the profile numbers behind it are
small-system smoke measurements that establish no production result. The
operator-centric and molecular-pipeline sections quote none either, for the same
reason in two forms: the worked example is an algebraic identity anyone can
recompute from the kernel, and the pipeline walkthrough describes what each
stage reads, writes and reuses rather than how fast it does so.

Records without a path prefix live under `benchmarks/reference_results/`. Of
these, only the block-commuting hierarchy overlaps a companion manuscript, and
this paper reads its `_v2` record; `paper_acase/` reads the `_v1` record of the
same study. Everything else — the preregistered QWC-versus-fully-commuting
comparison, the protocol-cost programme, the mapping axis, the second-instance
pricing and its refinement, the contextual screen and the structural
preconditioner — appears in no other manuscript.

## What the gate enforces

`check_manuscript.py` fails, rather than typesetting, when:

1. a generated fragment differs by a single byte from what the generator
   produces right now — the fragments are regenerated into a scratch tree
   during the check and compared. The provenance hashes each fragment carries
   digest its source records and the generator, so they catch a record
   regenerated without rerunning the generator; only regeneration catches a
   value edited into a fragment by hand;
2. a figure's manifest no longer matches its generator or its input digests —
   digests rather than bytes, because vector output is not reproducible across
   plotting and font builds while the digest of its inputs is;
3. any numeral appears in a table cell or in the body prose, outside a short
   allowlist of conceptual notation (a block size under discussion, a chemical
   subscript, the dimension of the algebra). There is no digit-count threshold:
   a three-digit setting count typed into a paragraph fails;
4. a generated macro is used without being defined, or defined without being
   used;
5. the committed source census no longer equals a census recomputed now — a
   module added without a layer, a benchmark gate added without a declared
   class, a gate added to or dropped from the CI workflow, or a record whose
   evidence declaration changed anywhere in the file, nested values and
   per-quantity maps included;
6. a citation, label, reference, input or table column count is broken;
7. one of the paper's evidence-language invariants has been edited out —
   including the statement that the lifetime bound is not a total-process one,
   so a later edit cannot leave the abstract asserting a discipline the body
   has dropped.

Points 1, 3 and 5 are the ones specific to this paper. Points 1 and 3 are what
make "every number is generated" checkable rather than aspirational, and point
5 is what keeps the architecture tables honest, since no experiment rebuilds
them.

`tests/test_paper_architecture_gate.py` pins the parts of this that have
already been wrong once: an evidence *role* counted as a tier, a nested tier
change that moved nothing the census compared, a hand-edited fragment that
passed a check over its inputs, and a rewrapped paragraph reported as a dropped
evidence commitment. Fault injection is still the quickest confirmation by
hand: rename a module without assigning it a layer, drop a gate from the
workflow, or type a number into a paragraph, and the checker names it.

`tests/test_paper_architecture_arxiv.py` does the same for the submission
package, where the failures are ones a local build cannot produce: a `.bbl`
left out of an upload arXiv will not rebuild, a `\pdfoutput` too far down the
preamble for arXiv to read, and a path that resolves here through a directory
the archive does not carry.
