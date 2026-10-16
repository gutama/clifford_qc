# arXiv submission

Everything on this page is about getting *this* manuscript onto arXiv
unchanged. The science is in `manuscript.tex`; what follows is the packaging,
the form fields, and the two or three decisions that are the author's rather
than the script's.

## Versions

| Version | arXiv | Source | Status |
|---|---|---|---|
| v1 | [2609.23059v1](https://arxiv.org/abs/2609.23059v1), 19 Sep 2026 | `manuscript.tex` at `fcceae2` (the posted source is byte-identical) | posted |
| v2 | 2609.23059v2 | this tree | ready to upload as a **replacement** |

v2 is a replacement of 2609.23059, not a new submission: use *Replace* on the
article from the arXiv user page, upload the new tarball, and keep the
identifier. What changed, for the replacement's comments field and for anyone
diffing the two versions:

- **New subsection, Residuals and folded-spectrum roots.** The second-moment
  block `K`, the true Ritz residual with its resolution floor, folded-spectrum
  roots with refined Ritz vectors, and the target and Weinstein inclusion
  intervals (which hold *some* eigenvalue, not a named one). Validated against
  dense oracles on the licensed banks; exact pairings only.
- **New measurement result, anticommuting cliques** (Table IX). The
  preregistered fixed-reference energy comparison against QWC on three
  eight-qubit banks: one censored-baseline reduction, one no-reduction, one
  reduction, and device prices that split again by card.
- **New negative result, the measured residual block.** Two post-hoc
  declarations price the residual against the energy under uniform QWC and
  fully commuting grouping; both read `NONE`, which is why no finite-shot
  second-moment estimator ships.
- **Coefficient storage now measured.** The process-isolated policy matrix
  (Table V) replaces v1's statement that the profiles established no
  process-memory reduction: on the four larger inputs, packing and streaming
  lower observed peak process memory while object retain stays fastest.
- **Capabilities brought up to date.** Sparse exterior operations in the
  kernel; exchange/CEO pools and shared-unitary multi-root ADAPT in exact
  sectors; time-evolved QSCI input; the fragment-solver callback; the
  versioned evidence contract and digest-bound annotations. The census tables
  are regenerated (more modules, gates and records; no JSON record now lacks an
  evidence declaration).
- **Corrections.** The abstract no longer says evidence labels are not
  validated uniformly; the gates section's spelled-out counts ("Five are
  preregistration gates", "Three further gates", "two compare records") had
  gone stale and are now generated, and the manuscript gate rejects such words.
  The gates table is full-width and several floats are anchored earlier, so no
  float is stuck at the end of the document.

## Build the package

```bash
python paper_architecture/make_tables.py      # tables/ and data/source_census.json
python paper_architecture/make_figures.py     # paper_assets/*.pdf (needs matplotlib)
python paper_architecture/check_manuscript.py # the manuscript gate
python paper_architecture/make_arxiv.py       # the submission package
```

The last command writes `paper_architecture/arxiv/` (git-ignored):

| File | What it is |
|---|---|
| `manuscript-arxiv.tar.gz` | the upload |
| `manuscript.pdf` | the PDF built **from the tarball**, not from the working tree |
| `abstract.txt` | the abstract as plain text, for the web form |
| `MANIFEST.txt` | a SHA-256 for every file in the archive |

`make_arxiv.py` builds the manuscript twice. The first build is the ordinary
local one, with BibTeX, and exists only to produce the `.bbl`. The second is
the one that matters: it unpacks the staged files on their own — no
`references.bib` anywhere in reach — and runs `pdflatex` three times, which is
exactly what arXiv does. If the two builds disagree on page count, or the
second leaves a citation or a cross-reference undefined, the script fails
rather than handing over an archive that would break on the other side.

`make_arxiv.py --check` runs the structural half of that without a TeX
installation, and runs in CI on every pull request. It also guards the
bibliography against the one BibTeX trap that costs a whole reference list:
BibTeX starts an entry at an at-sign wherever it finds one, comment or not, so
a header comment that mentions an entry type silently swallows the entry below
it. The manuscript gate cannot see that — it matches entry keys, and a comment
has none.

It also refuses a figure carrying Type 3 fonts. Type 3 is matplotlib's default,
it makes text unsearchable and renders badly at small sizes, and APS asks
authors not to send it; `make_figures.py` sets a TrueType font type instead.
Two schematics were nonetheless committed before that setting and kept their
Type 3 fonts through every later check, because a figure that declares no input
records is bound only to its own hash and to a generator digest that had not
moved. Regenerating them fixed it; the scan keeps it fixed.

## What is in the archive, and what is deliberately not

The archive holds `manuscript.tex`, `manuscript.bbl`, the eleven generated
`tables/*.tex` fragments the manuscript inputs, and the five
`paper_assets/*.pdf` figures it includes. Eighteen files, about 280 kB
uncompressed and 161 kB in the archive — far under arXiv's upload limit, which
`make_arxiv.py` checks anyway in case a raster figure ever arrives.

`references.bib` is **not** shipped. arXiv does not run BibTeX, so the `.bib`
would be dead weight, and leaving it out is what lets the verification build
prove the `.bbl` is really being read. The generators, the checker, the figure
manifest and the source census are not shipped either: they are how the paper
is built, not what it is.

## Form fields

**Title**

```
clifford_qc: A Python Toolkit for Quantum Simulation
```

**Authors**

```
Ginanjar Utama, Hermawan Kresno Dipojono
```

Both at the Department of Engineering Physics, Institut Teknologi Bandung.

**Abstract** — paste `arxiv/abstract.txt` verbatim; a replacement may change
the abstract, and v2's does. It is derived from the manuscript, so it cannot
drift from the typeset one. It is now about 1880 characters against arXiv's
1920-character limit — close, so any further addition to the abstract needs a
matching cut — and `make_arxiv.py` fails if a revision pushes it over.

**Categories**

- Primary: `quant-ph`, as v1 was posted. The subject is quantum simulation
  methodology, and both companion manuscripts are there.
- v1 is listed under `quant-ph` alone. Cross-lists to `physics.comp-ph` and
  `cs.MS` (Mathematical Software) remain reasonable for a toolkit paper; arXiv
  handles a cross-list request on an existing article separately from a
  replacement, so adding one is an independent decision.

**Comments**

v1 was posted with `15 pages, 5 figures`. For v2:

```
18 pages, 5 figures, 12 tables. v2: adds exact Ritz residuals and
folded-spectrum roots, a process-isolated storage measurement, an
anticommuting-clique comparison and a measured-residual closure; updates the
software census. Companion to arXiv:2607.17443. Source, records and manuscript
gate at https://github.com/gutama/clifford_qc
```

Re-count if the manuscript changes length; `make_arxiv.py` prints the page
count of the build it verified.

**ACM class** — `G.4; J.2`
**MSC class** — `81P68; 81-04; 65-04`

**Journal reference / DOI** — leave empty. No archival identifier or
package-index release exists yet, as the manuscript's Use and distribution
section explains; adding one here would contradict it.

**Report number** — none.

## Decisions that are not the script's

**Licence.** Decided at v1: arXiv's non-exclusive distribution licence 1.0.
A replacement keeps it. The repository is Apache-2.0, which covers the code but
says nothing about the manuscript; moving the text to CC BY 4.0 would be a
separate request to arXiv, not something the replacement form changes.

**Ordering against the companions.** Resolved: the manuscript cites
arXiv:2607.17443 as the operator-centric companion, and both are posted.

**Journal submission.** The source is REVTeX 4.2 with the `aps,pra` options, so
it is already in the shape Physical Review A wants. Nothing in the arXiv
package needs to change for a journal submission; the `.bbl` that arXiv needs
is one APS accepts too.

## Before uploading

- [ ] `python paper_architecture/check_manuscript.py` passes.
- [ ] `python -m pytest tests/test_paper_architecture_gate.py tests/test_paper_architecture_arxiv.py` passes.
- [ ] `python paper_architecture/make_arxiv.py` reports the same page count for
      the staged build as the local one.
- [ ] `arxiv/manuscript.pdf` — the one built from the tarball — reads correctly:
      figures present, references numbered, no `??` anywhere.
- [ ] The comments line's page, figure and table counts match that PDF.
- [ ] Uploaded through *Replace* on 2609.23059, not as a new submission.
- [ ] The revision is merged and pushed, so the URL in the manuscript resolves
      to the revision the paper describes.

## If arXiv's build fails anyway

arXiv returns its own log. Three failures are worth recognising on sight:

- *Undefined citations, references rendered as `[?]`* — the `.bbl` did not make
  it into the upload. The verification build here exists to catch that first.
- *A missing figure* — a path that resolved locally through a directory the
  archive does not contain. `--check` rejects any `\input` or
  `\includegraphics` reaching outside `paper_architecture/`.
- *A package not found* — arXiv's TeX Live differs from a local installation.
  Everything this manuscript loads (`revtex4-2`, `amsmath`, `amssymb`, `bm`,
  `graphicx`, `microtype`, `xcolor`, `hyperref`) has been in arXiv's
  distribution for years, but a newly added package is the first thing to
  suspect.
