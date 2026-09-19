# arXiv submission

Everything on this page is about getting *this* manuscript onto arXiv
unchanged. The science is in `manuscript.tex`; what follows is the packaging,
the form fields, and the two or three decisions that are the author's rather
than the script's.

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

The archive holds `manuscript.tex`, `manuscript.bbl`, the nine generated
`tables/*.tex` fragments the manuscript inputs, and the five
`paper_assets/*.pdf` figures it includes. Sixteen files, about 255 kB
uncompressed and 155 kB in the archive — far under arXiv's upload limit, which
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

**Abstract** — paste `arxiv/abstract.txt` verbatim. It is derived from the
manuscript, so it cannot drift from the typeset one. It is currently about
1750 characters against arXiv's 1920-character limit, and `make_arxiv.py`
fails if a revision pushes it over.

**Categories**

- Primary: `quant-ph`. The subject is quantum simulation methodology, and both
  companion manuscripts are there.
- Cross-list: `physics.comp-ph`, and `cs.MS` (Mathematical Software) if the
  moderators accept it. `cs.MS` is the natural home for a toolkit paper, but
  it is the more likely of the two to be reassigned; nothing is lost by
  proposing it.

**Comments**

```
16 pages, 5 figures, 10 tables. Companion to arXiv:2607.17443. Source,
records and manuscript gate at https://github.com/gutama/clifford_qc
```

Re-count if the manuscript changes length; `make_arxiv.py` prints the page
count of the build it verified.

**ACM class** — `G.4; J.2`
**MSC class** — `81P68; 81-04; 65-04`

**Journal reference / DOI** — leave empty. No archival identifier or
package-index release exists yet, and Sec. XI of the manuscript says so; adding
one here would contradict it.

**Report number** — none.

## Decisions that are not the script's

**Licence.** arXiv asks at submission and the choice is hard to reverse. The
repository is Apache-2.0, which covers the code but says nothing about the
manuscript. CC BY 4.0 matches the openness of the repository and lets the text
be reused and mirrored; arXiv's own non-exclusive licence is the conservative
default and is what most APS-bound preprints use. Pick before uploading —
arXiv will not let the licence be loosened afterwards without an administrative
request.

**Ordering against the companions.** The manuscript cites
arXiv:2607.17443 as the operator-centric companion. That reference resolves
whether or not this paper is posted first, so the order is a presentation
choice rather than a dependency.

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
- [ ] Licence chosen.
- [ ] The repository is pushed, so the URL in the manuscript resolves to the
      revision the paper describes.

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
