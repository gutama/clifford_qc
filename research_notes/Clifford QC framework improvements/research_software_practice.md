# Research-software practice for small academic quantum toolkits (status November 2026), applied to clifford_qc

Scope: what makes a small academic quantum-software toolkit usable, trusted and adopted, including packaging, documentation, API design, interoperability, benchmarking standards, hardware execution paths, publication venues and community. Every number below carries its source and retrieval date. PyPI release data come from the PyPI JSON API, and download counts come from the pypistats.org "recent" endpoint ("last_month" means the trailing 30 days). Both were retrieved on 2026-11-14. pypistats counts include CI and mirror traffic, so read them as relative scale, not as a count of users. GitHub star and contributor counts could not be retrieved: the GitHub API was blocked in this session.

Grounding facts about clifford_qc come from a read-only inspection of /home/user/clifford_qc on 2026-11-14:
- **Packaging:** setuptools with a flat layout, plus `MANIFEST.in` and `py.typed`. There are no `[project.scripts]` entry points.
- **Citation:** `CITATION.cff` exists, and its `preferred-citation` is arXiv:2608.00560.
- **Tags:** the only tag is `dacase-arxiv-v1.0.0` (2026-08-13). There is no `v0.3.0` tag in the local clone, which is shallow, so the full history could not be checked.
- **Missing community and docs files:** no `CONTRIBUTING.md`, no `CODE_OF_CONDUCT.md`, no issue templates, and no Sphinx or MkDocs configuration.
- **Repository size:** `benchmarks/` is 149 MB, `docs/` is 13 MB, `PLAN.md` is 408 KB and `REPRODUCING.md` is 340 KB.
- **CI:** a weekly cron, with the matrix on Python 3.10, 3.11, 3.13 and 3.14 and a record environment on 3.12.
- **Name:** `clifford-qc` and `clifford_qc` both return 404 on PyPI.
- **Authorship:** `git shortlog` on the shallow clone attributes 55 of 211 commits to "Claude" and 4 to "copilot-swe-agent[bot]".

## 1. What comparable small and mid-sized quantum toolkits do for distribution, docs and onboarding, and what adoption it bought them

### Takeaway
Every comparable toolkit that has measurable adoption is pip-installable from PyPI, and most publish a docs site organised as tutorials, how-to guides, explanations and an API reference. Download volume tracks two things: being a dependency of a big ecosystem (Qiskit, Google), and offering a narrowly useful primitive (Stim, ffsim, quimb). Tutorial volume matters less. Pauli-algebra libraries without an ecosystem host (symmer, PauliEngine, quri-parts, tequila) sit at roughly 150–2,100 downloads a month. The bar for credibility is low: a PyPI release, a docs site and a citable paper. Beyond that, adoption comes from being plugged into someone else's workflow.

### Cited Findings

**Adoption and release snapshot (PyPI JSON plus pypistats, retrieved 2026-11-14)**

| Package | Latest version (date) | Releases total / last 12 mo | Downloads, last 30 days | Notes |
|---|---|---|---|---|
| stim | 1.16.0 stable; dev builds to 2026-10-29 | 100 / 62 (dev builds included) | 575,463 | Single-purpose fast Clifford/QEC simulator ([PyPI](https://pypi.org/pypi/stim/json), [pypistats](https://pypistats.org/packages/stim)) |
| quimb | 1.16.0 (2026-11-10) | 39 / 6 | 149,281 | Tensor-network library, needs Python ≥3.11 ([PyPI](https://pypi.org/pypi/quimb/json), [pypistats](https://pypistats.org/packages/quimb)) |
| openfermion | 1.8.1 (2026-07-17) | 36 / 2 | 133,391 | "Production/Stable", needs h5py, Python ≥3.10 ([PyPI](https://pypi.org/pypi/openfermion/json), [pypistats](https://pypistats.org/packages/openfermion)) |
| qiskit-nature | 0.8.0 (2026-06-01) | 28 / 1 | 104,882 | ([PyPI](https://pypi.org/pypi/qiskit-nature/json), [pypistats](https://pypistats.org/packages/qiskit-nature)) |
| ffsim | 0.0.86 (2026-11-05) | 86 / 25 | 38,060 | Rust abi3 wheels; extras `cuda12`, `cuda13`, `qiskit`; Python ≥3.11 ([PyPI](https://pypi.org/pypi/ffsim/json), [pypistats](https://pypistats.org/packages/ffsim)) |
| qibo | 0.3.7 (2026-11-10) | 57 / 9 | 30,827 | Python ≥3.12,<3.15 ([PyPI](https://pypi.org/pypi/qibo/json), [pypistats](https://pypistats.org/packages/qibo)) |
| qiskit-addon-utils | 0.4.0 (2026-07-15) | 6 / 2 | 24,762 | ([PyPI](https://pypi.org/pypi/qiskit-addon-utils/json), [pypistats](https://pypistats.org/packages/qiskit-addon-utils)) |
| qiskit-addon-sqd | 0.14.1 (2026-11-06) | 15 / 4 | 22,421 | First release 2024-09-11; also on conda-forge ([PyPI](https://pypi.org/pypi/qiskit-addon-sqd/json), [pypistats](https://pypistats.org/packages/qiskit-addon-sqd)) |
| pennylane-qiskit | 0.45.1 (2026-10-29) | 60 / 4 | 13,905 | ([PyPI](https://pypi.org/pypi/pennylane-qiskit/json), [pypistats](https://pypistats.org/packages/pennylane-qiskit)) |
| qiskit-addon-obp | 0.3.0 (2025-05-22) | 3 / 0 | 11,077 | ([PyPI](https://pypi.org/pypi/qiskit-addon-obp/json), [pypistats](https://pypistats.org/packages/qiskit-addon-obp)) |
| mqt.bench | 2.4.0 (2026-11-13) | 47 / 5 | 5,813 | ([PyPI](https://pypi.org/pypi/mqt.bench/json), [pypistats](https://pypistats.org/packages/mqt-bench)) |
| tensorcircuit-ng | 1.10.0 (2026-10-02) | 16 / 7 | 5,144 | ([PyPI](https://pypi.org/pypi/tensorcircuit-ng/json), [pypistats](https://pypistats.org/packages/tensorcircuit-ng)) |
| mitiq | 1.1.0 (2026-09-04) | 61 / 4 | 4,729 | GPL-3.0; Beta; Typed; Python ≥3.11,<3.13 ([PyPI](https://pypi.org/pypi/mitiq/json), [pypistats](https://pypistats.org/packages/mitiq)) |
| clifford (pygae GA library) | n/a | n/a | 2,334 | A different project whose name is close to clifford_qc's ([pypistats](https://pypistats.org/packages/clifford)) |
| tequila-basic | 1.9.12 (2026-10-06) | 29 / 2 | 2,088 | No extras; heavy core dependencies (openfermion, sympy, autograd, pytest) ([PyPI](https://pypi.org/pypi/tequila-basic/json), [pypistats](https://pypistats.org/packages/tequila-basic)) |
| supermarq | 0.5.70 (2026-09-30) | 120 / 16 | 818 | ([PyPI](https://pypi.org/pypi/supermarq/json), [pypistats](https://pypistats.org/packages/supermarq)) |
| quri-parts | 0.28.0 (2026-10-29) | 44 / 7 | 775 | Metapackage of `quri-parts-*` subpackages; 15 backend extras; "Typing :: Typed" ([PyPI](https://pypi.org/pypi/quri-parts/json), [pypistats](https://pypistats.org/packages/quri-parts)) |
| symmer | 0.0.12 (2026-08-14) | 12 / 2 | 306 | Hard dependencies on qiskit ≥2,<3, openfermion, quimb and numba; no extras; Python ≥3.11 ([PyPI](https://pypi.org/pypi/symmer/json), [pypistats](https://pypistats.org/packages/symmer)) |
| metriq-gym | 0.9.0 (2026-11-06) | 16 / 7 | 176 | Alpha; first release 2025-06-25 ([PyPI](https://pypi.org/pypi/metriq-gym/json), [pypistats](https://pypistats.org/packages/metriq-gym)) |
| pauliengine | 0.1.4 (2026-09-29) | 3 / 3 | 158 | First release 2026-07-14; C++ with abi3 and platform wheels; "Beta", "Typed"; repository tequilahub/pauliengine; Python ≥3.10,<3.14 ([PyPI](https://pypi.org/pypi/pauliengine/json), [pypistats](https://pypistats.org/packages/pauliengine)) |
| tencirchem | 2024.11 (2024-11-14) | 5 / 0 | 28 | Sdist only, no wheel; dormant for two years ([PyPI](https://pypi.org/pypi/tencirchem/json), [pypistats](https://pypistats.org/packages/tencirchem)) |
| pauliarray | not on PyPI under this name (404) | — | — | Docs at algolab-quantique.github.io ([PyPI 404](https://pypi.org/pypi/pauliarray/json), [docs](https://algolab-quantique.github.io/pauliarray/user_guide/data_structures.html)) |

- **conda-forge presence** (anaconda.org API, 2026-11-14): qiskit, pennylane, pyscf, quimb, cirq, openfermion, qibo, stim and qiskit-addon-sqd have conda-forge packages. ffsim, mitiq, tequila-basic, symmer, quri-parts, pauliengine, mqt.bench and pytket returned none. Peers in this tier therefore treat conda-forge as optional. ([anaconda.org conda-forge API](https://api.anaconda.org/package/conda-forge/qiskit-addon-sqd))
- **ffsim docs:** the site is organised into Tutorials (2), Explanations (11), How-to guides (14) and an API reference per submodule. The landing page shows `pip install ffsim` and links an installation page covering source and container installs. It gives a BibTeX entry for "ffsim: Faster simulation of fermionic quantum circuits" (arXiv:2605.03123, 2026). The docs moved from qiskit-community.github.io to qiskit.github.io/ffsim. ([ffsim docs](https://qiskit.github.io/ffsim/), [redirect](https://qiskit-community.github.io/ffsim/))
- **ffsim versioning:** after 86 releases since 2023-08, ffsim is still at 0.0.x. Using a 0.0.x number has not held back adoption (38k downloads/month). ([PyPI](https://pypi.org/pypi/ffsim/json))
- **qiskit-addon-sqd docs:** the docs have a quickstart, tutorials hosted on IBM Quantum docs (SQD for chemistry Hamiltonians; sample-based Krylov diagonalisation for lattice models) and release notes that flag breaking changes. Examples: 0.7.0 changed the `hamming_right`/`hamming_left` parameters, and 0.11.0 added `diagonalize_fermionic_hamiltonian()`. ([IBM guide](https://docs.quantum.ibm.com/guides/qiskit-addons-sqd), [release notes](https://docs.quantum.ibm.com/api/qiskit-addon-sqd/release-notes))
- **qiskit-addon-sqd rename:** the docs at qiskit.github.io now redirect with the notice "`qiskit-addon-sqd` is now `qiskit-sqd`". The install page still says `pip install 'qiskit-addon-sqd'`, and `qiskit-sqd` does not exist on PyPI as of 2026-11-14. ([redirect](https://qiskit.github.io/qiskit-addon-sqd/install.html), [qiskit-sqd install page](https://qiskit.github.io/qiskit-sqd/install.html))
- **qiskit-addon-sqd notebooks:** notebooks live in `docs/` and are run in CI. The `nbtest` and `notebook-dependencies` extras exist for this. ([PyPI](https://pypi.org/pypi/qiskit-addon-sqd/json), [install page](https://qiskit.github.io/qiskit-sqd/install.html))
- **Tequila tutorials:** tutorials sit in a separate `tequila-tutorials` repository (basic usage, noisy simulation, FAQ, chemistry backends, research projects). A tutorials site lists notebooks by category with authors and dates. The pages found were dated 2022–2024. ([tequila-tutorials](https://www.github.com/tequilahub/tequila-tutorials), [tutorials site](https://tequilahub.github.io/tequila-tutorials/tutorials.html), [chemistry notebook](https://tequilahub.github.io/tequila/jupyter/Chemistry.html))
- **PauliArray:** a NumPy library that stores each Pauli string as two boolean arrays (X and Z) and supports broadcasting over multidimensional arrays of Pauli strings. It has a docs site with a user guide and an API reference, and a paper on arXiv:2405.19287 (May 2024). ([arXiv](https://arxiv.org/abs/2405.19287), [docs](https://algolab-quantique.github.io/pauliarray/api_reference/pauliarray.pauli.html))
- **PauliEngine:** a C++ backend bound to Python with nanobind. Its API "mimic[s] the QubitOperator from OpenFermion" so users can switch between PauliEngine, PauliArray and OpenFermion, and it claims near-seamless integration into Tequila. arXiv:2601.02233 was posted on 5 January 2026, and PyPI followed on 2026-07-14. Its paper benchmarks were run before any PyPI release and appeared alongside the arXiv release. ([arXiv HTML](https://arxiv.org/html/2601.02233v1), [PyPI](https://pypi.org/pypi/pauliengine/json))
- **Barriers to adoption (Unitary Foundation survey, 2024 figures on the 2025 results page):** "Poorly documented" is the most-cited reason respondents do not use a tool they want, at 50.8%. It is followed by price (47.5%), incompatibility with the current stack (35.3%) and "tool does not exist" (33.7%). ([UF survey 2025](https://unitaryfoundation.github.io/survey-2025))
- **Tool usage (same survey):** Qiskit 69.11%, Qiskit Aer 47.74%, PennyLane 30.79%, Cirq 17.45% and CUDA Quantum 16.15% of respondents. For planned use, Qiskit leads at 31.19% and CUDA Quantum follows at 25.28%. Python was used by 94.8% in 2024. ([UF survey 2025](https://unitaryfoundation.github.io/survey-2025))
- **Where developers find answers (2025 survey):** project documentation and source code still rank above chatbots as sources for technical answers, and video-based learning has overtaken hackathons. ([UF survey 2025](https://unitaryfoundation.github.io/survey-2025), [Quantum Insider coverage](https://thequantuminsider.com/2025/11/05/the-young-and-the-relentless-poll-shows-more-quantum-developers-are-young-independent-and-unpaid/))

### Inferences
- The most direct peers are symmer, PauliEngine, PauliArray and tequila, all Pauli-operator libraries that sit beside a chemistry pipeline. They reach their users through three things: a PyPI wheel, an API that looks like OpenFermion's `QubitOperator` or Qiskit's `SparsePauliOp`, and a paper with a reproducible speed or resource benchmark against those incumbents. clifford_qc has the paper and the benchmark culture but lacks the wheel and the familiar-looking conversion layer at the front door.
- ffsim and qiskit-addon-sqd show that a 0.x version number and frequent breaking changes do not deter users, provided releases are frequent, release notes flag breaks, and the docs follow a tutorials / how-to / explanations / API structure. clifford_qc's "Alpha" status does not by itself need to change before publishing to PyPI.
- Peers keep tutorials and paper reproductions in separate repositories: tequila-tutorials, and symmer's paper code in TimWeaving/N2-CS-VQE (section 7). clifford_qc keeps 149 MB of benchmark records and 750 KB of plan and ledger Markdown inside the library repository.
- tencirchem has been dormant since 2024-11, ships an sdist only and gets about 28 downloads a month. Stopping releases visibly erodes adoption, which argues for a release process cheap enough that two people can keep it up (section 2).
- The pygae `clifford` package (2,334 downloads a month) and the GA libraries galgebra and kingdon share the "Clifford" vocabulary. Docs and the PyPI summary should say early that clifford_qc is a quantum-simulation toolkit, not a general geometric-algebra library.

### Gaps
- GitHub stars, contributor counts and citation counts for these projects could not be retrieved because the GitHub API was blocked. Google Scholar and Semantic Scholar were not queried.
- No PyPI or docs data were collected for InQuanto (closed source) or CUDA-Q example repositories beyond the cuda-quantum PyPI version (0.15.1, 2026-08-03; [PyPI](https://pypi.org/pypi/cuda-quantum/json)).
- I found no source quantifying how much a docs site or a PyPI release increased adoption for any single project, so the effect is inferred from cross-sectional comparison only.

## 2. Research-software standards: JOSS, SoftwareX, SciPost Codebases, FAIR4RS, archiving, SPECs, versioning and what reviewers ask for

### Takeaway
As of 2026 the binding constraints for a software paper are these:
- **JOSS:** at least 6 months of *public* development history with releases and issues or PRs, an evidenced "research impact statement", a "software design" section and an explicit AI-usage disclosure. JOSS no longer accepts effort proxies such as lines of code or "three months of work".
- **pyOpenSci:** a PyPI or conda release, a docs site, `CONTRIBUTING.md` and a code of conduct.
- **SciPost Physics Codebases:** benchmark tests and one worked example application.

clifford_qc already exceeds the test, CI and reproducibility expectations. It misses the packaging and community-file requirements. Its private history and AI-heavy commit record have to be handled explicitly.

### Cited Findings

**JOSS**
- **Submission requirements:** "The repository must have been public for more than six months prior to submission." Development "concentrated into a few days or weeks" fails the check. The issue tracker must be readable without registration. "Minor utility" packages and thin API clients are out of scope, and software must be "feature-complete". ([JOSS submitting](https://joss.readthedocs.io/en/latest/submitting.html))
- **Policy change of 5 January 2026:** JOSS is "shifting from evaluating software through proxies for effort" toward "clear evidence of research impact, intellectual contribution, scholarly significance, and open-source software practices". The previous benchmark of roughly three months of developer time no longer applies. Projects developed privately are ineligible until they have "at least six months of public history prior to submission", with releases and public issues or PRs. Submissions awaiting review are judged under the new scope. ([JOSS blog, 2026-01-05](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future))
- **Paper format:** 750–1750 words. Required sections are Summary, Statement of need, State of the field (with a "build vs. contribute" justification), Software design, Research impact statement and AI usage disclosure. On AI: "If no AI tools were used, state this explicitly"; if tools were used, describe how and how correctness was verified. API documentation must not go in the paper. ([JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html))
- **Reviewer checklist:** a README with a statement of need; installation by an automated procedure, "ideally via a package manager"; example usage; API documentation (undocumented API is "Bad (not acceptable)"); community guidelines for contributing, reporting issues and seeking support; and automated tests in CI ("Good"). Multiple contributors are preferred, and single-author projects can show community engagement instead. Reviewers check that the author list is complete. ([JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html))
- **After acceptance:** make a tagged release and deposit it with Zenodo or figshare to get an archive DOI. Preprints do not count as prior publication. ([JOSS submitting](https://joss.readthedocs.io/en/latest/submitting.html))

**pyOpenSci**
- **Pre-review checks:** the package must be on PyPI (preferred) or a community conda channel, with docs that let editors judge scope "without installing the package". It needs installation guides, quickstart tutorials, API docs with standard docstrings, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, an OSI licence and tests in CI. Ruff, mypy and uv or Hatch are recommended. Generative-AI use must be disclosed. ([pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html))
- **Maintainer expectations:** maintain the package for 1–2 years after review. Submit when "the API should be stable or close to it". A package whose core method already has a peer-reviewed paper can get a one-reviewer fast-track. A pyOpenSci acceptance is accepted by JOSS (the author adds a `paper.md`). After acceptance, archive on Zenodo and add the badge. Timelines are 2 weeks for the editor, 3 weeks for reviewers and 2 weeks for the author response. ([pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html))

**SciPost Physics Codebases**
- **Acceptance criteria (all required):** "Benchmarking tests must be provided"; "At least one example application must be presented in detail"; "High-level programming standards"; a userguide that "must properly contextualize the software"; a demonstrable need; and complete documentation including download, install and run instructions. ([SciPost Phys. Codeb. about](https://scipost.org/SciPostPhysCodeb/about))
- **Publication mechanics:** both the userguide and the code release receive Crossref DOIs, and citations are "two-headed". The code is hosted on git.scipost.org. Small fixes go to the repository without new DOIs, a bigger release gets a short refereeing cycle and a new code DOI, and major upgrades get new DOIs for both objects. Apache-2.0, BSD, GPL, LGPL and MIT licences are accepted. ([SciPost Phys. Codeb. about](https://scipost.org/SciPostPhysCodeb/about))

**SoftwareX**
- The Original Software Publication template carries a code-metadata table: current version, a permanent link to the repository, licence, languages and dependencies. Code must be freely available, and the accepted version is copied to the journal's GitHub repository for archiving. A software-update article template (v6, March 2026) also exists. I could not retrieve the full Guide for Authors (HTTP 403). ([Elsevier update template](https://legacyfileshare.elsevier.com/promis_misc/softwarex-software-update-template.docx), [lesson example](https://softdev4research.github.io/4OSS-lesson/05-use-registry))

**FAIR4RS, archiving and citation metadata**
- The FAIR4RS principles are 17 principles for research software. The RDA/ReSA "Actionable FAIR4RS" task force (started December 2024) lists practical steps that include `codemeta.json` and `CITATION.cff` metadata and archiving on a DOI-issuing repository such as Zenodo. ([RDA VP24 slides, April 2025](https://www.rd-alliance.org/wp-content/uploads/2025/04/RDA_VP24_FAIR4RS_Three_Years_On_Reformatted.pdf))
- A 2026 software-publication course recommends providing both `codemeta.json` (comprehensive JSON-LD metadata) and `CITATION.cff` (citation), and notes that Zenodo and Software Heritage ingest them. ([IN2P3 course, 2026-01-21](https://indico.in2p3.fr/event/36319/contributions/167844/attachments/99324/153344/20260121_software_publication_vuillaume.pdf))
- Zenodo's GitHub integration documentation covers enabling a repository, describing software through `CITATION.cff` or `.zenodo.json`, archiving GitHub releases, and how software records are archived in Software Heritage. ([Zenodo GitHub docs](https://help.zenodo.org/docs/github/))
- A CASRAI editorial says Zenodo honours a well-formed `CITATION.cff` and otherwise falls back to repository-level metadata. ([CASRAI](https://casrai.org/news/datacite-github-zenodo-software-citation-stack)) This is commentary, not official guidance.

**Scientific Python SPECs**
- **SPEC 0:** drop a Python version 3 years after its release and a core package 2 years after a feature release. The 2026 Q4 schedule recommends dropping **Python 3.12** and **NumPy 2.2.0**; NumPy 2.0.0 was due for dropping in Q2 2026. A SPEC 0 badge is available. ([SPEC 0](https://scientific-python.org/specs/spec-0000/))
- **Endorsed SPECs:** SPEC 1 (lazy loading), SPEC 4 (nightly wheels), SPEC 6 ("Keys to the Castle"), SPEC 7 (seeding pseudo-random number generation) and SPEC 8 (securing the release process). SPEC 3 (accessibility), SPEC 5 (CI best practices) and SPEC 9 (governance) are drafts. ([SPECs index](https://scientific-python.org/specs/))

**Python versions required by peers in 2026 (PyPI `requires_python`, 2026-11-14)**
- Python ≥3.11: pennylane 0.45.1, pennylane-lightning 0.45.0, pennylane-catalyst 0.15.0, qiskit-aer 0.18.2, cirq-core 1.7.0, amazon-braket-sdk 1.127.8, pytket-qiskit 0.79.0, ffsim, quimb, symmer.
- qibo needs Python ≥3.12.
- mitiq 1.1.0 is capped at <3.13 and pauliengine at <3.14.
- qiskit 2.6.2, qiskit-ibm-runtime 0.52.1 and pytket 2.18.5 still allow Python 3.10.

([pennylane](https://pypi.org/pypi/pennylane/json), [qiskit-aer](https://pypi.org/pypi/qiskit-aer/json), [cirq-core](https://pypi.org/pypi/cirq-core/json), [amazon-braket-sdk](https://pypi.org/pypi/amazon-braket-sdk/json), [qibo](https://pypi.org/pypi/qibo/json), [mitiq](https://pypi.org/pypi/mitiq/json), [qiskit](https://pypi.org/pypi/qiskit/json))

**Packaging guidance**
- The Scientific Python development guide recommends:
  - a PEP 621 `[project]` table, with Hatchling as "an excellent choice" of build backend;
  - an SPDX `license` together with `license-files`;
  - `requires-python` set to the true minimum with no upper cap;
  - extras in `[project.optional-dependencies]` and CLIs in `[project.scripts]`;
  - developer tools (pytest, ruff, Sphinx) in `[dependency-groups]`, not in extras;
  - a `src/` layout, which avoids tests importing the local tree instead of the installed package;
  - VCS-derived versions through hatch-vcs;
  - checking sdist and wheel contents before release.
  ([Scientific Python packaging guide](https://learn.scientific-python.org/development/guides/packaging-simple/))
- PyPI Trusted Publishing uses OIDC from CI such as GitHub Actions to obtain a short-lived, project-scoped token valid for 15 minutes, so no long-lived API token is needed. ([PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/))
- Semantic Versioning: "Major version zero (0.y.z) is for initial development. Anything MAY change at any time. The public API SHOULD NOT be considered stable." ([semver.org §4](https://semver.org/#spec-item-4))

### Inferences
- **JOSS timing.** The JOSS six-month clock starts when the repository is public with releases and issues or PRs. If the public GitHub repository dates from recent months, the earliest credible JOSS submission is about six months after the first public release and issue activity. Reviewers will also see the commit timeline. clifford_qc should publish to PyPI, tag `v0.3.x` and start using public issues and PRs now, so that the clock and the "research impact" evidence build up together.
- **AI disclosure.** About a quarter of the commits visible in the local clone are attributed to an AI agent. JOSS (and pyOpenSci) will require a specific AI-usage disclosure that says how correctness was verified. clifford_qc's checker scripts, frozen records and CI gates are unusually strong evidence of human verification and could be cited directly in that section.
- **Which venue fits.** SciPost Physics Codebases wants benchmarks plus a worked example application, which matches clifford_qc's benchmark records better than JOSS's short-paper format. Its two-DOI model (userguide plus code release) would also give the architecture paper a home. The trade-off is the full refereeing a major upgrade triggers, and hosting on git.scipost.org. JOSS fits a later, stable-API release. pyOpenSci is a stepping stone that JOSS accepts, but it expects 1–2 years of maintenance and a near-stable API.
- **Python policy.** clifford_qc supports Python 3.10–3.14. That is wider than SPEC 0 (3.10 and 3.11 are already past their windows, and 3.12 drops in Q4 2026) and wider than most peers it bridges to: PennyLane, Cirq, Aer and Braket all need ≥3.11. Moving the floor to 3.11 now, or 3.12 in 2027, would cut the CI matrix and stop users on 3.10 from getting stale versions of the bridge extras. Stating SPEC 0 adoption in the README costs nothing.
- **Packaging changes.** The project already has `py.typed`, CI-enforced mypy, a pinned record environment and an SPDX licence. What it lacks for an index release is a `src/` layout (optional), developer tools moved out of the `test` and `release` extras into `[dependency-groups]`, a Trusted Publishing release workflow and VCS-derived versions. Each is a one-time change; after that, a release costs one `git tag` push.
- **Metadata.** Adding `codemeta.json` (generated from `CITATION.cff`, for example with cffconvert or FACILE-RS) plus Zenodo's GitHub integration satisfies FAIR4RS findability and gives the "persistent identifier" the project says it intends to have. Zenodo then archives into Software Heritage automatically.

### Gaps
- The full SoftwareX Guide for Authors (length limit, criteria for software already described in another paper) was not retrievable (HTTP 403).
- I found no official 2026 statement on whether JOSS treats a public mirror of a previously private repository, with history preserved, as satisfying the six-month rule. The 2026 blog post says only that privately developed projects become eligible after six months of public history.
- Specific deprecation-policy templates for 0.x scientific packages were not collected beyond SemVer §4. Qiskit's own deprecation policy was not fetched.

## 3. Interoperability targets in 2026, and which interchange formats matter most for a Pauli-operator and measurement-planning library

### Takeaway
For a Pauli-operator and measurement-planning library, these interfaces matter most in 2026, in order:
1. Qiskit `SparsePauliOp`/`SparseObservable` and the EstimatorV2/SamplerV2 primitive data model: the PUB inputs, the `PrimitiveResult`/`DataBin` outputs, and `BitArray` shot data.
2. OpenFermion `QubitOperator` and the HamLib HDF5 files built on it.
3. OpenQASM 3 for circuits.
4. PennyLane's device API, which needs a TOML capability file for Catalyst.

QIR and the Qiskit C API matter little for a Python-level operator library. FCIDUMP remains the lowest-common-denominator chemistry input, and TREXIO (HDF5) is the richer standard to follow.

### Cited Findings
- **Qiskit primitives data model:**
  - An Estimator PUB is (circuit, observables array, parameter values, optional precision) and a Sampler PUB is (circuit, parameter values, optional shots).
  - Observables can be `Pauli`, `SparsePauliOp`, `PauliList` or `str`, and broadcasting follows NumPy rules.
  - Each `SparsePauliOp` counts as one element of the broadcast shape.
  - Commuting observables in *different* PUBs are measured separately, so observables that should share measurements must go in the same PUB.
  - Observables must be mapped to the transpiled layout with `apply_layout`.
  - Results come as `PrimitiveResult` → `PubResult` → `DataBin`. For the Estimator these hold `evs` and `stds`; for the Sampler they hold one `BitArray` per register, with `get_counts`, slicing, post-selection masks and `expectation_values` for diagonal operators.
  ([IBM primitive I/O](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output))
- **Current versions (2026-11-14):** qiskit 2.6.2 (2026-11-05), qiskit-ibm-runtime 0.52.1 (2026-11-12) and qiskit-aer 0.18.2. ([qiskit PyPI](https://pypi.org/pypi/qiskit/json), [runtime PyPI](https://pypi.org/pypi/qiskit-ibm-runtime/json), [aer PyPI](https://pypi.org/pypi/qiskit-aer/json))
- **Qiskit C API:** the C API for `SparseObservable` arrived in Qiskit 2.0 as experimental. Qiskit 2.2 added standalone transpiler passes and `qk_transpile()` to the C API. Later 2.x releases added `SparseObservable.evolve` and scalar arithmetic. ([Qiskit 2.2 release notes](https://quantum.cloud.ibm.com/docs/api/qiskit/release-notes/2.2), [IBM blog 2.2](https://www.ibm.com/quantum/blog/qiskit-2-2-release-summary), [2.4.0 changelog](https://newreleases.io/project/github/Qiskit/qiskit/release/2.4.0))
- **Executor and the directed execution model (beta):**
  - The Executor takes a `QuantumProgram` built from a template circuit and a "samplex" made with Samplomatic. It gives explicit control of twirling, noise learning and injection, and basis changes, and returns a `QuantumProgramResult`.
  - Version requirements conflict between IBM pages (≥0.47.0, ≥0.48.0 or v0.50.0). The migration guide says there is no simulator support yet, but the local-testing page says local Executor support needs v0.50.0+.
  - samplomatic is at 0.21.0 (2026-08-25).
  ([IBM directed execution](https://quantum.cloud.ibm.com/docs/en/guides/directed-execution-model), [migration guide](https://quantum.cloud.ibm.com/docs/en/guides/migrate-from-sampler-to-executor.md), [local testing](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode), [samplomatic PyPI](https://pypi.org/pypi/samplomatic/json))
- **PennyLane devices (0.45.1):**
  - A device subclasses `pennylane.devices.Device`, and "you only need to override the execute() method". Preprocessing is a `CompilePipeline` with transforms such as `decompose`, `validate_measurements` and `split_non_commuting`.
  - Capabilities are declared in a TOML file (`schema = 3`), covering observables, measurement processes, and conditions such as `"terms-commute"` and `"finiteshots"`. The TOML file is "optional for PennyLane but required for Catalyst integration".
  - Devices register through the `pennylane.plugins` entry-point group.
  - PennyLane 0.45.1 and Catalyst 0.15.0 need Python ≥3.11.
  ([PennyLane plugin guide](https://docs.pennylane.ai/en/stable/development/plugins.html), [pennylane PyPI](https://pypi.org/pypi/pennylane/json), [catalyst PyPI](https://pypi.org/pypi/pennylane-catalyst/json))
- **pytket and Quantinuum:** pytket 2.18.5 (needs ≥3.10), pytket-qiskit 0.79.0 (needs ≥3.11), pytket-quantinuum 0.59.3, and the Quantinuum Nexus client qnexus 0.51.4. ([pytket](https://pypi.org/pypi/pytket/json), [pytket-qiskit](https://pypi.org/pypi/pytket-qiskit/json), [qnexus](https://pypi.org/pypi/qnexus/json))
- **Other frameworks:** cirq-core 1.7.0 (2026-11-11, needs ≥3.11), Stim 1.16.0, and the OpenQASM 3 reference AST/parser `openqasm3` 1.0.2 (2026-11-11). `qiskit-qasm3-import` 0.6.0 (2025-06-10) is Qiskit's OpenQASM 3 importer, and `pyqir` 0.12.8 (2026-10-30) handles QIR. ([cirq-core](https://pypi.org/pypi/cirq-core/json), [openqasm3](https://pypi.org/pypi/openqasm3/json), [qiskit-qasm3-import](https://pypi.org/pypi/qiskit-qasm3-import/json), [pyqir](https://pypi.org/pypi/pyqir/json))
- **What peers depend on:**
  - qibo makes `openqasm3[parser]` a core dependency and offers `qir`, `cudaq`, `qiskit` and `qulacs` extras.
  - quri-parts has extras for braket, cirq, ionq, itensor, openfermion, openqasm, pyscf, qiskit, quantinuum, qulacs, stim, tensornetwork and tket.
  - mitiq has qiskit, braket, pennylane, pyquil and qibo extras.

  ([qibo PyPI](https://pypi.org/pypi/qibo/json), [quri-parts PyPI](https://pypi.org/pypi/quri-parts/json), [mitiq PyPI](https://pypi.org/pypi/mitiq/json))
- **Cross-framework runtimes:** qbraid 0.13.0 ("platform-agnostic quantum runtime framework") and the Unitary Compiler Collection, `ucc` 0.4.12. UCC usage rose from 2% to 10% between the 2024 and 2025 UF surveys. ([qbraid PyPI](https://pypi.org/pypi/qbraid/json), [ucc PyPI](https://pypi.org/pypi/ucc/json), [Quantum Insider](https://thequantuminsider.com/2025/11/05/the-young-and-the-relentless-poll-shows-more-quantum-developers-are-young-independent-and-unpaid/))
- **HamLib format:**
  - Hamiltonians are stored as OpenFermion `QubitOperator` strings in HDF5, compressed in ZIP archives (about ten times smaller than the unpacked HDF5). v1.1 added Qiskit `SparsePauliOp` support.
  - Each file carries the attributes `nqubits`, `terms` and `one_norm`, and each subdataset has a CSV index of instances.
  - Fermionic encodings are JW, parity and BK, with little-endian ordering within each variable.
  - The data are hosted at portal.nersc.gov/cfs/m888/dcamps/hamlib/, and loader snippets ship as `hamlib_snippets.py`.

  ([HamLib arXiv v5](https://arxiv.org/html/2306.13126v5), [Quantum 8, 1559 (2024)](https://quantum-journal.org/papers/q-2024-12-11-1559/))
- **Chemistry formats:**
  - TREXIO is a C library with HDF5 and text back-ends for exchanging wavefunction data and integrals. `trexio-tools` converts to and from GAMESS, Gaussian, PySCF and FCIDUMP.
  - FCIDUMP is described as an older format that stores the minimum needed for post-HF methods.
  - OpenFermion 1.8.1 depends on h5py; its `MolecularData` is HDF5-backed.
  - MolSSI's `qcelemental` 0.51.4 (2026-10-21) provides the QCSchema data models.

  ([TREXIO arXiv](https://arxiv.org/abs/2302.14793), [FHI-aims TREXIO manual](https://fhi-aims.org/uploads/manual/A5/S5.html), [openfermion PyPI](https://pypi.org/pypi/openfermion/json), [qcelemental PyPI](https://pypi.org/pypi/qcelemental/json))
- **mitiq frontends:** Cirq (core), Qiskit, pyQuil, Braket, PennyLane, Qibo and OpenQASM 3. The integration contract is a user-supplied "executor" function that maps a circuit to an expectation value. ([mitiq docs](https://mitiq.readthedocs.io/en/stable/))
- **metriq-gym:** dispatches to providers through qBraid and the Qiskit stack. Its dependencies include `qiskit[qasm3-import]`, qiskit-ibm-runtime, qiskit-aer, `qbraid-core`, pydantic, jsonschema and typer. ([metriq-gym PyPI](https://pypi.org/pypi/metriq-gym/json))

### Inferences
- **Highest-value bridge.** The most useful bridge is a pair of exact converters: clifford_qc measurement plans to Qiskit EstimatorV2 PUBs (one PUB per commuting family, so the primitive measures groups together), and SamplerV2 `BitArray` back into clifford_qc's finite-shot estimators. With that pair, any Qiskit-compatible backend, simulator or IBM QPU can execute a clifford_qc plan unchanged. The existing bridges (Stim, OpenFermion, pytket, PennyLane, PyZX) appear to be operator and circuit converters, not execution-result importers.
- **Operator conversion.** `PauliSum` ↔ `SparsePauliOp`/`SparseObservable` and `PauliSum` ↔ OpenFermion `QubitOperator` round-trips cover HamLib, PauliEngine and PauliArray users. A `load_hamlib(path, key)` reader, an optional extra using h5py, would let clifford_qc report results on community instances (section 5) at low cost.
- **PennyLane.** A full PennyLane *device* is probably not worth it for an operator library; a converter to `qml.PauliSentence`/`qml.Hamiltonian` covers most users. Exposing clifford_qc's grouping and shot allocation as a PennyLane transform would be more distinctive, but it is lower priority.
- **QIR and C API.** QIR and the Qiskit C API are not priorities for a Python NumPy-core library. OpenQASM 3 export already exists. Adding a round-trip import test through `openqasm3` or `qiskit.qasm3` would strengthen the claim.
- **Chemistry input.** FCIDUMP import is adequate. An optional reader for TREXIO, or for OpenFermion `MolecularData` HDF5, would remove the "restricted real FCIDUMP" limitation for users who already have integrals from PySCF or other codes.

### Gaps
- I did not verify whether EstimatorV2 in Qiskit 2.6 or qiskit-ibm-runtime 0.52 accepts `SparseObservable` directly; the primitive I/O page lists only `Pauli`, `SparsePauliOp`, `PauliList` and `str`.
- No current data on QIR adoption by Python application libraries was collected.
- Azure Quantum (azure-quantum 3.13.0, [PyPI](https://pypi.org/pypi/azure-quantum/json)) target lists and pricing were not researched.

## 4. Hardware execution and noise: lowest-effort credible routes, error-mitigation integration, and what small projects published as a "first hardware result"

### Takeaway
The cheapest credible route is to emit Qiskit primitive PUBs. Development and noisy rehearsal then run in Qiskit Runtime local mode (fake backends plus Aer noise models), and the final run goes to IBM's free Open Plan:
- 10 minutes per 28-day rolling window, with Heron r2 `ibm_kingston` available;
- a 2026 opt-in promotion of 180 minutes over 12 months for active users.

Readout mitigation (TREX or M3) and ZNE are the standard first-paper mitigation stack. Amazon Braket per-shot pricing makes a small IQM or Rigetti run cost tens to hundreds of dollars. Peer "first hardware results" are small: a 5-qubit subspace problem replicated across a chip, honestly reported as missing chemical accuracy.

### Cited Findings
- **IBM Open Plan:** free, up to 10 minutes of runtime per 28-day rolling window. The Heron r2 processor `ibm_kingston` (up to 340k CLOPS, median two-qubit error 2.03×10⁻³) is open to all Open Plan users. ([IBM blog: Open Plan updates](https://www.ibm.com/quantum/blog/open-plan-updates), [IBM plans overview](https://quantum.cloud.ibm.com/docs/guides/plans-overview))
- **IBM promotion (16 March 2026):** Open Plan users who use 20 minutes within 12 months can opt in to 180 minutes over the next 12 months. ([Quantum Computing Report](https://quantumcomputingreport.com/ibm-offers-special-promotion-to-open-plan-users/)) The IBM announcement page itself returned 404 when fetched.
- **IBM paid tiers:** Pay-As-You-Go from $96/minute, Flex from $72 and Premium from $48 (per a GCN article of 2026-10-22, not checked against IBM's pricing page). ([search summary citing GCN](https://gcn.com/?p=22676))
- **Qiskit Runtime local mode:**
  - Fake backends from `qiskit_ibm_runtime.fake_provider` carry device snapshots and apply their noise automatically.
  - `AerSimulator.from_backend(real_backend)` builds a noise model from calibration data, but it "covers only some noise sources", so it is likely milder than the real device.
  - The `stabilizer` method handles large Clifford circuits, and `tensor_network` is GPU-only.
  - Laptops rarely go much beyond about 32 qubits for noisy simulation.
  - The client-side Sampler and Estimator (v0.50.0+) apply mitigation options such as `resilience.measure_mitigation` locally.

  ([IBM local testing mode](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode))
- **IBM error mitigation and suppression:**
  - TREX (twirled readout) has "a small overhead".
  - ZNE uses gate folding by default with linear or exponential extrapolation; the default three noise factors give "roughly 3x overhead".
  - PEA is "often the best choice" at utility scale but needs a separate noise-learning job.
  - PEC is unbiased but its cost "scales quadratically with γ, which in turn scales exponentially with the depth".
  - Pauli twirling and dynamical decoupling are available ("could even worsen performance" on dense circuits).
  - Newer options: Qiskit Mitigation with Samplomatic, propagated noise absorption (PNA), operator backpropagation (OBP), Paulice spacetime Pauli checks, and M3.

  ([IBM EM techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques))
- **Related packages (PyPI, 2026-11-14):** `qiskit-mitigation` 0.2.0 (2026-11-05), `qiskit-paulice` 0.3, `mthree` 3.0.0 (2025-03-27) and `qiskit-addon-pna` 0.2.0. ([qiskit-mitigation](https://pypi.org/pypi/qiskit-mitigation/json), [mthree](https://pypi.org/pypi/mthree/json), [qiskit-addon-pna](https://pypi.org/pypi/qiskit-addon-pna/json))
- **Amazon Braket on-demand pricing:**
  - Every task costs $0.30.
  - Per shot: IQM Garnet $0.00145, IQM Emerald $0.0016, Rigetti Cepheus $0.000425, AQT IBEX-Q1 $0.0235, QuEra Aquila $0.01 and IonQ Forte $0.08.
  - IonQ requires at least 2,500 shots per task when using error mitigation.
  - The SV1 simulator costs $0.075/min. The free tier gives 1 hour of simulation per month for the first 12 months.

  ([Braket pricing](https://aws.amazon.com/braket/pricing/))
- **Quantinuum:**
  - Nexus includes the H-Series emulator, so "new users … that don't have access to H-Series hardware can start experimenting right away".
  - The Selene emulator is the "digital sister" to Helios.
  - Hardware runs need Quantinuum credits. At ORNL, credit requests go through a project PI and are expected to be justified with emulator results.
  - Academic Helios access has come through institutional partnerships (SDU, UWA), not public free credits.

  ([Quantinuum Nexus launch](https://www.quantinuum.com/blog/announcing-the-launch-of-quantinuum-nexus-our-all-in-one-quantum-computing-platform), [Quantinuum stack blog](https://www.quantinuum.com/blog/built-for-all-introducing-our-new-software-stack), [ORNL OLCF Quantinuum docs](https://docs.olcf.ornl.gov/quantum/quantum_systems/quantinuum.html), [SDU](https://www.sdu.dk/en/forskning/qm/news/quantinuum-collaboration))
- **Braket usage (UF survey):** Braket is used by 16.4% of respondents and has the largest planned-use growth (22.7%). The IBM Quantum Platform is used by 65.5%. ([UF survey 2025](https://unitaryfoundation.github.io/survey-2025))
- **mitiq:** stable techniques are ZNE, PEC, (variable-noise) CDR, DDD, REM, QSE and LRE. Experimental techniques, not under semantic versioning, are shadows, PEA, virtual distillation, TREX and debiasing. mitiq is GPL-3.0, so it should be an optional extra in an Apache-2.0 project. ([mitiq docs](https://mitiq.readthedocs.io/en/stable/), [mitiq PyPI](https://pypi.org/pypi/mitiq/json))
- **vnCDR shot cost:** vnCDR unifies ZNE and CDR using near-Clifford training circuits at several noise levels, and more noise levels raise the shot cost. ([Bultrini et al., arXiv:2107.13470](https://arxiv.org/pdf/2107.13470)) This matters because clifford_qc already has exact Clifford and Stim machinery suited to building CDR training sets.
- **QURI Parts** documents readout error mitigation and other techniques in its how-to section. ([QURI Parts EM docs](https://quri-parts.qunasys.com/docs/category/error-mitigation))
- **Symmer group's first hardware result** (Weaving et al., arXiv:2312.04392; npj Quantum Information 11, 25 (2025)):
  - The problem was N₂ dissociation with the 15-qubit tapered Hamiltonian projected to a **5-qubit contextual subspace**.
  - It ran as Qiskit Runtime jobs on ibmq_guadalupe, ibm_hanoi, ibm_auckland and ibm_washington, with **circuit tiling** (up to 25 replicas on 125 of 127 qubits) as passive noise averaging.
  - Measurement used 5,000 shots per QWC clique (NetworkX graph colouring).
  - Mitigation: dynamical decoupling, `mthree` readout mitigation, and ZNE with CPhase-based noise amplification and inverse-variance-weighted extrapolation. No symmetry verification was used, because the symmetries collapse under the subspace projection.
  - Errors versus FCI were 47 meV to 1.2 eV, explicitly "not algorithmically accurate" (above the 43 meV threshold).
  - Data and code sit in a separate repository, TimWeaving/N2-CS-VQE.

  ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821530/), [arXiv](https://arxiv.org/pdf/2312.04392))
- **SQD flagship result:** Robledo-Moreno et al., Sci. Adv. 11, eadu9991 (2025), ran up to 77 qubits and 3.5k two-qubit gates on IBM Heron, with diagonalisation on Fugaku, for [2Fe-2S] and [4Fe-4S]. A critique argues SQD fails in some cases (Reinholdt et al., arXiv:2501.07231). ([inspirehep reprint](https://inspirehep.net/files/28aa1e5a91578a31de501a98ef2b5d10), [EurekAlert](https://www.eurekalert.org/news-releases/1088956), [critique](https://arxiv.org/html/2501.07231v2))

### Inferences
- **Hardware path for two maintainers:**
  1. A `clifford_qc.bridges.qiskit_runtime` (optional extra) that turns a measurement plan into EstimatorV2 PUBs and SamplerV2 circuits and maps `BitArray` back into clifford_qc's covariance and uncertainty machinery.
  2. A documented local-mode workflow (fake backend plus Aer, and Stim for Clifford readouts) as the noisy test bed.
  3. A single Open Plan run of a small instance (4–8 qubits) as a "first hardware result".

  The 10 minutes per 28 days plus the 180-minute promotion are enough for one small QWC- versus clifford_qc-grouping comparison. That is an inference: actual runtime depends on circuit count and shots.
- **Braket cost example (arithmetic from the published prices).** A 20-group plan at 4,000 shots per group is 80,000 shots in 20 tasks. That costs about $6 + $34 = $40 on Rigetti Cepheus, about $6 + $116 = $122 on IQM Garnet, and about $6 + $6,400 = $6,406 on IonQ Forte. Superconducting devices on Braket are therefore a viable paid fallback, and trapped-ion hardware is not affordable without credits.
- **Mitigation stack for the first result.** The credible first-paper stack, as in the symmer paper, is readout mitigation (TREX via Runtime options, or M3) plus ZNE, with DD left on. clifford_qc does not need to implement these; it needs to pass them through (Runtime `resilience` options) and to propagate mitigated means and standard errors into its finite-shot inference.
- **Where clifford_qc could add something.** A distinctive contribution would be mitigation that comes from its own structure: symmetry and sector post-selection on sampled bitstrings (it already tracks encodings and sectors), and CDR training sets from Clifford-projected circuits using Stim. mitiq could be the reference implementation for comparison.
- **What to report.** The symmer paper reported honest above-threshold errors, explained the hardware choices and kept data in a separate repository. Reviewers in 2025–2026 accept a modest-accuracy first hardware result if it is honest and fully reproducible, which fits clifford_qc's evidence culture.

### Gaps
- I did not confirm Azure Quantum credit programmes, IonQ or Quantinuum academic credit terms, or AWS Cloud Credit for Research terms; the Braket pricing page does not mention research credits.
- The exact 2026 Qiskit Runtime resilience-level defaults (what levels 0, 1 and 2 switch on) were not on the page fetched.
- I found no published example of a small Pauli-measurement-planning library (as opposed to an algorithm paper) reporting its own grouping strategy's hardware shot savings.

## 5. Benchmarking and evidence standards: suites, datasets, reporting expectations, preregistration and badges

### Takeaway
Community instance sets now exist and are cheap to adopt:
- HamLib (2–1000 qubits; HDF5 with OpenFermion strings; Quantum 2024);
- QB-GSEE (ground-state energy estimation, with hardness metrics and "guidestar" instances, 2025);
- MQT Bench (more than 70,000 circuits);
- metriq/metriq-gym (a schema-validated public results dataset, 2026).

Reviewers increasingly expect three things: runtime and tuning effort reported alongside accuracy, strong classical baselines, and no unjustified cherry-picking. Preregistration is almost absent in quantum computing; it has been proposed for advantage claims, but I found no adopted practice. clifford_qc's preregistered declarations are therefore unusual and could be a selling point if they are made legible.

### Cited Findings
- **HamLib** (Sawaya et al., Quantum 8, 1559, December 2024):
  - Instances run from 2 to 1000 qubits and cover Heisenberg, Fermi-Hubbard, Bose-Hubbard, electronic and vibrational structure, MaxCut, Max-k-SAT, Max-k-Cut, QMaxCut and TSP.
  - The stated goals include "reproducibility and standardization across research studies".
  - The authors argue scaling studies need instances "for every few qubits" because strong simulation stops at 40–50 qubits.

  ([Quantum](https://quantum-journal.org/papers/q-2024-12-11-1559/), [arXiv v5](https://arxiv.org/html/2306.13126v5))
- **Reporting with HamLib:** a user should cite the version (v1.0 or v1.1), the subdataset, the instance identifiers and the encoding. This is derived from the paper's guidance; the paper sets no formal protocol. ([arXiv v5](https://arxiv.org/html/2306.13126v5))
- **QB-GSEE benchmark** (Bellonzi et al., arXiv:2508.10873, August 2025):
  - The code is open at isi-usc-edu/qb-gsee-benchmark.
  - It evaluates SHCI, DMRG and double-factorized QPE resource estimates with hardness metrics, and includes known-solution instances and unsolved "guidestar" instances.
  - It finds many benchmark Hamiltonians are biased toward SHCI-friendly systems and proposes harder, strongly correlated cases.

  ([arXiv](https://arxiv.org/html/2508.10873v1), [Emergent Mind summary](https://www.emergentmind.com/papers/2508.10873), [Quantum Zeitgeist](https://quantumzeitgeist.com/shci-dmrg-and-df-qpe-algorithms-benchmarked-for-ground-state-energy-estimation/))
- **metriq** (arXiv:2603.08680, March 2026):
  - It has three parts: the runner (metriq-gym), a schema-validated dataset (metriq-data) and a web portal.
  - Submissions arrive as PRs to metriq-data and are reviewed.
  - Benchmarks have been run on more than ten QPUs (IBM, Quantinuum, Rigetti, IQM, OriginQ and others), and jobs can be dispatched to a local simulator.
  - The authors warn that the single "Metriq Score" may oversimplify.

  ([arXiv](https://arxiv.org/html/2603.08680v1), [metriq-gym docs](https://unitaryfoundation.github.io/metriq-gym/benchmarks/overview), [GitHub](https://github.com/unitaryfoundation/metriq-gym))
- **MQT Bench** (Quantum 7, 1062, July 2023): more than 70,000 circuits from 2 to 130 qubits at four abstraction levels. The current release is mqt.bench 2.4.0 (2026-11-13). ([Quantum](https://quantum-journal.org/papers/q-2023-07-20-1062/), [PyPI](https://pypi.org/pypi/mqt.bench/json))
- **SupermarQ** (HPCA 2022, pp. 587–603): an application-level, hardware-agnostic suite. The current release is supermarq 0.5.70. ([arXiv:2202.11045](https://arxiv.org/pdf/2202.11045), [Benchmark Zoo](https://quantumbenchmarkzoo.org/content/benchmarking-tools/benchmark-suites/Supermarq), [PyPI](https://pypi.org/pypi/supermarq/json))
- **QASMBench** is described in the MQT Bench paper as a target-independent circuit suite; its venue was not verified. ([MQT Bench arXiv](https://arxiv.org/pdf/2204.13719))
- **Reporting pitfalls:** McGeoch's "How NOT to Fool the Masses…" (arXiv:2411.08860, November 2024) sets four rules:
  1. "Don't claim superior performance without mentioning runtimes."
  2. "Don't report optimized results without mentioning the tuning time…"
  3. "Don't claim faster runtimes for … solvers running on imaginary platforms."
  4. "No cherry-picking (without justification and qualification)."

  ([arXiv](https://arxiv.org/html/2411.08860v1))
- **Fair comparisons:** quantum and classical approaches often get unequal tuning effort, and guidelines exist to correct this (arXiv:2405.07624). End-to-end runtime definitions and a method for choosing strong classical baselines are in arXiv:2510.06337, which found a claimed runtime advantage does not survive fuller benchmarking. Acuaviva et al. argue for a standard benchmarking framework (arXiv:2407.10941, v4 March 2026). ([2405.07624](https://export.arxiv.org/abs/2405.07624), [2510.06337](https://arxiv.org/pdf/2510.06337v3), [2407.10941](https://arxiv.org/html/2407.10941v4))
- **Preregistration:** the 2024 NQI Joint Algorithms Workshop report proposes registered reports for quantum-advantage claims, publishing the target calculation and algorithm before the experiment runs. ([arXiv:2508.13973](https://arxiv.org/pdf/2508.13973))
- **Registered-report uptake:** University of Surrey slides (2022) count only 5 of 281 journals offering registered reports in quantum technologies or physical sciences, naming Scientific Reports, Royal Society Open Science, Nature Communications and others. ([Surrey slides](https://www.surrey.ac.uk/sites/default/files/2022-06/considerations-open-practices-quantum-technologies-research.pdf))
- **ACM badges:** the scheme has "Artifacts Available" (deposit on a persistent repository such as Zenodo), "Artifacts Evaluated" and "Results Reproduced"; results that depend on closed-source artifacts are ineligible for most badges. I found no confirmation that ACM Transactions on Quantum Computing uses them. ([ACM badging](https://www.acm.org/publications/policies/artifact-review-badging/), [ACM reviewer course](https://reviewers.acm.org/training-course/artifact-review-and-badging))
- **SciPost Physics Codebases** makes benchmark tests mandatory for acceptance. ([SciPost](https://scipost.org/SciPostPhysCodeb/about))
- **PauliEngine's benchmark methodology** gives hardware (Apple M4 with 24 GB; Intel i9-11900KF with 32 GB), averaging (10, 100 or 1,000 runs) and sweep definitions (term count at string length 500; string length at 500 terms). It compares against OpenFermion, PauliArray and PennyLane, and states where a competitor wins (PauliArray is faster for small cases). ([arXiv HTML](https://arxiv.org/html/2601.02233v1))

### Inferences
- **Instances.** clifford_qc's committed records cover its own instances. Re-running headline comparisons on named HamLib instances (version, key and encoding cited) would make its measurement-cost and subspace results comparable with other papers at little cost. QB-GSEE "guidestar" instances above 12 qubits are out of exact reach, but its smaller known-solution instances could anchor accuracy claims.
- **Reporting checklist.** For every comparison, report wall-clock time and classical memory, optimiser and tuning budget, shot budget and estimator variance, and the strongest classical baseline (for example selected CI or DMRG through PySCF, block2 or quimb at the same active space). State instance-selection criteria up front. Much of this already exists in the declarations; the improvement is to put a one-table summary in the docs and papers.
- **Preregistration.** The preregistration apparatus is rare in the field and matches what the NQI report recommends. It reaches readers only if it is (a) summarised in a short "Evidence policy" docs page and (b) deposited on Zenodo or OSF with timestamps, so outside readers can verify the "declared before run" claim without reading a 340 KB ledger.
- **Results format.** Submitting a small measurement-cost dataset in a schema-validated format modelled on metriq-data (JSON Schema, PR-reviewed) would help. Alternatively, clifford_qc could align its record schema with metriq's, making results machine-comparable and reusable.

### Gaps
- I found no community benchmark specific to *measurement grouping and shot allocation* (Pauli-measurement planning). Existing suites target circuits, devices or ground-state energies. Whether such a suite exists in 2025–2026 remains unverified.
- No source confirmed any quantum journal (Quantum, PRX Quantum, ACM TQC, npj QI) offering reproducibility badges or registered reports in 2026.
- The full text of QB-GSEE was not read; its data format and submission process are unverified.

## 6. Scaling the simulation side cheaply past ~12–16 qubits, and how peers wire in optional backends

### Takeaway
Peers get past dense-statevector limits with four things:
- particle-number- and spin-sector-restricted vectors, as in ffsim, where the dimension is C(N,Nα)·C(N,Nβ) instead of 2^(2N);
- PySCF FCI and selected-CI kernels, plus external SCI solvers (Dice, SBD, Fulqrum) behind extras;
- matrix-product states through quimb or ITensor-style extras;
- GPU statevectors behind CUDA-specific extras (`cuda12`/`cuda13`, `cudaq`, Lightning-GPU, cuQuantum).

All of them expose heavy backends as optional extras and document them in dedicated how-to pages.

### Cited Findings
- **ffsim:**
  - A state vector is a 1D NumPy array of length C(N,Nα)·C(N,Nβ), against 2^(2N) for a qubit simulator. Ordering "matches PySCF's FCI module (`pyscf.fci`)".
  - Example: 26 orbitals with (5,5) electrons needs 64.48 GiB.
  - Rust kernels multithread through `RAYON_NUM_THREADS`, and GPU support comes through the `cuda12`/`cuda13` extras. Core dependencies include pyscf ≥2.15 and jax.

  ([ffsim explanation](https://qiskit.github.io/ffsim/explanations/state-vectors-and-gates.html), [ffsim docs](https://qiskit.github.io/ffsim/), [PyPI](https://pypi.org/pypi/ffsim/json))
- **qiskit-addon-sqd:** core dependencies are pyscf ≥2.9, jax and qiskit ≥2,<3. Extras `sbd` and `fulqrum` wire in HPC diagonalisation solvers, and the docs have guides for the Dice, Fulqrum and SBD solvers ("Scale workloads with the Dice solver"). A placeholder `qiskit-addon-dice-solver` 0.0.0.dev0 exists on PyPI. ([PyPI](https://pypi.org/pypi/qiskit-addon-sqd/json), [install page nav](https://qiskit.github.io/qiskit-sqd/install.html), [dice-solver PyPI](https://pypi.org/pypi/qiskit-addon-dice-solver/json))
- **symmer** hard-depends on quimb (MPS/tensor networks), numba and qiskit, with no optional extras. ([PyPI](https://pypi.org/pypi/symmer/json))
- **quri-parts** puts tensor-network (`itensor`, `tensornetwork`), `qulacs` and `stim` backends behind extras and ships a Rust core subpackage. ([PyPI](https://pypi.org/pypi/quri-parts/json))
- **qibo** puts `cudaq`, `qulacs` and `qir` behind extras. ([PyPI](https://pypi.org/pypi/qibo/json))
- **Current backend versions (PyPI, 2026-11-14):** quimb 1.16.0, physics-tenpy 1.1.1, block2 0.5.4 (MPO/DMRG for quantum chemistry), pyscf 2.15.0, qulacs 0.6.14, qsimcirq 0.22.1, cuquantum-python 26.11.0, pennylane-lightning 0.45.0 and qiskit-aer 0.18.2. ([quimb](https://pypi.org/pypi/quimb/json), [physics-tenpy](https://pypi.org/pypi/physics-tenpy/json), [block2](https://pypi.org/pypi/block2/json), [pyscf](https://pypi.org/pypi/pyscf/json), [qulacs](https://pypi.org/pypi/qulacs/json), [qsimcirq](https://pypi.org/pypi/qsimcirq/json), [cuquantum-python](https://pypi.org/pypi/cuquantum-python/json), [pennylane-lightning](https://pypi.org/pypi/pennylane-lightning/json))
- **Aer:** the `stabilizer` method handles large Clifford circuits, `tensor_network` is GPU-only, and noisy statevector simulation on laptops tops out around 32 qubits. ([IBM local testing](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode))
- **PauliEngine** keeps Pauli arithmetic in C++ with a binary-symplectic representation and bit-level multiplication and commutation, and scales "roughly quadratically in Hamiltonian size and near-linearly in string length". PauliArray (NumPy) is faster for small cases but memory-bound at large sizes. ([arXiv HTML](https://arxiv.org/html/2601.02233v1))

### Inferences
- **Lowest-effort scaling.** For A-CASE/DA-CASE and QSCI-type studies, the cheapest step is an optional ffsim or PySCF sector backend: evaluate subspace matrix elements and sample determinants in the C(N,Nα)·C(N,Nβ) space, and keep the MV/PauliSum path for validation at ≤12 qubits. clifford_qc already has "sector FCI" and "sector action" code, so the change is to delegate to ffsim or pyscf.fci kernels behind a `[ffsim]` extra. ffsim needs Python ≥3.11, which is another reason to raise the floor.
- **Large qubit counts.** For lattice models at 20–40 qubits, a quimb MPS backend behind an extra (as symmer and quri-parts do) gives approximate reference energies and sampled bitstrings for measurement-planning studies, without touching the NumPy core.
- **Sparse arithmetic.** If sparse Pauli arithmetic becomes the bottleneck, PauliEngine (pip-installable C++ wheel, MIT licence) or Qiskit's Rust-backed `SparsePauliOp`/`SparseObservable` could serve as an optional accelerated backend. That would need a differential test against the MV semantics, which the existing "no test, no bridge" rule already requires.
- **Docs for backends.** Each backend should get a how-to page in the qiskit-addon-sqd style ("Scale workloads with X") that states install, limits and a validation example. That reuses the existing conformance-test culture.

### Gaps
- No quantitative comparison was found of ffsim versus PySCF FCI versus Lightning-GPU for subspace-matrix construction at 16–30 qubits.
- I did not check whether Dice is pip-installable in 2026 (the PyPI `dice` package is unrelated) or how SBD and Fulqrum are distributed.

## 7. Usability patterns: one-function workflows versus primitives, result objects, CLI, logging, errors for missing extras, galleries, and keeping reproducibility material out of newcomers' way

### Takeaway
Successful peers offer a layered API: a high-level function for the common workflow (for example `diagonalize_fermionic_hamiltonian()` in SQD), composable primitives under it, and typed result containers with numpy arrays and metadata (Qiskit's `PrimitiveResult`/`DataBin`). Docs follow the tutorials / how-to / explanations / API split. Research reproduction material goes into separate repositories or datasets with their own DOIs. For a two-person project in 2026 the safest docs stack is Sphinx with an executable-notebook extension. MkDocs-Material entered maintenance mode in November 2025, and its critical-fix support ended on 5 November 2026.

### Cited Findings
- **High-level entry points with incremental breaks:** qiskit-addon-sqd added the high-level `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian()` in 0.11.0, keeping lower-level functions and flagging breaking parameter changes in release notes (0.7.0). ([SQD release notes](https://docs.quantum.ibm.com/api/qiskit-addon-sqd/release-notes))
- **Qiskit primitive results:** results are typed containers. `PrimitiveResult` holds `PubResult` objects; each has `.data` (a `DataBin` with `evs`/`stds` or `BitArray`s) and `.metadata`, and shapes follow the PUB broadcast shape. ([IBM primitive I/O](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output))
- **metriq-gym:** a `typer` CLI with `pydantic` models and `jsonschema` validation. Results become a public dataset through PRs to a separate repository (metriq-data). ([metriq-gym PyPI](https://pypi.org/pypi/metriq-gym/json), [metriq arXiv](https://arxiv.org/html/2603.08680v1))
- **ffsim docs layout:** 2 tutorials, 14 how-to guides, 11 explanations and a per-submodule API reference, plus links to a developer guide, release notes and development-branch docs. ([ffsim docs](https://qiskit.github.io/ffsim/))
- **Executable notebooks in CI:** qiskit-addon-sqd and qiskit-addon-obp ship `nbtest`, `docs`, `doctest` and `notebook-dependencies` extras, so tutorials run in CI. ([SQD PyPI](https://pypi.org/pypi/qiskit-addon-sqd/json), [OBP PyPI](https://pypi.org/pypi/qiskit-addon-obp/json))
- **Separate repositories for tutorials and paper code:** tequila keeps tutorials in `tequila-tutorials`. Symmer's hardware paper keeps data and code in TimWeaving/N2-CS-VQE, not in symmer. HamLib distributes data from a NERSC portal, not inside a Python package. ([tequila-tutorials](https://www.github.com/tequilahub/tequila-tutorials), [N2 paper data statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821530/), [HamLib](https://arxiv.org/html/2306.13126v5))
- **SciPost Codebases** splits a citable "userguide" from the code release, each with its own DOI. ([SciPost](https://scipost.org/SciPostPhysCodeb/about))
- **Material for MkDocs:** maintenance mode was announced on 5 November 2025, with critical bug and security fixes through 5 November 2026. New development moved to Zensical (MIT licence), which reads `mkdocs.yml` natively. Insiders features were made free on 11 November 2025. ([mkdocs-material issue #8523](https://github.com/squidfunk/mkdocs-material/issues/8523), [Material blog](https://squidfunk.github.io/mkdocs-material/blog/2025/11/11/insiders-now-free-for-everyone/), [pydevtools handbook](https://pydevtools.com/handbook/reference/mkdocs-material/))
- **Packaging guide on tooling:** dev tools (pytest, ruff, Sphinx) belong in `[dependency-groups]`, not extras, and CLIs belong in `[project.scripts]`. ([Scientific Python guide](https://learn.scientific-python.org/development/guides/packaging-simple/))
- **Docs requirements at review:** pyOpenSci requires quickstart tutorials and API docs with standard-format docstrings, viewable "without installing the package". JOSS calls an undocumented API "not acceptable". ([pyOpenSci](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html), [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html))
- **Lazy loading:** SPEC 1 (lazy loading of submodules) is endorsed by NumPy, SciPy, scikit-learn and others. It is relevant to import time when many optional bridges exist. ([SPECs](https://scientific-python.org/specs/))
- **Seeding:** SPEC 7 standardises seeding of pseudo-random number generation, relevant to finite-shot reproducibility. ([SPECs](https://scientific-python.org/specs/))

### Inferences
- **A newcomer layer.** clifford_qc's README is dense with contract language: "true Ritz residual norms … validated on the five banks the Phase 15 preflight licensed …". A newcomer layer would put three or four task-oriented functions (build model → plan measurements → estimate energy with uncertainty → export) at the top of a rendered docs site. Phase, record and preregistration detail would move to an "Evidence and reproducibility" section, and `PLAN.md`, `REPRODUCING.md` and `benchmarks/` would be linked from it rather than shown in the main navigation.
- **Moving the reproducibility weight.** Options, in order of effort:
  1. Keep the records in-repo and hide them from docs navigation. The sdist and wheel already exclude them: `MANIFEST.in` prunes `tests/` and ships only the library plus top-level documents, and its own comment notes that `benchmarks/` is about 140 MB.
  2. Move `benchmarks/` record data (149 MB) and paper directories to a companion `clifford_qc-evidence` repository, archived on Zenodo with its own DOI and pinned to library versions.
  3. Publish frozen record bundles as Zenodo datasets referenced by DOI from `REPRODUCING.md`.

  Option 2 mirrors tequila-tutorials and the symmer paper repository, and keeps the library repository light for contributors.
- **Docs stack.** Sphinx + MyST (myst-nb or sphinx-gallery for executable examples) + autodoc/autosummary + numpydoc, with docs deployed to GitHub Pages. This is the stack ffsim, the Qiskit addons and mitiq use, and it avoids building on MkDocs-Material, which is in maintenance mode. Converting the existing `examples/*.py` (13 scripts) into a sphinx-gallery is the cheapest way to get an executable gallery.
- **Result objects and CLI.** Results should be frozen dataclasses with `.to_dict()`/`.to_json()` and a published JSON Schema (the runner-artifact contract may already provide this). A small `clifford-qc` console script (for example `clifford-qc info`, which prints versions and installed extras, and `clifford-qc verify`, which runs the smoke check) helps bug reports. Missing extras should raise `ImportError` messages that name the pip extra, such as `pip install "clifford-qc[stim]"`.

### Gaps
- I did not find survey data quantifying how much executable docs or galleries increase adoption for quantum libraries specifically, beyond the UF survey's "poorly documented" barrier.
- I did not inspect peers' logging or progress-bar conventions; no sources were collected on this.

## 8. Community and sustainability for a two-person project: governance, contribution guides, templates, unitaryHACK, microgrants, and maintenance burden

### Takeaway
Funding and community programmes accessible to a small quantum project in 2026:
- Unitary Foundation microgrants (more than 100 awarded; secondary sources cite $4,000) and the Riverlane/UF Deltakit fund (up to $4,000, but QEC-specific);
- unitaryHACK, held 3–17 June 2026 (more than 1,000 developers, 165 bounties, about $18k paid);
- NumFOCUS Small Development Grants, open only to affiliated or fiscally sponsored projects.

The low-cost essentials are `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and PR templates, a documented support channel and a release automation workflow. JOSS and pyOpenSci review these, and pyOpenSci asks for 1–2 years of maintenance.

### Cited Findings
- **Unitary Foundation grants:** the grants page says more than 100 microgrants have been awarded across 31+ countries, with an "Apply Here" form. ([UF grants](https://unitary.foundation/grants/)) The $4,000 amount comes from secondary listings and a podcast, not the official page. ([search summary](https://www.startupgrantsindia.com/unitary-foundation-microgrant))
- **Deltakit Community Fund:** launched by Riverlane and the Unitary Foundation on 28 July 2026. It offers up to $4,000 for production-grade QEC features in Riverlane's open-source Deltakit, managed through UF's merit frameworks. ([Riverlane press release](https://www.riverlane.com/press-release/riverlane-and-unitary-foundation-launch-deltakit-community-fund), [Quantum Insider](https://thequantuminsider.com/2026/07/28/riverlane-unitary-foundation-deltakit-community-fund/))
- **unitaryHACK 2026:** ran 3–17 June 2026. More than 1,000 developers closed 165 bounties worth nearly $18,000; the 2025 edition listed 172 bounties and more than $19,000. Projects take part by labelling issues as bounties. ([Quantum Insider pre-event](https://thequantuminsider.com/2026/05/30/open-source-quantum-community-prepares-for-sixth-annual-unitaryhack/), [Quantum Zeitgeist on Riverlane](https://quantumzeitgeist.com/riverlane-funds-quantum-features-source/), [unitaryHACK maintainer guide](https://deploy-preview-13--unitaryhack2024.netlify.app/project-guide))
- **NumFOCUS Small Development Grants:** only fiscally sponsored or affiliated projects may apply, with one proposal per cycle and at most 2 grants per calendar year. The "up to $10,000" figure is from a third-party site and unverified. ([NumFOCUS SDG](https://numfocus.org/?p=959), [third-party listing](https://www.startupgrantsindia.com/numfocus-small-development-grants))
- **UF 2025 survey on the developer base:** the "no affiliation" share rose from 7% to 11%. The top open-ended needs were learning and mentorship (25 mentions) and collaboration and community building (22). Nearly 95% reported a positive open-source experience. ([UF survey 2025](https://unitaryfoundation.github.io/survey-2025), [Quantum Insider](https://thequantuminsider.com/2025/11/05/the-young-and-the-relentless-poll-shows-more-quantum-developers-are-young-independent-and-unpaid/))
- **pyOpenSci requirements:** `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`, a 1–2 year maintenance commitment, and a single long-term point of contact. pyOpenSci offers help finding co-maintainers or archiving if maintenance becomes unsustainable. ([pyOpenSci author guide](https://www.pyopensci.org/software-peer-review/how-to/author-guide.html))
- **JOSS:** community guidelines must give clear routes to contribute, report issues and seek support. Multiple contributors are preferred, and single-author projects can show external engagement instead. ([JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html), [JOSS blog 2026](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future))
- **Scientific Python governance and access SPECs:** SPEC 6 ("Keys to the Castle", on managing project credentials and access) and SPEC 8 (securing the release process) are endorsed; SPEC 9 (governance) is a draft. ([SPECs](https://scientific-python.org/specs/))
- **Release authentication:** PyPI Trusted Publishing removes long-lived tokens, which simplifies credential management for a small team. ([PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/))
- **Maintenance cost of dependency breadth:** mitiq 1.1.0 caps Python <3.13 and pins numpy <2.3 and cirq-core <1.7. metriq-gym pins upper bounds on more than a dozen dependencies. Several peers cap Python versions (pauliengine <3.14, qibo <3.15). ([mitiq PyPI](https://pypi.org/pypi/mitiq/json), [metriq-gym PyPI](https://pypi.org/pypi/metriq-gym/json), [pauliengine PyPI](https://pypi.org/pypi/pauliengine/json), [qibo PyPI](https://pypi.org/pypi/qibo/json))

### Inferences
- **Cheapest community setup** (a few hours, then near-zero upkeep): `CONTRIBUTING.md` (how to run the tests and what "no test, no bridge" means), a Contributor Covenant `CODE_OF_CONDUCT.md`, YAML issue forms for bugs (requiring `clifford-qc info` output) and bridge requests, a PR template, GitHub Discussions as the support channel, and a short `GOVERNANCE.md` stating that two maintainers decide by consensus.
- **Ongoing costs, roughly ranked:**
  1. Five optional bridges plus CI on five Python versions. Upstream releases (PennyLane quarterly, Qiskit 2.x minors, pytket monthly) will break bridges; the "no test, no bridge" rule turns this into CI failures rather than silent breakage, but someone must triage them.
  2. A docs site with executed notebooks, which needs re-execution on each release.
  3. PyPI releases, close to zero with Trusted Publishing and a tag-triggered workflow.
  4. conda-forge, a moderate burden: a feedstock and bot PRs. Most peers in this tier skip it.
  5. A hardware bridge, which carries the highest drift risk because qiskit-ibm-runtime has released 16 times in the last 12 months.
- **Reducing load.** Pin bridges to declared tested ranges, run a weekly "latest upstream" job (the weekly cron already exists), and drop bridges that no one uses (for example PyZX, if usage evidence is absent).
- **unitaryHACK.** Participation (June each year) is cheap: label 3–5 well-scoped issues, such as a HamLib loader, a sphinx-gallery conversion or a `SparseObservable` converter. It brings external contributors, which JOSS's research-impact and community review values. Mentoring time is the main cost.
- **Funding fit.** A UF microgrant application fits "docs site plus HamLib benchmark plus first hardware run" well. NumFOCUS SDGs would first need NumFOCUS affiliation, which is probably not worth pursuing at this stage.

### Gaps
- Official current UF microgrant terms (amount, reporting) were not visible on the official page fetched; the $4,000 figure is from secondary sources.
- I found no published data on the actual maintainer hours spent by comparable two-person quantum projects.
- Contributor counts for symmer, tequila, PauliArray and PauliEngine could not be obtained because the GitHub API was blocked.
