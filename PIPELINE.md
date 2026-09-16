# Reusable molecular calculations

Use this pipeline when running several A-CASE configurations on the same
molecular input. It caches model preparation, runs each solve independently,
and makes classical reference validation optional. Each command can run in its
own process. The historical benchmark producers remain the entry points for
reproducing their recorded experiments.

The CLI currently accepts restricted real FCIDUMP inputs and uses exact A-CASE
with determinant-excitation candidates. Other solvers and finite-shot methods
use the Python interfaces described in the [README](README.md).

## Run a small example

Install the core package from a source checkout and run these Bash commands
from the repository root. The included H4 input requires no chemistry extra:

```bash
python -m pip install -e .
mkdir -p /tmp/clifford-demo
python -m clifford_qc.pipeline prepare benchmarks/data/h4_sto3g_r0.9.FCIDUMP \
  --cache-directory /tmp/clifford-demo/cache > /tmp/clifford-demo/preparation.json
prepared_path=$(python -c 'import json; print(json.load(open("/tmp/clifford-demo/preparation.json"))["prepared"])')
python -m clifford_qc.pipeline solve "$prepared_path" \
  --output /tmp/clifford-demo/solve.json --max-additions 2
python -m clifford_qc.pipeline validate "$prepared_path" \
  --output /tmp/clifford-demo/validation.json --method dense --roots 1
```

The preparation response includes the artifact path, fingerprint, and cache-hit
status. Repeating the preparation command with unchanged inputs reuses the cache.
The two-addition solve is a small demonstration, not an accuracy target.
`--max-additions` counts additions beyond the identity direction;
`--max-rank` controls candidate excitation rank, not the number of eigenstates.
`--roots` belongs to reference validation.

Dense validation is appropriate for this small fixture. For larger sectors,
choose a matrix-free method such as `lanczos`, or install the `research` extra
for `eigsh`; consult `python -m clifford_qc.pipeline validate --help` for choices.
Validation can still be expensive and is not needed for every solve.

To try packed streaming storage on the same input:

```bash
python -m clifford_qc.pipeline solve "$prepared_path" \
  --output /tmp/clifford-demo/streaming.json --max-additions 2 \
  --storage packed --policy stream_recompute --frontier-pairs 4
```

## What is reused and recorded

Preparation does no FCI solve. Its immutable JSON includes ordered Hamiltonian
terms, reference program, model metadata, and a digest. The cache key covers
FCIDUMP bytes (including the orbital/active-space choice), sector overrides,
integral tolerance, model name, encoding, package source and NumPy version.
The implementation fingerprint is memoized per process; restart Python after
source changes (or explicitly clear `implementation_fingerprint.cache_clear()`
in development). Writes are atomic; a cache hit validates the digest and requested preparation
before skipping FCIDUMP parsing and mapping. The decoded payload is validated
once per prepared object; each consumer receives fresh model and metadata objects.
`prepare_fcidump` exposes sector/name overrides in Python.

Solve defaults to object storage with `retain_all`, preserving the existing
numerical path. It records candidate and solver configuration, preparation and
implementation fingerprints, energy history, resource scope, and wall time.
It does not calculate an exact ground-state oracle. Exact projected arithmetic
does not certify convergence to the full ground state. Validation is a separate,
optional sector reference calculation and reports eigenpair residuals; its cost
is not included in the solver record.

## Choosing storage and lifetime policies

Storage (`object` or `packed`) and lifetime (`retain_all` or
`stream_recompute`) are independent choices. Start with the default object
retain-all path. Try packing when coefficient payload is a bottleneck, and
streaming when keeping previously scored rows resident is too costly. Measure
both runtime and process memory on the actual workload; neither option is a
universal improvement.

| Policy | Persistent S/H coefficient rows | Additional retained state |
|---|---|---|
| `retain_all` | Every constructed pair | Scalars, supports and generator products |
| `stream_recompute` | Selected block plus at most `frontier_pairs` other pairs | Scalars, compact support history, word history/table and generator products |

For selected basis size `M` and frontier limit `F`, streaming bounds persistent
S/H rows by `2 * (M * (M + 1) / 2 + F)`. Two product rows can additionally exist
during construction. Projected observables are outside this bound. Support,
scalar and word histories grow cumulatively; this is **not a total-process RSS
bound**. Resource output separates resident/cumulative coefficients, physical
builds, recomputations, support payload, shallow containers and word-table
capacity. These components are scoped diagnostics, not an additive RSS estimate.
Resident-word reference counts are updated on row construction/eviction, and
support-history byte totals are accumulated once at first construction. Selected
word support is updated when the retained block changes. Normal resource queries
do not rescan coefficient occurrences. The extra word-index/set containers are
reported as shallow bytes; they add O(distinct words) bookkeeping. Arbitrary
nonretained subset queries still derive their selected word support on demand.
`retained_block_pairs + selection_pairs == pairs_built` partitions logical
history; `resident_retained_block_pairs` and `resident_selection_pairs` separately
partition live S/H pairs, including measurement-only sessions.

Object rows release dictionary references on eviction. Packed rows use independent
segments that release their NumPy buffer capacity when removed. Selection keeps
canonical arithmetic order and historical supports, including the word-cost
objective. A selected row evicted during scoring is rebuilt when retained.
Recomputation can cost time, so streaming stays opt-in. Disk-backed storage is
not implemented, and a full production molecular comparison of the new policies
has not been established.

Scoring reads row supports directly, and storage measurement never reconstructs
packed multivectors. Newly built exact pairs are contracted before packing.
`SharedMeasurement(bank, coefficient_storage="packed")` compiles independent,
ordered functional snapshots without evaluating exact matrix entries. Snapshots
remain usable after bank eviction. Reuse a session, or supply its `groups` to
another session for the same plan; assigned and shot-pooled covariance semantics
are unchanged. Functional arithmetic may create ordinary dictionaries, and
snapshot compilation has temporary allocations; it is not an out-of-core path.

## Python interfaces

`prepare_fcidump` returns a `PreparedProblem`, its artifact path, and a cache-hit
flag. `PreparedProblem.model()` returns a fresh model for each consumer.
`pipeline.solve_prepared` returns the adaptive result and its serializable run
record; `pipeline.validate_prepared` returns reference energies and residuals.
Python preparation also supports sector and name overrides. These are alpha
interfaces; identify the source revision when building downstream workflows.

## Verification and performance scope

`tests/test_pipeline_refactoring.py` checks matrix and selection equivalence,
word-cost scoring, multiple roots, eviction/rebuild, capacity release, optional
validation, cache integrity, and sampled reconstruction/covariance. Run:

```bash
python -m pytest tests/test_pipeline_refactoring.py tests/test_packed_bank.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/profile_pipeline.py --storage packed --policy stream_recompute
```

Run each profile arm in a fresh process. The profile reports numerical hashes,
materialization counts, solve time, process peak RSS and resident rows. Copy it
to an earlier checkout to measure a matched baseline. TFIM timings do not predict
the production molecular speedup; the historical stretched-H2O timings have not
been replaced. Historical `T_coeff/W_selected` ratios cannot establish a storage
saving based on `T_coeff/W_resident`, so the packed-storage verdict leaves those
historical banks ungraded while preserving their measured numbers.

The initial implementation at `eb5077e` has a
[three-process-per-arm TFIM(6) smoke profile](benchmarks/profile_results/pipeline_refactoring.json)
against main `64b06bc` measured median packed solve time `117.1 -> 54.2 ms`
(2.16x speedup) and packed materializations `3640 -> 0`. Streaming measured
`59.9 ms`, with persistent S/H rows `558 -> 50` and coefficients `6456 -> 369`.
All five arms returned identical trajectory/matrix hashes. Object retain-all
remained fastest at `27.0 ms`; these small-system measurements support keeping
the storage options explicit. They do not establish a production molecular
speedup or an RSS reduction.
