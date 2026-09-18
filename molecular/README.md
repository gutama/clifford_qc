# Molecular simulations

This folder owns the seven-molecule STO-3G comparison suite: geometries, PySCF
preparation, simulations, historical results, and reports. Installed APIs remain
in `clifford_qc.prepared` and `clifford_qc.pipeline`; see [PIPELINE.md](PIPELINE.md)
for independent FCIDUMP preparation, solving, and optional reference validation.

## Run a comparison

From a source checkout on Linux (RSS sampling uses `/proc/self/statm`):

```bash
python -m pip install -e '.[molecular]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m molecular.run --molecules lih --max-additions 2 --complete-sd off
```

This is a bounded smoke run, not a chemical-accuracy claim. The default output
is `molecular/runs/`, ignored by Git. Use `--output-directory` to keep different
configurations separate. `python /path/to/repo/molecular/run.py` also works
outside the repository root. `--help` does not import PySCF.

The suite computes RHF, CISD, CCSD, sector FCI, a spin-checked determinant CISD
cross-check, and adaptive A-CASE with observables. Its default stopping rule uses
FCI and records that oracle use. For solves without an FCI calculation, use the
separate `clifford_qc.pipeline solve` command in [PIPELINE.md](PIPELINE.md).

| Option | Default and effect |
|---|---|
| `--molecules` | All seven catalog entries; accepts one or more keys |
| `--max-additions` | Catalog budget, otherwise candidate count; explicit values override the catalog. `0` keeps only the identity |
| `--max-candidates` | `0` uses the full SD pool; a positive value truncates generator order |
| `--target-error-mha` | `1.5936`; `0` disables oracle stopping, but the comparison still computes FCI |
| `--complete-sd` | `auto`: on for equilibrium, off for stretched molecules; `on`/`off` override |
| `--reference-method` | `dense`; also `auto`, `eigsh`, `lanczos` |
| `--storage` | `object`; `packed` changes adaptive coefficient storage |
| `--policy` | `retain_all`; `stream_recompute` bounds retained adaptive coefficient rows |
| `--frontier-pairs` | `32`; positive streaming frontier capacity |
| `--resume` | Skip only matching, verified results |
| `--refresh-chemistry` | Rebuild chemistry and rerun even with `--resume` |

The complete-SD arm retains its existing object-storage solver independently of
adaptive storage options. It can take minutes and gigabytes. Packing/streaming
do not guarantee lower total RSS or runtime.

## Reuse and recovery

The chemistry cache under `cache/chemistry/` covers geometry, basis, charge,
spin, units, iteration limit, chemistry source, dependency versions, and thread
settings. Solver budgets/storage do not invalidate it. Integral bytes and
metadata are verified before reuse. Unconverged RHF aborts the molecule;
correlated convergence flags remain visible in the result. FCIDUMP mapping is
cached separately under `cache/prepared/`.

Each molecule runs in a fresh process; the adaptive result is released before
complete SD starts. Determinant CISD restricts sector operators directly to its
selected block instead of building a full `2^n` sparse matrix.

Each completed record is written atomically before the summary. Restarting
recovers the summary from per-molecule files, including records written just
before an interruption. Corrupt records fail with an error. An exclusive output
lock lasts while the driver or its worker is alive. Resume verifies checksums,
configuration, source code, dependency/thread environment, and FCIDUMP contents.
Historical records without provenance are readable but never resumable.

Repeat this command to resume an interrupted run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m molecular.run --molecules lih beh2 --max-additions 2 \
  --complete-sd off --storage packed --policy stream_recompute \
  --frontier-pairs 4 --resume
```

## Files and historical evidence

| Path | Purpose |
|---|---|
| `catalog.py` | Geometries and historical budgets |
| `chemistry.py` | Cached PySCF baselines and FCIDUMP export |
| `simulation.py` | One molecule's numerical calculations |
| `run.py` | CLI, isolated workers, resume checks, and persistence |
| [results/](results/) | Original FCIDUMPs and numerical records |
| [report.md](report.md), [report.pdf](report.pdf) | Generated comparison and limitations |
| [plan.md](plan.md) | Scientific scope and experiment plan |

Historical numerical records and timings are unchanged by the move. They do
not measure the refactored runner. Regenerate reports without rerunning chemistry:

```bash
python benchmarks/check_molecular.py
python benchmarks/summarize_molecular.py
```

To replace the committed results deliberately, run
`python -m molecular.run --output-directory molecular/results`, then regenerate
and review reports. A full seven-molecule run is expensive. The old
`--max-subspace` name becomes `--max-additions`: it counts directions added
after the identity, not total basis size.

Regression checks:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_molecular_workflow.py tests/test_pipeline_refactoring.py -q
```
