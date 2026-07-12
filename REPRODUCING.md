# Reproducing the benchmark artifacts

Every file under `benchmarks/reference_results/` regenerates from a clean
environment with the commands below. All experiments are seeded; JSONL
rows should match up to floating-point noise in wall-clock fields, and the
summary tables should match exactly.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[test,research]      # numpy core + scipy optimizer
pip install -e .[stim]               # stabilizer backend / Phase 4
pip install -e .[chemistry]          # openfermion + pyscf / Phase 5
pytest                               # 230+ tests should pass
```

The core package imports with numpy alone; without SciPy the optimizer
falls back to pure-Python Adam (numerically equivalent results at looser
tolerance — the committed artifacts were produced with SciPy's L-BFGS-B).

## Spin-model matrices (Phase 3)

```bash
# 30-seed development matrix (TFIM n=4 + random-field Ising n=4, 7 methods)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_small.json \
    --out benchmarks/reference_results/spin_small.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_small.jsonl \
    --csv benchmarks/reference_results/spin_small_summary.csv

# 100-seed headline matrix (TFIM h-sweep, periodic TFIM, random Ising; n=4)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_headline_n4.json \
    --out benchmarks/reference_results/spin_headline_n4.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_headline_n4.jsonl \
    --csv benchmarks/reference_results/spin_headline_n4_summary.csv

# 20-seed exploratory matrix at n=6 (TFIM, random Ising, XXZ)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_n6.json \
    --out benchmarks/reference_results/spin_n6.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_n6.jsonl \
    --csv benchmarks/reference_results/spin_n6_summary.csv
```

Long sweeps shard across k workers: run k processes with
`--shard 0/k … --shard k-1/k`, each with its own `--out`, and concatenate
the JSONL files (the committed 100-seed matrix used 4 shards; row order
differs between shardings but content does not). Approximate costs: one
noisy run is ~3.5 s at n=4 and ~5 min at n=6 on one laptop-class core;
n ≥ 8 sweeps need dedicated hardware.

## Stabilizer seeding go/no-go (Phase 4, `stim` extra)

```bash
python benchmarks/run_stabilizer_seeding.py \
    --out benchmarks/reference_results/stabilizer_seeding.jsonl
```

## Chemistry (Phase 5, `chemistry` extra)

```bash
python benchmarks/run_chemistry.py --seeds 3 \
    --out benchmarks/reference_results/chemistry.jsonl
```

PySCF computes SCF/FCI on the fly (no cached integrals); tiny numerical
differences in the last decimals of pyscf energies across platforms do not
change selection decisions at the committed seeds.

## Demos (not committed as artifacts)

```bash
python benchmarks/demo_shared_word_selection.py --seeds 5
```

## Predeclared experiment parameters

Seeds are `range(n_seeds)` per (model, method); model disorder seeds equal
the run seed. Selection uses delta = 0.05, near-optimality tolerance 0.05,
ADAPT gradient threshold 1e-6 (1e-5/1e-6 chemistry), shot escalation
base 256 doubling to 64x, variance-proportional round budget 4096
(growth 2, 7 rounds), operator budgets 8 (n=4/6 spin), 12 (chemistry and
seeding), optimizer L-BFGS-B (gtol 1e-8; maxiter 150 for the n=6 matrix
and 200 for chemistry, unlimited-default elsewhere). Chemical accuracy is
1.6e-3 Ha against the active-space FCI energy. Resource metrics follow
RESEARCH_PLAN.md section 7 (shots, circuits, unique words, operators,
optimizer evaluations, peak Pauli support).
