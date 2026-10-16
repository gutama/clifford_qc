"""Instrument the existing mean-root A-CASE implementation; no new selector."""
from __future__ import annotations

from dataclasses import asdict

from clifford_qc.algorithms.exchange import exchange_pools
from clifford_qc.algorithms.more_adapt import matrix_record
from clifford_qc.backends import ExactMVBackend
from clifford_qc.subspace import (
    Generator, MatrixElementBank, configuration_generator, run_acase,
)


class TraceBank(MatrixElementBank):
    """Observe native pair creation and solved pencils without extra solves."""
    def __init__(self, *args, **kwargs):
        self.prefixes = []
        self.pair_ledger = []
        self.last_matrices = None
        super().__init__(*args, **kwargs)

    def _build_pair(self, i, j, *, evaluate=False):
        super()._build_pair(i, j, evaluate=evaluate)
        self.pair_ledger.append([len(self.prefixes), i, j,
                                 self._row_nnz('overlap', (i, j)),
                                 self._row_nnz('element', (i, j))])

    def matrices(self, indices=None):
        s, h = super().matrices(indices)
        self.last_matrices = (s, h)
        return s, h

    def solve(self, indices=None, **kwargs):
        result = super().solve(indices, **kwargs)
        s, h = self.last_matrices
        self.prefixes.append({'indices': list(result.indices),
                              'S': matrix_record(s), 'H': matrix_record(h),
                              'coefficients': matrix_record(result.coefficients),
                              'energies': list(result.energies),
                              'effective_rank': result.effective_rank,
                              'condition_number': result.condition_number,
                              'resources': bank_resources(self, result.indices)})
        return result


def generators(model, reference_bits):
    """Same seed span and QE family; candidate T_i P_j on the single HF ket."""
    initial = []
    for j, word in enumerate(reference_bits):
        occupied = [q for q in range(model.n) if word & (1 << (model.n - 1 - q))]
        initial.append(configuration_generator(model.reference, occupied, label=f'ref:{j}'))
    qe, _ = exchange_pools(model.n)
    candidates = [Generator(f'{op.label}@ref:{j}', op.generator * seed.mv)
                  for op in qe for j, seed in enumerate(initial)]
    return initial, candidates


def bank_resources(bank, indices):
    observed = bank.resources(indices)
    names = ('pairs_built', 'coefficient_occurrences', 'cached_operator_bytes',
             'resident_operator_rows', 'word_universe', 'resident_word_universe',
             'retained_block_pairs', 'selection_pairs', 'operator_products',
             'registered_generators', 'basis_size')
    result = {name: observed[name] for name in names}
    words = [code for code in bank.word_set() if code]
    result['independent_word_readout_settings'] = len(words)
    result['local_readout_cnot_bound'] = 0
    result['local_readout_clifford_gate_bound'] = sum(
        sum(1 if (code >> (2 * j)) & 3 == 1 else
            2 if (code >> (2 * j)) & 3 == 2 else 0 for j in range(bank.n)) for code in words)
    return result


def run_acase_mean(model, reference_bits, options):
    initial, candidates = generators(model, reference_bits)
    rho = ExactMVBackend().state(model.reference, ())
    bank = TraceBank(rho, model.hamiltonian)
    roots = len(reference_bits)
    run = run_acase(rho, model.hamiltonian, candidates, initial=initial, bank=bank,
                    roots=roots, aggregation='mean', **options)
    rows = []
    for record, pencil in zip(run.records, bank.prefixes[1:]):
        row = asdict(record)
        row.pop('step_seconds')
        row['iteration'] = row.pop('step')
        row['objective'] = row.pop('energy')
        row['root_energies'] = list(row['root_energies'])
        row['per_root_lowering'] = list(row['per_root_lowering'])
        row['pencil'] = pencil
        rows.append(row)
    retained = set(run.indices)
    total_candidates = len(bank) - len(initial)
    terminal = (0 if run.stopped_reason in ('basis budget reached', 'candidate pool exhausted')
                else total_candidates - len(run.records))
    resources = bank_resources(bank, run.indices)
    resources.update({'candidate_score_forms': sum(row.candidates_scored for row in run.records) + terminal,
                      'terminal_candidate_score_forms': terminal,
                      'root_score_forms': roots * (sum(row.candidates_scored for row in run.records) + terminal),
                      'root_readout_eigensolves': len(bank.prefixes),
                      'overlap_eigendecompositions': len(bank.prefixes),
                      'projected_h_eigendecompositions': len(bank.prefixes),
                      'optimizer_energy_gradient_evaluations': 0,
                      'parameter_derivatives': 0, 'nonlinear_parameters': 0,
                      'reference_preparation_x_gates_per_measurement': model.metadata['n_electrons'],
                      'retained_ritz_coefficient_entries': len(run.indices) * roots,
                      'retained_generator_support': sum(bank.generator(i).support() for i in retained)})
    return {'arm': 'acase_mean', 'reference_bits': list(reference_bits), 'roots': roots,
            'initial_root_energies': bank.prefixes[0]['energies'][:roots],
            'initial_pencil': bank.prefixes[0], 'root_energies': list(run.root_energies),
            'objective': run.energy, 'trajectory': rows,
            'stopped_reason': run.stopped_reason, 'resources': resources,
            'generators': [{'label': bank.generator(i).label,
                            'coefficients': [[code, float(c.real).hex(), float(c.imag).hex()]
                                             for code, c in sorted(bank.generator(i).mv.terms.items())]}
                           for i in range(len(bank))],
            'candidate_indices': [i for i in range(len(initial), len(bank))],
            'pair_ledger': bank.pair_ledger, 'quantum_shots': None}
