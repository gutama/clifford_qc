"""Generate observation-driven reports for the original and revised pilots."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.check_orthogonal_active_space import decode, independent_spin_squared
from benchmarks.metadata_io import read_json

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'benchmarks/reference_results/orthogonal_active_space.json'
REPORT = ROOT / 'benchmarks/results/orthogonal_active_space.md'
V2_RECORD = ROOT / 'benchmarks/reference_results/orthogonal_active_space_v2.json'
V2_REPORT = ROOT / 'benchmarks/results/orthogonal_active_space_v2.md'
ARMS = ('orthogonal', 'same_span_mixed', 'dimension_matched_determinants', 'support_matched_determinants')
NOISE_ARMS = ('orthogonal_known_overlap', 'mixed_known_overlap', 'mixed_noisy_overlap')


def serialized_bases(case, prefix):
    """Reconstruct the recorded support, never rethreshold independent sums."""
    dimension = len(case['sector_words'])
    classical = np.eye(dimension, dtype=complex)[:, case['classical_indices']]
    by_pattern = {row['pattern']: row for row in case['packet_ranking']}
    extra = []
    for pattern in prefix['patterns']:
        row = by_pattern[pattern]
        column = np.zeros(dimension, complex)
        column[row['indices']] = decode(row['coefficients'])
        extra.append(column)
    basis = np.column_stack([*classical.T, *extra])
    transform = np.eye(basis.shape[1], dtype=complex)
    transform[:4, 4:] = case.get('_mixing', 4.) * decode(case['classical_coefficients'])[:, None]
    simple = case['classical_indices'] + case['determinant_ranking'][:len(extra)]
    support = np.flatnonzero(np.any(basis != 0, axis=1))
    return dict(zip(ARMS, (basis, basis @ transform, np.eye(dimension)[:, simple], np.eye(dimension)[:, support])))


def spin_readout(case, prefix, mixing):
    s2 = independent_spin_squared(case['sector_words'], case['n_qubits'])
    bases = serialized_bases(dict(case, _mixing=mixing), prefix)
    result = {}
    for name, basis in bases.items():
        state = basis @ decode(prefix['arms'][name]['ground_coefficients'])
        state /= np.linalg.norm(state)
        result[name] = float(np.vdot(state, s2 @ state).real)
    return result


def render(record):
    config, cases = record['config'], record['cases']
    revised = record['schema'].endswith('.v2')
    title = 'Revised singlet' if revised else 'Original'
    target = config['accuracy_target_hartree'] * 1000
    rows = [f'# {title} orthogonal active-space pilot', '',
            f"Exact classical simulation on {len(cases)} declared molecular inputs.",
            'Errors are against the independently checked full-sector ground energy.',
            'The reference is the complete four-determinant CAS(2e,2o), with other occupations frozen at HF.', '',
            '| Input | Classical error (mHa) | Packet error (mHa) | Equal-dimension determinant error (mHa) | Support-containing determinant error (mHa) | Packet gain over equal-dimension (mHa) | Corrections |',
            '|---|---:|---:|---:|---:|---:|---:|']
    reached, initial, caps = [], [], []
    monotonic, containment, coordinate_errors, gains = [], [], [], []
    for case in cases:
        trajectory = case['trajectory']
        first, last = trajectory[0]['arms']['orthogonal'], trajectory[-1]
        a = last['arms']
        gain = (a['dimension_matched_determinants']['energy'] - a['orthogonal']['energy']) * 1000
        gains.append(gain)
        rows.append(f"| {case['id']} | {first['error']*1000:.6f} | {a['orthogonal']['error']*1000:.6f}"
                    f" | {a['dimension_matched_determinants']['error']*1000:.6f}"
                    f" | {a['support_matched_determinants']['error']*1000:.6f} | {gain:+.6f}"
                    f" | {last['retained_corrections']} / {last['requested_corrections']} |")
        if abs(a['orthogonal']['error']) <= config['accuracy_target_hartree']:
            reached.append(case['id'])
        if abs(first['error']) <= config['accuracy_target_hartree']:
            initial.append(case['id'])
        if last['retained_corrections'] < last['requested_corrections']:
            caps.append(f"{case['id']}: {len(case['packet_ranking'])} candidates, retained {last['retained_corrections']}")
        energies = [p['arms']['orthogonal']['energy'] for p in trajectory]
        monotonic.append(all(b <= a + 1e-9 for a, b in zip(energies, energies[1:])))
        containment.append(all(p['arms']['support_matched_determinants']['energy'] <= p['arms']['orthogonal']['energy'] + 1e-9 for p in trajectory))
        coordinate_errors.extend(abs(p['arms']['orthogonal']['energy'] - p['arms']['same_span_mixed']['energy']) for p in trajectory)
    rows += ['', f"Final target ({target:g} mHa): {len(reached)}/{len(cases)} inputs ({', '.join(reached) or 'none'}).",
             f"Already within target at the classical reference: {', '.join(initial) or 'none'}.",
             f"Capped requests: {'; '.join(caps) or 'none'}.",
             f"Nested energy monotonicity holds on {sum(monotonic)}/{len(cases)} inputs (1e-9 Ha tolerance).",
             f"Support containment energy bound holds on {sum(containment)}/{len(cases)} inputs.",
             f"Maximum same-span energy difference: {max(coordinate_errors, default=0):.3g} Ha.",
             f"Packet gains over equal-dimension determinants range from {min(gains, default=0):+.6f} to {max(gains, default=0):+.6f} mHa; negative values favor determinants.",
             'Column counts alone do not equate amplitude support, symmetry, selection work or computational cost.', '',
             '## Spin and measured overlap', '',
             '| Input | Packet ⟨S²⟩ | Mixed ⟨S²⟩ | Equal-dimension determinant ⟨S²⟩ | Support-containing determinant ⟨S²⟩ | Packet measured Gram defect |',
             '|---|---:|---:|---:|---:|---:|']
    for case in cases:
        prefix = case['trajectory'][-1]
        spins = spin_readout(case, prefix, config['same_span_mixing'])
        basis = serialized_bases(dict(case, _mixing=config['same_span_mixing']), prefix)['orthogonal']
        defect = np.max(abs(basis.conj().T @ basis - np.eye(basis.shape[1])))
        rows.append(f"| {case['id']} | " + ' | '.join(f'{spins[a]:.6g}' for a in ARMS) + f' | {defect:.3g} |')
    if revised:
        rows += ['', 'The revised packet arms enforce singlet S² action norms at every prefix; individual-determinant controls are not constrained to be singlets.',
                 f"Packet norm cutoff: max({config['packet_zero_tolerance']:g}, {config['packet_relative_tolerance']:g} × full external preconditioned residual norm), applied before normalization and after singlet projection.",
                 'The producer records measured overlaps and spin diagnostics for every arm.']
    else:
        rows += ['', 'These spin and Gram diagnostics are retrospective calculations from the serialized states, not new original observation fields.',
                 'The original external spin-orbital partition can split spin partners; its Ritz states are not constrained to be singlets.',
                 'The original absolute 1e-12 packet cutoff can normalize round-off-only groups. These limitations remain part of the preserved v1 pilot.',
                 'Original orthogonal-arm overlap fields assumed identity; the table above measures the Gram defect from serialized vectors.',
                 'The original complement-spectrum scalar is permutation round-off, not independent evidence of partition correctness.']
    rows += ['', '## Illustrative centered matrix-noise diagnostic', '',
             'RMSE is conditional on a successful solve and measured around the noiseless retained-span energy.',
             'Full-sector variational violations count successful estimates below the exact ground energy by more than 1e-9 Ha.',
             'Unconditional target fractions include failed solves in the denominator.', '',
             '| Input | H noise (Ha) | S noise | Estimator | RMSE (mHa) | Failures / replicas | Variational violations / successes | Unconditional target fraction |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    differences = []
    for case in cases:
        for block in case['noise']['scales']:
            scale = config['noise']['scales'][block['scale_index']]
            for name in NOISE_ARMS:
                summary = block['summary'][name]
                rmse = 'unresolved' if summary['conditional_rmse'] is None else f"{summary['conditional_rmse']*1000:.6f}"
                rows.append(f"| {case['id']} | {scale['hamiltonian_hartree']:g} | {scale['overlap_dimensionless']:g} | {name}"
                            f" | {rmse} | {summary['failures']} / {summary['replicas']}"
                            f" | {summary['variational_violations']} / {summary['successes']}"
                            f" | {summary['unconditional_target_fraction']:.6f} |")
            differences.extend(abs(r['orthogonal_known_overlap']['error'] - r['mixed_known_overlap']['error']) for r in block['replicas'])
    rows += ['', f"Maximum replica difference between exact-overlap coordinate representations: {max(differences, default=0):.3g} Ha.",
             'Gaussian matrix perturbations have no conversion here to Pauli shots, hardware noise or confidence certificates.',
             'A small estimation RMSE does not repair retained-space energy bias or guarantee a variational estimate.', '',
             '## Scope and provenance', '',
             'Explicit classical sector vectors, full residuals and sector diagonals are used. Candidate selection and Hamiltonian actions are counted; payload bytes are not peak RSS.',
             'Quantum preparation, physical measurements, scalable selection and total quantum cost remain unimplemented and unpriced.',
             'No superiority over SHCI/DMRG implementations or quantum advantage is established.', '',
             f"- Declaration: `{record['declaration_commit']}`.", f"- Execution: `{record['source_commit']}`.",
             f"- Configuration SHA-256: `{record['config_sha256']}`.",
             '- The checker binds original observation bytes to their observation commit and independently replays the science.',
             '- Both versions have digest-bound execution bundles retaining declaration, execution and observation commits across merging. V2 preserves all v1 observations.', '']
    return '\n'.join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--record', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.record:
        args.record = args.record.resolve()
    if args.output and args.record is None:
        pairs = [(RECORD, args.output)]
    elif args.record:
        pairs = [(args.record, args.output or (V2_REPORT if args.record == V2_RECORD else REPORT))]
    else:
        pairs = [(RECORD, REPORT), (V2_RECORD, V2_REPORT)]
    failed = False
    for record_path, report_path in pairs:
        expected = render(read_json(record_path))
        if args.check:
            if not report_path.exists() or report_path.read_text() != expected:
                print(f'FAIL stale report: {report_path.name}')
                failed = True
        else:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(expected)
    if args.check and not failed:
        print('VALID observation-driven orthogonal active-space reports')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
