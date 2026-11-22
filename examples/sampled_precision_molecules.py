"""Opt-in ideal joint-shot workflow smoke on already declared molecular banks.

Run from the repository root: python examples/sampled_precision_molecules.py
Add --stretched for the larger engineered near-cutoff LiH/BeH2 banks (SciPy).
This is an integration check, not a preregistered coverage or cost experiment.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_cross_domain_ratio import build_case
from benchmarks.run_sampled_ratio import JointShotSimulator
from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.models.fcidump import model_from_fcidump
from clifford_qc.multivector import MV
from clifford_qc.sampled_reference import PreparedSampledReference
from clifford_qc.subspace import PrecisionAcquisitionConfig
from clifford_qc.subspace.sampled import SampledProjectedProvider
from clifford_qc.workflows import export_workflow_evidence, run_workflow


def stretched_bank(case):
    # Reuse the declared integral contraction and operator bank, but construct
    # only a statevector at the simulated preparation boundary. Do not call
    # the historical density-backed setup or exact/CV certification routines.
    from benchmarks.run_stretched_molecular_stop import active_integrals

    n = case['n']
    h = model_from_fcidump(active_integrals(case)).hamiltonian.to_mv()
    state = np.zeros(2**n, complex)
    a, b = (int(bits, 2) for bits in case['reference_bits'])
    state[a] = math.cos(case['theta'])
    state[b] = np.exp(1j*case['phi'])*math.sin(case['theta'])
    ident = MV.scalar(n, 1)
    generators = [('I', ident)]
    source = case['pair_source']
    for i, target in enumerate(case['pair_targets']):
        t = (cdag_op(n, 2*target)*cdag_op(n, 2*target+1)
             * c_op(n, 2*source+1)*c_op(n, 2*source))
        g = t-t.dagger()
        generators.append((f'G{source}{target}', ident+case['weak_mix']*g if i == 0 else g))
    return h, generators, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stretched', action='store_true')
    parser.add_argument('--export-dir', type=Path, help='existing directory for optional evidence archives')
    args = parser.parse_args()
    name = ('stretched_molecular_stop_declaration.json' if args.stretched
            else 'cross_domain_ratio_declaration.json')
    declaration = json.loads((ROOT/'docs/restart'/name).read_text())
    for case in declaration['cases']:
        if not args.stretched and case['kind'] != 'molecule':
            continue
        h, generators, state = (stretched_bank(case) if args.stretched else build_case(case))
        sampler = JointShotSimulator(state, case['n'])
        reference = PreparedSampledReference(case['n'], case['name'], sampler.draw)
        provider = SampledProjectedProvider(reference, h, generators, block_size=2)
        settings = len(provider.plan.settings)
        pilot, rounds = (40, (40*settings, 100*settings)) if args.stretched else (200, (200*settings, 500*settings))
        config = PrecisionAcquisitionConfig(pilot, rounds, target_std_error=.02)
        result = run_workflow(provider, path='sampled_precision', problem=case['name'], unit='Ha',
                              precision_config=config, seed=20261122,
                              solve_options=case['solver'] if args.stretched else {'tau_s': .005})
        record = result.as_dict()
        outcome = record['outcome']
        looks = outcome['method_details']['looks']
        print(json.dumps({'case': case['name'], 'settings': settings,
                          'words': len(provider.session.words), 'status': outcome['status'],
                          'rank': [look['retained_rank'] for look in looks],
                          'rank_margin': [look['rank_guard']['lower_margin'] for look in looks],
                          'std_error_Ha': [look['std_error'] for look in looks],
                          'measurement': record['cost_ledger']['measurement']}, indent=2))
        if args.export_dir is not None:
            print(export_workflow_evidence(result, args.export_dir/f"{case['name']}.zip"))
        if outcome['status'] == 'failed':
            raise RuntimeError(outcome['method_details']['error'])


if __name__ == '__main__':
    main()
