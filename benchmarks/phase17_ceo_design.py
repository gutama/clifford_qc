"""Frozen inputs and result-free structure of the Phase 17 CEO comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clifford_qc.algorithms.exchange import exchange_pools
from clifford_qc.models.fcidump import fcidump_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'benchmarks/configs/phase17_ceo.json'
RECORD = ROOT/'benchmarks/reference_results/phase17_ceo.json'
ARMS = ('qe', 'ovp_ceo', 'mixed_ceo')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                   allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure(path):
    model = fcidump_model(path, integral_tolerance=1e-12)
    qe, ceo = exchange_pools(model.n)
    def rows(pool):
        return [{'label':op.label,'support':list(op.support),
                 'creation':list(op.creation),'annihilation':list(op.annihilation),
                 'parents':list(op.parents),'sign':op.sign,
                 'coefficients':[[code,float(c.real).hex(),float(c.imag).hex()]
                                 for code,c in sorted(op.generator.terms.items())]}
                for op in pool]
    return {'n':model.n,'n_electrons':model.metadata['n_electrons'],
            'sz':model.metadata['sz'],'spin_convention':'interleaved',
            'reference_occupied':model.metadata['reference_occupied_spin_orbitals'],
            'qe':rows(qe),'ovp_ceo':rows(ceo)}


def target_summary(run, target):
    """Observed first frozen prefix; no interpolated or aggregate winner."""
    base_error = run['reference_energy']-run['sector_ground_energy']
    if base_error <= target:
        return {'status':'REFERENCE_ALREADY_WITHIN_TARGET','iteration':0,
                'error':base_error,'parameters':0,'screened_gradients':0,
                'optimizer_evaluations':0,'parameter_derivatives':0,'generic_ladder_cnots':0}
    evaluations = derivatives = 0
    for row in run['trajectory']:
        evaluations += row['optimizer_evaluations']
        derivatives += len(row['parameters'])*row['optimizer_evaluations']
        error = row['energy']-run['sector_ground_energy']
        if error <= target:
            return {'status':'REACHED','iteration':row['iteration'],'error':error,
                    'parameters':len(row['parameters']),
                    'screened_gradients':sum(r['screened_gradients'] for r in run['screening']
                                              if r['iteration']<=row['iteration']),
                    'optimizer_evaluations':evaluations,'parameter_derivatives':derivatives,
                    'generic_ladder_cnots':row['generic_ladder_cnot_upper_bound']}
    return {'status':'RIGHT_CENSORED','iteration':None,'error':run['energy']-run['sector_ground_energy']}
