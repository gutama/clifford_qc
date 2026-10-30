"""Diagnose exact plan identities on one unchanged, already-published pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys


def child():
    import numpy as np
    from benchmarks import run_sampled_ratio as r
    from clifford_qc.measurement.ratio import uniform_setting_plan
    c = r.load_declaration()
    case = c['cases'][0]
    record = json.loads((r.ROOT/r.RESULT).read_text())
    _, _, _, reference, provider, _, _, _ = r.setup(case, c)
    budget = c['total_shot_budgets'][0]
    pilot_budget = int(budget*c['pilot_fraction'])
    pilot = reference.sample(provider.plan.settings,
                             uniform_setting_plan(provider.plan.settings, pilot_budget),
                             seed=c['master_seed'])
    plan = provider.freeze_ratio(pilot, budget-pilot_budget, tau_s=c['tau_s'],
                                max_condition=c['max_condition'], delta=c['ratio_delta'],
                                denominator_floor=c['denominator_floor'], allocation='ratio')
    def array(a):
        return {'shape': a.shape, 'values': [[float(x.real).hex(), float(x.imag).hex()] for x in a.ravel()]}
    print(json.dumps({'profile': os.environ.get('OPENBLAS_CORETYPE'),
                      'numpy_disabled': os.environ.get('NPY_DISABLE_CPU_FEATURES'),
                      'numpy': np.__version__, 'python': sys.version.split()[0],
                      'expected_identity': record['rows'][0]['plan_identity'],
                      'actual_identity': plan.identity,
                      'matches': plan.identity == record['rows'][0]['plan_identity'],
                      'pilot_identity': pilot.identity,
                      'pilot_energy': float(plan.pilot_energy).hex(),
                      'coefficients': array(plan.coefficients),
                      'retained_frame': array(plan.retained_frame),
                      'counts': plan.counts,
                      'numerator': [plan.numerator.constant, sorted(plan.numerator.coefficients.items())],
                      'denominator': [plan.denominator.constant, sorted(plan.denominator.coefficients.items())]}))
    if not os.environ.get('OPENBLAS_CORETYPE'):
        np.show_config()
        np.show_runtime()


if '--child' in sys.argv:
    child()
else:
    root = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'benchmarks' else Path.cwd()
    for core, disabled in [(None, None), ('HASWELL', None), ('SANDYBRIDGE', None),
                           ('NEHALEM', None), ('PRESCOTT', None),
                           ('HASWELL', 'X86_V3,X86_V4,AVX512_ICL,AVX512_SPR')]:
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if core:
            env['OPENBLAS_CORETYPE'] = core
        if disabled:
            env['NPY_DISABLE_CPU_FEATURES'] = disabled
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=60)
        print('PROFILE', core, disabled, 'EXIT', result.returncode, flush=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
