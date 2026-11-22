"""Preferred fixed-bank acquisition; bootstrap width is a separately priced guard.

Run from the repository root after installing the package:
  python examples/acase_precision_acquisition.py
  python examples/acase_precision_acquisition.py --bootstrap-half-width 0.02 --bootstrap-replicates 64
"""
import argparse

from clifford_qc import I, X, Z, ket_density
from clifford_qc.backends.finite_shot import FiniteShotBackend
from clifford_qc.subspace import MatrixElementBank, SharedMeasurement, acquire_ritz_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-half-width', type=float)
    parser.add_argument('--bootstrap-replicates', type=int)
    args = parser.parse_args()
    # Independent but mixed generators: the bank has a weak overlap mode.
    bank = MatrixElementBank(ket_density(2, '00'), .5*Z(2, 0)+.3*X(2, 0)+.1*Z(2, 1),
                             [('I', I(2)), ('mixed', I(2)+.3*X(2, 0)+.2*Z(2, 1))])
    shared = SharedMeasurement(bank)
    groups = len(shared.groups)
    result = acquire_ritz_precision(shared, FiniteShotBackend(19), pilot_shots_per_group=400,
                                    round_budgets=[1500*groups, 3000*groups, 6000*groups],
                                    target_std_error=.02, solver_kwargs={'tau_s': .001},
                                    max_bootstrap_half_width=args.bootstrap_half_width,
                                    bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=29)
    for look in result.looks:
        guard = look['rank_guard']
        print(f"look {look['look']}: shots={look['shots']:,} rank={look['retained_rank']} "
              f"SE={look['std_error']:.5g} rank_guard={guard['passed']} "
              f"width_guard={look.get('bootstrap_width_passed', 'not checked')}")
    print(f'status={result.stopped}; energy={result.energy:.7f}; shots={result.shots:,}')
    if result.bootstrap_interval is not None:
        print('bootstrap interval:', result.bootstrap_interval)
    if result.stopped != 'target_met':
        print('Budget exhausted before the guards passed; estimate is diagnostic only.')


if __name__ == '__main__':
    main()
