"""Audit opt-in graph compilation on frozen dyadic banks; no sampled experiment.

Rebuild both compilers on identical committed word universes and partitions.
The output binds software/inputs and illustrative device-card accounting, not
hardware measurements, matched-accuracy shot savings or a new frozen campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from benchmarks.run_clifford_hierarchy import SYSTEMS
from clifford_qc.prepared import implementation_fingerprint
from clifford_qc.measurement.block_commuting import block_commuting_partition
from clifford_qc.measurement.block_synthesis import synthesize_block_settings
from clifford_qc.measurement.graph_state import compile_graph_settings
from clifford_qc.measurement.cost import DeviceCard,setting_fidelity,setting_duration_us


def digest_file(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def summarize(synthesis):
    depths=synthesis.two_qubit_depth_per_setting
    return {'gate_counts':dict(sorted(synthesis.gate_counts.items())),
            'two_qubit_per_sweep':synthesis.logical_two_qubit_per_sweep,
            'mean_two_qubit_critical_depth':sum(depths)/len(depths),
            'max_two_qubit_critical_depth':max(depths),
            'max_two_qubit_per_setting':max(synthesis.two_qubit_per_setting),
            'mean_scheduled_two_qubit_layers':sum(s.d_2q for s in synthesis.settings)/len(synthesis.settings)}


def comparison():
    sources=('clifford_qc/measurement/graph_state.py','clifford_qc/measurement/block_synthesis.py',
             'clifford_qc/measurement/planning.py','clifford_qc/measurement/cost.py',
             'benchmarks/compare_graph_readout.py','benchmarks/run_clifford_hierarchy.py')
    result={'schema':'clifford_qc.graph_readout_comparison.v1',
            'evidence':'exact discrete compilation/counts; illustrative device-card surrogates',
            'scope':'software compilation audit, not a preregistered sampling/hardware experiment',
            'quantum_advantage_claim':False,'implementation_fingerprint':implementation_fingerprint(),'source_sha256':{p:digest_file(p) for p in sources},
            'inputs':{},'rows':[]}
    for name,build in SYSTEMS.items():
        bank=build();n,codes=bank['n_qubits'],bank['codes']
        path=f'benchmarks/reference_results/clifford_hierarchy_{name}_v2.json'
        frozen=json.loads((ROOT/path).read_text())
        if len(codes)!=frozen['word_universe']:raise ValueError('frozen word universe differs')
        result['inputs'][name]={'record_path':path,'record_sha256':digest_file(path),
            'n':n,'words':len(codes),'basis_labels':bank['basis_labels'],
            'codes_sha256':hashlib.sha256(json.dumps(codes,separators=(',',':')).encode()).hexdigest()}
        for k in (2,4,8):
            groups=block_commuting_partition(n,codes,k)
            original=synthesize_block_settings(n,codes,groups,k)
            settings,graph=compile_graph_settings(n,codes,groups,k)
            oldrow=next(r for r in frozen['rows'] if r['block_size']==k)
            if (len(groups)!=oldrow['settings'] or original.logical_cx_per_sweep!=oldrow['logical_cx_per_sweep']
                    or sum(original.cx_depth_per_setting)!=oldrow['logical_cx_depth_sum']):
                raise ValueError('current default no longer reproduces frozen elimination counts')
            before,after=summarize(original),summarize(graph)
            cards={}
            for file in sorted((ROOT/'benchmarks/configs/device_cards').glob('*.json')):
                card=DeviceCard.load(file)
                old=[setting_fidelity(card,s,n) for s in original.settings]
                new=[setting_fidelity(card,s,n) for s in graph.settings]
                cards[card.name]={'card_sha256':card.sha256,'calibration_status':card.calibration_status,
                    'fidelity_floor':card.fidelity_floor,'elimination_min_fidelity':min(old),'graph_min_fidelity':min(new),
                    'elimination_admissible_settings':sum(v>=card.fidelity_floor for v in old),
                    'graph_admissible_settings':sum(v>=card.fidelity_floor for v in new),
                    'newly_admissible_settings':sum(a<card.fidelity_floor<=b for a,b in zip(old,new)),
                    'lost_admissible_settings':sum(b<card.fidelity_floor<=a for a,b in zip(old,new)),
                    'elimination_sweep_duration_us':sum(setting_duration_us(card,s) for s in original.settings),
                    'graph_sweep_duration_us':sum(setting_duration_us(card,s) for s in graph.settings)}
            row={'system':name,'block_size':k,'settings':len(groups),'elimination':before,'graph_state':after,
                 'two_qubit_gate_reduction':1-after['two_qubit_per_sweep']/before['two_qubit_per_sweep'],
                 'critical_depth_sum_reduction':1-sum(graph.two_qubit_depth_per_setting)/sum(original.cx_depth_per_setting),
                 'group_indices_sha256':hashlib.sha256(json.dumps(groups,separators=(',',':')).encode()).hexdigest(),
                 'assigned_signed_readouts_verified':sum(len(s.assigned_word_codes) for s in settings),
                 'compatibility_matrices_equal':bool((original.compatibility==graph.compatibility).all()),
                 'readable_setting_word_pairs':{'elimination':int(original.compatibility.sum()),
                     'graph_state':int(graph.compatibility.sum()),
                     'gained':int((~original.compatibility & graph.compatibility).sum()),
                     'lost':int((original.compatibility & ~graph.compatibility).sum())},
                 'cards':cards}
            result['rows'].append(row)
            print(f'{name} k={k}: {before["two_qubit_per_sweep"]} CX -> {after["two_qubit_per_sweep"]} CZ',file=sys.stderr,flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',type=Path,help='rebuild and compare an existing audit report')
    args=parser.parse_args(argv)
    result=comparison()
    if args.check is not None:
        if json.loads(args.check.read_text())!=result:raise SystemExit('graph compilation audit differs')
    elif args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
