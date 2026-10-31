"""Independent dense/histogram validation of bounded Phase 15/16 completion.

No import of the producer, sampled second-moment provider, time-family builder,
confidence functions or native projected solver. Dense problem definitions,
Pauli decomposition, circuit action, histogram reconstruction, Hoeffding
regions, interval arithmetic and polynomial propagation are recomputed here.
Native coefficient/product-work counters are range-checked, not independently
recounted. Hardware IID assumptions are not established by a record checker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json
from benchmarks.check_observable_noise import I2, X, Y, Z, hop, number, site
from benchmarks.replay_verification import verify_source_bindings

DESIGN = 'docs/restart/phase15_16_completion_design.json'
RESULT = 'docs/restart/phase15_16_completion_results.json.gz'
CASES = ('tfim2', 'tfim3', 'hubbard2')
BUDGETS = (16384, 262144, 4194304)


def decode(a):
    a = np.asarray(a)
    return a[..., 0]+1j*a[..., 1]


def problem(name):
    n = {'tfim2': 2, 'tfim3': 3, 'hubbard2': 4}[name]
    identity = np.eye(2**n, dtype=complex)
    if name.startswith('tfim'):
        h = -sum((site(n, {q: Z, q+1: Z}) for q in range(n-1)), np.zeros_like(identity))
        h -= 1.2*sum((site(n, {q: X}) for q in range(n)), np.zeros_like(identity))
        psi = np.ones(2**n, complex)/np.sqrt(2**n)
        gens = [identity] + [site(n, {q: Z, q+1: Z}) for q in range(n-1)]
    else:
        numbers = [number(n, q) for q in range(n)]
        a, b = hop(n, 0, 2), hop(n, 1, 3)
        h = -a-b+4*(numbers[0]@numbers[1]+numbers[2]@numbers[3])-2*sum(numbers)
        psi = np.zeros(16, complex); psi[9] = 1
        gens = [identity, a, b, a@b]
    return n, h, psi, gens


def paulis(n):
    return [site(n, {q: (I2, X, Y, Z)[(code >> (2*q)) & 3] for q in range(n)})
            for code in range(4**n)]


def entries(h, gens):
    return [[[(a.conj().T @ b) if kind == 0 else
              (a.conj().T @ h @ b) if kind == 1 else
              (h @ a).conj().T @ (h @ b) for b in gens] for a in gens] for kind in range(3)]


def setting_unitary(n, setting, basis):
    u = np.eye(2**n, dtype=complex)
    for name, qs in setting['gates']:
        if name in ('H', 'S', 'SDG') and len(qs) == 1 and 0 <= qs[0] < n:
            gate = ((X+Z)/math.sqrt(2) if name == 'H' else
                    np.diag([1, 1j if name == 'S' else -1j]))
            u = site(n, {qs[0]: gate}) @ u
        elif name == 'CZ' and len(qs) == 2 and qs[0] != qs[1] and all(0 <= q < n for q in qs):
            diagonal = [(-1. if all((k >> (n-1-q)) & 1 for q in qs) else 1.) for k in range(2**n)]
            u = np.diag(diagonal) @ u
        else:
            raise ValueError('invalid measurement gate')
    for word, (sign, positions) in setting['readouts'].items():
        if sign not in (-1, 1) or any(not 0 <= p < n for p in positions):
            raise ValueError('invalid signed readout')
        diagonal = [sign*(-1)**sum((k >> (n-1-p)) & 1 for p in positions) for k in range(2**n)]
        if not np.allclose(u @ basis[int(word)] @ u.conj().T, np.diag(diagonal), atol=1e-10, rtol=0):
            raise ValueError('compiled circuit does not implement its signed Pauli readout')
    return u


def reconstruct(row, settings, ops, psi, n, delta):
    basis = paulis(n)
    size = len(ops[0])
    coefficients = {}
    family = 0
    needed = set()
    for t in range(3):
        for a in range(size):
            for b in range(a, size):
                cs = np.array([np.trace(p @ ops[t][a][b])/(2**n) for p in basis])
                cs[np.abs(cs) < 1e-10] = 0.
                coefficients[t, a, b] = cs
                needed.update(int(k) for k in np.flatnonzero(cs) if k)
                family += int(np.any(cs[1:].real)) + int(np.any(cs[1:].imag))
    assignments = [word for setting in settings for word in setting['assigned']]
    if (len(assignments) != len(set(assignments)) or 0 in assignments
            or not needed.issubset(assignments) or len(assignments) > 4**n-1):
        raise ValueError('settings do not cover the needed S/H/K universe exactly once')
    # Sparse arithmetic can retain tiny extra words or functionals that vanish
    # in a dense polynomial product. Extra assigned words still pay their full
    # physical cost. A larger Bonferroni family is conservative; an undersized
    # family is rejected. Tiny coefficient differences are checked numerically.
    declared_family = row.get('confidence_family_size', max(1, family))
    if (type(declared_family) is not int or declared_family < max(1, family)
            or declared_family > 3*size*(size+1)):
        raise ValueError('the confidence family is undersized or exceeds the entry count')
    family = declared_family
    counts = row['counts']
    if len(counts) != len(settings) or len(row['histograms']) != len(settings):
        raise ValueError('missing setting or histogram')
    if sum(counts) != row['budget'] or row['physical_shots'] != row['budget'] or row['circuits'] != len(settings):
        raise ValueError('physical-shot or circuit accounting differs')
    quotient, remainder = divmod(row['budget'], len(settings))
    if counts != [quotient+(j < remainder) for j in range(len(settings))]:
        raise ValueError('allocation differs from the declared uniform endpoint')
    means, owners = {}, {}
    rng = np.random.default_rng(row['seed'])
    for j, (setting, shots, histogram) in enumerate(zip(settings, counts, row['histograms'])):
        if type(shots) is not int or shots < 1:
            raise ValueError('invalid setting count')
        if any(type(v) is not int or v < 0 or len(bits) != n or set(bits)-{'0','1'}
               for bits, v in histogram.items()) or sum(histogram.values()) != shots:
            raise ValueError('invalid histogram')
        u = setting_unitary(n, setting, basis)
        p = np.abs(u @ psi)**2; p /= p.sum()
        # Tiny differences at a mathematically zero probability can change a
        # multinomial generator's stream consumption. Validate the published
        # floating-point draw table independently, then replay those exact
        # inputs, instead of pretending equivalent BLAS paths have equal bits.
        draw_table = np.asarray(setting.get('sampling_probabilities', p), float)
        if (draw_table.shape != p.shape or not np.isfinite(draw_table).all()
                or np.any(draw_table < 0) or abs(draw_table.sum()-1) > 1e-12
                or not np.allclose(draw_table, p, atol=1e-14, rtol=1e-12)):
            raise ValueError('sampling table disagrees with independent circuit probabilities')
        fresh = rng.multinomial(shots, draw_table)
        replay = {f'{k:0{n}b}': int(value) for k, value in enumerate(fresh) if value}
        if histogram != replay:
            raise ValueError('seeded joint histograms do not reproduce')
        for word in setting['assigned']:
            sign, positions = setting['readouts'][str(word)]
            means[word] = sum(sign*(-1)**sum(bits[p] == '1' for p in positions)*count
                              for bits, count in histogram.items())/shots
            owners[word] = j
    matrices = [np.zeros((size,size),complex) for _ in range(3)]
    radii = [np.zeros((size,size),float) for _ in range(3)]
    for (t,a,b), cs in coefficients.items():
        estimate = cs[0] + sum(cs[word]*means[word] for word in owners)
        component_radii = []
        for real in (cs.real, cs.imag):
            widths = [2*sum(abs(real[word]) for word in s['assigned']) for s in settings]
            component_radii.append(math.sqrt(.5*math.log(2*max(1,family)/delta)
                                            *sum(width*width/count for width,count in zip(widths,counts))))
        radius = math.hypot(*component_radii)
        matrices[t][a,b] = estimate.real if a==b else estimate
        matrices[t][b,a] = matrices[t][a,b].conjugate()
        radii[t][a,b] = radii[t][b,a] = radius
    return matrices,radii


def expected_report(matrices, radii, c, target, config):
    c=np.asarray(c,complex); c/=np.linalg.norm(c); v=np.abs(c)
    n,h,k=[float(np.vdot(c,m @ c).real) for m in matrices]
    rs,rh,rk=[float(v @ r @ v) for r in radii]
    ni=(n-rs,n+rs)
    report={'normalization':n,'normalization_interval':ni,
            'energy':h/n if n>0 else None,'variance':k/n-(h/n)**2 if n>0 else None,
            'status':'normalization_unresolved','energy_interval':None,'variance_interval':None,
            'residual_interval':None,'folded_value':None,'folded_interval':None,
            'variance_numerical_allowance':None,'folded_numerical_allowance':None,
            'target_eigenvalue_interval':None,'energy_eigenvalue_interval':None}
    if ni[0]<=config['normalization_floor']: return report
    def ratios(lo,hi):
        values=[x/y for x in (lo,hi) for y in ni]
        return min(values),max(values)
    ei=ratios(h-rh,h+rh); ki=ratios(k-rk,k+rk)
    emin=0. if ei[0]<=0<=ei[1] else min(x*x for x in ei)
    cancellation=1e-12*max(abs(ki[0]),abs(ki[1]),max(x*x for x in ei))
    vi=(ki[0]-max(x*x for x in ei)-cancellation,ki[1]-emin+cancellation)
    report['variance_numerical_allowance']=cancellation
    report.update(energy_interval=ei,variance_interval=vi,status='inconsistent_moments')
    if vi[1]<0: return report
    ri=(math.sqrt(max(0.,vi[0])),math.sqrt(vi[1]))
    report.update(residual_interval=ri,energy_eigenvalue_interval=(ei[0]-ri[1],ei[1]+ri[1]))
    report['status']=('within_tolerance' if ri[1]<=config['residual_tolerance'] else
                      'above_tolerance' if ri[0]>config['residual_tolerance'] else 'unresolved')
    f=k-2*target*h+target*target*n; radius=rk+2*abs(target)*rh+target*target*rs
    numerical=1e-12*(abs(k)+2*abs(target*h)+target*target*abs(n))
    radius+=numerical
    report['folded_numerical_allowance']=numerical
    fi=ratios(f-radius,f+radius)
    report.update(folded_value=f/n,folded_interval=fi)
    if fi[1]<0:
        report.update(status='inconsistent_moments',residual_interval=None,energy_eigenvalue_interval=None)
    else:
        distance=math.sqrt(fi[1]); report['target_eigenvalue_interval']=(target-distance,target+distance)
    return report


def close(a,b,path,problems):
    if b is None or isinstance(b,(str,bool)):
        if type(a) is not type(b) or a!=b: problems.append(path+': discrete value differs')
    elif isinstance(b,(tuple,list)):
        if not isinstance(a,(tuple,list)) or len(a)!=len(b): problems.append(path+': shape differs')
        else:
            for j,(x,y) in enumerate(zip(a,b)): close(x,y,path+'/'+str(j),problems)
    elif isinstance(a,bool) or not isinstance(a,(int,float,np.number)) or not np.isfinite(a) or not np.isclose(a,b,atol=2e-9,rtol=1e-8):
        problems.append(path+': numeric value differs')


def retained_solve(s, h, config):
    """Independent diagonal normalization, overlap cut and whitening."""
    scales=np.sqrt(np.maximum(np.diag(s).real,0))
    live=scales>1e-14
    if not np.any(live): raise ValueError('no live generator')
    index=np.flatnonzero(live); scales=scales[index]
    a=s[np.ix_(index,index)]/scales[:,None]/scales[None,:]
    b=h[np.ix_(index,index)]/scales[:,None]/scales[None,:]
    values,vectors=np.linalg.eigh(a)
    keep=values>max(config['tau_s'],values[-1]/config['max_condition'])
    if not np.any(keep): raise ValueError('no retained overlap mode')
    frame=vectors[:,keep]/np.sqrt(values[keep])
    reduced=frame.conj().T @ b @ frame
    energies=np.linalg.eigvalsh((reduced+reduced.conj().T)/2)
    return energies,int(np.count_nonzero(keep)),float(values[keep][-1]/values[keep][0])


def check_row(row, settings, h, psi, gens, config, problems, *, integration=False):
    n=int(round(math.log2(len(psi)))); ops=entries(h,gens)
    matrices,radii=reconstruct(row,settings,ops,psi,n,config['delta'])
    for label,matrix,radius in zip(('S','H','K'),matrices,radii):
        if not np.allclose(decode(row['matrices'][label]),matrix,atol=2e-9,rtol=1e-8):
            problems.append(row['case']+'/'+label+': histogram reconstruction differs')
        if not np.allclose(row['matrices']['r'+label],radius,atol=2e-9,rtol=1e-8):
            problems.append(row['case']+'/r'+label+': Hoeffding radius differs')
    basis=np.column_stack([g @ psi for g in gens])
    truths=[basis.conj().T @ basis,basis.conj().T @ h @ basis,(h @ basis).conj().T @ (h @ basis)]
    covered=all(np.all(np.abs(a-b)<=r+1e-10) for a,b,r in zip(matrices,truths,radii))
    if not integration: close(row['entry_region_covers'],bool(covered),'entry coverage',problems)
    inference=row['inference']
    try:
        energies,rank,condition=retained_solve(matrices[0],matrices[1],config)
    except (ValueError,np.linalg.LinAlgError):
        if inference['status']!='solve_failed': problems.append('reported a solve for a failed pencil')
        return
    if inference['status']=='solve_failed':
        problems.append('solvable pencil was hidden as solve_failed')
        return
    if inference['status']!='solved': raise ValueError('unknown inference status')
    close(inference['energy'],float(energies[0]),'Ritz value',problems)
    close(inference['retained_rank'],rank,'retained rank',problems)
    close(inference['retained_condition'],condition,'retained condition',problems)
    target=config['folded_targets'][row['case']]
    w=target
    f=matrices[2]-2*w*matrices[1]+w*w*matrices[0]
    folded_values,folded_rank,folded_condition=retained_solve(matrices[0],f,config)
    close(inference['folded']['effective_rank'],folded_rank,'folded rank',problems)
    close(inference['folded']['condition_number'],folded_condition,'folded condition',problems)
    close(inference['folded']['confidence_delta'],config['delta'],'folded delta',problems)
    for j,root in enumerate(inference['folded']['roots']):
        close(root['noisy_folded_eigenvalue'],float(folded_values[j]),'folded value',problems)
    directions=[(decode(inference['coefficients']),inference['convergence'])]
    if inference['folded']['ordered_root_certificate'] is not False: problems.append('unlicensed ordered-root claim')
    directions.extend((decode(r['coefficients']),r['assessment']) for r in inference['folded']['roots'])
    for c,report in directions:
        if report['ground_state_certificate'] is not False: problems.append('unlicensed ground-root claim')
        close(report['confidence_delta'],config['delta'],'confidence delta',problems)
        close(report['residual_tolerance'],config['residual_tolerance'],'residual tolerance',problems)
        close(report['target'],target,'folded target',problems)
        if report['evidence']!='simultaneous_fixed_endpoint_iid_bounded_shots':
            problems.append('unexpected confidence evidence label')
        expected=expected_report(matrices,radii,c.copy(),target,config)
        for key,value in expected.items(): close(report[key],value,'assessment/'+key,problems)
        if covered:
            state=basis @ c; state/=np.linalg.norm(state)
            energy=float(np.vdot(state,h @ state).real)
            variance=float(np.linalg.norm(h @ state-energy*state)**2)
            for key,value in (('energy_interval',energy),('variance_interval',variance)):
                interval=report[key]
                if interval is not None and not interval[0]-1e-8<=value<=interval[1]+1e-8:
                    problems.append('confidence event holds but '+key+'misses truth')
            interval=report['target_eigenvalue_interval']
            if interval is not None and not any(interval[0]-1e-8<=e<=interval[1]+1e-8 for e in np.linalg.eigvalsh(h)):
                problems.append('confidence event holds but target interval misses every eigenvalue')
            if report['status']=='within_tolerance' and math.sqrt(variance)>config['residual_tolerance']+1e-8:
                problems.append('false convergence on the simultaneous-confidence event')


def check_time(record, config, problems):
    from scipy.linalg import expm
    options=config['time_family']; times=options['times']; degree=options['degree']
    generated={}
    for row in record['time_families']:
        name=row['case']; n,h,psi,_=problem(name); identity=np.eye(2**n)
        shift=np.trace(h).real/(2**n); centered=h-shift*identity
        generators=[]
        for t in times:
            term=identity.astype(complex); polynomial=term.copy()
            for k in range(1,degree+1):
                term=(-1j*t/k)*(centered @ term); polynomial+=term
            generators.append(np.exp(-1j*t*shift)*polynomial)
        generated[name]=generators
        basis=np.column_stack([g @ psi for g in generators])
        np.testing.assert_allclose(decode(row['overlap']),basis.conj().T @ basis,atol=2e-9,rtol=1e-8)
        np.testing.assert_allclose(decode(row['hamiltonian']),basis.conj().T @ h @ basis,atol=2e-9,rtol=1e-8)
        energies,rank,condition=retained_solve(basis.conj().T @ basis,basis.conj().T @ h @ basis,config)
        close(row['projected_energies'],[float(e) for e in energies],name+'/projected energies',problems)
        for j,(t,action) in enumerate(zip(times,row['validation']['actions'])):
            ideal=expm(-1j*t*h) @ psi; v=basis[:,j]; norm=np.linalg.norm(v)
            error=np.linalg.norm(v-ideal); infidelity=1-abs(np.vdot(ideal,v/norm))**2
            energy_error=abs(np.vdot(v,h @ v).real/norm**2-np.vdot(ideal,h @ ideal).real)
            if (error>options['norm_error_budget']+1e-10 or infidelity>options['infidelity_budget']+1e-10
                    or energy_error>options['energy_error_budget']+1e-10): problems.append(name+': Taylor budgets fail')
            if action['norm_error_bound']!=action['taylor_bound']+action['pruning_allowance']:
                problems.append(name+': truncation/pruning accounting differs')
            if error>action['norm_error_bound']+1e-10: problems.append(name+': action exceeds analytic/pruning bound')
            close(action['fidelity'],float(1-infidelity),name+'/fidelity',problems)
            if action['norm_error_bound']>options['norm_error_budget']: problems.append(name+': unadmitted truncation')
        resources=row['validation']['resources']
        if (not 0<=resources['product_pairs']<=options['max_product_pairs']
                or not 1<=resources['peak_single_operator_terms']<=options['max_terms']
                or resources['unitary_preparation_claim'] or resources['quantum_advantage_claim']):
            problems.append(name+': invalid work/capability claim')
        condition=row['validation']['conditioning']
        if not options['minimum_retained_rank']<=condition['retained_rank']<=len(times) or condition['retained_condition']>options['max_condition']:
            problems.append(name+': conditioning admission fails')
    return generated


def check(record, *, root=ROOT, history=True):
    config=read_json(root/DESIGN); problems=[]
    if record['schema']!='clifford_qc.phase15_16_completion.v1' or record['evidence']['advantage_claim'] is not False:
        raise ValueError('unexpected schema or unlicensed advantage claim')
    if record['design']!=config: raise ValueError('design differs from the committed contract')
    if (tuple(config['cases'])!=CASES or tuple(config['shot_budgets'])!=BUDGETS
            or config['replicas']!=32 or config['master_seed']!=2026103115 or config['delta']!=.05):
        raise ValueError('fixed engineering-validation grid differs')
    grid=[(name,budget,replica) for name in CASES for budget in BUDGETS for replica in range(32)]
    if [(r['case'],r['budget'],r['replica']) for r in record['rows']]!=grid:
        raise ValueError('the 288 sampled outcomes are incomplete or out of order')
    if [r['case'] for r in record['cases']]!=list(CASES) or [r['case'] for r in record['time_families']]!=list(CASES):
        raise ValueError('missing case or real-time family')
    by_case={row['case']:row for row in record['cases']}
    for row in record['rows']:
        index=CASES.index(row['case']); bi=BUDGETS.index(row['budget'])
        seed=int(np.random.SeedSequence([config['master_seed'],index,bi,row['replica']]).generate_state(1)[0])
        if row['seed']!=seed: raise ValueError('seed differs from the declaration')
        n,h,psi,gens=problem(row['case'])
        check_row(row,by_case[row['case']]['settings'],h,psi,gens,config,problems)
    time_generators=check_time(record,config,problems)
    if [r['case'] for r in record['sampled_time_integration']]!=list(CASES):
        raise ValueError('missing sampled time-family integration')
    for index,row in enumerate(record['sampled_time_integration']):
        n,h,psi,_=problem(row['case'])
        seed=int(np.random.SeedSequence([config['master_seed'],index,99]).generate_state(1)[0])
        if row['seed']!=seed or row['budget']!=BUDGETS[-1]: raise ValueError('integration schedule differs')
        check_row(row,row['settings'],h,psi,time_generators[row['case']],config,problems,integration=True)
    if [s['case'] for s in record['summaries']]!=list(CASES):
        raise ValueError('missing case summary')
    for summary in record['summaries']:
        rows=[r for r in record['rows'] if r['case']==summary['case']]
        coverage=sum(r['entry_region_covers'] for r in rows)/len(rows)
        statuses=Counter(r['inference']['convergence']['status'] if r['inference']['status']=='solved' else 'solve_failed' for r in rows)
        if summary!={'case':summary['case'],'outcomes':len(rows),'entry_region_coverage':coverage,
                     'convergence_statuses':dict(statuses),'coverage_acceptance_passed':coverage>=.90}:
            problems.append('summary accounting differs')
        if coverage<.90: problems.append(summary['case']+': declared empirical coverage check fails')
    historical={f'benchmarks/reference_results/{name}.json' for name in (
        'phase15_h2_preflight','phase15_residual_successor','phase15_certifying_screen',
        'phase15_certifying_krylov','phase15_fc_residual','phase16a_te_qsci',
        'phase16b_feasibility','phase16b_v2_feasibility','phase16b_v3_feasibility')}
    if set(record['historical_bindings'])!=historical:
        raise ValueError('historical binding inventory differs')
    for path,digest in record['historical_bindings'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest: problems.append('historical bytes changed: '+path)
    if history:
        fixed={'benchmarks/run_phase15_16_completion.py','benchmarks/check_phase15_16_completion.py',
               'benchmarks/run_sampled_ratio.py','benchmarks/check_observable_noise.py',DESIGN,
               'docs/restart/PHASE15_16_COMPLETION_CONTRACT.md'}
        package=subprocess.check_output(['git','ls-tree','-r','--name-only',record['execution_commit'],
                                         '--','clifford_qc'],cwd=root,text=True).splitlines()
        expected_sources=fixed|{p for p in package if p.endswith('.py')}
        if set(record['source_sha256'])!=expected_sources:
            raise ValueError('execution source inventory is incomplete')
        verify_source_bindings(record,root)
        declared=subprocess.check_output(['git','show',record['declaration_commit']+':'+DESIGN],cwd=root)
        if hashlib.sha256(declared).hexdigest()!=record['source_sha256'][DESIGN]:
            raise ValueError('execution design differs from the prior declaration')
        publication=subprocess.check_output(['git','log','--diff-filter=A','--format=%H','--',RESULT],cwd=root,text=True).splitlines()
        if not publication: raise ValueError('published record ancestry is unavailable')
        for first,second in ((record['declaration_commit'],record['execution_commit']),
                             (record['execution_commit'],publication[-1])):
            if first==second or subprocess.run(['git','merge-base','--is-ancestor',first,second],cwd=root).returncode:
                problems.append('declaration/execution/observation chronology differs')
    return problems


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--record',type=Path,default=ROOT/RESULT)
    parser.add_argument('--no-history',action='store_true',help='local development only; CI uses full history')
    args=parser.parse_args()
    problems=check(read_json(args.record),history=not args.no_history)
    for problem in problems[:30]: print('FAIL',problem)
    if problems: raise SystemExit(f'{len(problems)} independent completion-validation failures')
    print('OK: 288 joint-shot outcomes, three time families, sampled integration and historical bindings independently checked')


if __name__=='__main__':
    main()
