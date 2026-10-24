"""Bounded SciPy recovery: numerical, spin, tracking and ground evidence differ.

Every configured start runs. Independent spectra assess returned states only;
multiple converged starts do not certify a ground root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np

from ..backends.sector_reference import SectorActionProvider, immutable_array
from .bounded_actions import ActionMemoryLimit, ActionMemoryPlan, BoundedActionSpace, RSSMonitor, positive_integer
from .convergence import ConvergenceConfig, residual_convergence_report


@dataclass(frozen=True)
class SectorSolveConfig:
    roots: int = 1
    ncv: int = 32
    max_iterations: int = 300
    recovery_attempts: int = 2
    residual_tolerance: float = 1e-8
    spin_tolerance: float = 1e-8
    seed: int = 20261024
    max_owned_bytes: int = 256*2**20
    max_rss_bytes: int | None = None
    rss_headroom_bytes: int = 0

    def __post_init__(self):
        for name in ("roots","ncv","max_iterations","max_owned_bytes"):
            positive_integer(getattr(self,name),name)
        for name in ("recovery_attempts","seed","rss_headroom_bytes"):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("residual_tolerance","spin_tolerance"):
            value = getattr(self,name)
            if isinstance(value,bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_rss_bytes is not None:
            positive_integer(self.max_rss_bytes,"max_rss_bytes")
        if self.recovery_attempts > 16:
            raise ValueError("recovery_attempts must not exceed the bounded log capacity of 16")
        if self.ncv <= self.roots+1:
            raise ValueError("ncv must exceed roots + 1")


@dataclass(frozen=True)
class ActionResidual:
    root: int
    energy: float
    rayleigh_energy: float
    second_moment: float
    variance: float
    residual_norm: float
    cancellation_scale: float
    resolution_norm: float

    @property
    def resolved(self): return self.residual_norm > self.resolution_norm

    def as_dict(self):
        return {**asdict(self),"resolved":self.resolved,"source":"cached_actions",
                "resolution_scope":"128 eps times (norm(Hpsi) + abs(E)*norm(psi))/norm(psi); rounding convention without a forward-error proof; not a statistical confidence bound"}


def action_residual(state, acted, energy, *, root=0):
    state,acted = np.asarray(state,complex),np.asarray(acted,complex)
    if (state.ndim != 1 or acted.shape != state.shape or not np.isfinite(state).all()
            or not np.isfinite(acted).all() or not np.isfinite(energy)):
        raise ValueError("finite matching state/action and energy required")
    norm = float(np.linalg.norm(state))
    if norm <= np.finfo(float).tiny:
        raise ValueError("action residual requires a nonzero normalized state")
    rayleigh = complex(np.vdot(state,acted))/norm**2
    if abs(rayleigh.imag) > 1e-10*max(1.,abs(rayleigh.real)):
        raise ValueError("Hamiltonian action has a complex Rayleigh quotient")
    second = float(np.vdot(acted,acted).real/norm**2)
    residual = float(np.linalg.norm(acted-energy*state)/norm)
    resolution = float(128*np.finfo(float).eps*(np.linalg.norm(acted)+abs(energy)*norm)/norm)
    return ActionResidual(root,float(energy),float(rayleigh.real),second,residual**2,
                          residual,second+energy**2,resolution)


def spin_diagnostics(provider, state, *, check=None):
    """An S² mean alone can hide mixtures; evaluate the action defect."""
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("spin diagnostics require a nonzero state")
    state = state/norm
    words,spec = provider.reference.occupation_words,provider.reference.spec
    numbers = np.fromiter((int(w).bit_count() for w in words),dtype=float,count=words.size)
    up_mask = sum(1 << (spec.n-1-q) for q in range(0,spec.n,2))
    z = np.fromiter(((int(w)&up_mask).bit_count()-int(w).bit_count()/2 for w in words),
                    dtype=float,count=words.size)
    weights = abs(state)**2
    if check is not None: check()
    squared = provider.apply_spin_squared(state)
    if check is not None: check()
    target = provider.target_spin*(provider.target_spin+1)
    return {"particle_number_mean":float(weights@numbers),
            "particle_number_defect":float(np.linalg.norm((numbers-spec.n_electrons)*state)),
            "spin_z_mean":float(weights@z),"spin_z_defect":float(np.linalg.norm((z-spec.sz)*state)),
            "spin_squared_mean":float(np.vdot(state,squared).real),
            "spin_action_defect":float(np.linalg.norm(squared-target*state))}


def root_tracking_overlap(previous, current):
    """Ray overlaps and block principal cosines, not ground-root identification."""
    previous,current = np.asarray(previous,complex),np.asarray(current,complex)
    if (previous.ndim != 2 or current.ndim != 2 or previous.shape[0] != current.shape[0]
            or not previous.shape[1] or not current.shape[1]
            or not np.isfinite(previous).all() or not np.isfinite(current).all()):
        raise ValueError("finite matching nonempty orthonormal blocks required")
    for block in (previous,current):
        if not np.allclose(block.conj().T@block,np.eye(block.shape[1]),atol=1e-9,rtol=0):
            raise ValueError("tracking blocks must be orthonormal")
    overlap = previous.conj().T@current
    return {"principal_cosines":np.clip(np.linalg.svd(overlap,compute_uv=False),0,1).tolist(),
            "ray_overlaps_squared":(abs(overlap)**2).tolist()}


@dataclass(frozen=True)
class SectorSolveResult:
    energies: tuple[float,...]
    states: np.ndarray | None
    actions: np.ndarray | None
    attempts: tuple[dict,...]
    residuals: tuple[ActionResidual,...]
    spin: tuple[dict,...]
    status: str
    resources: dict

    def as_dict(self, *, ground_reference=None):
        config = self.resources['config']
        if ground_reference is not None and any(
                max(s['particle_number_defect'],s['spin_z_defect'],s['spin_action_defect']) > config['spin_tolerance']
                for s in self.spin):
            raise ValueError("sector spectral assessment requires verified quantum-number consistency")
        reports = [residual_convergence_report(
            residual,ConvergenceConfig(config['residual_tolerance'],ground_reference),
            basis_size=len(self.energies),effective_rank=len(self.energies),condition_number=1.,
            residual_resolution=residual.resolution_norm,stopped_reason=self.status).as_dict()
            for residual in self.residuals]
        return {"schema":"clifford_qc.sector_solve.v1","energies":list(self.energies),
                "attempts":list(self.attempts),"convergence":reports,"spin":list(self.spin),
                "status":self.status,"root_target":"lowest_in_declared_N_Sz_S_sector",
                "ground_state_status":reports[0]['ground_state_status'] if ground_reference is not None and reports else 'not_established',
                "stopping_uses_spectral_oracle":False,"resources":self.resources}


def solve_sector(provider, *, config=None, initial=None, read_rss=None):
    """Assess the initial block, then run every independently seeded recovery.

    Public ARPACK does not expose internal restart counts; every callback action
    is charged. Tiny sectors use an admitted complete projection, not a hidden
    library dense fallback. Spectral references are accepted only by as_dict.
    """
    if not isinstance(provider,SectorActionProvider):
        raise TypeError("a SectorActionProvider is required")
    import scipy
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
    cfg = SectorSolveConfig() if config is None else config
    if not isinstance(cfg,SectorSolveConfig):
        raise TypeError("config must be a SectorSolveConfig")
    started = time.perf_counter()
    if initial is not None and not isinstance(initial,np.ndarray):
        raise TypeError("initial must be an explicit sector ndarray")
    initial = provider.reference.amplitudes[:,None] if initial is None else initial
    if initial.ndim == 1:
        initial = initial[:,None]
    if (initial.ndim != 2 or initial.shape[0] != provider.dimension or not initial.shape[1]
            or initial.shape[1] > cfg.ncv):
        raise ValueError("finite initial columns within the Krylov capacity are required")
    d,k = provider.dimension,cfg.roots
    if k > provider.target_dimension:
        raise ValueError("requested roots exceed the target total-spin dimension")
    small = d <= k+1
    capacity = d if small else max(k,initial.shape[1])
    plan = ActionMemoryPlan.admit(d,capacity,roots=k,ncv=0 if small else min(d,cfg.ncv),
        input_bytes=provider.input_payload_bytes,max_owned_bytes=cfg.max_owned_bytes,
        max_rss_bytes=cfg.max_rss_bytes,rss_headroom_bytes=cfg.rss_headroom_bytes)
    if not np.isfinite(initial).all():
        raise ValueError("initial columns must be finite")
    monitor = RSSMonitor(plan,read_rss=read_rss)
    attempts,best = [],None
    initial_identity = None
    counters = provider.resources()
    orthogonal_projections = recovery_started = 0
    space = None

    def checked_action(vector):
        monitor.check()
        acted = provider.apply_hamiltonian(vector)
        monitor.check()
        return acted

    def prepare(columns):
        nonlocal orthogonal_projections
        space.size = 0
        for column in columns.T:
            monitor.check()
            direction = provider.project_spin(column,check=monitor.check)
            monitor.check()
            for _ in range(2):
                for j in range(space.size):
                    orthogonal_projections += 1
                    direction -= space.basis[:,j]*np.vdot(space.basis[:,j],direction)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-10:
                space.extend((direction/norm)[:,None])
        return bool(space.size)

    def assess(label, solver_status):
        nonlocal best
        before = time.perf_counter()
        result = space.solve()
        roots = min(k,result.effective_rank)
        states,actions = space.ritz_actions(result,roots)
        residuals = tuple(action_residual(states[:,j],actions[:,j],result.energies[j],root=j) for j in range(roots))
        monitor.check()
        spin = tuple(spin_diagnostics(provider,states[:,j],check=monitor.check) for j in range(roots))
        monitor.check()
        consistent = all(max(s['particle_number_defect'],s['spin_z_defect'],s['spin_action_defect']) <= cfg.spin_tolerance for s in spin)
        converged = roots == k and all(r.residual_norm+r.resolution_norm <= cfg.residual_tolerance for r in residuals)
        tracking = None if best is None else root_tracking_overlap(best[1],states)
        attempts.append({"kind":label,"solver_status":solver_status,"energies":list(result.energies[:roots]),
            "residual_norms":[r.residual_norm for r in residuals],"spin_consistent":consistent,
            "numerically_converged":converged,"ground_state_status":"not_established",
            "tracking_principal_cosines":None if tracking is None else tracking['principal_cosines'],
            "tracking_ray_overlaps_squared":None if tracking is None else tracking['ray_overlaps_squared'],
            "tracking_scope":"overlap with previous best retained block; not ground-root evidence",
            "h_action_columns_cumulative":provider.h_action_columns-counters['h_action_columns'],
            "spin_action_columns_cumulative":provider.spin_action_columns-counters['spin_action_columns'],
            "diagnostic_seconds":time.perf_counter()-before})
        # Lowest physically consistent variational candidate; assess its
        # numerical convergence independently. No spectral oracle enters selection.
        key = (not consistent,float(result.energies[0]))
        if best is None or key < best[0]:
            best = (key,states.copy(),actions.copy(),tuple(float(e) for e in result.energies[:roots]),residuals,spin)

    status = 'root_unresolved'
    try:
        monitor.check(admission=True)
        initial = immutable_array(initial,complex)
        initial_identity = hashlib.sha256(str(initial.shape).encode()+initial.tobytes()).hexdigest()
        space = BoundedActionSpace(checked_action,plan,monitor=monitor)
        initial_started = time.perf_counter()
        if prepare(initial):
            assess('initial_block','projected')
        else:
            attempts.append({'kind':'initial_block','solver_status':'zero_spin_weight'})
        attempts[-1].update(wall_seconds=time.perf_counter()-initial_started,
            h_action_columns=provider.h_action_columns-counters['h_action_columns'],
            spin_action_columns=provider.spin_action_columns-counters['spin_action_columns'])
        if small:
            space.size = 0
            for j in range(d):
                v = np.zeros(d,complex); v[j] = 1
                projected = provider.project_spin(v,check=monitor.check)
                for _ in range(2):
                    for i in range(space.size):
                        orthogonal_projections += 1
                        projected -= space.basis[:,i]*np.vdot(space.basis[:,i],projected)
                norm = float(np.linalg.norm(projected))
                if norm > 1e-10:
                    space.extend((projected/norm)[:,None])
            assess('complete_small_sector','complete_projection')
        else:
            bound = sum(abs(v) for v in provider.hamiltonian.terms.values())
            penalty,shift = 2*bound+1,3*bound+2
            def shifted(vector):
                monitor.check()
                target = provider.project_spin(vector,check=monitor.check)
                acted = checked_action(target)+penalty*(vector-target)-shift*vector
                monitor.check()
                return acted
            operator = LinearOperator((d,d),matvec=shifted,dtype=complex)
            diagonal = provider.diagonal()
            low = int(np.argmin(diagonal))
            for attempt in range(cfg.recovery_attempts):
                recovery_started += 1
                attempt_started = time.perf_counter()
                count_before,spin_before = provider.h_action_columns,provider.spin_action_columns
                monitor.check()
                rng = np.random.default_rng(np.random.SeedSequence([cfg.seed,attempt]))
                v0 = rng.normal(size=d)+1j*rng.normal(size=d)
                v0 /= np.linalg.norm(v0); v0[low] += 1
                v0 = provider.project_spin(v0,check=monitor.check)
                norm = float(np.linalg.norm(v0))
                if norm <= 1e-10:
                    attempts.append({'kind':'recovery','seed_index':attempt,'solver_status':'zero_spin_start',
                        'h_action_columns':provider.h_action_columns-count_before,
                        'spin_action_columns':provider.spin_action_columns-spin_before,
                        'wall_seconds':time.perf_counter()-attempt_started})
                    continue
                v0 /= norm
                try:
                    _,vectors = eigsh(operator,k=k,which='LM',v0=v0,ncv=min(d,cfg.ncv),
                        maxiter=cfg.max_iterations,tol=min(cfg.residual_tolerance*.01,1e-11))
                    solver_status = 'returned'
                except ArpackNoConvergence as exc:
                    vectors,solver_status = exc.eigenvectors,'iteration_limit'
                monitor.check()
                if vectors is None or vectors.shape[1] == 0:
                    attempts.append({'kind':'recovery','seed_index':attempt,'solver_status':solver_status,
                        'h_action_columns':provider.h_action_columns-count_before,
                        'spin_action_columns':provider.spin_action_columns-spin_before,
                        'wall_seconds':time.perf_counter()-attempt_started})
                    continue
                if not prepare(vectors):
                    attempts.append({'kind':'recovery','seed_index':attempt,'solver_status':'zero_spin_weight',
                        'h_action_columns':provider.h_action_columns-count_before,
                        'spin_action_columns':provider.spin_action_columns-spin_before,
                        'wall_seconds':time.perf_counter()-attempt_started})
                    continue
                assess('recovery',solver_status)
                attempts[-1].update(seed_index=attempt,start_type='lowest_diagonal_plus_seeded_random',
                    low_diagonal_index=low,h_action_columns=provider.h_action_columns-count_before,
                    spin_action_columns=provider.spin_action_columns-spin_before,
                    wall_seconds=time.perf_counter()-attempt_started)
        converged = (best is not None and not best[0][0] and len(best[3]) == k
            and all(r.residual_norm+r.resolution_norm <= cfg.residual_tolerance for r in best[4]))
        status = 'numerically_converged_root_unresolved' if converged else 'accuracy_unresolved'
    except ActionMemoryLimit:
        status = 'resource_limited'
        attempts.append({'kind':'limit_breach','solver_status':'resource_limited',
            'h_action_columns_cumulative':provider.h_action_columns-counters['h_action_columns'],
            'spin_action_columns_cumulative':provider.spin_action_columns-counters['spin_action_columns']})
    resources = {**provider.resources(),'config':asdict(cfg),'memory_plan':plan.as_dict(),
        'rss':monitor.as_dict(),'wall_seconds':time.perf_counter()-started,
        'orthogonal_projections':orthogonal_projections,'recovery_attempts_run':recovery_started,
        'solve_h_action_columns':provider.h_action_columns-counters['h_action_columns'],
        'solve_spin_action_columns':provider.spin_action_columns-counters['spin_action_columns'],
        'projection_resets':1+sum(a['kind']=='recovery' and 'energies' in a for a in attempts),
        'deflation_steps':0,'admitted_ncv':0 if small else min(d,cfg.ncv),'internal_restarts':None,
        'internal_restart_scope':'ARPACK public API does not expose a restart count; ncv/maxiter bound it and every H/S² action is charged',
        'initial_block_identity':initial_identity,'solver':'scipy.sparse.linalg.eigsh',
        'scipy_version':scipy.__version__,'target_spin_projection':'polynomial_in_S_squared',
        'arpack_workspace_scope':'complex Arnoldi vector/work-array envelope; vendor-library and allocator overhead remain outside owned payload admission'}
    if best is None:
        return SectorSolveResult((),None,None,tuple(attempts),(),(),status,resources)
    return SectorSolveResult(best[3],immutable_array(best[1],complex),immutable_array(best[2],complex),
                             tuple(attempts),best[4],best[5],status,resources)
