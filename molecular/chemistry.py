"""Cache RHF orbitals, correlated baselines and FCIDUMP independently of A-CASE."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile

from clifford_qc.prepared import _atomic_json, _digest


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pyscf"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    versions["threads"] = {key: os.environ.get(key) for key in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return versions


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(Path(source).read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_chemistry(key, spec, output_directory, *, refresh=False):
    """Return (baselines, exported FCIDUMP, cache hit); fail on unconverged RHF.

    PySCF is imported only on a cache miss. Solver budgets/storage do not change
    the chemistry key. Cached JSON and integral bytes are both verified.
    """
    output = Path(output_directory)
    chemistry_spec = {field: spec[field] for field in ("atom", "basis", "charge", "spin")}
    if chemistry_spec["spin"] != 0:
        raise ValueError("the molecular comparison currently requires closed-shell RHF (spin=0)")
    request = {"schema": "clifford_qc.chemistry.v1", "spec": chemistry_spec,
               "unit": "Angstrom", "max_cycle": 300, "environment": environment(),
               "implementation": file_digest(__file__)}
    fingerprint = _digest(request)
    cache = output / "cache" / "chemistry"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / f"{fingerprint}.json"
    integral = cache / f"{fingerprint}.fcidump"
    exported = output / f"{key}.fcidump"
    if metadata.exists() and not refresh:
        saved = json.loads(metadata.read_text())
        digest = saved.pop("digest", None)
        if (digest != _digest(saved) or saved.get("request") != request
                or not integral.is_file()
                or saved.get("fcidump_sha256") != file_digest(integral)):
            raise ValueError("chemistry cache is inconsistent; rerun with --refresh-chemistry")
        atomic_copy(integral, exported)
        return saved["baselines"], exported, True

    try:
        from pyscf import cc, ci, gto, scf, tools
    except ImportError as exc:
        raise RuntimeError("molecular chemistry requires PySCF: pip install '.[molecular]'") from exc
    mol = gto.M(**chemistry_spec, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.max_cycle = 300
    mf.run()
    if not mf.converged:
        raise RuntimeError(f"{key}: RHF did not converge; no correlated result was published")
    cisd, ccsd = ci.CISD(mf), cc.CCSD(mf)
    cisd.max_cycle = ccsd.max_cycle = 300
    cisd.run()
    ccsd.run()
    baselines = {"e_rhf": float(mf.e_tot), "e_cisd": float(cisd.e_tot),
                 "e_ccsd": float(ccsd.e_tot), "rhf_converged": True,
                 "cisd_converged": bool(cisd.converged), "ccsd_converged": bool(ccsd.converged)}
    fd, temporary = tempfile.mkstemp(prefix="integrals.", dir=cache)
    os.close(fd)
    try:
        tools.fcidump.from_scf(mf, temporary)
        atomic_copy(temporary, integral)
    finally:
        os.unlink(temporary)
    saved = {"request": request, "baselines": baselines, "fcidump_sha256": file_digest(integral)}
    _atomic_json(metadata, {**saved, "digest": _digest(saved)})
    atomic_copy(integral, exported)
    return baselines, exported, False
