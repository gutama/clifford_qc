"""Session benchmark: MV.__mul__ against an array-at-a-time NumPy product.

Run from anywhere: python "research_notes/Clifford QC framework improvements/bench_kernel.py".
Random Pauli sums, not chemistry-structured banks; a one-off measurement, not a repository record.
"""
import sys, time, platform, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clifford_qc.multivector import MV
from clifford_qc import pauli_kernel as pk

print("python", platform.python_version(), "numpy", np.__version__, "bitwise_count", hasattr(np, "bitwise_count"), "cpu", platform.processor() or platform.machine(), "ncpu", os.cpu_count())
rng = np.random.default_rng(0)
PH = np.array([1, 1j, -1, -1j])

def lanes(n):
    return np.int64(sum(1 << (2*j) for j in range(n)))

def split(c, lo):
    z = (c >> 1) & lo
    return (c & lo) ^ z, z

def bc(a):
    return np.bitwise_count(a).astype(np.int64)

def vec_products(n, ca, wa, cb, wb):
    lo = lanes(n)
    xa, za = split(ca, lo); xb, zb = split(cb, lo)
    c = ca[:, None] ^ cb[None, :]
    xc, zc = split(c, lo)
    e = (bc(xa & za)[:, None] + bc(xb & zb)[None, :] - bc(xc & zc) + 2*bc(za[:, None] & xb[None, :])) % 4
    w = np.multiply.outer(wa, wb) * PH[e]
    return c.ravel(), w.ravel()

def dense_accumulate(n, c, w):
    re = np.bincount(c, weights=w.real, minlength=4**n)
    im = np.bincount(c, weights=w.imag, minlength=4**n)
    acc = re + 1j*im
    keep = np.flatnonzero(np.abs(acc) > 1e-12)
    return keep, acc[keep]

def sort_accumulate(c, w):
    u, inv = np.unique(c, return_inverse=True)
    re = np.bincount(inv, weights=w.real); im = np.bincount(inv, weights=w.imag)
    acc = re + 1j*im
    keep = np.abs(acc) > 1e-12
    return u[keep], acc[keep]

for n, t in [(8, 300), (8, 1000), (8, 3000), (12, 1000), (12, 3000)]:
    ca = rng.choice(4**n, size=t, replace=False).astype(np.int64)
    cb = rng.choice(4**n, size=t, replace=False).astype(np.int64)
    wa = rng.normal(size=t) + 1j*rng.normal(size=t)
    wb = rng.normal(size=t) + 1j*rng.normal(size=t)
    A = MV(n, {int(k): complex(v) for k, v in zip(ca, wa)})
    B = MV(n, {int(k): complex(v) for k, v in zip(cb, wb)})
    pk.clear_kernel_caches()
    t0 = time.perf_counter(); C = A * B; t_cold = time.perf_counter() - t0
    t0 = time.perf_counter(); C2 = A * B; t_warm = time.perf_counter() - t0
    t0 = time.perf_counter(); c, w = vec_products(n, ca, wa, cb, wb); t_prod = time.perf_counter() - t0
    t0 = time.perf_counter(); kd, vd = dense_accumulate(n, c, w); t_dense = time.perf_counter() - t0
    t0 = time.perf_counter(); ks, vs = sort_accumulate(c, w); t_sort = time.perf_counter() - t0
    ref = C.terms
    ok_d = len(ref) == len(kd) and all(abs(ref.get(int(k), 0) - v) < 1e-9 for k, v in zip(kd, vd))
    ok_s = len(ref) == len(ks) and all(abs(ref.get(int(k), 0) - v) < 1e-9 for k, v in zip(ks, vs))
    print(f"n={n} t={t} pairs={t*t:,} out_words={len(ref):,} | MV cold {t_cold:.3f}s warm {t_warm:.3f}s | numpy products {t_prod:.4f}s + dense-bincount {t_dense:.4f}s (match {ok_d}) | + unique-sort {t_sort:.4f}s (match {ok_s}) | speedup cold/(prod+dense) {t_cold/(t_prod+t_dense):.0f}x, warm/(prod+sort) {t_warm/(t_prod+t_sort):.0f}x")
