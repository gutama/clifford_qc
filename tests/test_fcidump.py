"""Core, dependency-free checks for the FCIDUMP interchange boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models import fcidump_model, read_fcidump

DATA = (Path(__file__).parents[1] / "benchmarks" / "data"
        / "h4_sto3g_r0.9.FCIDUMP")
H4_RHF = -2.124259738972812
H4_FCI = -2.180316614323862


def test_committed_h4_fcidump_is_a_real_active_space_oracle():
    data = read_fcidump(DATA)
    assert (data.n_orbitals, data.n_electrons, data.ms2) == (4, 4, 0)
    assert data.orbsym == (1, 1, 1, 1)
    assert data.core_energy == pytest.approx(2.547890274800001)
    assert data.source_sha256 == (
        "bf32568461477cf1ec8e7c4aaae737b0c6c80bcbd83d51f783ec3a3daed5a922")

    model = fcidump_model(DATA, name="H4 CAS(4e,4o)")
    assert model.n == 8
    assert len(model.hamiltonian.terms) == 185
    assert model.hamiltonian.is_hermitian()
    assert model.metadata["reference_occupied_spin_orbitals"] == [0, 1, 2, 3]
    hf = ExactMVBackend().expectation(model.reference, model.hamiltonian, ())
    assert hf == pytest.approx(H4_RHF, abs=1e-11)

    sector = SectorStatevectorBackend(model.n, n_electrons=4, sz=0.0)
    values, _ = sector.ground_state(model.hamiltonian, k=4, method="dense")
    assert sector.dimension == 36
    assert values[0] == pytest.approx(H4_FCI, abs=1e-11)
    assert np.all(np.diff(values) >= -1e-12)


def test_packed_integrals_restore_all_real_orbital_symmetries():
    data = read_fcidump(DATA)
    h, eri = data.one_body, data.two_body
    assert np.allclose(h, h.T, atol=0.0)
    assert np.allclose(eri, eri.transpose(1, 0, 2, 3), atol=0.0)
    assert np.allclose(eri, eri.transpose(0, 1, 3, 2), atol=0.0)
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=0.0)


def test_source_digest_is_stable_across_line_endings(tmp_path):
    """Scientific provenance must not depend on Git's Windows checkout mode."""
    text = DATA.read_text(encoding="utf-8")
    windows = tmp_path / "windows.FCIDUMP"
    windows.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert read_fcidump(windows).source_sha256 == read_fcidump(DATA).source_sha256


def test_fortran_exponents_and_open_shell_reference_are_supported(tmp_path):
    path = tmp_path / "open_shell.FCIDUMP"
    path.write_text(
        "&FCI NORB=2, NELEC=1, MS2=1, ORBSYM=1,1, ISYM=1, &END\n"
        "-1.0D+00 1 1 0 0\n"
        "-5.0D-01 2 2 0 0\n"
        "0.0 0 0 0 0\n",
        encoding="utf-8")
    model = fcidump_model(path)
    assert model.metadata["n_electrons"] == 1
    assert model.metadata["sz"] == pytest.approx(0.5)
    assert model.metadata["reference_occupied_spin_orbitals"] == [0]
    assert ExactMVBackend().expectation(
        model.reference, model.hamiltonian, ()) == pytest.approx(-1.0)


def test_sector_overrides_must_be_explicit_integers():
    with pytest.raises(TypeError, match="n_electrons"):
        fcidump_model(DATA, n_electrons=2.5)
    with pytest.raises(TypeError, match="ms2"):
        fcidump_model(DATA, ms2=False)


@pytest.mark.parametrize("body, message, error", [
    (
        "&FCI NORB=2,NELEC=2,MS2=0,IUHF=1,&END\n0.0 0 0 0 0\n",
        "unrestricted",
        NotImplementedError,
    ),
    (
        "&FCI NORB=2,NELEC=2,MS2=0,&END\n1.0 1 0 0 0\n",
        "zero-index sentinel",
        ValueError,
    ),
    (
        "&FCI NORB=2,NELEC=2,MS2=0,&END\n"
        "1.0 1 1 0 0\n2.0 1 1 0 0\n",
        "conflicts",
        ValueError,
    ),
    (
        "&FCI NORB=2,NELEC=2,MS2=1,&END\n0.0 0 0 0 0\n",
        "integer spin sector",
        ValueError,
    ),
])
def test_invalid_or_ambiguous_fcidump_is_rejected(tmp_path, body, message, error):
    path = tmp_path / "bad.FCIDUMP"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(error, match=message):
        read_fcidump(path)


def test_active_space_benchmark_reproduces_the_committed_record():
    root = Path(__file__).parents[1]
    path = root / "benchmarks" / "run_fcidump_h4.py"
    spec = importlib.util.spec_from_file_location("run_fcidump_h4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    actual = module.build_record()
    expected = json.loads(
        (root / "benchmarks" / "reference_results" / "fcidump_h4.json")
        .read_text(encoding="utf-8"))
    assert actual["input"] == expected["input"]
    assert actual["system"] == expected["system"]
    assert actual["evidence"] == expected["evidence"]
    assert actual["adaptive_acase"]["labels"] == expected["adaptive_acase"]["labels"]
    assert actual["adaptive_acase"]["word_universe"] == 7371
    assert actual["adaptive_acase"]["ground_energy"] == pytest.approx(
        expected["adaptive_acase"]["ground_energy"], abs=1e-10)
    assert actual["complete_singles_doubles"]["ground_energy"] == pytest.approx(
        expected["complete_singles_doubles"]["ground_energy"], abs=1e-10)
    assert actual["mapped_oracle"]["ground_energy"] == pytest.approx(
        expected["mapped_oracle"]["ground_energy"], abs=1e-10)
