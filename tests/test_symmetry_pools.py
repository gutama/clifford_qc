"""Global symmetry and independent occupation-space completeness checks."""
import dataclasses
from functools import lru_cache
import json
import shutil
import subprocess

import numpy as np
import pytest

from clifford_qc.fermion import c_op, cdag_op, total_number_op, total_sz_op
from clifford_qc.matrix import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    Generator, MatrixElementBank, determinant_excitations, identity_generator,
)
from clifford_qc.subspace.symmetry_pools import (
    closed_shell_occupied_spatial, point_group_filter, spin_adapted_excitations,
)

PAIR_FCIDUMP = ('&FCI NORB=2, NELEC=2, MS2=0, ORBSYM=1,2, ISYM=1 /\n'
                '-0.8 1 1 0 0\n0.2 2 2 0 0\n0.1 2 1 2 1\n0.0 0 0 0 0\n')


def sites(n_spatial, ordering):
    """Independent (alpha, beta) qubits of each spatial orbital."""
    if ordering == "interleaved":
        return [(2*p, 2*p+1) for p in range(n_spatial)]
    return [(p, n_spatial+p) for p in range(n_spatial)]


def fock_matrix(n, operations):
    """Independent fermion signs: no Pauli/JW or MV multiplication."""
    matrix = np.zeros((1 << n, 1 << n), complex)
    for source in range(1 << n):
        word, sign = source, 1
        for site, creation in reversed(operations):
            mask = 1 << (n - 1 - site)
            if bool(word & mask) == creation:
                break
            sign *= (-1) ** ((word >> (n - site)).bit_count())
            word ^= mask
        else:
            matrix[word, source] = sign
    return matrix


@lru_cache(None)
def spin_matrices(n, ordering="interleaved"):
    pairs = sites(n // 2, ordering)
    up = {alpha for alpha, _ in pairs}
    plus = sum((fock_matrix(n, ((alpha, True), (beta, False))) for alpha, beta in pairs),
               np.zeros((1 << n, 1 << n), complex))
    z = np.diag([sum((.5 if site in up else -.5)
                     for site in range(n) if word & (1 << (n-1-site)))
                 for word in range(1 << n)])
    return plus, z @ z + .5 * (plus @ plus.conj().T + plus.conj().T @ plus)


def character(word, n, labels, ordering="interleaved"):
    value = 0
    for p, pair in enumerate(sites(n // 2, ordering)):
        for site in pair:
            if word & (1 << (n-1-site)):
                value ^= labels[p]
    return value


@pytest.mark.parametrize("n_spatial,occupied,labels,rank,ordering", [
    (2, (0,), (0, 0), 1, "interleaved"), (2, (0,), (0, 1), 2, "interleaved"),
    (3, (0,), (0, 0, 0), 2, "interleaved"), (4, (0, 1), (0, 0, 0, 0), 2, "interleaved"),
    (4, (0, 1), (0, 1, 0, 1), 2, "interleaved"), (4, (1, 3), (0, 1, 2, 3), 2, "interleaved"),
    (3, (1,), (0, 0, 0), 2, "blocked"), (4, (1, 3), (0, 1, 2, 3), 2, "blocked"),
])
def test_images_span_the_independent_point_group_singlet_cisd_space(
        n_spatial, occupied, labels, rank, ordering):
    n = 2 * n_spatial
    pairs = sites(n_spatial, ordering)
    reference = sum(1 << (n-1-site) for p in occupied for site in pairs[p])
    # Every same-N, Sz=0 determinant at excitation rank <= rank. S² is
    # independently built in occupation space, then restricted to this span.
    indices = [word for word in range(1 << n)
               if word.bit_count() == 2*len(occupied)
               and sum(bool(word & (1 << (n-1-alpha))) for alpha, _ in pairs) == len(occupied)
               and (word & ~reference).bit_count() <= rank
               and character(word, n, labels, ordering) == 0]
    _, s2 = spin_matrices(n, ordering)
    values, vectors = np.linalg.eigh(s2[np.ix_(indices, indices)])
    expected = np.eye(1 << n)[:, indices] @ vectors[:, abs(values) < 1e-10]
    pool = spin_adapted_excitations(n_spatial, occupied, max_rank=rank, orbital_irreps=labels,
                                    spin_ordering=ordering)
    images = np.column_stack([np.eye(1 << n)[:, reference]]
                             + [to_matrix(g.mv)[:, reference] for g in pool])
    u, singular, _ = np.linalg.svd(images, full_matrices=False)
    actual = u[:, singular > 1e-10]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual @ actual.conj().T, expected @ expected.conj().T, atol=2e-12)


@pytest.mark.parametrize("ordering", ["interleaved", "blocked"])
def test_every_generator_is_antihermitian_and_globally_spin_scalar(ordering):
    n = 6
    number, sz = total_number_op(n), total_sz_op(n, spin_ordering=ordering)
    plus, s2 = spin_matrices(n, ordering)
    for generator in spin_adapted_excitations(3, (0,), spin_ordering=ordering):
        g = generator.mv
        assert (g + g.dagger()).is_zero(1e-12)
        assert (g * number - number * g).is_zero(1e-12)
        assert (g * sz - sz * g).is_zero(1e-12)
        dense = to_matrix(g)
        np.testing.assert_allclose(dense @ plus, plus @ dense, atol=1e-12)
        np.testing.assert_allclose(dense @ s2, s2 @ dense, atol=1e-12)
    # Keeping alpha and beta excitations separate only conserves Sz.
    _, s2 = spin_matrices(n)
    raw = to_matrix(determinant_excitations(n, (0, 1), max_rank=1)[0].mv)
    assert np.linalg.norm(raw @ s2 - s2 @ raw) > 1


def test_forbidden_singles_still_form_allowed_pair_doubles_with_correct_factor():
    pool = spin_adapted_excitations(2, (0,), orbital_irreps=(0, 1))
    assert [g.label for g in pool] == ["SF(1<-0;1<-0)"]
    pair = 2 * fock_matrix(4, ((2, True), (3, True), (1, False), (0, False)))
    np.testing.assert_allclose(to_matrix(pool[0].mv), pair - pair.conj().T, atol=1e-12)


def _dense_point_group_oracle(pool, n, labels, ordering):
    parities = [np.diag([(-1) ** bool(character(word, n, labels, ordering) & bit)
                         for word in range(1 << n)]) for bit in (1, 2)]
    oracle = []
    for g in pool:
        dense = to_matrix(g.mv)
        if all(np.allclose(dense @ parity, parity @ dense, atol=1e-12) for parity in parities):
            oracle.append(g)
    return oracle


def test_point_group_filter_matches_global_dense_commutators_and_preserves_objects():
    labels = (0, 1, 2, 3)
    n = 8
    pool = determinant_excitations(n, (0, 1, 2, 3))
    selected = point_group_filter(pool, labels)
    oracle = _dense_point_group_oracle(pool, n, labels, "interleaved")
    assert 0 < len(selected) < len(pool)
    assert [id(g) for g in selected] == [id(g) for g in oracle]
    allowed = selected[0]
    forbidden = next(g for g in pool if id(g) not in {id(item) for item in selected})
    mixed = Generator("mixed", allowed.mv + forbidden.mv)
    assert point_group_filter([mixed], labels) == []


def test_blocked_point_group_filter_uses_the_declared_spin_pairs():
    labels = (0, 1, 2, 3)
    n = 8
    # Spatial orbitals 0 and 1 doubly occupied under blocked ordering.
    pool = determinant_excitations(n, (0, 1, 4, 5), conserve_sz=False)
    oracle = _dense_point_group_oracle(pool, n, labels, "blocked")
    selected = point_group_filter(pool, labels, spin_ordering="blocked")
    assert 0 < len(selected) < len(pool)
    assert [id(g) for g in selected] == [id(g) for g in oracle]
    # Reading the same blocked operators as interleaved keeps forbidden ones:
    # E(2,6<-0,5) touches blocked orbitals 0,1,2,2 (character 1) but would be
    # read as orbitals 0,2,1,3 (character 0), and drops allowed ones.
    misread = {g.label for g in point_group_filter(pool, labels)}
    assert 'E(2,6<-0,5)' in misread
    assert 'E(2,6<-0,5)' not in {g.label for g in selected}
    assert misread != {g.label for g in selected}
    with pytest.raises(ValueError, match="spin_ordering"):
        point_group_filter(pool, labels, spin_ordering="Blocked")
    with pytest.raises(ValueError, match="spin_ordering"):
        spin_adapted_excitations(4, (0,), spin_ordering=("up", "down") * 4)


def test_optional_hamiltonian_guard_rejects_a_broken_declared_symmetry_even_for_empty_pool():
    transfer = cdag_op(4, 2) * c_op(4, 0)
    h = transfer + transfer.dagger()
    with pytest.raises(ValueError, match="Hamiltonian does not preserve"):
        point_group_filter([], (0, 1), hamiltonian=h)
    with pytest.raises(ValueError, match="Hamiltonian does not preserve"):
        spin_adapted_excitations(2, (0,), orbital_irreps=(0, 1), hamiltonian=h)
    assert point_group_filter([], (0, 0), hamiltonian=h) == []
    # Without characters the guard still checks the register, not the terms.
    with pytest.raises(ValueError, match="two spin orbitals"):
        spin_adapted_excitations(3, (0,), hamiltonian=h)


def test_hamiltonian_guard_tolerance_admits_only_declared_integral_noise():
    n = 4
    number = [cdag_op(n, p) * c_op(n, p) for p in range(n)]
    transfer = cdag_op(n, 2) * c_op(n, 0)
    noise = 1e-11 * (transfer + transfer.dagger())
    h = -.8*(number[0]+number[1]) + .2*(number[2]+number[3]) + noise
    with pytest.raises(ValueError, match=r"largest forbidden \|coefficient\| is 5e-12"):
        point_group_filter([], (0, 1), hamiltonian=h)
    assert point_group_filter([], (0, 1), hamiltonian=h, hamiltonian_atol=1e-10) == []
    pool = spin_adapted_excitations(2, (0,), orbital_irreps=(0, 1), hamiltonian=h,
                                    hamiltonian_atol=1e-10)
    assert [g.label for g in pool] == ["SF(1<-0;1<-0)"]
    # The tolerance applies to the Hamiltonian check only, never to generators.
    leaky = Generator("leaky", 1e-11 * (transfer - transfer.dagger()))
    assert point_group_filter([leaky], (0, 1), hamiltonian_atol=1.0) == []
    for bad in (-1e-12, float("nan"), float("inf"), True, "1e-10"):
        with pytest.raises(ValueError, match="hamiltonian_atol"):
            point_group_filter([], (0, 1), hamiltonian=h, hamiltonian_atol=bad)


def test_closed_shell_spatial_occupation_is_read_not_assumed():
    assert closed_shell_occupied_spatial((2, 3, 6, 7), 4) == (1, 3)
    assert closed_shell_occupied_spatial((np.int64(5), 1, 4, 0), 4,
                                         spin_ordering="blocked") == (0, 1)
    assert closed_shell_occupied_spatial((), 2) == ()
    for occupied, message in [((0, 3), "not closed shell"), ((0, 0, 1), "distinct"),
                              ((0, 1, 8, 9), "in range"), ((-1,), "occupied spin orbital")]:
        with pytest.raises(ValueError, match=message):
            closed_shell_occupied_spatial(occupied, 4)
    with pytest.raises(ValueError, match="not closed shell"):
        closed_shell_occupied_spatial((0, 1), 4, spin_ordering="blocked")


def test_closed_shell_pair_model_preserves_energy_and_counts_the_complete_bank():
    n = 4
    number = [cdag_op(n, p) * c_op(n, p) for p in range(n)]
    pair = cdag_op(n, 2) * cdag_op(n, 3) * c_op(n, 1) * c_op(n, 0)
    h = -.8*(number[0]+number[1]) + .2*(number[2]+number[3]) - .4*(pair+pair.dagger())
    rho = ket_density(n, "1100")
    raw = determinant_excitations(n, (0, 1))
    adapted = spin_adapted_excitations(2, (0,), orbital_irreps=(0, 1), hamiltonian=h)
    assert (len(raw), len(adapted)) == (3, 1)
    banks = [MatrixElementBank(rho, h, [identity_generator(n)] + pool) for pool in (raw, adapted)]
    results = [bank.solve() for bank in banks]
    sector = [word for word in range(1 << n) if word.bit_count() == 2]
    exact = np.linalg.eigvalsh(to_matrix(h)[np.ix_(sector, sector)])[0]
    for result in results:
        assert result.ground_energy == pytest.approx(exact, abs=1e-12)
    counts = [bank.resources() for bank in banks]
    assert [r['pairs_built'] for r in counts] == [10, 3]
    assert all(r['word_universe'] == r['resident_word_universe'] for r in counts)
    assert counts[1]['word_universe'] < counts[0]['word_universe']


@pytest.mark.parametrize("kwargs", [
    {'n_spatial': 0}, {'n_spatial': True}, {'n_spatial': 2.5},
    {'occupied_spatial': (0, 0)}, {'occupied_spatial': (-1,)},
    {'occupied_spatial': (3,)}, {'occupied_spatial': (True,)},
    {'max_rank': 0}, {'max_rank': 3}, {'max_rank': True}, {'max_rank': 1.5},
    {'orbital_irreps': ()}, {'orbital_irreps': (0,)},
    {'orbital_irreps': (0, -1, 0)}, {'orbital_irreps': (0, True, 0)},
    {'orbital_irreps': (0, 1.5, 0)}, {'spin_ordering': 'qubits'},
    {'hamiltonian_atol': -1.0},
])
def test_invalid_orbital_partitions_and_character_conventions_are_rejected(kwargs):
    inputs = dict(n_spatial=3, occupied_spatial=(0,))
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        spin_adapted_excitations(**inputs)


def test_numpy_integers_empty_partitions_and_deterministic_orbital_order():
    one = spin_adapted_excitations(np.int64(4), (np.int64(2), np.int64(0)), max_rank=np.int64(2))
    two = spin_adapted_excitations(4, (0, 2), orbital_irreps=(0, 0, 0, 0))
    assert [(g.label, g.mv.terms) for g in one] == [(g.label, g.mv.terms) for g in two]
    assert spin_adapted_excitations(3, ()) == []
    assert spin_adapted_excitations(3, (0, 1, 2)) == []


def test_filter_requires_matching_spin_register_and_nonempty_integer_characters():
    with pytest.raises(ValueError, match="two spin orbitals"):
        point_group_filter([Generator('bad', MV.scalar(3, 1))], (0, 1))
    with pytest.raises(ValueError, match="one irrep"):
        point_group_filter([], ())
    with pytest.raises(ValueError, match="orbital irrep"):
        point_group_filter([], (np.bool_(True),))


def test_structural_audit_counts_screening_and_complete_candidate_banks(tmp_path, monkeypatch):
    from benchmarks import benchmark_symmetry_pools as audit
    from benchmarks.benchmark_symmetry_pools import build_report
    path = tmp_path/'pair.FCIDUMP'
    path.write_text(PAIR_FCIDUMP)
    report = build_report(path, orbital_irreps=(0, 1))
    assert report['physical_shots'] == 0 and report['finite_shot_cost_claim'] is False
    assert report['fcidump_orbsym_inferred'] is False
    assert report['orbital_irreps'] == [0, 1]
    assert report['spin_ordering'] == 'interleaved' and report['occupied_spatial'] == [0]
    assert report['hamiltonian_atol'] == 0.0
    assert report['named_sources_hashed_before_import'] is False
    assert [row['arm'] for row in report['rows']] == [
        'number_sz', 'spin_free', 'number_sz_point_group', 'spin_free_point_group']
    assert [row['generators_excluding_identity'] for row in report['rows']] == [3, 2, 1, 1]
    assert [row['initial_screen']['pairs_built'] for row in report['rows']] == [7, 5, 3, 3]
    assert [row['complete_candidate_bank']['pairs_built'] for row in report['rows']] == [10, 6, 3, 3]
    assert all(row['energy'] == pytest.approx(report['rows'][0]['energy'], abs=1e-12) for row in report['rows'])
    for row in report['rows']:
        # Stage timings partition the run; the completion stage no longer
        # carries the screen's grouping.
        assert row['total_seconds'] == pytest.approx(
            row['initial_screen_seconds'] + row['completion_seconds'], rel=1e-9, abs=1e-12)
        assert min(row['initial_screen_seconds'], row['completion_seconds']) >= 0
    assert set(report['named_source_sha256']) >= {'clifford_qc/subspace/symmetry_pools.py'}
    # A snapshot that no longer matches disk means the loaded code is unbound.
    stale = dict(report['named_source_sha256'], **{'clifford_qc/subspace/symmetry_pools.py': '0'*64})
    with pytest.raises(RuntimeError, match='source changed after import'):
        build_report(path, loaded_sources=stale)
    bound = build_report(path, loaded_sources=report['named_source_sha256'])
    assert bound['named_sources_hashed_before_import'] is True
    # Changing a bound file during real matrix assembly must prevent a report.
    source = tmp_path/'execution.py'
    input_text = path.read_text()
    original_ledger = audit._ledger
    for changed in ('source', 'input'):
        source.write_text('before')
        path.write_text(input_text)
        def mutate(bank):
            ledger = original_ledger(bank)
            if changed == 'source':
                source.write_text('after')
            else:
                path.write_text(input_text.replace('-0.8', '-0.7'))
            return ledger
        with monkeypatch.context() as patch:
            patch.setattr(audit, 'ROOT', tmp_path)
            patch.setattr(audit, 'NAMED_SOURCES', ('execution.py',))
            patch.setattr(audit, '_ledger', mutate)
            with pytest.raises(RuntimeError, match=changed+' changed during audit'):
                build_report(path, orbital_irreps=(0, 1))


def test_structural_audit_excites_the_reference_it_was_given(tmp_path, monkeypatch):
    """Both arms must use the model's determinant, not an assumed prefix."""
    from benchmarks import benchmark_symmetry_pools as audit
    from clifford_qc.models.fcidump import _reference_program
    path = tmp_path/'tri.FCIDUMP'
    path.write_text('&FCI NORB=3, NELEC=2, MS2=0, ORBSYM=1,1,1, ISYM=1 /\n'
                    '0.6 1 1 1 1\n0.5 2 2 2 2\n0.4 3 3 3 3\n0.1 2 1 2 1\n0.1 3 1 3 1\n'
                    '0.12 3 2 3 2\n0.3 2 2 1 1\n0.3 3 3 1 1\n0.3 3 3 2 2\n'
                    '-0.9 1 1 0 0\n-1.0 2 2 0 0\n0.3 3 3 0 0\n0.05 3 2 0 0\n0.0 0 0 0 0\n')
    load = audit.fcidump_model
    model = load(path)
    # Orbital 1 doubly occupied: not the aufbau prefix. Assuming one excites
    # from empty orbital 0, so transfers into orbital 2 annihilate this
    # reference and half the singlet space is lost.
    monkeypatch.setattr(audit, 'fcidump_model', lambda p: dataclasses.replace(
        load(p), reference=_reference_program(6, (2, 3))))
    report = audit.build_report(path)
    assert report['occupied_spatial'] == [1]
    spin_free = report['rows'][1]
    # With two electrons singlet CISD is the whole six-dimensional singlet
    # space, so the arm must reach its exact ground energy. (The number/Sz arm
    # also reaches a lower triplet here, so it is not the comparator.)
    h = to_matrix(model.hamiltonian.to_mv())
    _, s2 = spin_matrices(6)
    sector = [word for word in range(1 << 6) if word.bit_count() == 2]
    values, vectors = np.linalg.eigh(s2[np.ix_(sector, sector)])
    singlets = vectors[:, abs(values) < 1e-10]
    assert singlets.shape[1] == spin_free['retained_rank'] == 6
    exact = np.linalg.eigvalsh(singlets.conj().T @ h[np.ix_(sector, sector)] @ singlets)[0]
    assert spin_free['energy'] == pytest.approx(exact, abs=1e-10)


def test_audit_cli_writes_strict_json_once(tmp_path):
    from benchmarks.benchmark_symmetry_pools import main
    path = tmp_path/'pair.FCIDUMP'
    path.write_text(PAIR_FCIDUMP)
    output = tmp_path/'nested'/'audit.json'
    main(['--fcidump', str(path), '--orbital-irreps', '0', '1',
          '--hamiltonian-atol', '1e-10', '--output', str(output)])
    record = json.loads(output.read_text())
    assert record['hamiltonian_atol'] == 1e-10 and len(record['rows']) == 4
    with pytest.raises(FileExistsError):
        main(['--fcidump', str(path), '--output', str(output)])
    with pytest.raises(SystemExit):
        main(['--fcidump', str(path), '--hamiltonian-atol', '-1', '--output', str(tmp_path/'x')])


@pytest.mark.skipif(shutil.which('git') is None, reason='git executable unavailable')
def test_exported_audit_tree_cannot_borrow_a_parent_repository_commit(tmp_path):
    from benchmarks.benchmark_symmetry_pools import _git_provenance
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    '-c', 'commit.gpgsign=false',
                    'commit', '-qm', 'parent', '--allow-empty'], cwd=tmp_path, check=True)
    archive = tmp_path/'archive'
    archive.mkdir()
    (archive/'source.txt').write_text('exported source')
    assert _git_provenance(tmp_path)['git_provenance_verified'] is True
    assert _git_provenance(archive) == {
        'git_head': None, 'dirty_tree': None, 'git_provenance_verified': False}
