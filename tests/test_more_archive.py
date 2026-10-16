"""MORE archives verify historical content without certifying Git chronology."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from benchmarks.archive_provenance import MANIFEST, history_kind
from benchmarks.check_phase17_more_declaration import history_problems
from benchmarks.metadata_io import binding, read_json
from benchmarks.phase17_more_design import CONFIG, RECORD, ROOT, digest

RECORD_PATH = RECORD.relative_to(ROOT)
CONFIG_PATH = CONFIG.relative_to(ROOT)


@pytest.fixture
def more_export():
    # TMPDIR may select a Git-free scratch root on hosts whose /tmp or
    # /workspace is itself beneath Git metadata.
    with tempfile.TemporaryDirectory(prefix='more-archive-') as directory:
        exported = Path(directory)
        paths = {Path(MANIFEST), RECORD_PATH, CONFIG_PATH}
        for package in ('clifford_qc', 'benchmarks'):
            paths.update(path.relative_to(ROOT) for path in (ROOT / package).rglob('*.py')
                         if 'provenance' not in path.parts)
        config = read_json(CONFIG)
        paths.add(Path(config['environment_reference']))
        paths.update(Path(case['path']) for case in config['cases'])
        manifest = read_json(ROOT / MANIFEST)
        entry = manifest['records'][str(RECORD_PATH)]
        paths.update(Path(item['snapshot']) for item in entry['sources'].values())
        for relative in paths:
            target = exported / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        assert history_kind(exported, archive=True) == 'archive'
        yield exported


def _problems(exported, *, archive=True):
    return history_problems(read_json(exported / CONFIG_PATH),
                            read_json(exported / RECORD_PATH), exported, archive=archive)


def _cli(exported, script, *arguments):
    environment = dict(os.environ, PYTHONPATH=str(exported))
    environment.update({key: '1' for key in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    return subprocess.run(
        [sys.executable, str(exported / 'benchmarks' / f'{script}.py'), *arguments],
        cwd=exported, env=environment, capture_output=True, text=True, timeout=90,
    )


@pytest.mark.parametrize('script', ['check_phase17_more_declaration', 'check_phase17_more'])
def test_more_gates_verify_export_only_with_explicit_archive_mode(more_export, script):
    done = _cli(more_export, script, '--archive')
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'commit chronology UNVERIFIED' in done.stdout
    assert 'preregistration ancestry is not verified' in done.stdout
    for arguments in ((), ('--require-git',)):
        strict = _cli(more_export, script, *arguments)
        assert strict.returncode != 0
        assert 'full Git history required' in strict.stderr
        assert 'content bindings verified' not in strict.stdout


@pytest.mark.parametrize('script', ['check_phase17_more_declaration', 'check_phase17_more'])
def test_more_archive_gates_require_the_published_record(more_export, script):
    (more_export / RECORD_PATH).unlink()
    done = _cli(more_export, script, '--archive')
    assert done.returncode == 1
    assert 'archive verification requires the published MORE record' in done.stderr
    assert 'content bindings verified' not in done.stdout
    assert 'outcome-free design only' not in done.stdout


def test_more_archive_uses_historical_sources_even_when_current_sources_evolve(more_export):
    (more_export / 'benchmarks/run_phase17_more.py').write_text('# later maintenance\n')
    assert _problems(more_export) == []


def test_more_export_refuses_an_already_published_campaign_before_git(more_export):
    original = (more_export / RECORD_PATH).read_bytes()
    done = _cli(more_export, 'run_phase17_more', '--execute')
    assert done.returncode == 1
    assert 'record exists; another campaign is not authorized' in done.stderr
    assert 'full Git history required' not in done.stderr
    assert (more_export / RECORD_PATH).read_bytes() == original


@pytest.mark.parametrize('change', ['record', 'config', 'source', 'missing_source',
                                   'manifest_source', 'declaration_binding'])
def test_more_archive_rejects_tampered_bound_content(more_export, change):
    manifest_path = more_export / MANIFEST
    manifest = read_json(manifest_path)
    entry = manifest['records'][str(RECORD_PATH)]
    source = next(iter(entry['sources'].values()))
    if change in ('record', 'config'):
        path = more_export / (RECORD_PATH if change == 'record' else CONFIG_PATH)
        path.write_bytes(path.read_bytes() + b'\n')
    elif change == 'source':
        (more_export / source['snapshot']).write_text('# changed historical source\n')
    elif change == 'missing_source':
        (more_export / source['snapshot']).unlink()
    elif change == 'manifest_source':
        source['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        path = more_export / RECORD_PATH
        record = read_json(path)
        record['config_sha256'] = '0' * 64
        path.write_text(json.dumps(record))
        # Updating the archive's record digest cannot hide a false declaration
        # binding carried by the record itself.
        entry['record_sha256'] = binding(path)['sha256']
        manifest_path.write_text(json.dumps(manifest))
    try:
        problems = _problems(more_export)
    except ValueError:
        pass
    else:
        assert problems


@pytest.mark.parametrize('change', ['timing', 'science', 'configuration'])
def test_more_archive_binds_the_actual_inspected_objects(more_export, change):
    config = read_json(more_export / CONFIG_PATH)
    record = read_json(more_export / RECORD_PATH)
    if change == 'timing':
        record['cells'][0]['worker']['prepare_seconds'] += 0.25
    elif change == 'science':
        record['cells'][0]['run']['root_energies'][0] += 0.25
    else:
        config = deepcopy(config)
        config['common_caps']['new_operator_directions'] += 1
        record['config_sha256'] = digest(config)
    assert history_problems(config, record, more_export, archive=True)


@pytest.mark.parametrize('archive', [False, True])
def test_more_archive_mode_cannot_bypass_incomplete_git_metadata(more_export, archive):
    (more_export / '.git').write_text('gitdir: missing-history\n')
    with pytest.raises(ValueError, match='unavailable/incomplete'):
        _problems(more_export, archive=archive)


@pytest.mark.parametrize('archive', [False, True])
def test_more_archive_mode_cannot_bypass_shallow_history(more_export, archive):
    # A real shallow boundary is enough: --archive must fail before considering
    # any packaged fallback, even if all source snapshots are present.
    for arguments in (
            ('init', '-b', 'main'), ('config', 'user.name', 'MORE archive test'),
            ('config', 'user.email', 'archive@example.invalid'),
            ('config', 'commit.gpgsign', 'false'), ('add', str(CONFIG_PATH)),
            ('commit', '-m', 'fixture')):
        subprocess.run(['git', *arguments], cwd=more_export,
                       capture_output=True, check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=more_export, text=True).strip()
    (more_export / '.git/shallow').write_text(head + '\n')
    with pytest.raises(ValueError, match='full git history'):
        _problems(more_export, archive=archive)


def test_more_archive_mode_cannot_bypass_a_complete_git_worktree(more_export):
    for arguments in (
            ('init', '-b', 'main'), ('config', 'user.name', 'MORE archive test'),
            ('config', 'user.email', 'archive@example.invalid'),
            ('config', 'commit.gpgsign', 'false'), ('add', str(CONFIG_PATH)),
            ('commit', '-m', 'fixture')):
        subprocess.run(['git', *arguments], cwd=more_export,
                       capture_output=True, check=True)
    with pytest.raises(ValueError, match='cannot bypass'):
        _problems(more_export)
