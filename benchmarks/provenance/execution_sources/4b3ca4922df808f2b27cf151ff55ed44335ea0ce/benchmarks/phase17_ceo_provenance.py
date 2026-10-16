"""Historical source bindings and explicit Git lineage evidence for CEO records."""
import json
import subprocess

from benchmarks.check_phase15_preregistration import _first_commit, _last_commit, _shallow_boundary
from benchmarks.phase17_ceo_design import digest
from benchmarks.archive_provenance import commit_binding, HistoryError


def _git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, timeout=30)


def history_problems(config, record, root, config_path, record_path, notes):
    """Audit recorded sources; current core modules may evolve after execution.

    Missing history in an export or shallow clone is explicitly unverified.
    Missing commits/paths in a complete repository are failures.
    """
    try:
        if _git(root, 'rev-parse', '--show-toplevel').returncode:
            notes.append('provenance: SKIP (export without Git history)')
            return []
        source = record['provenance']['git_sha']
        boundary = _shallow_boundary(root=root)
        available = _git(root, 'cat-file', '-e', source+'^{commit}').returncode == 0
        config_commit = _last_commit(config_path, root=root)
        record_commit = _first_commit(record_path, root=root)
        if boundary and (not available or {config_commit, record_commit} & boundary):
            notes.append('provenance: SKIP (shallow history; lineage/source bindings unverified)')
            return []
        problems = []
        if not available:
            return ['execution commit missing from complete history; preserve merge ancestry']
        if config_commit is None:
            problems.append('declaration has no committed history')
        elif config_commit == source or _git(root, 'merge-base', '--is-ancestor', config_commit, source).returncode:
            problems.append('declaration last change must strictly precede execution')
        if record_commit is None:
            problems.append('record has no committed adding history')
        elif record_commit == source or _git(root, 'merge-base', '--is-ancestor', source, record_commit).returncode:
            problems.append('record must strictly follow execution source')
        historical = _git(root, 'show', f'{source}:{config_path}')
        if historical.returncode or digest(json.loads(historical.stdout)) != digest(config):
            problems.append('execution commit lacks frozen declaration')
        bindings = config['source_bindings'] | record['execution_source_bindings']
        for path, expected in bindings.items():
            if commit_binding(root, source, path)['sha256'] != expected:
                problems.append('historical source binding mismatch: '+path)
        if record['provenance']['git_dirty']:
            problems.append('dirty execution')
        if not problems:
            notes.append('provenance: verified declaration last change < execution < record')
        return problems
    except OSError as error:
        notes.append(f'provenance: SKIP (Git unavailable: {type(error).__name__})')
        return []
    except subprocess.SubprocessError as error:
        return [f'Git provenance audit failed: {type(error).__name__}']
    except HistoryError as error:
        return [f'Git provenance audit failed: {error}']
