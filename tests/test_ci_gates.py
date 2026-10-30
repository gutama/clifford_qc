"""Routing must retain evidence protection when sources or history are unknown."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import re
import subprocess

import pytest

from benchmarks import ci_gates as ci


def names(rows):
    return {g["gate"] for g in rows}


def test_registry_preserves_every_previous_command_environment_and_timeout():
    # Immutable pre-extraction policy. A later intentional migration can revise
    # this acceptance boundary explicitly; an accidental omission fails here.
    raw = (ci.ROOT/'tests/data/ci_before_shared_runner.yml').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'f4179bf1a144b7309fc1a458de5d36eddbc5ea88d9131c175a7d8c5457ebc5ec'
    old = raw.decode()
    registry = ci.load_registry()
    commands = {tuple(command) for g in registry for command in g["commands"]}
    direct = {tuple(line.split()[1:]) for line in old.splitlines()
              if re.match(r"\s*(?:- run: )?python (?:benchmarks|paper_)", line) and "${{" not in line
              for line in [line.strip().removeprefix("- run: ")]}
    assert direct <= commands
    entries = re.findall(r"- gate: (\w+)\s+timeout: (\d+)\s+environment_record: (\S+)", old)
    by_name = {g["gate"]: g for g in registry}
    for name, timeout, environment in entries:
        assert by_name[name]["timeout"] == int(timeout)
        assert by_name[name]["environment_record"] == environment
        assert by_name[name]["commands"] == [[f"benchmarks/{name}.py"]]
    assert len(entries) > 40  # A parser returning nothing cannot pass vacuously.
    assert {"replay_sampled_ratio", "replay_cross_domain_ratio"} <= names(registry)


def test_document_edit_keeps_deterministic_checks_without_sampled_replays():
    gates = ci.load_registry()
    chosen = ci.affected(gates, ["README.md"])
    assert names(chosen) == names(g for g in gates if g["tier"] in {"core", "structural"})
    assert {"check_docs", "check_evidence", "paper_architecture_check_manuscript"} <= names(chosen)


@pytest.mark.parametrize("path", ["benchmarks/run_cross_domain_ratio.py",
                                  "docs/restart/cross_domain_ratio_results.json.gz"])
def test_changed_sampled_sources_and_evidence_select_their_required_replay(path):
    chosen = names(ci.affected(ci.load_registry(), [path]))
    assert "check_cross_domain_ratio" in chosen
    assert "replay_cross_domain_ratio" in chosen


def test_changed_imported_runner_selects_its_other_campaign_consumer():
    chosen = names(ci.affected(ci.load_registry(), ["benchmarks/run_sampled_ratio.py"]))
    # Cross-domain replay imports the spin runner's joint sampler; matching
    # only a campaign basename would miss this edge.
    assert "replay_cross_domain_ratio" in chosen


def test_deleted_helper_cannot_erase_its_dependency_edge():
    gates = ci.load_registry()
    chosen = names(ci.affected(gates, ["benchmarks/removed_phase15_helper.py"]))
    assert names(g for g in gates if g["tier"] != "environment") <= chosen


@pytest.mark.parametrize("path", ["clifford_qc/pauli.py", "benchmarks/provenance/archive_manifest.json",
                                  "benchmarks/new_unknown_helper.py"])
def test_shared_unknown_or_archived_sources_fail_closed_to_all_campaigns(path):
    gates = ci.load_registry()
    chosen = names(ci.affected(gates, [path]))
    assert names(g for g in gates if g["tier"] != "environment") <= chosen


@pytest.mark.parametrize("path", ["benchmarks/ci_gates.json", ".github/workflows/ci.yml",
                                  "benchmarks/experiment_runner.py"])
def test_transport_change_runs_all_deterministic_gates(path):
    gates = ci.load_registry()
    chosen = names(ci.affected(gates, [path]))
    assert names(g for g in gates if g["tier"] in {"core", "structural"}) <= chosen
    assert {"replay_sampled_ratio", "replay_cross_domain_ratio"} <= chosen


def test_unavailable_git_ancestry_selects_full_replay_instead_of_skipping():
    result = ci.plan("pull_request", {})
    assert result["has_structural"] and result["has_sampled"]
    assert "replay_cross_domain_ratio" in names(result["sampled"]["include"])


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_scheduled_and_explicit_full_replay_cover_every_registered_campaign(event):
    gates = ci.load_registry()
    result = ci.plan(event, {})
    for tier in ("structural", "sampled"):
        assert names(result[tier]["include"]) == names(g for g in gates if g["tier"] == tier and (event == "workflow_dispatch" or not g.get("manual_only", False)))


def test_deleted_and_renamed_evidence_paths_are_both_routed(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    source = tmp_path/"benchmarks/reference_results"
    source.mkdir(parents=True)
    (source/"cross_domain_ratio.json").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "before")
    before = git("rev-parse", "HEAD")
    (source/"cross_domain_ratio.json").rename(source/"unrecognized.json")
    git("add", "-A")
    git("commit", "-qm", "rename")
    changed = ci.changed_paths(tmp_path, before, git("rev-parse", "HEAD"))
    assert len(changed) == 2
    assert "replay_cross_domain_ratio" in names(ci.affected(ci.load_registry(), changed))


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "missing_script", "unknown_tier", "manual_contract"])
def test_registry_rejects_corruption(tmp_path, mutation):
    registry = json.loads(ci.REGISTRY.read_text())
    if mutation == "duplicate": registry["gates"].append(deepcopy(registry["gates"][0]))
    elif mutation == "traversal": registry["gates"][0]["commands"] = [["../bad.py"]]
    elif mutation == "missing_script": registry["gates"][0]["commands"] = [["benchmarks/missing.py"]]
    elif mutation == "unknown_tier": registry["gates"][0]["tier"] = "quietly_skip"
    elif mutation == "manual_contract":
        next(g for g in registry['gates'] if g['gate']=='check_r3d_qr3_refinement')['manual_only'] = False
    path = tmp_path/"registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        ci.load_registry(path)


def test_workflow_uses_the_registry_and_retains_history_pins_failure_logs_and_schedule():
    text = (ci.ROOT/".github/workflows/ci.yml").read_text()
    assert 'cron:' in text
    assert "--plan" in text and "fromJSON(needs.plan.outputs.structural)" in text
    assert "fromJSON(needs.plan.outputs.sampled)" in text
    assert "python -m pytest --hypothesis-profile=ci" in text
    assert "--tier core" in text
    assert text.count("fetch-depth: 0") == 4
    assert text.count("fail-fast: false") == 2
    assert "reference-record: ${{ matrix.environment_record }}" in text
    assert text.count("actions/upload-artifact@v4") == 3
    assert "pull_request_target" not in text


def test_full_molecular_replay_uses_its_exact_declared_runtime():
    gate = next(g for g in ci.load_registry() if g["gate"] == "replay_cross_domain_ratio")
    value = ci.execution_environment(gate["execution_declaration"])
    assert value == {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0", "threads": 1}
    matrix = ci.plan("workflow_dispatch", {})["sampled"]["include"]
    row = next(g for g in matrix if g["gate"] == gate["gate"])
    assert row["execution_declaration"] == gate["execution_declaration"]
    action = (ci.ROOT/".github/actions/record-environment/action.yml").read_text()
    assert action.count("RECORD_EXECUTION_DECLARATION: ${{ inputs.execution-declaration }}") == 3
    assert "--emit python" in action and "--emit constraints" in action and "--emit verify" in action


def test_replay_environment_rejects_mismatched_versions_and_thread_settings(monkeypatch):
    value = {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0", "threads": 1}
    monkeypatch.setattr(ci.platform, "python_version", lambda: value["python"])
    monkeypatch.setattr(ci.importlib.metadata, "version", lambda name: value[name])
    for p in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(p, "1")
    ci.verify_execution_environment(value)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="thread"):
        ci.verify_execution_environment(value)
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(ci.platform, "python_version", lambda: "3.12.99")
    with pytest.raises(ValueError, match="runtime"):
        ci.verify_execution_environment(value)


@pytest.mark.parametrize('record', ['protocol_cost.json', 'second_moment_validation.json'])
def test_environment_pin_changes_select_every_record_consumer(record):
    gates = ci.load_registry()
    path = 'benchmarks/reference_results/'+record
    chosen = names(ci.affected(gates, [path]))
    assert names(g for g in gates if g['environment_record'] == record) <= chosen
    assert all(path in ci.gate_inputs(g) for g in gates if g['environment_record'] == record)


@pytest.mark.parametrize('path', ['benchmarks/check_record_environment.py',
                                 'benchmarks/experiment_runner.py',
                                 '.github/actions/record-environment/action.yml',
                                 '.github/workflows/ci.yml'])
def test_environment_and_transport_edits_exercise_both_real_replay_paths(path):
    chosen = ci.affected(ci.load_registry(), [path])
    assert {'replay_sampled_ratio', 'replay_cross_domain_ratio'} <= names(chosen)
    assert 'check_record_environment' in names(chosen)


@pytest.mark.parametrize('path', ['docs/restart/sampled_ratio_results.json',
                                 'docs/restart/sampled_ratio_declaration.json',
                                 'benchmarks/run_sampled_ratio.py'])
def test_cross_domain_literal_byte_bindings_are_explicit_dependencies(path):
    gates = ci.load_registry()
    for name in ('check_cross_domain_ratio', 'replay_cross_domain_ratio'):
        gate = next(g for g in gates if g['gate'] == name)
        assert path in ci.gate_inputs(gate)
        assert name in names(ci.affected(gates, [path]))


@pytest.mark.parametrize('path', ['PHASE_STATUS.json', 'molecular/README.md',
                                 'paper/manuscript.tex', 'CITATION.cff', 'examples/new.py',
                                 'tests/test_protocol_cost.py', '.gitignore'])
def test_routine_edits_do_not_spend_sampled_replays(path):
    chosen = ci.affected(ci.load_registry(), [path])
    assert not any(g['tier'] == 'sampled' for g in chosen)


@pytest.mark.parametrize('event', ['pull_request', 'push', 'schedule'])
def test_frozen_manual_only_replays_are_never_automatic(event):
    result = ci.plan(event, {})
    assert names(result['sampled']['include']) == {'replay_sampled_ratio', 'replay_cross_domain_ratio'}
    assert 'check_r3d_qr3_refinement' not in names(result['sampled']['include'])
    config = json.loads((ci.ROOT/'benchmarks/configs/r3d_qr3_refinement.json').read_text())
    assert config['reporting_contract']['sampled_ci_is_manual_dispatch_only'] is True


def test_main_runs_are_unique_and_uploads_run_after_cancellation():
    text = (ci.ROOT/'.github/workflows/ci.yml').read_text()
    assert "github.event_name == 'pull_request' && github.ref || github.run_id" in text
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text
    assert text.count('if: ${{ always() }}') == 3
    assert text.count('CI_JOB_STARTED_EPOCH=$(date +%s)') == 3


def test_executor_uses_per_gate_logs_and_elapsed_job_budget(tmp_path, monkeypatch):
    captured = {}
    def fake_run(tasks, **kwargs):
        captured.update(kwargs, tasks=tasks)
        return {'summary': {'failed': 0}}
    monkeypatch.setattr(ci, 'run', fake_run)
    monkeypatch.setattr(ci.subprocess, 'check_output', lambda *a, **kw: b'')
    monkeypatch.setenv('CI_JOB_STARTED_EPOCH', '100')
    monkeypatch.setenv('CI_JOB_TIMEOUT_SECONDS', '1800')
    gate = next(g for g in ci.load_registry() if g['gate'] == 'replay_cross_domain_ratio')
    ci.execute([gate], output=tmp_path/'run')
    assert captured['deadline_epoch'] == 1720
    assert captured['tasks'][0].max_output_bytes == gate['max_output_bytes'] == 64*2**20


def test_baseline_inventory_works_without_git_history(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    test_registry_preserves_every_previous_command_environment_and_timeout()


@pytest.mark.parametrize('path', ['benchmarks/run_protocol_cost.py',
                                 'benchmarks/run_exact_shot_search.py',
                                 'benchmarks/configs/r3c_lih_full_cost.json'])
def test_r3d_non_imported_source_bindings_keep_the_preregistration_gate(path):
    gates = ci.load_registry()
    gate = next(g for g in gates if g['gate'] == 'check_r3d_preregistration')
    assert path in ci.gate_inputs(gate)
    assert gate['gate'] in names(ci.affected(gates, [path]))
