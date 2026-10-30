"""Routing must retain evidence protection when sources or history are unknown."""
from copy import deepcopy
import json
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
    old = subprocess.check_output([
        "git", "show", "8e7eec35d6c888d86b5697a1310146f9fcbd2d9a:.github/workflows/ci.yml"
    ], cwd=ci.ROOT, text=True)
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


def test_document_edit_keeps_core_checks_without_spending_campaigns():
    gates = ci.load_registry()
    chosen = ci.affected(gates, ["README.md"])
    assert names(chosen) == names(g for g in gates if g["tier"] == "core")
    assert {"check_docs", "check_evidence", "paper_architecture_check_manuscript"} <= names(chosen)


@pytest.mark.parametrize("path", ["benchmarks/run_cross_domain_ratio.py",
                                  "docs/restart/cross_domain_ratio_results.json.gz",
                                  "tests/test_cross_domain_checker.py"])
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
                                  "benchmarks/new_unknown_helper.py", "unclassified.dat"])
def test_shared_unknown_or_archived_sources_fail_closed_to_all_campaigns(path):
    gates = ci.load_registry()
    chosen = names(ci.affected(gates, [path]))
    assert names(g for g in gates if g["tier"] != "environment") <= chosen


@pytest.mark.parametrize("path", ["benchmarks/ci_gates.json", ".github/workflows/ci.yml",
                                  "benchmarks/experiment_runner.py", ".gitignore"])
def test_transport_change_runs_all_deterministic_gates(path):
    gates = ci.load_registry()
    chosen = names(ci.affected(gates, [path]))
    assert names(g for g in gates if g["tier"] in {"core", "structural"}) <= chosen
    assert not names(g for g in gates if g["tier"] == "sampled") & chosen


def test_unavailable_git_ancestry_selects_full_replay_instead_of_skipping():
    result = ci.plan("pull_request", {})
    assert result["has_structural"] and result["has_sampled"]
    assert "replay_cross_domain_ratio" in names(result["sampled"]["include"])


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_scheduled_and_explicit_full_replay_cover_every_registered_campaign(event):
    gates = ci.load_registry()
    result = ci.plan(event, {})
    for tier in ("structural", "sampled"):
        assert names(result[tier]["include"]) == names(g for g in gates if g["tier"] == tier)


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


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "missing_script", "unknown_tier"])
def test_registry_rejects_corruption(tmp_path, mutation):
    registry = json.loads(ci.REGISTRY.read_text())
    if mutation == "duplicate": registry["gates"].append(deepcopy(registry["gates"][0]))
    elif mutation == "traversal": registry["gates"][0]["commands"] = [["../bad.py"]]
    elif mutation == "missing_script": registry["gates"][0]["commands"] = [["benchmarks/missing.py"]]
    elif mutation == "unknown_tier": registry["gates"][0]["tier"] = "quietly_skip"
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
    assert text.count("fetch-depth: 0") == 5
    assert text.count("fail-fast: false") == 2
    assert "reference-record: ${{ matrix.environment_record }}" in text
    assert text.count("actions/upload-artifact@v4") == 4
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
