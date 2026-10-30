"""Transport invariants: failures, crashes and corruption cannot become success."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from benchmarks import experiment_runner as runner
from benchmarks.experiment_artifacts import (
    ArtifactStore, decode_json, json_bytes, local_path, read_json, write_json,
)


def command(key, code, **kwargs):
    return runner.Task(key, (sys.executable, "-c", code), **kwargs)


def test_byte_addressed_artifacts_preserve_existing_json_and_detect_corruption(tmp_path):
    store = ArtifactStore(tmp_path)
    raw = b'{"signed_zero": -0.0, "historical_spacing":  3}\n'
    ref = store.put(raw)
    assert store.get(ref) == raw
    assert store.put(raw) == ref
    assert ref["sha256"] == hashlib.sha256(raw).hexdigest()
    path = store._path(ref["sha256"])
    path.write_bytes(raw.replace(b"3", b"4"))
    with pytest.raises(ValueError, match="digest"):
        store.get(ref)
    with pytest.raises(ValueError, match="digest"):
        store.put(raw)


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{"a":[NaN]}', b'{"a":1e999}'])
def test_new_metadata_rejects_ambiguous_or_nonfinite_json(payload):
    with pytest.raises(ValueError):
        decode_json(payload)


def test_atomic_publication_is_exclusive_and_gzip_repeatable(tmp_path):
    a, b = tmp_path/"a.json.gz", tmp_path/"b.json.gz"
    write_json(a, {"z": -0.0, "a": [1, 2]})
    write_json(b, {"a": [1, 2], "z": -0.0})
    assert a.read_bytes() == b.read_bytes()
    assert gzip.decompress(a.read_bytes()) == json_bytes({"a": [1, 2], "z": -0.0})
    with pytest.raises(FileExistsError):
        write_json(a, {"different": True})
    assert read_json(a)["a"] == [1, 2]


@pytest.mark.parametrize("relative", ["/tmp/out", "../out", "a/../../out", "a\\b", ""])
def test_artifact_paths_reject_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        local_path(tmp_path, relative)


def test_artifact_path_rejects_symlink_escape_and_oversized_objects(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    (root/"escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        local_path(root, "escape/out")
    store = ArtifactStore(root, max_object_bytes=3)
    with pytest.raises(ValueError, match="byte budget"):
        store.put(b"four")


def test_exit_failure_timeout_and_remaining_schedule_are_all_retained(tmp_path):
    tasks = (command("bad", "print('failure output'); raise SystemExit(7)"),
             command("slow", "import time; time.sleep(20)", timeout_seconds=.1),
             command("last", "print('last still ran')"))
    result = runner.run(tasks, root=tmp_path, destination=tmp_path/"run", sources=())
    assert [r["status"] for r in result["rows"]] == ["failed", "timed_out", "passed"]
    assert result["summary"] == {"scheduled": 3, "attempted": 3, "passed": 1, "failed": 2}
    assert result["rows"][0]["returncode"] == 7
    assert result["scientific_evidence_claim"] is False
    store = ArtifactStore(tmp_path/"run")
    assert b"failure output" in store.get(result["rows"][0]["log"])
    assert b"last still ran" in store.get(result["rows"][-1]["log"])


def test_log_limit_is_a_failure_and_preserves_bounded_prefix(tmp_path):
    tasks = (command("loud", "print('x'*100000)", max_output_bytes=128),)
    result = runner.run(tasks, root=tmp_path, destination=tmp_path/"run", sources=())
    row = result["rows"][0]
    assert row["status"] == "output_limited"
    assert row["observed_log_bytes"] > 128
    assert row["log"]["size_bytes"] == 128


def test_completed_resume_never_reexecutes_and_publication_cannot_be_overwritten(tmp_path):
    tasks = (command("once", "from pathlib import Path; p=Path('count'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"),)
    kwargs = dict(root=tmp_path, destination=tmp_path/"run", sources=())
    result = runner.run(tasks, **kwargs)
    assert runner.run(tasks, **kwargs, resume=True) == result
    assert (tmp_path/"count").read_text() == "x"
    with pytest.raises(FileExistsError):
        runner.run(tasks, **kwargs)


def test_interrupted_resume_keeps_failure_and_runs_only_unstarted_tasks(tmp_path, monkeypatch):
    tasks = (command("crashed", "raise AssertionError('must not retry')"),
             command("remaining", "print('remaining')"))
    kwargs = dict(root=tmp_path, destination=tmp_path/"run", sources=())
    original = runner._execute
    def crash(task, root, log):
        log.write_bytes(b"partial observations")
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "_execute", crash)
    with pytest.raises(KeyboardInterrupt):
        runner.run(tasks, **kwargs)
    monkeypatch.setattr(runner, "_execute", original)
    result = runner.run(tasks, **kwargs, resume=True)
    assert [r["status"] for r in result["rows"]] == ["interrupted", "passed"]
    assert result["rows"][0]["elapsed_seconds"] is None
    assert result["rows"][0]["costs_complete"] is False
    assert ArtifactStore(tmp_path/"run").get(result["rows"][0]["log"]) == b"partial observations"
    assert result["summary"]["failed"] == 1


def test_resume_rejects_changed_source_schedule_environment_and_log(tmp_path, monkeypatch):
    source = tmp_path/"source.py"
    source.write_text("original")
    tasks = (command("once", "print('once')"),)
    kwargs = dict(root=tmp_path, destination=tmp_path/"run", sources=("source.py",))
    result = runner.run(tasks, **kwargs)
    source.write_text("changed")
    with pytest.raises(ValueError, match="differ"):
        runner.run(tasks, **kwargs, resume=True)
    source.write_text("original")
    with pytest.raises(ValueError, match="differ"):
        runner.run((command("once", "print('changed')"),), **kwargs, resume=True)
    monkeypatch.setenv("OMP_NUM_THREADS", "123")
    with pytest.raises(ValueError, match="differ"):
        runner.run(tasks, **kwargs, resume=True)
    monkeypatch.undo()
    ArtifactStore(tmp_path/"run")._path(result["rows"][0]["log"]["sha256"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="size differs"):
        runner.run(tasks, **kwargs, resume=True)


def test_source_change_stops_before_next_command(tmp_path):
    (tmp_path/"bound").write_text("before")
    tasks = (command("mutate", "from pathlib import Path; Path('bound').write_text('after')"),
             command("forbidden", "from pathlib import Path; Path('unexpected').touch()"))
    with pytest.raises(ValueError, match="source changed"):
        runner.run(tasks, root=tmp_path, destination=tmp_path/"run", sources=("bound",))
    assert not (tmp_path/"unexpected").exists()


def test_duplicate_schedule_and_concurrent_writer_are_rejected(tmp_path):
    task = command("same", "pass")
    with pytest.raises(ValueError, match="duplicate"):
        runner.run((task, task), root=tmp_path, destination=tmp_path/"run", sources=())
    assert not (tmp_path/"run").exists()
    (tmp_path/"run").mkdir()
    with runner._run_lock(tmp_path/"run/active.lock"):
        with pytest.raises(BlockingIOError):
            runner.run((task,), root=tmp_path, destination=tmp_path/"run", sources=(), resume=True)


def test_resume_rejects_schedule_row_tampering(tmp_path):
    tasks = (command("one", "pass"),)
    kwargs = dict(root=tmp_path, destination=tmp_path/"run", sources=())
    runner.run(tasks, **kwargs)
    checkpoint = tmp_path/"run/checkpoint.json"
    receipt = read_json(checkpoint)
    receipt["rows"][0]["key"] = "invented"
    write_json(checkpoint, receipt, exclusive=False)
    with pytest.raises(ValueError, match="schedule differs"):
        runner.run(tasks, **kwargs, resume=True)


def test_required_artifacts_are_byte_bound_and_missing_artifacts_fail(tmp_path):
    tasks = (command("produce", "from pathlib import Path; Path('result.json').write_bytes(b'{  }\\n')",
                     artifacts=("result.json",)),
             command("missing", "pass", artifacts=("absent.json",)))
    result = runner.run(tasks, root=tmp_path, destination=tmp_path/"run", sources=())
    assert [r["status"] for r in result["rows"]] == ["passed", "failed"]
    ref = result["rows"][0]["artifacts"]["result.json"]
    assert ArtifactStore(tmp_path/"run").get(ref) == b'{  }\n'
    assert "absent.json" in result["rows"][1]["artifact_errors"]


@pytest.mark.skipif(os.name != "posix", reason="process group contract is POSIX")
def test_timeout_terminates_descendant_work(tmp_path):
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "Path('child').write_text(str(p.pid)); time.sleep(30)")
    result = runner.run((command("tree", code, timeout_seconds=.4),),
                        root=tmp_path, destination=tmp_path/"run", sources=())
    assert result["rows"][0]["status"] == "timed_out"
    pid = int((tmp_path/"child").read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"
