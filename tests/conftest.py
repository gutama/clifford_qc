import os
from pathlib import Path
import shutil

import pytest

from hypothesis import HealthCheck, settings

from clifford_qc.record_environment import ALLOW_MIGRATION_ENV

settings.register_profile(
    "ci", derandomize=True, max_examples=200,
    suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=50)

# Let the suite stamp records under any supported interpreter.
#
# ``stamp_record`` refuses an environment no committed record declares, which is
# what keeps a producer from starting a version split.  Tests that stamp write
# into ``tmp_path`` and are not producing committed evidence, so the suite opts
# out rather than failing on every interpreter but the one the records name.
#
# At import time rather than in a fixture: a fixture can only be function
# scoped if it wants ``monkeypatch``, and module- and session-scoped fixtures
# run before that, so they would stamp before the opt-out was in place.  Set
# here it also reaches any subprocess a test starts.  ``setdefault`` leaves an
# explicit choice from the environment alone, and the guard's own tests take
# the variable back off.
os.environ.setdefault(ALLOW_MIGRATION_ENV, "1")

ROOT = Path(__file__).resolve().parents[1]

# Some tests have a precondition that belongs to the *environment* rather than
# to the code: the exact stack a frozen record named, or the commit history a
# chronology check walks.  Off that environment they cannot hold, so they skip
# and say why -- a shallow clone or a newer NumPy is not a defect in the tree.
#
# A skip is only safe if it cannot go unnoticed where the precondition is
# supposed to hold.  The record-environment CI job sets this variable, and there
# the same condition is a failure: its checkout is complete and its stack is the
# pinned one, so a skip would mean the job no longer tests what it is for.
STRICT_ENVIRONMENT_ENV = "CLIFFORD_QC_STRICT_TEST_ENVIRONMENT"


def _environment_skip(reason: str) -> None:
    if os.environ.get(STRICT_ENVIRONMENT_ENV) == "1":
        pytest.fail(f"{reason} ({STRICT_ENVIRONMENT_ENV}=1 forbids skipping "
                    "an environment precondition)")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def environment_skip():
    """``environment_skip(reason)``: skip here, fail under the strict CI job."""
    return _environment_skip


@pytest.fixture(scope="session")
def full_git_history():
    """Require a complete Git checkout, for tests that read commit history.

    Record chronology, execution-commit bindings and bundle restoration all walk
    ancestry.  A ``--depth`` clone, a source archive and an installed sdist do
    not have it, and without this the failure is a ``CalledProcessError`` from
    ``git`` several frames down rather than a statement of what is missing.
    """
    from benchmarks.archive_provenance import history_kind

    try:
        history_kind(ROOT)
    except ValueError as error:  # HistoryError and its shallow case included
        _environment_skip(f"needs a full Git clone of the repository: {error}")


@pytest.fixture
def frozen_phase19_checkout(tmp_path):
    """A pristine frozen declaration tree, independent of today's live kernel."""
    from benchmarks import check_phase19_preregistration as gate, phase19_structure as structure
    from benchmarks.archive_provenance import MANIFEST, load_archive_manifest, lineage_source_path
    root = tmp_path / "frozen"
    root.mkdir()
    config = gate.load_config()
    manifest = load_archive_manifest(structure.ROOT)
    for row in config["lineage"]["files"] + config["bank_manifests"]:
        relative = row["path"]
        source = (lineage_source_path(structure.ROOT, structure.RECORD_PATH, row, manifest=manifest)
                  if relative in structure.IMPLEMENTATIONS else structure.ROOT / relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    paths = {MANIFEST, structure.CONFIG_PATH, structure.RECORD_PATH}
    paths.update(item["snapshot"] for item in manifest["records"][structure.RECORD_PATH]["lineage_sources"].values())
    paths.update(item["snapshot"] for item in manifest["records"][structure.RECORD_PATH]["sources"].values())
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(structure.ROOT / relative, target)
    return root
