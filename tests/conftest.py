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
