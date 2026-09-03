import pytest
from hypothesis import HealthCheck, settings

from clifford_qc.record_environment import ALLOW_MIGRATION_ENV

settings.register_profile(
    "ci", derandomize=True, max_examples=200,
    suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=50)


@pytest.fixture(autouse=True)
def _stamp_outside_the_committed_record_set(monkeypatch):
    """Let the suite stamp records under any supported interpreter.

    ``stamp_record`` refuses an environment no committed record declares, which
    is what keeps a producer from starting a version split.  Tests that stamp
    write into ``tmp_path`` and are not producing committed evidence, so the
    suite opts out globally rather than failing on every interpreter but the
    one the records happen to name.  The guard's own tests take the variable
    back off.
    """
    monkeypatch.setenv(ALLOW_MIGRATION_ENV, "1")
