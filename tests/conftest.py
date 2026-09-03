import os

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
