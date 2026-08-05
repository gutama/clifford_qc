from hypothesis import HealthCheck, settings


settings.register_profile(
    "ci", derandomize=True, max_examples=200,
    suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=50)
