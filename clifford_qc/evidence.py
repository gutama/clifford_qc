"""Versioned per-quantity evidence declarations; labels never infer results."""
from __future__ import annotations

SCHEMA = "clifford_qc.evidence.v1"
TIERS = frozenset({"exact_algebra", "exact_small_sector_simulation",
                   "finite_sample_empirical", "illustrative_model",
                   "classical_process_measurement", "asymptotic_bound",
                   "published_literature"})


def validate_evidence(record: dict) -> None:
    """Reject absent, malformed or unknown labels in the opt-in v1 contract."""
    if record.get("evidence_schema") != SCHEMA:
        raise ValueError(f"evidence_schema must be {SCHEMA!r}")
    quantities = record.get("evidence")
    if not isinstance(quantities, dict) or not quantities:
        raise ValueError("evidence must be a nonempty per-quantity mapping")
    for quantity, declaration in quantities.items():
        if not isinstance(quantity, str) or not quantity.strip():
            raise ValueError("evidence quantity must be a nonempty string")
        if not isinstance(declaration, dict) or set(declaration) != {"tier", "scope"}:
            raise ValueError(f"{quantity}: evidence must contain exactly tier and scope")
        if not isinstance(declaration["tier"], str) or declaration["tier"] not in TIERS:
            raise ValueError(f"{quantity}: unknown evidence tier {declaration['tier']!r}")
        if not isinstance(declaration["scope"], str) or not declaration["scope"].strip():
            raise ValueError(f"{quantity}: evidence scope must be nonempty")


def declare_evidence(record: dict, quantities: dict) -> dict:
    """Return a labelled copy for producers; validate before expensive execution."""
    labelled = dict(record, evidence_schema=SCHEMA, evidence=quantities)
    validate_evidence(labelled)
    return labelled
