"""Untimed observation of the candidate masks the live blade check builds/visits."""
from __future__ import annotations

import builtins
from unittest.mock import patch

from clifford_qc import multivector as kernel


def audit_blade(value):
    generated, visited = [], []
    buckets = kernel._blade_contraction_buckets
    encode = kernel.mask_to_code

    def observe_buckets(*args):
        result = buckets(*args)
        generated.append(sorted(result))
        return result

    def observe_visit(n, mask):
        visited.append(mask)
        return encode(n, mask)

    def sparse_range(*args):
        if args == (4 ** value.n,):
            raise ValueError("live blade check enumerated the whole algebra")
        return builtins.range(*args)

    def no_whole_grade(*args):
        raise ValueError("live blade check requested a whole grade")

    with patch.object(kernel, "_blade_contraction_buckets", observe_buckets), \
         patch.object(kernel, "mask_to_code", observe_visit), \
         patch.object(kernel, "range", sparse_range, create=True), \
         patch.object(kernel, "codes_of_grade", no_whole_grade):
        decision = value.is_blade()
    if len(generated) != 1:
        raise ValueError("live blade check did not use one observed active-support pass")
    if len(visited) != len(set(visited)) or not set(visited).issubset(generated[0]):
        raise ValueError("live blade check visited unexpected/repeated contraction masks")
    return decision, generated[0], visited
