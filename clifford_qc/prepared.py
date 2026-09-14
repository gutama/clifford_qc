"""Content-addressed preparation for repeated FCIDUMP solver experiments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .ir import PauliSum, PauliWord, Program
from .models.spin import Model

SCHEMA = "clifford_qc.prepared_problem.v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def implementation_fingerprint():
    """Conservatively invalidate preparation after any package source change."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_canonical(data) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class PreparedProblem:
    """Immutable serialized model; each consumer receives fresh mutable objects."""

    payload: str
    fingerprint: str

    @classmethod
    def from_model(cls, model, *, preparation):
        data = {
            "schema": SCHEMA, "preparation": preparation,
            "name": model.name, "n": model.n,
            # Preserve emission order, including the original real/complex bits.
            "hamiltonian": [[int(code), complex(c).real, complex(c).imag]
                            for code, c in model.hamiltonian.to_mv().terms.items()],
            "reference": model.reference.to_dict(),
            "hva_layers": [[label, [w.code for w in words]] for label, words in model.hva_layers],
            "metadata": model.metadata,
        }
        return cls(_canonical(data), _digest(data))

    def model(self):
        data = json.loads(self.payload)
        if data.get("schema") != SCHEMA or _digest(data) != self.fingerprint:
            raise ValueError("prepared problem schema or content digest mismatch")
        n = data["n"]
        return Model(data["name"], n,
                     PauliSum(n, {int(code): complex(real, imag)
                                  for code, real, imag in data["hamiltonian"]}),
                     Program.from_dict(data["reference"]),
                     tuple((label, tuple(PauliWord(n, code) for code in words))
                           for label, words in data["hva_layers"]), data["metadata"])

    @property
    def preparation(self):
        return json.loads(self.payload)["preparation"]

    def save(self, path):
        self.model()  # validate before replacing a file
        _atomic_json(path, {"fingerprint": self.fingerprint, "problem": json.loads(self.payload)})

    @classmethod
    def load(cls, path, *, expected_preparation=None):
        data = json.loads(Path(path).read_text())
        result = cls(_canonical(data["problem"]), data["fingerprint"])
        result.model()
        if expected_preparation is not None and result.preparation != expected_preparation:
            raise ValueError("prepared problem does not match requested source/options/implementation")
        return result


def prepare_fcidump(source, cache_directory, *, name=None, n_electrons=None,
                     ms2=None, integral_tolerance=1e-12):
    """Return (prepared problem, artifact path, cache hit) without any FCI solve.

    FCIDUMP bytes already encode the orbital/active-space choice. Sector and
    tolerance overrides, model name and source implementation also enter the key.
    Cached bytes are digest-checked before use; mismatch fails closed.
    """
    from .models.fcidump import fcidump_model

    source = Path(source)
    source_bytes = source.read_bytes()
    specification = {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "name": name or source.stem, "n_electrons": n_electrons, "ms2": ms2,
        "integral_tolerance": float(integral_tolerance),
        "implementation": implementation_fingerprint(), "encoding": "jw",
    }
    path = Path(cache_directory) / (_digest(specification) + ".json")
    if path.exists():
        return PreparedProblem.load(path, expected_preparation=specification), path, True
    # Parse exactly the bytes hashed above, even if the source changes concurrently.
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / "source.FCIDUMP"
        snapshot.write_bytes(source_bytes)
        model = fcidump_model(snapshot, name=specification["name"], n_electrons=n_electrons,
                              ms2=ms2, integral_tolerance=integral_tolerance)
    # Do not retain the temporary pathname as scientific provenance.
    model.metadata["prepared_source_sha256"] = specification["source_sha256"]
    model.metadata.pop("path", None)
    model.metadata.pop("source_path", None)
    prepared = PreparedProblem.from_model(model, preparation=specification)
    prepared.save(path)
    return prepared, path, False
