"""Optional bridges from the Pauli-rotor IR to external ecosystems.

Each submodule imports its third-party dependency at module import time,
so import only the bridges whose extras are installed:

- ``clifford_qc.bridges.stim_bridge``        (``pip install -e '.[stim]'``)
- ``clifford_qc.bridges.openfermion_bridge`` (``'.[openfermion]'``)
- ``clifford_qc.bridges.pytket_bridge``      (``'.[pytket]'``)
- ``clifford_qc.bridges.pennylane_bridge``   (``'.[pennylane]'``)
- ``clifford_qc.bridges.pyzx_bridge``        (``'.[pyzx]'``)

The package is installed from a source checkout; ``python -m
clifford_qc.capabilities`` prints the exact command for each missing extra.

The QASM3 exporter lives in ``clifford_qc.qasm3`` and has no extra
dependency.
"""
