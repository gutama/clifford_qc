"""Optional bridges from the Pauli-rotor IR to external ecosystems.

Each submodule imports its third-party dependency at module import time,
so import only the bridges whose extras are installed:

- ``clifford_qc.bridges.stim_bridge``        (``pip install clifford-qc[stim]``)
- ``clifford_qc.bridges.openfermion_bridge`` (``clifford-qc[openfermion]``)
- ``clifford_qc.bridges.pytket_bridge``      (``clifford-qc[pytket]``)
- ``clifford_qc.bridges.pennylane_bridge``   (``clifford-qc[pennylane]``)
- ``clifford_qc.bridges.pyzx_bridge``        (``clifford-qc[pyzx]``)

The QASM3 exporter lives in ``clifford_qc.qasm3`` and has no extra
dependency.
"""
