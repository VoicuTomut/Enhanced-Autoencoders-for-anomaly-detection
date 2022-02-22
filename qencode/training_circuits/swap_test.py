"""
Swap test circuit
"""

import pennylane as qml


def swap_t(spec):
    """
    :spec: QubitsArrangement specification
    """
    for i in spec.swap_qubits:
        qml.Hadamard(wires=i)
    for i in range(len(spec.trash_qubits)):
        qml.CSWAP(wires=[*spec.swap_qubits, spec.aux_qubits[i], spec.trash_qubits[i]])
    for i in spec.swap_qubits:
        qml.Hadamard(wires=i)
