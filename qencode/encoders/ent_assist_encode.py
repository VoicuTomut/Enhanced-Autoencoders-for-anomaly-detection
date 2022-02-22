import pennylane as qml
from base import e1_classic


def ent_assisted_encode_no_interation(params, wires, ent_params, ent_wires):
    # Prepare the standard e1 circuit
    e1_classic(params, wires)

    # Should always be even
    assert len(ent_wires) % 2 == 0
    for i in range(len(ent_wires) // 2):
        qml.Hadamard(wires=[i])
        qml.CNOT(wires=[i, len(ent_wires) // 2 + i])

    # Apply encoding on ent_qubits
    e1_classic(ent_params, ent_wires)
