"""
The encoder circuit inspired  from the: https://arxiv.org/pdf/2112.12563.pdf
but with a different ansats
"""

import pennylane as qml


def e5_patch(params1, params2, wires1, wires2):
    e2_layer(params1, wires1)
    e2_layer(params2, wires2)


def e2_layer(params, wires):
    """
    :params: an array of gate parameters of len. nr_params
    :wires: list of qubits on which decoder is applied
    nr_params = 15 * len(wires)*(len(wires)-1)/2
    """
    idx = 0

    for j in range(1, len(wires)):
        i = 0
        while i + j < len(wires):
            # qml.Rot(phi, theta, omega, wire)
            qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
            qml.Rot(params[idx + 3], params[idx + 4], params[idx + 5], wires=i + j)
            qml.CNOT(wires=[i + j, i])
            qml.RZ(params[idx + 6], wires=i)
            qml.RY(params[idx + 7], wires=[i + j])
            qml.CNOT(wires=[i, i + j])
            qml.RY(params[idx + 8], wires=[i + j])
            qml.CNOT(wires=[i + j, i])
            qml.Rot(params[idx + 9], params[idx + 10], params[idx + 11], wires=i)
            qml.Rot(params[idx + 12], params[idx + 13], params[idx + 14], wires=i + j)

            i += 1
            idx += 15
