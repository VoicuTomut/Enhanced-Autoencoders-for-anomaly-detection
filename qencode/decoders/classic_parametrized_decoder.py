"""
An example of a parametrized decoder
"""

import pennylane as qml


# parametrized decoders
def d1_classic(params, wires):
    """
    :params: an array of gate parameters of len. nr_params
    :wires: list of qubits on which decoder is applied
    nr_params = 2*3*len(wires) + 3*(len(wires)-1) * len(wires)
    """

    # Add the first rotational gates:
    idx = 0
    for i in wires:
        # qml.Rot(phi, theta, omega, wire)
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3

    # Add the controlled rotational gates
    for i in wires:
        for j in wires:
            if i != j:
                qml.CRot(params[idx], params[idx + 1], params[idx + 2], wires=[i, j])
                idx += 3

    # Add the last rotational gates:
    for i in wires:
        # qml.Rot(phi, theta, omega, wire)
        qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
        idx += 3
