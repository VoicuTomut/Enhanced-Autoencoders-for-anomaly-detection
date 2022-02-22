"""
The encoder circuit inspired  from the: https://arxiv.org/abs/2010.06599
"""
import pennylane as qml


def e3_enhance(params, x, spec, nr_layers=2):
    """
    :params: an array of gate parameters of len. nr_params
    :x: scalar x = 1 or 2 or 3 ...
    :spec: QubitsArrangement object with .trash_qubits and .latent_qubits
    :nr_layers: int number of layers.
    nr_params = nr_layers*(2*len(wires))+ 2*len(spec.trash_qubits)
    """

    wires = [*spec.latent_qubits, *spec.trash_qubits]

    # Add the first rotational gates:
    idx = 0

    for l in range(nr_layers):
        # first Ry layer
        for i in wires:
            # qml.Rot(phi, wire)
            qml.RY(params[idx] * x + params[idx + 1], wires=i)
            idx += 2

        # Add the controlled rotational gates
        for i in spec.trash_qubits:
            for j in wires:
                if i != j:
                    qml.CZ(wires=[j, i])

    # Ry on the trash:
    for i in spec.trash_qubits:
        qml.RY(params[idx] * x + params[idx + 1], wires=i)
        idx += 2
