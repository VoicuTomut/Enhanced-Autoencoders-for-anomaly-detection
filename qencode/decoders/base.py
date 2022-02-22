"""
The most basic decoder possible is
the inverse of the encoder.
"""

import pennylane as qml


def decoder_adjoint(encoder_circuit, params, wires):
    """
    :encoder_circuit: the circuit used to encode the data
    :params: the encoder parameters
    :wires: list of qubits on which decoder is applied

    This decoder is basically the adjoint of the encoder.
    """
    qml.adjoint(encoder_circuit)(params, wires)
