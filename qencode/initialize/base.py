"""
Here we have some basic initializers for Pennylane
applied on our target registers.
"""

import pennylane as qml


# AB initialization #

# Amplitude encoding:
def setAB_amplitude(spec, inputs):
    """
    :spec: QubitsArrangement specification
    :inputs: vector of amplitudes
    """
    qml.templates.embeddings.AmplitudeEmbedding(
        inputs,
        wires=[*spec.latent_qubits, *spec.trash_qubits],
        normalize=True,
        pad_with=0.0j,
    )


# Angle encoding:
def setAB_angle(spec, inputs, rotation='X'):
    """
    :spec: QubitsArrangement specification
    :inputs: vector of amplitudes
    :rotation: 'X', 'Y', 'Z' ax of rotation
    """
    qml.templates.embeddings.AngleEmbedding(
        inputs, wires=[*spec.latent_qubits, *spec.trash_qubits], rotation=rotation
    )


# State encoding:
def setAB_state(spec, inputs):
    """
    :spec: QubitsArrangement specification
    :inputs: vector of amplitudes
    """
    qml.MottonenStatePreparation(
        inputs, wires=[*spec.latent_qubits, *spec.trash_qubits]
    )


# Auxiliary Qubits initialization #

# Prepare the state in which trash-qubits will be reinitialized:
def setAux(spec, inputs):
    """
    :spec: QubitsArrangement specification
    :inputs: vector of amplitudes
    """
    qml.MottonenStatePreparation(inputs, wires=spec.aux_qubits)


# Entanglement preparation #

# Prepare the entagled qubits:
# we can aso decide to use  H CX or something else for the final submmision
# for tests this option is quite eazy to customize
def setEnt(spec, inputs):
    """
    :spec: QubitsArrangement specification
    :inputs: vector of amplitudes
    """
    if len(spec.ent_qubits) > 0:
        qml.MottonenStatePreparation(inputs, wires=spec.ent_qubits)
