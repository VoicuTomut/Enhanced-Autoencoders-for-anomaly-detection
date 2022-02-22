

import pennylane as qml
from .base import e2_classic

# Requires 15*(n - ent_pairs choose 2) parameters where n is total number of qubits

def ent_assist_encode_sean(params, wires, ent_pairs):
    
    encode_wires = wires
    
    #Create EPR pairs, Alice to Bob
    for j in range(ent_pairs):
        
        qml.Hadamard(wires = len(wires) - 2j - 2)
        qml.CNOT(wires = [len(wires) - 2j- 2, len(wires) - 2*j - 1])
        
        # Remove Bob's qubits from encoding list
        del encode_wires[len(wires) - 2*j - 1]
    
    # e2 encoding with Alices qubits included 
    e2_classic(params, encode_wires)
    