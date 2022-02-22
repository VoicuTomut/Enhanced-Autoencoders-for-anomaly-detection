from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, assemble, transpile
import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import COBYLA
from sympy import I, Matrix, symbols, limit
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


class QAOA():
 
    def __init__(self, n, mu, sigma, optimizer, backend, layers = 4, shots = 1024):
        
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.optimizer = optimizer
        self.backend = backend 
        self.layers = layers
        self.shots = shots

        m = np.random.rand(2*self.layers)
        self.params = m

    # Define the quantum circuit representing the QAOA 
    def layer_circuit(params, n, layers):
  
        qr = QuantumRegister(self.n)
        cr = ClassicalRegister(self.n)
        qc = QuantumCircuit(qr, cr) 
    
        # State preparation. Apply hadamard to all qubits to achieve equal superposition |+>_N
        for i in range(self.n):
            qc.h(qr[i])
    
        # Loops for multiple layers 
        for i in range(self.layers):
        
            # Define parameters for each layer 
            gamma = self.params[2*i]
            beta = self.params[(2*i)+1]
  
            #########################
            # Apply Cost Hamiltonian#
            #########################
        
            
            for qubit in range(n):
                qc.rz(-2*gamma*self.mu[qubit], qr[qubit])
        
            for qubit in range(self.n): 
            
                if qubit < self.n-1:
                    for j in range(qubit, self.n-1):
                    
                        qc.barrier()
    
                        # Apply ZZ interaction
                        qc.cx(qr[qubit], qr[qubit+1])
                        qc.rz(2*gamma*self.sigma[qubit][j], qr[qubit+1])
                        qc.cx(qr[qubit], qr[qubit+1]) 
                    
                        qc.barrier()
                    
        
            ############################
            # Apply Mixing  Hamiltonian#
            ############################
        
            for qubit in range(self.n):
                qc.rx(2*beta, qr[qubit])
        
            qc.barrier()
    
        # Measure the state 
        qc.measure(qr, cr)
    
        return qc
    
    
    # Create the objective function that will be optimized
    def objective_function(self, params):
        # Define the quantum circuit with updated parameters
        qc = self.layer_circuit()
    
        t_qc = transpile(qc, self.backend)
        qobj = assemble(t_qc, shots = self.shots)
        result = self.backend.run(qobj).result()
    
        states = result.get_counts(qc)

        max_key = int(max(states, key=states.get))  # Obtains the state with the highest counts
    
        max_string = str(max_key)                      # Turns it into string and corrects the truncation of 0's by Qiskit: (Qiskit will truncate a '010' into a '10' and this fixes that)
        if len(max_string) < self.n: 
            while len(max_string) < self.n:
                max_string = "0" + max_string
   
        # Turns the corrected string into a vector describing the choice of assets
    
        choices = np.array([int(x) for x in max_string])
        print(choices)

        # Calculating expectation value of the cost Hamiltonian on the trail state
        avg = 0
        sum_count = 0
    
        for bitstring, count in states.items():
        
            string = str(bitstring)                      # Turns it into string and corrects the truncation of 0's by Qiskit: (Qiskit will truncate a '010' into a '10' and this fixes that)
            if len(string) < self.n: 
                while len(string) < self.n:
                    string = "0" + string
            choice = np.array([int(x) for x in string])
        
        
            cost = np.dot(np.dot(choice, self.sigma), Dagger(choice)) - np.dot(self.mu, Dagger(choice))
            avg += cost * count
            sum_count += count
    
        expectation_value = avg/sum_count
    
        # Keep track of the most measured state and its coresponding cost
        bchoice.append(choices)
        lcost.append(np.dot(np.dot(choices, self.sigma), Dagger(choices)) - np.dot(self.mu, Dagger(choices)))
    
        # Print to see the progress of the QAOA 
        print( expectation_value)
        print(np.dot(np.dot(choices, self.sigma), Dagger(choices)) - np.dot(self.mu, Dagger(choices)))
        print("")
    
    
        # Return the expectation value 
        return  expectation_value


    # Run the QAOA and classically update parameters 
    def run(self):
        
        ret = self.optimizer.optimize(num_vars = len(self.params), objective_function = self.objective_function, initial_point = self.params)
        
        self.params = ret[0]
                                  

        qc = self.Ansatz()
        t_qc = transpile(qc, self.backend)
        qobj = assemble(t_qc, shots = self.shots)


        # Find the lowest cost found and the associated choices from all iterations of QAOA.
        lowest_cost = min(lcost)
        best_choice = bchoice[lcost.index(lowest_cost)]
        
        return lowest_cost, best_choice

