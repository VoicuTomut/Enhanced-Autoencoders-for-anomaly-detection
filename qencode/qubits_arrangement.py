"""
The only thing that's really important.
While we use this class to keep our qubit register in order,
we should have no problem keeping everything together.
"""


class QubitsArrangement:
    def __init__(self, nr_trash, nr_latent, nr_swap=1, nr_ent=2, nr_aux=None):
        """
        :nr_trash: number of trash qubits
        :nr_latent: number of qubits in latent space
        :nr_swap: nr of qubits used for swap tests 1
        :nr_ent: nr of qubits used for entanglement advantage
        """

        # qubits in register A
        self.latent_qubits = [*range(nr_latent)]
        # qubits in register B
        self.trash_qubits = [*range(nr_latent, nr_latent + nr_trash)]
        # qubits in a (nr_q(a)=nr_q(B))
        if nr_aux!=None:
            self.aux_qubits = [*range(nr_latent + nr_trash, nr_latent +  nr_trash+nr_aux)]
        else:
            self.aux_qubits = [*range(nr_latent + nr_trash, nr_latent + 2 * nr_trash)]
            
        # qubits used for swap tests
        self.swap_qubits = [*range(nr_latent + 2 * nr_trash, nr_latent + 2 * nr_trash + nr_swap)]
        self.ent_qubits = [
            *range(
                nr_latent + 2 * nr_trash + nr_swap,
                nr_latent + 2 * nr_trash + nr_swap + nr_ent,
            )
        ]  # qubits used for entanglement
        self.qubits = [*range(nr_latent + 2 * nr_trash + nr_swap + nr_ent)]
        self.num_qubits = len(self.qubits)
        self.num_trash = len(self.trash_qubits)
        self.num_latent = len(self.latent_qubits)
