"""
Reduced Bose-Hubbard Hamiltonian construction.

Builds the Bose-Hubbard Hamiltonian for a 1D lattice with open boundary
conditions, restricted to the fixed total-boson-number sector (the
"reduced" Hilbert space).
"""

import numpy as np
from scipy.sparse import coo_matrix
from qutip import Qobj

__all__ = ["ReducedBoseHubbard"]




class ReducedBoseHubbard:
    """
    Represent a one-dimensional Bose-Hubbard model with a fixed total
    number of bosons.

    Each site has N + 1 possible occupations. For L = 5 and N = 5,
    the full Hilbert space contains (N+1)^L = 6^5 = 7776 states. Fixing the
    total boson number to N = 5 reduces it to 126 states.
    """

    def __init__(self, L=5, N=5, J=1.0, U=0.0):
        """
        Initialize the Bose-Hubbard model.

        Parameters
        ----------
        L : int, optional
            Number of lattice sites.
        N : int, optional
            Total number of bosons.
        J : float, optional
            Hopping strength.
        U : float, optional
            On-site interaction strength.
        """

        self.L = L
        self.N = N
        self.J = J
        self.U = U

        # Occupation numbers range from 0 to N.
        self.Nmax = self.N + 1

        # Generate the fixed-particle-number basis.
        self.basis_states = self.generate_fixed_N_occupation_basis()

        # Construct the Hamiltonian in this basis.
        self.H = self.build_reduced_bose_hubbard_hamiltonian()

    def generate_fixed_N_occupation_basis(self):
        """
        Generate all states containing exactly N bosons.

        Returns
        -------
        list of tuples
            Fixed-particle-number basis states.
        """

        def compositions(remaining, sites_left, prefix=()): # Weak compositions: it generates every way to distribute N bosons among L sites.

            if sites_left == 1:
                yield prefix + (remaining,)
                return

            for occupation in range(remaining + 1): 
                yield from compositions(
                    remaining - occupation,
                    sites_left - 1,
                    prefix + (occupation,)
                )

        basis_states = list(
            compositions(
                self.N,
                self.L
            )
        )

        return basis_states

    def build_reduced_bose_hubbard_hamiltonian(self):
        """
        Construct the reduced Bose-Hubbard Hamiltonian.

        The Hamiltonian is

            H = -J sum_j (bdag_{j+1} b_j + bdag_j b_{j+1})
                + (U/2) sum_j n_j(n_j - 1).

        Open boundary conditions are used.

        Returns
        -------
        qutip.Qobj
            Hamiltonian in the fixed-particle-number basis.
        """

        dim = len(self.basis_states)

        # Map each basis state to its matrix index.
        state_index = {
            state: index
            for index, state in enumerate(self.basis_states)
        }

        rows = []
        columns = []
        data = []

        
        for column, state in enumerate(self.basis_states):

            # On-site interaction term.
            interaction_energy = 0.0

            for site in range(self.L):

                number_at_site = state[site]

                interaction_energy += (
                    self.U
                    / 2.0
                    * number_at_site
                    * (number_at_site - 1)
                )

            rows.append(column)
            columns.append(column)
            data.append(interaction_energy)

            # Hopping term.
            for site in range(self.L - 1):

                number_at_site = state[site]
                number_at_next_site = state[site + 1]

                # Hop from site to site + 1.
                if number_at_site > 0:

                    new_state = list(state)

                    new_state[site] -= 1
                    new_state[site + 1] += 1

                    new_state = tuple(new_state)

                    row = state_index[new_state]

                    hopping_amplitude = (
                        -self.J
                        * np.sqrt(
                            number_at_site
                            * (number_at_next_site + 1)
                        )
                    )

                    rows.append(row)
                    columns.append(column)
                    data.append(hopping_amplitude)

                # Hop from site + 1 to site.
                if number_at_next_site > 0:

                    new_state = list(state)

                    new_state[site] += 1
                    new_state[site + 1] -= 1

                    new_state = tuple(new_state)

                    row = state_index[new_state]

                    hopping_amplitude = (
                        -self.J
                        * np.sqrt(
                            number_at_next_site
                            * (number_at_site + 1)
                        )
                    )

                    rows.append(row)
                    columns.append(column)
                    data.append(hopping_amplitude)

        hamiltonian_matrix = coo_matrix(
            (data, (rows, columns)),
            shape=(dim, dim),
            dtype=complex
        ).tocsr()

        return Qobj(hamiltonian_matrix)
