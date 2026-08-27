  
"""
Reduced Bose-Hubbard Hamiltonian construction.

Builds the Bose-Hubbard Hamiltonian for a supported two-dimensional lattice,
restricted to the fixed total-boson-number sector (the "reduced" Hilbert
space).
"""

from itertools import combinations
from math import comb

import numpy as np
import qse
from scipy.sparse import coo_matrix
from qutip import Qobj

__all__ = ["ReducedBoseHubbard2D"]


class ReducedBoseHubbard2D:
    """
    Reduced Bose-Hubbard model for a two-dimensional lattice.

    Constructs the fixed-particle-number basis, identifies nearest-neighbor
    bonds, and builds the corresponding sparse Bose-Hubbard Hamiltonian.
    """

    supported_shapes = {
        "square",
        "hexagonal",
        "triangular",
        "kagome",
        "ring",
    }

    def __init__(
        self,
        shape,
        lattice_parameters,
        number_of_bosons=5,
        hopping_amplitude=1.0,
        on_site_interaction=0.0,
    ):
        """
        Initialize the reduced Bose-Hubbard model.

        Parameters
        ----------
        shape : str
            Shape of the lattice.
        lattice_parameters : dict
            Parameters passed to the lattice constructor. Must contain
            ``lattice_spacing``.
        number_of_bosons : int, optional
            Total number of bosons in the system.
        hopping_amplitude : float, optional
            Hopping amplitude between neighboring lattice sites.
        on_site_interaction : float, optional
            Strength of the on-site interaction.

        Raises
        ------
        ValueError
            If the lattice shape is unsupported, ``lattice_spacing`` is
            missing, or ``number_of_bosons`` is invalid.
        """
        self.shape = shape.strip().lower()

        if self.shape not in self.supported_shapes:
            shapes = ", ".join(sorted(self.supported_shapes))
            raise ValueError(
                f"Unsupported lattice shape: {shape}. Choose from: {shapes}."
            )

        if self.shape == "ring":
            if "spacing" not in lattice_parameters:
                raise ValueError(
                    "Ring lattice parameters must contain spacing."
                )
        elif "lattice_spacing" not in lattice_parameters:
            raise ValueError(
                "lattice_parameters must contain lattice_spacing."
            )

        if not isinstance(number_of_bosons, int) or number_of_bosons < 0:
            raise ValueError("number_of_bosons must be a non-negative integer.")

        self.number_of_bosons = number_of_bosons
        self.hopping_amplitude = hopping_amplitude
        self.on_site_interaction = on_site_interaction

        if self.shape == "ring":
            self.lattice_spacing = lattice_parameters["spacing"]
        else:
            self.lattice_spacing = lattice_parameters["lattice_spacing"]

        lattice_constructor = getattr(qse.lattices, self.shape)
        self.lattice = lattice_constructor(**lattice_parameters)

        self.number_of_sites = self.lattice.nqbits
        self.distances = self.lattice.get_all_distances()

        self.bonds = self.get_bonds()

        self.basis_dimension = comb(
            self.number_of_bosons + self.number_of_sites - 1, ###### It counts the possible distributions of 'N' bosons across the  'L' lattice sites (another way to find the dimension of the reduced Hilbert space) #######
            self.number_of_bosons,
        )

        self.basis_states = self.generate_basis()
        self.H = self.build_H()

    def generate_basis(self):
        """
        Generate the fixed-particle-number occupation basis.

        Returns
        -------
        list of tuple of int
            Occupation-number states containing all weak compositions of the
            total number of bosons across the lattice sites.
        """
        number_of_bosons = self.number_of_bosons
        number_of_sites = self.number_of_sites

        def compositions(remaining, sites_left, prefix=()):
            """
            Generate weak compositions recursively.

            Parameters
            ----------
            remaining : int
                Number of bosons remaining to be assigned.
            sites_left : int
                Number of sites still requiring an occupation number.
            prefix : tuple of int, optional
                Occupation numbers assigned to preceding sites.

            Yields
            ------
            tuple of int
                Complete occupation-number state.
            """
            if sites_left == 1:
                yield prefix + (remaining,)
                return

            for occupation in range(remaining + 1):
                yield from compositions(
                    remaining - occupation,
                    sites_left - 1,
                    prefix + (occupation,),
                )

        return list(compositions(number_of_bosons, number_of_sites))

    def get_bonds(self):
        """
        Find pairs of nearest-neighbor lattice sites.

        Returns
        -------
        list of tuple of int
            Pairs of site indices separated by one lattice spacing.
        """
        bonds = []

        for first_site, second_site in combinations( ##### Generates every unique pair of lattice sites without repeating these pairs. #####
            range(self.number_of_sites), 2
        ):
            distance = self.distances[first_site, second_site]

            if (
                distance > 0
                and np.isclose(
                    distance,
                    self.lattice_spacing,
                    rtol=1e-7,
                    atol=1e-10,
                )
            ):
                bonds.append((first_site, second_site))

        return bonds

    def build_H(self):
        """
        Construct the reduced Bose-Hubbard Hamiltonian.

        Returns
        -------
        qutip.Qobj
            Sparse Hamiltonian containing the on-site interaction and
            nearest-neighbor hopping terms.
        """
        basis = self.basis_states
        dimension = len(basis)
        state_index = {
            state: index for index, state in enumerate(basis)
        }

        rows = []
        columns = []
        values = []

        for column, state in enumerate(basis):
            # Diagonal interaction energy:
            # U/2 * sum_i n_i(n_i - 1)
            diagonal = (
                0.5
                * self.on_site_interaction
                * sum(n * (n - 1) for n in state)
            )

            if diagonal != 0:
                rows.append(column)
                columns.append(column)
                values.append(diagonal)

            for first_site, second_site in self.bonds:
                n_first = state[first_site]
                n_second = state[second_site]

                # first_site -----> second_site
                if n_first > 0:
                    new_state = list(state)
                    new_state[first_site] -= 1
                    new_state[second_site] += 1

                    row = state_index[tuple(new_state)]
                    amplitude = (
                        -self.hopping_amplitude
                        * np.sqrt(n_first * (n_second + 1))
                    )

                    rows.append(row)
                    columns.append(column)
                    values.append(amplitude)

                # second_site -----> first_site
                if n_second > 0:
                    new_state = list(state)
                    new_state[second_site] -= 1
                    new_state[first_site] += 1

                    row = state_index[tuple(new_state)]
                    amplitude = (
                        -self.hopping_amplitude
                        * np.sqrt(n_second * (n_first + 1))
                    )

                    rows.append(row)
                    columns.append(column)
                    values.append(amplitude)

        sparse_H = coo_matrix(
            (values, (rows, columns)),
            shape=(dimension, dimension),
            dtype=complex,
        ).tocsr()

        return Qobj(sparse_H)