"""
Test the construction of the reduced two-dimensional Bose-Hubbard Hamiltonian.

The tests verify the diagonal form of the Hamiltonian when hopping is
disabled and compare a small Hamiltonian with its analytical result.
"""

import numpy as np
import pytest
from scipy.sparse import diags

from hamiltonian_2d import ReducedBoseHubbard2D


def test_hamiltonian_is_diagonal_when_j_is_zero():
    """
    Verify that the Hamiltonian is diagonal when hopping is disabled.

    Setting ``J = 0`` removes all hopping terms. The Hamiltonian must
    therefore contain only diagonal on-site interaction terms.
    """
    print("\nFirst test: J = 0, check that H is diagonal")

    BH_model_test_1 = ReducedBoseHubbard2D(
        shape="hexagonal",
        lattice_parameters={
            "lattice_spacing": 1.0,
            "repeats_x": 3,
            "repeats_y": 3,
        },
        number_of_bosons=5,
        hopping_amplitude=0.0,
        on_site_interaction=2.0,
    )

    H_sparse = BH_model_test_1.H.data.as_scipy()

    off_diagonal = (
        H_sparse
        - diags(H_sparse.diagonal())
    )

    off_diagonal.eliminate_zeros()
    is_diagonal = off_diagonal.nnz == 0

    print("Hamiltonian is diagonal:", is_diagonal)

    assert is_diagonal


def test_hamiltonian_matches_expected_matrix():
    """
    Verify the Hamiltonian against an analytical result.

    The calculated Hamiltonian for one boson on four square-lattice sites is
    compared with the expected four-dimensional matrix.
    """
    print("\nSecond test: compare H with the expected matrix")

    BH_model_test_2 = ReducedBoseHubbard2D(
        shape="square",
        lattice_parameters={
            "lattice_spacing": 1.0,
            "repeats_x": 2,
            "repeats_y": 2,
        },
        number_of_bosons=1,
        hopping_amplitude=1.0,
        on_site_interaction=2.0,
    )

    expected = np.array([
        [0.0, -1.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, -1.0, 0.0],
    ])

    calculated = BH_model_test_2.H.full()

    matrices_match = np.allclose(
        calculated,
        expected,
    )

    print("Do they match?", matrices_match)

    assert matrices_match