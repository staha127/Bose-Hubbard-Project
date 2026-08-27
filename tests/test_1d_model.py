"""
Test the construction of the reduced Bose-Hubbard Hamiltonian.

The tests verify the diagonal form of the Hamiltonian when hopping is
disabled and compare a small Hamiltonian with its analytical result.
"""

import numpy as np
from scipy.sparse import diags

from hamiltonian_1d import ReducedBoseHubbard


def test_hamiltonian_is_diagonal_when_j_is_zero():
    """
    Verify that the Hamiltonian is diagonal when hopping is disabled.

    Setting ``J = 0`` removes all hopping terms. The Hamiltonian must
    therefore contain only diagonal on-site interaction terms.
    """
    print("\nFirst test: J = 0, check that H is diagonal")

    BH_model_test_1 = ReducedBoseHubbard(
        L=10,
        N=10,
        J=0.0,
        U=2.0,
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

    The calculated Hamiltonian for two bosons on two lattice sites is
    compared with the expected three-dimensional matrix.
    """
    print("\nSecond test: compare H with the expected matrix")

    BH_model_test_2 = ReducedBoseHubbard(
        L=2,
        N=2,
        J=1.0,
        U=2.0,
    )

    expected = np.array([
        [2, -np.sqrt(2), 0],
        [-np.sqrt(2), 0, -np.sqrt(2)],
        [0, -np.sqrt(2), 2],
    ])

    calculated = BH_model_test_2.H.full()

    matrices_match = np.allclose(
        calculated,
        expected,
    )

    print("Do they match?", matrices_match)

    assert matrices_match