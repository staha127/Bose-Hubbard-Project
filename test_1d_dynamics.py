"""
Test the initial state used in the Bose-Hubbard dynamics.

The tests verify that the initial state is a ket, is normalized, and
represents a pure quantum state.
"""

import numpy as np
from qutip import ket2dm
import pytest

from hamiltonian_1d import ReducedBoseHubbard
from dynamics_1d import BoseHubbardDynamics


def create_initial_state():
    """
    Create the initial state used by every test.

    Returns
    -------
    qutip.Qobj
        Initial state represented as a ket.
    """

    model = ReducedBoseHubbard(
        L=10,
        N=5,
        J=1.0,
        U=0.0,
    )

    dynamics = BoseHubbardDynamics(
        model,
        t_start=0,
        t_end=10,
        num_times=300,
    )

    psi0, _, _, _ = dynamics.run_dynamics(
        random_fock_state=False,
        random_superposition_of_fock_states=False,
        state_loaded_from_numpy_file=None,
        localized_state_site=2,
    )

    return psi0


def test_initial_state_is_ket():
    """
    Verify that the initial state is a ket.
    """

    print("\nFirst test: check that psi0 is a ket")

    psi0 = create_initial_state()

    is_ket = psi0.isket

    print("psi0 is a ket:", is_ket)

    assert is_ket


def test_initial_state_is_normalized():
    """
    Verify that the initial state has unit norm.
    """

    print("\nSecond test: check that psi0 is normalized")

    psi0 = create_initial_state()

    state_norm = psi0.norm()

    is_normalized = np.isclose(
        state_norm,
        1.0,
    )

    print("State norm:", state_norm)
    print("State is normalized:", is_normalized)

    assert is_normalized


def test_initial_state_is_pure():
    """
    Verify that the initial state has unit purity.

    The density matrix is stored in sparse CSR format. Its purity is
    calculated using the trace of its square.
    """

    print("\nThird test: check that psi0 is pure")

    psi0 = create_initial_state()

    rho = ket2dm(psi0).to("csr") # It would be better to get the CSR format of the psi0, as psi0 could be in a dense format depending of the choice of inital state [whether it is Fock state or a normalized state!]

    purity = float(
        np.real_if_close(
            (rho * rho).tr()
        )
    )

    is_pure = np.isclose(
        purity,
        1.0,
    )

    print(
        "Density-matrix format:",
        rho.data.__class__.__name__,
    )
    print("Purity:", purity)
    print("State is pure:", is_pure)

    assert is_pure