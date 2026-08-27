"""
Time evolution and visualization for the reduced two-dimensional
Bose-Hubbard model.

This module solves the time-dependent Schrödinger equation, calculates
boson-number expectation values, and provides plotting and animation
methods for a fixed-particle-number two-dimensional Bose-Hubbard system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from qutip import Qobj, basis, qeye, sesolve
from scipy.sparse import diags

__all__ = ["BoseHubbardDynamics2D"]



class BoseHubbardDynamics2D:
    """
    Represent the dynamics of a reduced two-dimensional Bose-Hubbard model.

    The class creates local and total boson-number operators, prepares
    different initial states, solves the time-dependent Schrödinger
    equation, and calculates boson-number expectation values.

    Parameters
    ----------
    model : ReducedBoseHubbard2D
        Reduced two-dimensional Bose-Hubbard model containing the
        Hamiltonian, basis, lattice, and bonds.
    t_start : float, optional
        Initial simulation time.
    t_end : float, optional
        Final simulation time.
    num_times : int, optional
        Number of simulation times.

    Attributes
    ----------
    model : ReducedBoseHubbard2D
        Reduced two-dimensional Bose-Hubbard model.
    dimension : int
        Dimension of the reduced Hilbert space.
    tlist : numpy.ndarray
        Times at which the state is calculated.
    state_index : dict
        Mapping from occupation states to reduced-basis indices.
    N_op : qutip.Qobj
        Total boson-number operator.
    n_ops : list of qutip.Qobj
        Local boson-number operators.
    """

    random_fock_seed = 42
    random_superposition_seed = 123

    def __init__(
        self,
        model,
        t_start=0,
        t_end=10,
        num_times=100
    ):
        """
        Initialize the dynamics and construct the number operators.

        Parameters
        ----------
        model : ReducedBoseHubbard2D
            Reduced two-dimensional Bose-Hubbard model containing the
            Hamiltonian and fixed-particle-number basis.
        t_start : float, optional
            Initial simulation time.
        t_end : float, optional
            Final simulation time.
        num_times : int, optional
            Number of simulation times.
        """
        self.model = model
        self.dimension = len(
            self.model.basis_states
        )

        self.tlist = np.linspace(
            t_start,
            t_end,
            num_times
        )

        self.state_index = {
            state: index
            for index, state in enumerate(
                self.model.basis_states
            )
        }

        self.N_op = (
            self.model.number_of_bosons
            * qeye(self.dimension)
        )

        self.n_ops = []

        for site in range(
            self.model.number_of_sites
        ):
            diagonal = np.fromiter(
                (
                    state[site]
                    for state in self.model.basis_states
                ),
                dtype=np.float64,
                count=self.dimension
            )

            sparse_operator = diags(
                diagonal,
                offsets=0,
                format="csr"
            )

            self.n_ops.append(
                Qobj(sparse_operator)
            )

    def run_dynamics(
        self,
        random_fock_state=False,
        random_superposition_of_fock_states=True,
        state_loaded_from_numpy_file=None,
        localized_state_site=None
    ):
        """
        Prepare an initial state and solve the Schrödinger equation.

        Exactly one initial-state option must be selected. The initial
        state can be a random Fock state, a random superposition, a state
        loaded from a NumPy file, or a state localized at one site.

        Parameters
        ----------
        random_fock_state : bool, optional
            If True, use a randomly selected reduced-basis Fock state.
        random_superposition_of_fock_states : bool, optional
            If True, use a normalized random complex superposition of
            all reduced-basis Fock states.
        state_loaded_from_numpy_file : str or path-like or None, optional
            NumPy file containing either occupation numbers or
            reduced-basis coefficients.
        localized_state_site : int or None, optional
            Site at which all bosons are initially localized.

        Returns
        -------
        psi0 : qutip.Qobj
            Initial state ket.
        result : qutip.solver.Result
            Result returned by the QuTiP Schrödinger-equation solver.
        site_number_expect : numpy.ndarray
            Local boson-number expectation values. The first index
            identifies the site and the second identifies time.
        total_number_expect : numpy.ndarray
            Total boson-number expectation value at every time.

        Raises
        ------
        ValueError
            If exactly one initial-state option is not selected.
        ValueError
            If a loaded state is invalid.
        ValueError
            If the localized-state site is invalid.
        """
        number_of_choices = sum([
            random_fock_state,
            random_superposition_of_fock_states,
            state_loaded_from_numpy_file is not None,
            localized_state_site is not None
        ])

        if number_of_choices != 1:
            raise ValueError(
                "Choose exactly one initial-state option."
            )

        if random_fock_state:
            generator = np.random.default_rng(
                self.random_fock_seed
            )

            state_index = generator.integers(
                0,
                self.dimension
            )

            psi0 = basis(
                self.dimension,
                state_index
            )

        elif random_superposition_of_fock_states:
            generator = np.random.default_rng(
                self.random_superposition_seed
            )

            coefficients = (
                generator.normal(
                    size=self.dimension
                )
                + 1j
                * generator.normal(
                    size=self.dimension
                )
            )

            coefficients /= np.linalg.norm(
                coefficients
            )

            psi0 = Qobj(
                coefficients.reshape(
                    (self.dimension, 1)
                )
            )

        elif state_loaded_from_numpy_file is not None:
            loaded_state = np.asarray(
                np.load(
                    state_loaded_from_numpy_file,
                    allow_pickle=False
                )
            ).flatten()

            is_fock_state = (
                len(loaded_state)
                == self.model.number_of_sites

                and np.all(
                    np.isreal(loaded_state)
                )

                and np.all(
                    loaded_state >= 0
                )

                and np.all(
                    loaded_state
                    == loaded_state.astype(int)
                )

                and np.sum(loaded_state)
                == self.model.number_of_bosons
            )

            if is_fock_state:
                initial_state = tuple(
                    loaded_state.astype(int)
                )

                if initial_state not in self.state_index:
                    raise ValueError(
                        "The loaded Fock state is not in the basis."
                    )

                state_index = self.state_index[
                    initial_state
                ]

                psi0 = basis(
                    self.dimension,
                    state_index
                )

                # print(
                #     "Fock state loaded from NumPy file:",
                #     initial_state
                # )

            elif len(loaded_state) == self.dimension:
                coefficients = loaded_state.astype(
                    complex
                )

                norm = np.linalg.norm(
                    coefficients
                )

                if norm == 0:
                    raise ValueError(
                        "The initial state cannot be the zero vector."
                    )

                coefficients /= norm

                psi0 = Qobj(
                    coefficients.reshape(
                        (self.dimension, 1)
                    )
                )

                # print(
                #     "State loaded from NumPy file."
                # )

            else:
                raise ValueError(
                    "The NumPy file must contain either site "
                    "occupations or reduced-basis coefficients."
                )

        else:
            if not isinstance(
                localized_state_site,
                (int, np.integer)
            ):
                raise ValueError(
                    "The localized-state site must be an integer."
                )

            if not (
                0
                <= localized_state_site
                < self.model.number_of_sites
            ):
                raise ValueError(
                    "The localized-state site is outside the lattice."
                )

            initial_state = (
                [0] * self.model.number_of_sites
            )

            initial_state[
                localized_state_site
            ] = self.model.number_of_bosons

            initial_state = tuple(
                initial_state
            )

            if initial_state not in self.state_index:
                raise ValueError(
                    "The localized state is not in the basis."
                )

            state_index = self.state_index[
                initial_state
            ]

            psi0 = basis(
                self.dimension,
                state_index
            )

        result = sesolve(
            self.model.H,
            psi0,
            self.tlist,
            e_ops=self.n_ops + [self.N_op]
        )

        site_number_expect = np.real(
            np.asarray(
                result.expect[
                    :self.model.number_of_sites
                ]
            )
        )

        site_number_expect = np.round(
            site_number_expect,
            12
        )

        total_number_expect = np.real(
            result.expect[
                self.model.number_of_sites
            ]
        )

        total_number_expect = np.round(
            total_number_expect,
            12
        )

        return (
            psi0,
            result,
            site_number_expect,
            total_number_expect
        )

    def plot_site_numbers(
        self,
        site_number_expect
    ):
        """
        Plot the local boson-number expectation values.

        Parameters
        ----------
        site_number_expect : numpy.ndarray
            Local boson-number expectation values. The first index
            identifies the lattice site and the second identifies time.

        Returns
        -------
        None
            The plot is displayed directly.
        """
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        number_of_sites = self.model.number_of_sites

        colors = plt.colormaps["tab20"].resampled(
            number_of_sites
        )

        for site in range(number_of_sites):
            ax.plot(
                self.tlist,
                site_number_expect[site],
                color=colors(site),
                label=fr"$\langle n_{{{site}}}\rangle$"
            )

        ax.set_xlim(
            0,
            self.tlist[-1]
        )

        highest_value = np.max(
            site_number_expect
        )

        top_limit = max(
            highest_value * 1.35,
            1
        )

        ax.set_ylim(
            0,
            top_limit
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Expected boson number")

        ax.set_title(
            "Boson number at each lattice site"
        )

        ax.legend(
            loc="upper right",
            ncol=min(
                number_of_sites,
                5
            ),
            frameon=True
        )

        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_total_number(
        self,
        total_number_expect
    ):
        """
        Plot the total boson-number expectation value.

        Parameters
        ----------
        total_number_expect : numpy.ndarray
            Total boson-number expectation value at every simulation
            time.

        Returns
        -------
        None
            The plot is displayed directly.
        """
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            self.tlist,
            total_number_expect,
            color="blue",
            linewidth=1.2,
            label=r"$\langle N\rangle$"
        )

        ax.set_xlim(
            0,
            self.tlist[-1]
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        ax.ticklabel_format(
            axis="y",
            style="plain",
            useOffset=False
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Total number of bosons")

        ax.set_title(
            "Expectation value of the total number operator"
        )

        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def animate_site_numbers(
        self,
        site_number_expect
    ):
        """
        Animate the local boson-number expectation values on the
        two-dimensional lattice.

        Purple represents low occupation and yellow represents high
        occupation.

        Parameters
        ----------
        site_number_expect : numpy.ndarray
            Local boson-number expectation values. The first index
            identifies the lattice site and the second identifies time.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation showing the boson-number distribution over time.
        """
        from matplotlib.colors import PowerNorm

        positions = np.asarray(
            self.model.lattice.positions
        )

        figure, axis = plt.subplots(
            figsize=(6, 7)
        )

        color_maximum = np.max(
            site_number_expect
        )

        if color_maximum <= 0:
            color_maximum = 1.0

        color_normalization = PowerNorm(
            gamma=0.3,
            vmin=0,
            vmax=color_maximum
        )

        for first_site, second_site in self.model.bonds:
            axis.plot(
                [
                    positions[first_site, 0],
                    positions[second_site, 0]
                ],
                [
                    positions[first_site, 1],
                    positions[second_site, 1]
                ],
                color="0.65",
                linewidth=2,
                zorder=1
            )

        points = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            c=site_number_expect[:, 0],
            s=120,
            cmap="viridis",
            norm=color_normalization,
            edgecolors="black",
            linewidths=0.5,
            zorder=2
        )

        colorbar = figure.colorbar(
            points,
            ax=axis,
            fraction=0.05,
            pad=0.04
        )

        colorbar.set_label(
            "Expected boson number",
            fontsize=11
        )

        axis.set_aspect("equal")
        axis.axis("off")

        horizontal_range = np.ptp(
            positions[:, 0]
        )

        vertical_range = np.ptp(
            positions[:, 1]
        )

        horizontal_padding = max(
            0.15 * horizontal_range,
            1.0
        )

        vertical_padding = max(
            0.15 * vertical_range,
            1.0
        )

        axis.set_xlim(
            positions[:, 0].min()
            - horizontal_padding,

            positions[:, 0].max()
            + horizontal_padding
        )

        axis.set_ylim(
            positions[:, 1].min()
            - vertical_padding,

            positions[:, 1].max()
            + vertical_padding
        )

        title = axis.set_title(
            f"Boson hopping, t = {self.tlist[0]:.2f}"
        )

        figure.patch.set_facecolor("white")
        axis.set_facecolor("white")
        figure.tight_layout()

        def update(frame):
            points.set_array(
                site_number_expect[:, frame]
            )

            title.set_text(
                f"Boson hopping, "
                f"t = {self.tlist[frame]:.2f}"
            )

            return points, title

        animation = FuncAnimation(
            figure,
            update,
            frames=len(self.tlist),
            interval=25,
            blit=False
        )

        plt.close(figure)

        return animation

