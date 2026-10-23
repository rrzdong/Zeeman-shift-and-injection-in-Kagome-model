#!/usr/bin/env python3
"""In-plane Zeeman injection conductivity of the Kagome model."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kagome_120_model import KagomeModel


# One independent Hamiltonian for this calculation.
model = KagomeModel(lam=0.2)


# Tensor order: (a, b, beta) = (current, electric, magnetic).
# Response parameters
all_inplane_components = ("111", "112", "121", "122", "211", "212", "221", "222")
components = all_inplane_components
nk = 800
omega_min = 1.5           # photon energy (eV)
omega_max = 3.00           # photon energy (eV)
nomega = 400
eta = 0.050                # Gaussian width (eV)
g = 2.0
tau_ps = 1.0
workers = 8


# Constants used only in the final SI prefactor
e = 1.602176634e-19        # C
hbar = 1.054571817e-34     # J s
mu_B = 9.2740100783e-24    # J/T

root = Path(__file__).resolve().parent


def injection_conductivity_components(
    tensors: tuple[str, ...] = components,
    fractional_rows: np.ndarray | None = None,
    eta_width: float = eta,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Calculate several components in one traversal of the k mesh."""

    photon_energy = np.linspace(omega_min, omega_max, nomega)

    if fractional_rows is None and workers > 1:
        full_grid = (np.arange(nk) + 0.5) / nk - 0.5
        tasks = [
            (tensors, rows, eta_width)
            for rows in np.array_split(full_grid, workers)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partial_results = list(pool.map(calculate_injection_rows, tasks))

        spectra = {
            tensor: sum(result[1][tensor] for result in partial_results)
            for tensor in tensors
        }
        minimum_transition = min(result[2] for result in partial_results)
        return photon_energy, spectra, minimum_transition

    conductivity = np.zeros((len(tensors), len(photon_energy)), dtype=complex)
    minimum_transition = np.inf

    if fractional_rows is None:
        fractional_rows = (np.arange(nk) + 0.5) / nk - 0.5
    fractional_grid = (np.arange(nk) + 0.5) / nk - 0.5
    reciprocal_area = abs(
        np.linalg.det(np.column_stack((model.b1, model.b2)))
    )
    k_weight = reciprocal_area / ((2.0 * np.pi) ** 2 * nk**2)

    for u in fractional_rows:
        for v in fractional_grid:
            k = u * model.b1 + v * model.b2
            energies, states = np.linalg.eigh(model.hamiltonian(k[0], k[1]))

            occupations = (energies < model.E_F).astype(float)

            bra = states.conj().T
            spin_directions = {beta for a, b, beta in tensors}
            sigma = {
                direction: bra @ model.SPIN_OPERATORS[direction] @ states
                for direction in spin_directions
            }
            derivative = {
                direction: (
                    bra @ model.dh_dk(direction, k[0], k[1]) @ states
                )
                for direction in "12"
            }

            # n is the occupied initial band; m is the empty final band.
            for n in range(len(energies) - 1):
                for m in range(n + 1, len(energies)):
                    f_nm = float(occupations[n] - occupations[m])
                    if f_nm == 0.0:
                        continue

                    transition_energy = energies[m] - energies[n]
                    minimum_transition = min(
                        minimum_transition,
                        transition_energy,
                    )

                    # r^b_nm = v^b_nm/(i omega_nm)
                    #        = (dH/dk_b)_nm/[i(E_n-E_m)]
                    r_nm = {
                        direction: derivative[direction][n, m]
                        / (-1.0j * transition_energy)
                        for direction in "12"
                    }
                    derivative_difference = {
                        direction: (
                            derivative[direction][m, m]
                            - derivative[direction][n, n]
                        )
                        for direction in "12"
                    }
                    # sigma[a,b,beta]: s^beta_mn r^b_nm (v^a_mm-v^a_nn).
                    matrix_element = np.asarray(
                        [
                            sigma[magnetic_direction][m, n]
                            * r_nm[electric_direction]
                            * derivative_difference[current_direction]
                            for current_direction, electric_direction, magnetic_direction
                            in tensors
                        ]
                    )

                    energy_difference = transition_energy - photon_energy
                    gaussian_delta = (
                        np.exp(-(energy_difference / eta_width) ** 2)
                        / (eta_width * np.sqrt(np.pi))
                    )
                    conductivity += (
                        f_nm
                        * matrix_element[:, np.newaxis]
                        * gaussian_delta[np.newaxis, :]
                        * k_weight
                    )

    final_si_prefactor = (
        1.0
        * np.pi
        * g
        * mu_B
        * e**2
        * (tau_ps * 1.0e-12)
        / hbar**2
    )
    conductivity *= final_si_prefactor
    spectra = {
        tensor: conductivity[index]
        for index, tensor in enumerate(tensors)
    }
    return photon_energy, spectra, float(minimum_transition)


def calculate_injection_rows(
    task: tuple[tuple[str, ...], np.ndarray, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Calculate one group of k-grid rows in a worker process."""

    tensors, fractional_rows, eta_width = task
    return injection_conductivity_components(tensors, fractional_rows, eta_width)


def main() -> None:
    photon_energy, spectra, minimum_transition = (
        injection_conductivity_components(eta_width=eta)
    )
    output = root / "data" / f"Zeeman_injection-eta={eta * 1.0e3:g}meV-Nk={nk}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [photon_energy]
    labels = ["Energy"]
    for tensor in components:
        columns.extend((spectra[tensor].real, spectra[tensor].imag))
        labels.extend((f"Re {tensor}", f"Im {tensor}"))

    header = (
        "Units: energy=eV; conductivity=A/(V T)\n"
        "Kagome Zeeman injection; tensor order=(current,electric,magnetic); "
        "indices=(a,b,beta); "
        f"components={','.join(components)}; t={model.t} eV; J={model.J} eV; "
        f"lambda={model.lam} eV; kappa={model.kappa}; "
        f"DeltaPhi={model.delta_phi} deg; theta={model.theta} deg; "
        f"a={model.a} A; "
        f"Nk={nk}x{nk}; EF={model.E_F} eV; "
        f"occupation=step(EF-E); T=0 K; eta={eta} eV; g={g}; tau={tau_ps} ps\n"
        + "\t".join(labels)
    )
    header = header.replace("; ", "\n")
    np.savetxt(output, np.column_stack(columns), header=header)

    print(f"Minimum transition: {minimum_transition:.6f} eV")
    for tensor in components:
        print(f"{tensor}: peak |sigma| = {np.max(np.abs(spectra[tensor])):.6e} A/(V T)")
    print(f"Data: {output}")


if __name__ == "__main__":
    main()
