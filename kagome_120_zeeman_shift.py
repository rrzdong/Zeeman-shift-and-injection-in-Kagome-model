#!/usr/bin/env python3
"""Kagome Zeeman shift current from the full covariant derivative."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kagome_120_model import KagomeModel


# One independent Hamiltonian for this calculation.
model = KagomeModel(lam=0.2)


# Response parameters
all_inplane_components = ("111", "112", "121", "122", "211", "212", "221", "222")
components = all_inplane_components
nk = 800
omega_min = 1.5           # photon energy (eV)
omega_max = 3.00          # photon energy (eV)
nomega = 400
eta = 0.050               # Gaussian width (eV)
sumrule_smear = 1.0e-3    # intermediate-band principal-value width (eV)
g = 2.0
workers = 8


# Constants used only in the final SI prefactor
e = 1.602176634e-19        # C
hbar = 1.054571817e-34     # J s
hbar_ev = hbar / e         # eV s
mu_B = 9.2740100783e-24    # J/T

root = Path(__file__).resolve().parent


def shift_conductivity_components(
    tensors: tuple[str, ...] = components,
    fractional_rows: np.ndarray | None = None,
    sumrule_width: float = sumrule_smear,
    eta_width: float = eta,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Calculate several components in one traversal of the k mesh."""

    photon_energy = np.linspace(omega_min, omega_max, nomega)

    if fractional_rows is None and workers > 1:
        full_grid = (np.arange(nk) + 0.5) / nk - 0.5
        tasks = [
            (tensors, rows, sumrule_width, eta_width)
            for rows in np.array_split(full_grid, workers)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            partial_results = list(pool.map(calculate_shift_rows, tasks))

        spectra = {
            tensor: sum(result[1][tensor] for result in partial_results)
            for tensor in tensors
        }
        minimum_transition = min(result[2] for result in partial_results)
        return photon_energy, spectra, minimum_transition

    conductivity = np.zeros((len(tensors), nomega), dtype=complex)
    minimum_transition = np.inf

    if fractional_rows is None:
        fractional_rows = (np.arange(nk) + 0.5) / nk - 0.5
    fractional_grid = (np.arange(nk) + 0.5) / nk - 0.5
    reciprocal_area = abs(np.linalg.det(np.column_stack((model.b1, model.b2))))
    k_weight = reciprocal_area / ((2.0 * np.pi) ** 2 * nk**2)

    spin_directions = set(tensor[2] for tensor in tensors)
    derivative_pairs = set((tensor[1], tensor[0]) for tensor in tensors)
    spin_derivative_pairs = set((tensor[2], tensor[0]) for tensor in tensors)
    derivative_directions = set(direction for pair in derivative_pairs for direction in pair)

    for u in fractional_rows:
        for v in fractional_grid:
            k = u * model.b1 + v * model.b2
            energies, states = np.linalg.eigh(model.hamiltonian(k[0], k[1]))

            occupations = (energies < model.E_F).astype(float)

            bra = states.conj().T
            spin = {
                direction: bra @ model.SPIN_OPERATORS[direction] @ states    # < m | \sigma | n >
                for direction in spin_directions
            }
            derivative = {
                direction: bra @ model.dh_dk(direction, k[0], k[1]) @ states  # < m | d H / dk | n >
                for direction in derivative_directions
            }
            second_derivative = {
                pair: bra @ model.d2h_dk2(*pair, k[0], k[1]) @ states       # < p | d2 H / dkb dka | q >
                for pair in derivative_pairs
            }

            for n in range(len(energies) - 1):
                for m in range(n + 1, len(energies)):
                    if occupations[n] == occupations[m]:
                        continue

                    # n is the initial band; m is the final band.
                    energy_mn = energies[m] - energies[n]
                    f_nm = float(occupations[n] - occupations[m])
                    minimum_transition = min(minimum_transition, energy_mn)

                    # Generalized derivative sigma^beta_mn;a.
                    spin_derivative = {}
                    inverse_mn = energy_mn / (
                        energy_mn**2 + sumrule_width**2
                    )
                    for magnetic_direction, current_direction in spin_derivative_pairs:
                        derivative_a = derivative[current_direction]
                        sigma_beta = spin[magnetic_direction]
                        value = (
                            -derivative_a[m, n]
                            * (
                                sigma_beta[m, m]
                                - sigma_beta[n, n]
                            )
                            * inverse_mn
                        )

                        for l in range(len(energies)):
                            if l in (n, m):
                                continue
                            energy_ml = energies[m] - energies[l]
                            energy_ln = energies[l] - energies[n]
                            inverse_ml = energy_ml / (
                                energy_ml**2 + sumrule_width**2
                            )
                            inverse_ln = energy_ln / (
                                energy_ln**2 + sumrule_width**2
                            )
                            value += (
                                derivative_a[m, l]
                                * sigma_beta[l, n]
                                * inverse_ml
                                - sigma_beta[m, l]
                                * derivative_a[l, n]
                                * inverse_ln
                            )

                        spin_derivative[
                            (magnetic_direction, current_direction)
                        ] = value

                    # Generalized derivative r^b_nm;a.
                    r_derivative = {}
                    energy_nm = -energy_mn
                    for electric_direction, current_direction in derivative_pairs:
                        derivative_b = derivative[electric_direction]
                        derivative_a = derivative[current_direction]

                        # Bracket in r^b_nm;a: second H derivative, both
                        # diagonal-velocity terms, then intermediate bands.
                        bracket = (
                            second_derivative[
                                (electric_direction, current_direction)
                            ][n, m]
                            + derivative_b[n, m]
                            * (
                                derivative_a[m, m]
                                - derivative_a[n, n]
                            )
                            / energy_nm
                            + derivative_a[n, m]
                            * (
                                derivative_b[m, m]
                                - derivative_b[n, n]
                            )
                            / energy_nm
                        )

                        for l in range(len(energies)):
                            if l in (n, m):
                                continue
                            energy_nl = energies[n] - energies[l]
                            energy_lm = energies[l] - energies[m]
                            inverse_nl = energy_nl / (energy_nl**2 + sumrule_width**2)
                            inverse_lm = energy_lm / (energy_lm**2 + sumrule_width**2)
                            bracket += (
                                derivative_a[n, l]
                                * derivative_b[l, m]
                                * inverse_nl
                                - derivative_b[n, l]
                                * derivative_a[l, m]
                                * inverse_lm
                            )

                        r_derivative[(electric_direction, current_direction)] = (
                            bracket / (1.0j * energy_nm)
                        )

                    kernel = []
                    for a, b, beta in tensors:
                        r_nm = (
                            derivative[b][n, m]
                            / (1.0j * energy_nm)
                        )
                        kernel.append(
                            spin_derivative[(beta, a)] * r_nm
                            - spin[beta][m, n]
                            * r_derivative[(b, a)]
                        )
                    kernel = np.asarray(kernel)
                    energy_difference = energy_mn - photon_energy
                    gaussian_delta = (
                        np.exp(-(energy_difference / eta_width) ** 2)
                        / (eta_width * np.sqrt(np.pi))
                    )
                    conductivity += (
                        f_nm
                        * kernel[:, np.newaxis]
                        * gaussian_delta[np.newaxis, :]
                        * k_weight
                    )

    final_si_prefactor = 0.5j * np.pi * g * mu_B * e**2 * hbar_ev / hbar**2
    conductivity *= final_si_prefactor
    spectra = {
        tensor: conductivity[index]
        for index, tensor in enumerate(tensors)
    }
    return photon_energy, spectra, float(minimum_transition)


def calculate_shift_rows(
    task: tuple[tuple[str, ...], np.ndarray, float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Calculate one group of k-grid rows in a worker process."""

    tensors, fractional_rows, sumrule_width, eta_width = task
    return shift_conductivity_components(
        tensors, fractional_rows, sumrule_width, eta_width
    )


def main() -> None:
    photon_energy, spectra, minimum_transition = shift_conductivity_components(
        sumrule_width=sumrule_smear,
        eta_width=eta,
    )
    output = root / "data" / (
        f"Zeeman_shift-eta={eta * 1.0e3:g}meV-"
        f"sum_smear={sumrule_smear * 1.0e3:g}meV-Nk={nk}.txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [photon_energy]
    labels = ["Energy"]
    for tensor in components:
        columns.extend((spectra[tensor].real, spectra[tensor].imag))
        labels.extend((f"Re {tensor}", f"Im {tensor}"))

    header = (
        "Units: energy=eV; conductivity=A/(V T)\n"
        "Kagome Zeeman shift v1; "
        "tensor order=(current,electric,magnetic); "
        "indices=(a,b,beta); "
        f"components={','.join(components)}; t={model.t} eV; J={model.J} eV; "
        f"lambda={model.lam} eV; kappa={model.kappa}; "
        f"DeltaPhi={model.delta_phi} deg; theta={model.theta} deg; "
        f"a={model.a} A; "
        f"Nk={nk}x{nk}; EF={model.E_F} eV; occupation=step(EF-E); T=0 K; "
        f"eta={eta} eV; sumrule_smear={sumrule_smear} eV; g={g}\n"
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
