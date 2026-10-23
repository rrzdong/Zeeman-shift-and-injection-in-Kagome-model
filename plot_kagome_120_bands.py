#!/usr/bin/env python3
"""Plot the SOC bands and the Dirac gap of the current magnetic configuration."""

from pathlib import Path

import numpy as np

import kagome_120_model as parameters
from kagome_120_model import KagomeModel


points_per_segment = 1000
soc_strength = parameters.lam
band_color = "#111111"

root = Path(__file__).resolve().parent
output_directory = root / "figure"

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update({
    "font.size": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
})


def main() -> None:
    kappa = parameters.kappa
    delta_phi = parameters.delta_phi
    theta = parameters.theta
    model = KagomeModel(
        lam=soc_strength,
        kappa=kappa,
        delta_phi=delta_phi,
        theta=theta,
    )

    m_point = (model.b1 + model.b2) / 2.0
    k_point = (2.0 * model.b1 + model.b2) / 3.0
    gamma = np.zeros(2)
    # Inversion-related path through the center of the first Brillouin zone.
    corners = (m_point, k_point, gamma, -k_point, -m_point)

    path = [corners[0]]
    distance = [0.0]
    ticks = [0.0]
    for start, stop in zip(corners[:-1], corners[1:]):
        for k in np.linspace(start, stop, points_per_segment + 1)[1:]:
            distance.append(distance[-1] + np.linalg.norm(k - path[-1]))
            path.append(k)
        ticks.append(distance[-1])

    output_directory.mkdir(parents=True, exist_ok=True)

    bands = np.asarray(
        [np.linalg.eigvalsh(model.hamiltonian(*k)) for k in path]
    ) - model.E_F

    figure, axis = plt.subplots(figsize=(8, 6))
    for band in bands.T:
        axis.plot(distance, band, color=band_color, linestyle="-", linewidth=3)

    for tick in ticks:
        axis.axvline(tick, color="0.75", linewidth=0.7, zorder=0)
    axis.axhline(
        0.0, color="black", linestyle="--", linewidth=0.9, zorder=0
    )
    axis.set_xlim(distance[0], distance[-1])
    axis.set_ylim(-4.2, 5.2)
    axis.set_xticks(ticks, ("M", "K", r"$\Gamma$", "−K", "−M"))
    axis.set_ylabel(r"$E-E_F$ (eV)")
    axis.yaxis.set_major_locator(MultipleLocator(2.0))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=2.0,
    )
    axis.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=1.5,
    )
    figure.tight_layout()

    output = output_directory / (
        f"kagome_bands_kappa{kappa:+d}_dphi{delta_phi:+.0f}deg_"
        "soc_M_K_Gamma_minusK_minusM.png"
    )
    figure.savefig(output, dpi=500)
    plt.show()
    plt.close(figure)
    print(f"Figure: {output}")

    # Find and plot the minimum SOC Dirac gap between bands 1 and 2
    # in a local window around K.
    distance_array = np.asarray(distance)
    k_window = 0.1
    k_distance = ticks[1]  # K on the M-K-Gamma-(-K)-(-M) path
    gap_region = (distance_array >= k_distance - k_window) & (
        distance_array <= k_distance + k_window
    )
    gap = bands[gap_region, 1] - bands[gap_region, 0]
    gap_index = np.flatnonzero(gap_region)[np.argmin(gap)]
    gap_energy = 0.5 * (bands[gap_index, 0] + bands[gap_index, 1])

    figure, axis = plt.subplots(figsize=(8, 6))
    for band_index in (0, 1):
        axis.plot(
            distance_array[gap_region],
            bands[gap_region, band_index],
            color=band_color,
            linestyle="-",
            linewidth=3,
        )

    for tick in (k_distance,):
        axis.axvline(tick, color="0.75", linewidth=0.7, zorder=0)
    axis.set_xlim(k_distance - k_window, k_distance + k_window)
    axis.set_ylim(gap_energy - 0.15, gap_energy + 0.15)
    axis.set_xticks((k_distance,))
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelbottom=False,
        labelleft=False,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()

    gap_output = output_directory / (
        f"kagome_gap_kappa{kappa:+d}_dphi{delta_phi:+.0f}deg_"
        "soc_M_K_Gamma_minusK_minusM.png"
    )
    figure.savefig(gap_output, dpi=500)
    plt.show()
    plt.close(figure)
    print(
        f"Gap figure: {gap_output} "
        f"(minimum SOC gap E2-E1 = {1.0e3 * gap.min():.3f} meV)"
    )

if __name__ == "__main__":
    main()
