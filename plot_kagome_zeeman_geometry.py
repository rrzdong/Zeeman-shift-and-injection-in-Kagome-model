#!/usr/bin/env python3
"""Plot a two-band Zeeman geometry over the 2D Kagome Brillouin zone.

The plotted gauge-invariant band-pair quantity is

    Z_mn^(bβ)(k) = r_nm^b(k) s_mn^β(k),

with r_nm^b = (dH/dk_b)_nm / [i(E_n-E_m)].  Band numbers below are
one-based, while the model spin operators use Pauli matrices.  Therefore
Z has units of Angstrom.  A band-resolved value is not defined at an exact
degeneracy, so those points are masked.
"""

from pathlib import Path

import numpy as np

from kagome_120_model import KagomeModel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as PolygonPath

# Plot parameters; model parameters are set in kagome_120_model.py.
band_n = 1                  # one-based initial-band number
band_m = 6                  # one-based final-band number
electric_direction = "x"   # electric-field direction b: x or y
magnetic_direction = "x"   # Zeeman-field direction beta: x, y, or z
nk = 301                    # rectangular sampling of the hexagonal first BZ
band_isolation_tolerance = 1.0e-3  # eV; mask m/n if either touches another band
part = "Im"                 # plot "Im" or "Re"

direction_indices = {"x": "1", "y": "2", "z": "3", "1": "1", "2": "2", "3": "3"}


def direction_index(direction: str) -> str:
    """Convert a displayed x/y/z direction to the model's 1/2/3 index."""
    return direction_indices[direction]

def direction_axis(direction: str) -> str:
    return ("x", "y", "z")[int(direction_index(direction)) - 1]

root = Path(__file__).resolve().parent
output_directory = root / "figure"

model = KagomeModel()

plt.rcParams.update({"font.size": 24})


def first_bz() -> np.ndarray:
    """Return the six K-point corners of the first hexagonal BZ."""

    k1 = (2.0 * model.b1 + model.b2) / 3.0
    k2 = (model.b1 + 2.0 * model.b2) / 3.0
    k3 = (-model.b1 + model.b2) / 3.0
    return np.asarray((k1, k2, k3, -k1, -k2, -k3))


def calculate_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Z_mn^(bβ) on a rectangular grid covering the first BZ."""

    n = band_n - 1
    m = band_m - 1
    electric_index = direction_index(electric_direction)
    magnetic_index = direction_index(magnetic_direction)

    corners = first_bz()
    kx = np.linspace(corners[:, 0].min(), corners[:, 0].max(), nk)
    ky = np.linspace(corners[:, 1].min(), corners[:, 1].max(), nk)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    points = np.column_stack((kx_grid.ravel(), ky_grid.ravel()))
    inside = PolygonPath(corners).contains_points(points, radius=1.0e-12)
    inside = inside.reshape(kx_grid.shape)

    geometry = np.full(kx_grid.shape, np.nan + 1.0j * np.nan, dtype=complex)
    spin_operator = model.SPIN_OPERATORS[magnetic_index]

    for iy, ix in np.argwhere(inside):
        kx_value = kx_grid[iy, ix]
        ky_value = ky_grid[iy, ix]
        energies, states = np.linalg.eigh(model.hamiltonian(kx_value, ky_value))
        transition_energy = energies[m] - energies[n]

        separation_m = np.abs(energies - energies[m])
        separation_n = np.abs(energies - energies[n])
        separation_m[m] = np.inf
        separation_n[n] = np.inf
        if (
            separation_m.min() < band_isolation_tolerance
            or separation_n.min() < band_isolation_tolerance
        ):
            continue

        state_n = states[:, n]
        state_m = states[:, m]
        s_mn = state_m.conj() @ spin_operator @ state_n
        dh_nm = (
            state_n.conj()
            @ model.dh_dk(electric_index, kx_value, ky_value)
            @ state_m
        )
        r_nm = dh_nm / (-1.0j * transition_energy)
        geometry[iy, ix] = s_mn * r_nm

    return kx_grid, ky_grid, geometry


def plot_part(
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    values: np.ndarray,
    part: str,
    title: str | None = None,
    filename_prefix: str = "kagome_zeeman_geometry",
) -> Path:
    """Plot one normalized real or imaginary geometry panel."""

    finite_values = values[np.isfinite(values)]
    plotted_values = values / np.max(np.abs(finite_values))

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.pcolormesh(
        kx_grid,
        ky_grid,
        plotted_values,
        shading="auto",
        cmap=LinearSegmentedColormap.from_list(
            "RedWhiteBlue", ("#0000cc", "#ffffff", "#cc0000")
        ),
        vmin=-1.0,
        vmax=1.0,
        rasterized=True,
    )

    corners = first_bz()
    boundary = np.vstack((corners, corners[0]))
    axis.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=3)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
    axis.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    axis.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=8,
        width=2,
    )
    if title is None:
        direction_label = (
            rf"{direction_axis(electric_direction)}"
            rf"\tilde{{{direction_axis(magnetic_direction)}}}"
        )
        title = rf"$\mathrm{{{part}}}\,\mathcal{{Z}}^{{{direction_label}}}_{{mn}}$"
        if model.lam == 0.0:
            title += " w/o SOC"
    axis.set_title(title)

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.65,
        aspect=14,
    )
    colorbar.set_ticks((-1.0, 0.0, 1.0))
    colorbar.set_ticklabels((r"$-$", r"$0$", r"$+$"))
    colorbar.ax.tick_params(direction="in", length=0, width=2)
    figure.tight_layout()

    output_directory.mkdir(parents=True, exist_ok=True)
    if filename_prefix == "kagome_zeeman_geometry":
        filename = (
            f"{filename_prefix}_{part.lower()}_"
            f"Z{direction_axis(electric_direction)}"
            f"{direction_axis(magnetic_direction)}_bands{band_n}-{band_m}_"
        )
    else:
        filename = f"{filename_prefix}_"
    filename += (
        f"kappa{model.kappa:+d}_dphi{model.delta_phi:+.0f}deg_Nk{nk}"
        f"{'_normalized' if filename_prefix == 'kagome_zeeman_geometry' else ''}.png"
    )
    output = output_directory / filename
    figure.savefig(output, dpi=500)
    plt.show()
    return output


def main() -> None:
    kx_grid, ky_grid, geometry = calculate_geometry()
    values = geometry.imag if part == "Im" else geometry.real
    output = plot_part(kx_grid, ky_grid, values, part)

    reflected = geometry[::-1, ::-1]
    valid = np.isfinite(geometry) & np.isfinite(reflected)
    scale = np.max(np.abs(geometry[valid]))
    odd_residual = np.max(np.abs(geometry[valid] + reflected[valid])) / scale

    print(
        f"kappa={model.kappa:+d}, DeltaPhi={model.delta_phi:+.0f} deg: "
        f"max|Z^{direction_axis(electric_direction)}"
        f"{direction_axis(magnetic_direction)}_"
        f"{band_m}{band_n}|={scale:.6e} Angstrom; "
        f"inversion-odd residual={odd_residual:.3e}"
    )
    print(f"{part} figure: {output}")


if __name__ == "__main__":
    main()
