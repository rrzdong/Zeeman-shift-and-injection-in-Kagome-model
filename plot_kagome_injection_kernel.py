#!/usr/bin/env python3
"""Plot the normalized two-band Zeeman kernel Z_mn^(bβ) Delta_mn^a."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as PolygonPath
import numpy as np

from kagome_120_model import KagomeModel


# Plot parameters; model parameters are set in kagome_120_model.py.
band_n = 1                  # one-based initial-band number
band_m = 6                  # one-based final-band number
electric_direction = "x"   # electric-field direction b: x or y
magnetic_direction = "x"   # Zeeman-field direction beta: x, y, or z
current_direction = "x"    # current direction a: x or y
nk = 301                    # rectangular sampling of the hexagonal first BZ
band_isolation_tolerance = 1.0e-3  # eV; mask m/n if either touches another band
part = "Im"                 # plot "Im" or "Re" of Z times Delta

direction_indices = {"x": "1", "y": "2", "z": "3", "1": "1", "2": "2", "3": "3"}
root = Path(__file__).resolve().parent
model = KagomeModel()

plt.rcParams.update({"font.size": 24})


def main() -> None:
    n = band_n - 1
    m = band_m - 1
    electric_index = direction_indices[electric_direction]
    magnetic_index = direction_indices[magnetic_direction]
    current_index = direction_indices[current_direction]
    electric_axis = ("x", "y", "z")[int(electric_index) - 1]
    magnetic_axis = ("x", "y", "z")[int(magnetic_index) - 1]
    current_axis = ("x", "y", "z")[int(current_index) - 1]

    k1 = (2.0 * model.b1 + model.b2) / 3.0
    k2 = (model.b1 + 2.0 * model.b2) / 3.0
    k3 = (-model.b1 + model.b2) / 3.0
    corners = np.asarray((k1, k2, k3, -k1, -k2, -k3))
    kx = np.linspace(corners[:, 0].min(), corners[:, 0].max(), nk)
    ky = np.linspace(corners[:, 1].min(), corners[:, 1].max(), nk)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    points = np.column_stack((kx_grid.ravel(), ky_grid.ravel()))
    inside = PolygonPath(corners).contains_points(points, radius=1.0e-12)
    inside = inside.reshape(kx_grid.shape)

    kernel = np.full(kx_grid.shape, np.nan)
    spin_operator = model.SPIN_OPERATORS[magnetic_index]

    for iy, ix in np.argwhere(inside):
        kx_value = kx_grid[iy, ix]
        ky_value = ky_grid[iy, ix]
        energies, states = np.linalg.eigh(model.hamiltonian(kx_value, ky_value))

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
        r_nm = dh_nm / (-1.0j * (energies[m] - energies[n]))

        derivative_a = model.dh_dk(current_index, kx_value, ky_value)
        delta_v = np.real(
            state_m.conj() @ derivative_a @ state_m
            - state_n.conj() @ derivative_a @ state_n
        )
        zeeman_geometry = s_mn * r_nm
        kernel[iy, ix] = (
            zeeman_geometry.imag if part == "Im" else zeeman_geometry.real
        ) * delta_v

    finite = kernel[np.isfinite(kernel)]
    scale = np.max(np.abs(finite))

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.pcolormesh(
        kx_grid,
        ky_grid,
        kernel / scale,
        shading="auto",
        cmap=LinearSegmentedColormap.from_list(
            "RedWhiteBlue", ("#0000cc", "#ffffff", "#cc0000")
        ),
        vmin=-1.0,
        vmax=1.0,
        rasterized=True,
    )
    boundary = np.vstack((corners, corners[0]))
    axis.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=3)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
    axis.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
    axis.tick_params(
        which="major", direction="in", top=True, right=True,
        bottom=True, left=True, length=8, width=2,
    )
    direction_label = rf"{electric_axis}\tilde{{{magnetic_axis}}}"
    title = (
        rf"$\mathrm{{{part}}}\,\mathcal{{Z}}^{{{direction_label}}}_{{{band_m}{band_n}}}"
        rf"\Delta^{{{current_axis}}}_{{{band_m}{band_n}}}$"
    )
    if model.lam == 0.0:
        title += " w/o SOC"
    axis.set_title(title)

    colorbar = figure.colorbar(image, ax=axis, shrink=0.65, aspect=14)
    colorbar.set_ticks((-1.0, 0.0, 1.0))
    colorbar.set_ticklabels((r"$-$", r"$0$", r"$+$"))
    colorbar.ax.tick_params(direction="in", length=0, width=2)
    figure.tight_layout()

    output = root / "figure" / (
        f"kagome_injection_kernel_{part}Z{electric_axis}{magnetic_axis}_"
        f"Delta{current_axis}_bands{band_n}-{band_m}_"
        f"kappa{model.kappa:+d}_dphi{model.delta_phi:+.0f}deg_Nk{nk}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=500)
    plt.show()

    reflected = kernel[::-1, ::-1]
    valid = np.isfinite(kernel) & np.isfinite(reflected)
    even_residual = np.max(np.abs(kernel[valid] - reflected[valid])) / scale
    print(
        f"max |{part} Z^{electric_axis}~{magnetic_axis}_{band_m}{band_n} "
        f"Delta^{current_axis}_{band_m}{band_n}| = "
        f"{scale:.6e} eV Angstrom^2"
    )
    print(f"inversion-even residual = {even_residual:.3e}")
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
