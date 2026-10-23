#!/usr/bin/env python3
"""Plot circular and linear Zeeman-injection spectra with SOC."""

import os
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
soc_file = root / "data" / "Zeeman_injection-eta=100meV-Nk=800.txt"
output = root / "figure" / "Zeeman-injection-maintxt.png"
os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update({"font.size": 26, "axes.titlesize": 26})

def main() -> None:
    soc = np.loadtxt(soc_file, comments="#")
    energy = soc[:, 0]

    soc_circular = soc[:, 2] * 1.0e6             # Im sigma_111
    soc_linear = soc[:, 11] * 1.0e6              # Re sigma_212 in a;b beta order

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(energy, soc_linear, color="#d62728", linestyle="-", linewidth=4,zorder=10)
    axis.plot(energy, soc_circular, color="#1f77b4", linestyle="--", linewidth=4,zorder=10)
    axis.axhline(0.0, color="gray", linestyle="--", linewidth=2,zorder=1)
    axis.set_xlim(energy[0], energy[-1])
    axis.set_ylim(-5.5, 0.5)
    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel(r"$\eta_\mathrm{2D}$ ($\mu\mathrm{A\,V^{-1}\,T^{-1}}$)")
    axis.set_title("Zeeman injection current")
    axis.yaxis.set_major_locator(MultipleLocator(1.0))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="major", direction="in", top=True, right=True,
                     length=6, width=2.0)
    axis.tick_params(which="minor", direction="in", top=True, right=True,
                     length=3, width=1.5)

    legend = axis.legend(
        handles=[
            Line2D([], [], color="#d62728", linestyle="-", linewidth=4,
                   label=r"$\eta_{\mathrm{L}}^{y;x\tilde{y}}$"),
            Line2D([], [], color="#1f77b4", linestyle="--", linewidth=4,
                   label=r"$\eta_{\mathrm{C}}^{x;x\tilde{x}}$"),
        ],
        loc="lower left",
        ncol=1,
        fontsize=26,
        frameon=False,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=500)
    plt.show(block=True)
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
