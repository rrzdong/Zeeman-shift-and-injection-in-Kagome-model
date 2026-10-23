#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "k48-sc-eta-0p01"

curves = [
    ("zeeman_shift_122.OUT", r"$\eta_L^{y;y\tilde{x}}$", "#d62728", "--"),
    ("zeeman_shift_212.OUT", r"$\eta_L^{y;x\tilde{y}}$", "#1f77b4", "-"),
]

fig, ax = plt.subplots(figsize=(8, 6))
for filename, label, color, linestyle in curves:
    energy, values = np.loadtxt(DATA / filename, usecols=(0, 1), unpack=True)
    ax.plot(energy, values, color=color, linestyle=linestyle,
            linewidth=3.5, label=label)

ax.axhline(0, color="0.45", linewidth=1.1, zorder=0)
ax.set_xlim(energy[0], 2.0)
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.set_xlabel("Photon energy (eV)", fontsize=26)
ax.set_ylabel(r"$\eta$ ($\mathrm{Am^{-1}T^{-1}V^{-1}}$)", fontsize=26)
ax.set_title("Zeeman shift current", fontsize=26, pad=12)
ax.tick_params(which="both", direction="in", top=True, right=True,
               labelsize=26, width=1.4, length=7)
ax.tick_params(which="minor", length=4)
ax.minorticks_on()
for spine in ax.spines.values():
    spine.set_linewidth(1.4)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=22,
          frameon=False)
fig.tight_layout()
fig.savefig(HERE / "Mn3Sn-shift-k48-sceta0p01-maintext.png",
            dpi=600, bbox_inches="tight")
plt.show()
