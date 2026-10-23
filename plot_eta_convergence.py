#!/usr/bin/env python3
"""Plot injection eta and shift eta/sum-rule-width comparisons."""

import os
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
data_directory = root / "data"
figure_directory = root / "figure"
os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update({"font.size": 26, "axes.titlesize": 26})


def draw_spectra(
    files: tuple[Path, ...],
    labels: tuple[str, ...],
    column: int,
    scale: float,
    ylabel: str,
    title: str,
    output_name: str,
    legend_title: str,
    y_major: float,
    pos
) -> None:
    colors = ("#d62728", "#1f77b4", "#2ca02c")
    styles = ("-", "-.", "--")
    figure, axis = plt.subplots(figsize=(8, 6))

    for data_file, label, color, style in zip(files, labels, colors, styles):
        data = np.loadtxt(data_file, comments="#")
        axis.plot(
            data[:, 0],
            data[:, column] * scale,
            color=color,
            linestyle=style,
            linewidth=4,
            label=label,
        )

    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlim(1.5, 3.0)
    axis.set_xlabel("Photon energy (eV)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.yaxis.set_major_locator(MultipleLocator(y_major))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.xaxis.set_major_locator(MultipleLocator(0.5))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(
        which="major", direction="in", top=True, right=True,
        length=6, width=2.0,
    )
    axis.tick_params(
        which="minor", direction="in", top=True, right=True,
        length=3, width=1.5,
    )
    axis.legend(
        loc=pos, ncol=1, fontsize=14, frameon=False,
        handlelength=2.4, labelspacing=0.35,
        title=legend_title, title_fontsize=14,
    )
    figure.tight_layout()

    output = figure_directory / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=500)
    print(f"Figure: {output}")


def main() -> None:
    injection_files = (
        data_directory / "Zeeman_injection-eta=100meV-Nk=800.txt",
        data_directory / "Zeeman_injection-eta=10meV-Nk=800.txt",
        data_directory / "Zeeman_injection-eta=1meV-Nk=5000.txt",
    )
    draw_spectra(
        injection_files,
        (r"$0.1$ eV", r"$0.01$ eV", r"$0.001$ eV"),
        column=2,  # Im sigma_111 = Im sigma^{x;x~x}
        scale=1.0e6,
        ylabel=r"$\eta$ ($\mu\mathrm{A\,V^{-1}\,T^{-1}}$)",
        title="Zeeman injection current",
        output_name="Zeeman-injection-eta-comparison.png",
        legend_title=r"$\eta_{\mathrm{C}}^{x;x\tilde{x}}$",
        y_major=1.0,
        pos='lower left'
    )

    shift_eta_files = (
        data_directory / "Zeeman_shift-eta=100meV-sum_smear=1meV-Nk=800.txt",
        data_directory / "Zeeman_shift-eta=50meV-sum_smear=1meV-Nk=800.txt",
        data_directory / "Zeeman_shift-eta=1meV-sum_smear=1meV-Nk=3000.txt",
    )
    draw_spectra(
        shift_eta_files,
        (r"$0.1$ eV", r"$0.01$ eV", r"$0.001$ eV"),
        column=11,  # Re sigma_212 = Re sigma^{y;x~y}
        scale=1.0e9,
        ylabel=r"$\eta$ ($\mathrm{nA\,V^{-1}\,T^{-1}}$)",
        title="Zeeman shift current",
        output_name="Zeeman-shift-eta-comparison.png",
        legend_title=r"$\eta_{\mathrm{L}}^{y;x\tilde{y}}$",
        y_major=5.0,
        pos='lower left'
    )

    sum_smear_files = (
        data_directory / "Zeeman_shift-eta=10meV-sum_smear=100meV-Nk=800.txt",
        data_directory / "Zeeman_shift-eta=10meV-sum_smear=10meV-Nk=800.txt",
        data_directory / "Zeeman_shift-eta=10meV-sum_smear=1meV-Nk=800.txt",
    )
    draw_spectra(
        sum_smear_files,
        (
            r"$0.1$ eV",
            r"$0.01$ eV",
            r"$0.001$ eV",
        ),
        column=11,
        scale=1.0e9,
        ylabel=r"$\eta$ ($\mathrm{nA\,V^{-1}\,T^{-1}}$)",
        title=r"Zeeman shift current",
        output_name="Zeeman-shift-sum-smear-comparison.png",
        legend_title=r"$\eta_{\mathrm{L}}^{y;x\tilde{y}}$",
        y_major=2.0,
        pos='lower left'
    )

    plt.show(block=True)


if __name__ == "__main__":
    main()
