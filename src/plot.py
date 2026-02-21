"""
plot.py — Band structure & PDOS quick-plot utilities

Usage (from project root):
    python src/plot.py --bands data/bands.dat.gnu --fermi 2.3467
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bands(bands_file: str, fermi_energy: float = 0.0,
               color: str = "#4a9896", ylim=(-10, 10),
               kpoints: dict = None) -> plt.Figure:
    """
    2D band structure plot from QE bands.dat.gnu output.

    Parameters
    ----------
    bands_file : str  Path to *.dat.gnu file
    fermi_energy : float  Fermi energy shift (eV)
    color : str  Matplotlib color string
    ylim : tuple  y-axis limits in eV
    kpoints : dict  {label: k_position} for high-symmetry points

    Returns
    -------
    fig : matplotlib.Figure
    """
    data = np.loadtxt(bands_file)
    k = np.unique(data[:, 0])
    bands = np.reshape(data[:, 1], (-1, len(k)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for band in bands:
        ax.plot(k, band - fermi_energy, lw=1.2, alpha=0.7, color=color)
    ax.axhline(0, ls="--", lw=0.8, color="gray", label="E_F")
    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(*ylim)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("k")
    if kpoints:
        positions = list(kpoints.values())
        labels = list(kpoints.keys())
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        for xv in positions:
            ax.axvline(xv, lw=0.7, color="k", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_pdos(pdos_file: str, fermi_energy: float = 0.0,
              color: str = "#4a9896", ylim=(-10, 10)) -> plt.Figure:
    """Quick PDOS plot from QE pdos_tot output."""
    data = np.loadtxt(pdos_file)
    E = data[:, 0] - fermi_energy
    dos = data[:, 1]
    fig, ax = plt.subplots(figsize=(3, 5))
    ax.plot(dos, E, color=color, lw=1.5)
    ax.fill_betweenx(E, dos, alpha=0.15, color=color)
    ax.axhline(0, ls="--", lw=0.8, color="gray")
    ax.set_ylim(*ylim)
    ax.set_xlabel("DOS (states/eV)")
    ax.set_ylabel("Energy (eV)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick band/DOS plotter")
    parser.add_argument("--bands", type=str, help="Path to bands.dat.gnu")
    parser.add_argument("--pdos",  type=str, help="Path to pdos_tot file")
    parser.add_argument("--fermi", type=float, default=0.0,
                        help="Fermi energy shift (eV)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output image path (optional)")
    args = parser.parse_args()

    if args.bands:
        fig = plot_bands(args.bands, fermi_energy=args.fermi)
        fig.suptitle(Path(args.bands).stem)
        if args.out:
            fig.savefig(args.out, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.out}")
        else:
            plt.show()

    if args.pdos:
        fig = plot_pdos(args.pdos, fermi_energy=args.fermi)
        fig.suptitle(Path(args.pdos).stem)
        if args.out:
            fig.savefig(args.out.replace(".png", "_dos.png"),
                        dpi=150, bbox_inches="tight")
        else:
            plt.show()
