"""
Phase 0 EDA — Step 3.
Visualize:
  (1) dz distribution
  (2) dist_xy distribution
  (3) (dx, dy) scatter — expect 3 high-symmetry clusters (AA / AB / BA)
  (4) Spatial maps colored by dz, dist_xy
  (5) Moire periodicity sanity check
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "out"
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)


def main():
    pairs = pd.read_parquet(OUT_DIR / "pairs.parquet")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(pairs['dz'], bins=80, color='steelblue', edgecolor='k', lw=0.3)
    ax[0].set_xlabel('dz (Å)'); ax[0].set_ylabel('count')
    ax[0].set_title('dz distribution (final timestep)')
    ax[1].hist(pairs['dist_xy'], bins=80, color='indianred', edgecolor='k', lw=0.3)
    ax[1].set_xlabel('dist_xy (Å)'); ax[1].set_ylabel('count')
    ax[1].set_title('dist_xy distribution')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_dz_distxy_hist.png", dpi=120)
    plt.close()

    # (dx, dy) scatter — color by bot_type
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, sub in enumerate([1, 2]):
        sel = pairs[pairs['bot_type'] == sub]
        sc = axes[i].scatter(sel['dx'], sel['dy'], s=2,
                             c=sel['dz'], cmap='coolwarm', alpha=0.6)
        axes[i].set_aspect('equal')
        axes[i].set_xlabel('dx (Å)'); axes[i].set_ylabel('dy (Å)')
        axes[i].set_title(f'bot_type={sub}: (dx, dy) colored by dz')
        plt.colorbar(sc, ax=axes[i], label='dz (Å)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_dxdy_scatter.png", dpi=120)
    plt.close()

    # Spatial maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc1 = axes[0].scatter(pairs['x'], pairs['y'], c=pairs['dz'],
                          s=3, cmap='coolwarm')
    axes[0].set_aspect('equal'); axes[0].set_title('dz spatial map')
    axes[0].set_xlabel('x (Å)'); axes[0].set_ylabel('y (Å)')
    plt.colorbar(sc1, ax=axes[0], label='dz (Å)')
    sc2 = axes[1].scatter(pairs['x'], pairs['y'], c=pairs['dist_xy'],
                          s=3, cmap='viridis')
    axes[1].set_aspect('equal'); axes[1].set_title('dist_xy spatial map')
    axes[1].set_xlabel('x (Å)'); axes[1].set_ylabel('y (Å)')
    plt.colorbar(sc2, ax=axes[1], label='dist_xy (Å)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_spatial_maps.png", dpi=120)
    plt.close()

    # Type-pair spatial map
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_codes = {'1->3': 0, '2->4': 1, '1->4': 2, '2->3': 3}
    color_map = pairs['type_pair'].map(pair_codes)
    sc = ax.scatter(pairs['x'], pairs['y'], c=color_map, s=3,
                    cmap='tab10', alpha=0.8)
    ax.set_aspect('equal')
    ax.set_title('type_pair spatial map (0:B-B, 1:N-N, 2:B-N, 3:N-B)')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_typepair_spatial.png", dpi=120)
    plt.close()

    # Estimate moire periodicity: peaks of dz spatial map ~ AA regions
    print("Figures saved to", FIG_DIR)
    print("\nKey numbers:")
    print(f"  dz range: {pairs['dz'].min():.3f} – {pairs['dz'].max():.3f}")
    print(f"  dist_xy range: {pairs['dist_xy'].min():.3e} – "
          f"{pairs['dist_xy'].max():.3f}")
    a_bn = 2.504 / np.sqrt(3)
    print(f"  Expected B-N bond ≈ a/√3 = {a_bn:.3f} Å (matches dist_xy max)")

    # Moire angle from CELL geometry (dump-derived, supersedes input file).
    # Input file hbn_twist_input.inp says θ=6.01° but is stale provenance.
    a_hbn = 2.504
    L_cell = 132.339  # |A| from triclinic-corrected box
    theta = 2 * np.arcsin(a_hbn / (2 * L_cell))
    print(f"  Twist angle (from cell |A|): θ = {theta:.6f} rad = "
          f"{np.degrees(theta):.4f}°")
    print(f"  Moiré period L = |A| = {L_cell:.2f} Å → box = 1 moiré supercell")


if __name__ == "__main__":
    main()
