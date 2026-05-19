"""
Phase 1 — Voronoi labeling on registry plane (v2).

Key fix: (dx, dy) from any-type nearest pairing is bounded by Wigner-Seitz
cell of hBN primitive lattice (|d| <= a/sqrt(3)). The 6 WS-cell corners
correspond to AB and BA stackings (3 each, alternating at 60-deg).
AA = origin. Center set:
  - AA at (0, 0)
  - 6 WS corners at (a/sqrt(3)) * (cos(30+60k), sin(30+60k)) for k=0..5
    Even k -> S1, odd k -> S2 (names AB/BA assigned after sanity check).

Voronoi: nearest of 7 centers; merge 6 corner labels into S1/S2.
Confidence margin = (2nd-nearest) - (1st-nearest) in Cartesian.
Boundary atoms: margin below 25% quantile.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
EDA_OUT = ROOT / "eda" / "out"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "visualization"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

A = 2.504
R_CORNER = A / np.sqrt(3)  # distance from origin to WS hexagon corner


def build_centers():
    centers = [(0.0, 0.0)]
    names = ['AA']
    for k in range(6):
        ang = np.deg2rad(30 + 60 * k)
        centers.append((R_CORNER * np.cos(ang), R_CORNER * np.sin(ang)))
        names.append('S1' if k % 2 == 0 else 'S2')
    return np.array(centers), np.array(names)


def main():
    pairs = pd.read_parquet(EDA_OUT / "pairs.parquet")
    d = pairs[['dx', 'dy']].to_numpy()

    centers, names = build_centers()
    tree = cKDTree(centers)
    dists, idxs = tree.query(d, k=2)
    label = names[idxs[:, 0]]
    margin = dists[:, 1] - dists[:, 0]

    pairs['label_voronoi'] = label
    pairs['margin'] = margin
    q25 = np.quantile(margin, 0.25)
    pairs['region'] = np.where(margin < q25, 'boundary', 'core')

    counts = pairs['label_voronoi'].value_counts()
    fracs = counts / len(pairs)
    print("--- Voronoi label fractions ---")
    for k in ['AA', 'S1', 'S2']:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({100*fracs.get(k, 0):.2f}%)")
    print(f"\nmargin: median={np.median(margin):.3f}, "
          f"q25={q25:.3f}, max={margin.max():.3f}")
    print(f"boundary atoms: {(pairs['region']=='boundary').sum()} "
          f"({100*(pairs['region']=='boundary').mean():.1f}%)")
    print(f"\nPaper sanity (θ=1.08° BN/BN relaxed):")
    print(f"  expected AA ~10%, AB ~45%, BA ~45%")

    pairs.to_csv(PROC / "pairs_labeled_voronoi.csv", index=False)
    pairs.to_parquet(PROC / "pairs_labeled_voronoi.parquet")
    print(f"\nsaved -> {PROC}/pairs_labeled_voronoi.csv")

    # Plot
    colors = {'AA': 'tab:red', 'S1': 'tab:blue', 'S2': 'tab:green'}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for k, c in colors.items():
        sel = pairs['label_voronoi'] == k
        ax[0].scatter(pairs.loc[sel, 'dx'], pairs.loc[sel, 'dy'],
                      s=3, c=c, alpha=0.5, label=k)
    ax[0].scatter(centers[:, 0], centers[:, 1], marker='x', s=120, c='black')
    for (cx, cy), n in zip(centers, names):
        ax[0].annotate(n, (cx, cy), xytext=(4, 4),
                       textcoords='offset points', fontsize=9)
    ax[0].set_aspect('equal'); ax[0].set_xlabel('dx (Å)'); ax[0].set_ylabel('dy (Å)')
    ax[0].set_title('Registry plane: 7-center Voronoi labels')
    ax[0].legend()

    for k, c in colors.items():
        sel = pairs['label_voronoi'] == k
        ax[1].scatter(pairs.loc[sel, 'x'], pairs.loc[sel, 'y'],
                      s=3, c=c, alpha=0.7, label=k)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('x (Å)'); ax[1].set_ylabel('y (Å)')
    ax[1].set_title('Spatial domain map')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "voronoi_registry_plane.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for k, c in colors.items():
        sel = pairs['label_voronoi'] == k
        ax[0].scatter(pairs.loc[sel, 'x'], pairs.loc[sel, 'y'],
                      s=3, c=c, alpha=0.7, label=k)
    ax[0].set_aspect('equal'); ax[0].set_title('Voronoi domain map')
    ax[0].legend()
    sc = ax[1].scatter(pairs['x'], pairs['y'], c=pairs['margin'],
                       s=3, cmap='viridis')
    ax[1].set_aspect('equal')
    ax[1].set_title('Margin map (low=boundary)')
    plt.colorbar(sc, ax=ax[1], label='margin (Å)')
    plt.tight_layout()
    plt.savefig(FIG / "voronoi_domain_map.png", dpi=120)
    plt.close()
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
