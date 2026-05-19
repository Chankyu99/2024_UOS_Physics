"""
Phase 1 (v3) — hybrid labeler.

Rationale:
  Paper Fig 3(g)/(h) uses dz to localize AA cores (AA has larger interlayer
  separation because B-B / N-N stacks are repulsive).
  Within stable region (low dz), AB vs BA distinguished by 3-fold symmetric
  registry direction.

Scheme:
  1. dz percentile threshold -> AA (top 10% by dz)  [matches paper θ=1.08° ~10%]
  2. Remaining 90% -> assign AB or BA by sector of (dx, dy):
       sector_AB = angles 30, 150, 270 deg (alternating)
       sector_BA = angles 90, 210, 330 deg
     bot_type=1 (B) and bot_type=2 (N) have mirrored sectors -> handle per type.
  3. Confidence:
       - AA: dz value (higher -> stronger AA)
       - AB/BA: angular margin to nearest AB/BA center
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EDA_OUT = ROOT / "eda" / "out"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "visualization"

A = 2.504
R_CORNER = A / np.sqrt(3)


def assign_sector(dx, dy, bot_type):
    """Assign 6-sector label by angular position in (dx, dy)."""
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    # 6 sectors centered at 0, 60, 120, 180, 240, 300 deg
    # boundaries at 30, 90, 150, 210, 270, 330
    # sector k covers [60k - 30, 60k + 30)
    sector = ((ang + 30) // 60).astype(int) % 6
    return ang, sector


def main():
    pairs = pd.read_parquet(EDA_OUT / "pairs.parquet")
    dz = pairs['dz'].to_numpy()
    dx = pairs['dx'].to_numpy()
    dy = pairs['dy'].to_numpy()
    bot_type = pairs['bot_type'].to_numpy()

    # 1) dz threshold for AA (top 10%)
    dz_thresh = np.quantile(dz, 0.90)
    is_aa = dz > dz_thresh
    print(f"dz threshold (90th percentile) = {dz_thresh:.4f} A")
    print(f"AA atoms by dz: {is_aa.sum()} ({100*is_aa.mean():.2f}%)")

    # 2) Sector assignment for non-AA
    ang, sector = assign_sector(dx, dy, bot_type)

    # Inspect sector distribution by bot_type for non-AA atoms
    not_aa = ~is_aa
    print("\n--- Sector (60-deg bins) distribution for non-AA ---")
    for bt in [1, 2]:
        sel = not_aa & (bot_type == bt)
        print(f"bot_type={bt}: sectors {np.bincount(sector[sel], minlength=6)}")

    # AB vs BA: empirically, for bot_type=1 (B), the 3 AB corners are at
    # ~60-deg apart starting from some phase. We pick alternating sectors:
    # sectors 0, 2, 4 -> S1; sectors 1, 3, 5 -> S2.
    # For bot_type=2 (N), the registry pattern is mirrored, so AB/BA may swap.
    # Decision deferred: tag both as S1/S2, check spatial alignment.
    s_label = np.where(sector % 2 == 0, 'S1', 'S2')

    label = np.full(len(pairs), 'X', dtype='<U2')
    label[is_aa] = 'AA'
    label[not_aa] = s_label[not_aa]

    pairs['label_hybrid'] = label
    pairs['dz_thresh'] = dz_thresh
    pairs['sector6'] = sector
    pairs['ang_deg'] = ang

    counts = pd.Series(label).value_counts()
    print("\n--- Hybrid label fractions ---")
    for k in ['AA', 'S1', 'S2']:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({100*c/len(pairs):.2f}%)")

    pairs.to_csv(PROC / "pairs_labeled_hybrid.csv", index=False)
    pairs.to_parquet(PROC / "pairs_labeled_hybrid.parquet")
    print(f"\nsaved -> {PROC}/pairs_labeled_hybrid.parquet")

    # Plot
    colors = {'AA': 'tab:red', 'S1': 'tab:blue', 'S2': 'tab:green'}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, c in colors.items():
        sel = pairs['label_hybrid'] == k
        ax[0].scatter(pairs.loc[sel, 'dx'], pairs.loc[sel, 'dy'],
                      s=3, c=c, alpha=0.5, label=k)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('dx (Å)'); ax[0].set_ylabel('dy (Å)')
    ax[0].set_title('Registry plane (hybrid label: dz + sector)')
    ax[0].legend()

    for k, c in colors.items():
        sel = pairs['label_hybrid'] == k
        ax[1].scatter(pairs.loc[sel, 'x'], pairs.loc[sel, 'y'],
                      s=3, c=c, alpha=0.7, label=k)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('x (Å)'); ax[1].set_ylabel('y (Å)')
    ax[1].set_title('Spatial map (hybrid label)')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "hybrid_labels.png", dpi=120)
    plt.close()
    print(f"figure -> {FIG}/hybrid_labels.png")


if __name__ == "__main__":
    main()
