"""
Phase 1 (v4) — type_pair-based labeling.

Insight from EDA:
  At AB stack core: top-B sits directly above bot-N (cross-sub pair 2->3),
    while bot-B has 3 equidistant top atoms (no unique nearest).
  At BA stack core: top-N sits directly above bot-B (cross-sub pair 1->4),
    while bot-N has 3 equidistant top atoms.
  At AA stack core: both sublattices have same-sub atom directly above
    (pairs 1->3 and 2->4).

Labeling:
  same-sub pair (1->3 or 2->4) AND dz > q90  -> AA core
  cross-sub pair 2->3                          -> AB region
  cross-sub pair 1->4                          -> BA region
  same-sub pair AND low dz                     -> assign by closer AB/BA core
                                                  using which corner (sector)
                                                  of registry plane
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EDA_OUT = ROOT / "eda" / "out"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "visualization"


def main():
    pairs = pd.read_parquet(EDA_OUT / "pairs.parquet")
    dz = pairs['dz'].to_numpy()
    type_pair = pairs['type_pair'].to_numpy()

    same_sub = np.isin(type_pair, ['1->3', '2->4'])
    is_pair_ab = type_pair == '2->3'
    is_pair_ba = type_pair == '1->4'

    # AA = same-sub AND dz high
    # threshold chosen so AA core is ~10% of all atoms
    dz_thresh = np.quantile(dz, 0.90)
    is_aa = same_sub & (dz > dz_thresh)
    print(f"dz threshold (q90) = {dz_thresh:.4f}")
    print(f"AA atoms (same-sub & dz>q90): {is_aa.sum()} "
          f"({100*is_aa.mean():.2f}%)")

    # Same-sub atoms NOT classified as AA (low dz): atoms in transition between
    # AA and AB or BA. Resolve via registry sector.
    same_sub_lowdz = same_sub & ~is_aa
    print(f"same-sub atoms in transition (low dz): {same_sub_lowdz.sum()}")

    # Sector at (dx, dy): use which of 6 corners is nearest.
    dx = pairs['dx'].to_numpy()
    dy = pairs['dy'].to_numpy()
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    sector = ((ang + 30) // 60).astype(int) % 6  # 0..5

    # For same-sub low-dz atoms, sector assignment maps to AB vs BA.
    # We learn the convention from cross-sub atoms (which are unambiguously
    # labeled AB or BA). Inspect their sector distribution:
    print("\n--- sector distribution by cross-sub label ---")
    for lab, mask in [('AB (2->3)', is_pair_ab), ('BA (1->4)', is_pair_ba)]:
        s = np.bincount(sector[mask], minlength=6)
        dominant = np.argsort(-s)[:3]
        print(f"  {lab}: sectors {s}, top-3 sectors {sorted(dominant.tolist())}")

    # From this we'll define AB-sectors and BA-sectors.
    # Then for same-sub low-dz atoms, assign by which sector their (dx,dy) is in.
    s_counts_ab = np.bincount(sector[is_pair_ab], minlength=6)
    s_counts_ba = np.bincount(sector[is_pair_ba], minlength=6)
    ab_sectors = set(np.argsort(-s_counts_ab)[:3].tolist())
    ba_sectors = set(np.argsort(-s_counts_ba)[:3].tolist())
    print(f"\nAB sectors (from cross-sub): {sorted(ab_sectors)}")
    print(f"BA sectors (from cross-sub): {sorted(ba_sectors)}")

    label = np.full(len(pairs), 'X', dtype='<U2')
    label[is_aa] = 'AA'
    label[is_pair_ab] = 'AB'
    label[is_pair_ba] = 'BA'
    # Same-sub low-dz: assign by sector
    in_ab_sector = np.isin(sector, list(ab_sectors))
    in_ba_sector = np.isin(sector, list(ba_sectors))
    label[same_sub_lowdz & in_ab_sector] = 'AB'
    label[same_sub_lowdz & in_ba_sector] = 'BA'
    # Edge case: |d|~0 -> sector arbitrary. Use type_pair instead.
    # For same-sub low-dz with |d| very small: shouldn't happen often, but
    # fall back to AB (will be tiny minority).

    pairs['label_tp'] = label
    counts = pd.Series(label).value_counts()
    print("\n--- Final label fractions ---")
    for k in ['AA', 'AB', 'BA', 'X']:
        c = counts.get(k, 0)
        print(f"  {k}: {c} ({100*c/len(pairs):.2f}%)")
    print(f"\nPaper θ=1.08° BN/BN relaxed: AA~10%, AB~45%, BA~45%")

    pairs.to_csv(PROC / "pairs_labeled_typepair.csv", index=False)
    pairs.to_parquet(PROC / "pairs_labeled_typepair.parquet")

    colors = {'AA': 'tab:red', 'AB': 'tab:blue', 'BA': 'tab:green',
              'X': 'gray'}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, c in colors.items():
        sel = pairs['label_tp'] == k
        if sel.sum() == 0:
            continue
        ax[0].scatter(pairs.loc[sel, 'dx'], pairs.loc[sel, 'dy'],
                      s=3, c=c, alpha=0.5, label=k)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('dx (Å)'); ax[0].set_ylabel('dy (Å)')
    ax[0].set_title('Registry plane (type_pair label)')
    ax[0].legend()
    for k, c in colors.items():
        sel = pairs['label_tp'] == k
        if sel.sum() == 0:
            continue
        ax[1].scatter(pairs.loc[sel, 'x'], pairs.loc[sel, 'y'],
                      s=3, c=c, alpha=0.8, label=k)
    ax[1].set_aspect('equal')
    ax[1].set_title('Spatial domain map (type_pair label)')
    ax[1].set_xlabel('x (Å)'); ax[1].set_ylabel('y (Å)')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "typepair_labels.png", dpi=120)
    plt.close()
    print(f"figure -> {FIG}/typepair_labels.png")


if __name__ == "__main__":
    main()
