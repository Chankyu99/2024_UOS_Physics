"""
Run the stacking-domain labeler on the final LAMMPS dump frame.
Outputs:
  data/processed/pairs_labeled.parquet
  data/processed/pairs_labeled.csv
  img/visualization/label_registry_plane.png
  img/visualization/label_spatial_map.png
  img/visualization/confidence_map.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labeling import (Box, pair_atoms, label_stacking, smooth_labels,
                          add_spatial_blocks, label_fractions)

DUMP = ROOT / "data" / "hbn_lammps_dump.dat"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "visualization"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def read_final_frame(path: Path) -> tuple[pd.DataFrame, Box]:
    with open(path) as fp:
        last_df, last_box = None, None
        while True:
            line = fp.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            _ = int(fp.readline().strip())
            assert fp.readline().startswith("ITEM: NUMBER OF ATOMS")
            n = int(fp.readline().strip())
            _ = fp.readline()  # box header
            bx = [fp.readline().split() for _ in range(3)]
            cols = fp.readline().replace("ITEM: ATOMS", "").split()
            rows = [fp.readline().split() for _ in range(n)]
            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns:
                df[c] = (df[c].astype(int) if c in ('id', 'type', 'ix', 'iy', 'iz')
                         else df[c].astype(float))
            xlo_b, xhi_b, xy = map(float, bx[0])
            ylo_b, yhi_b, xz = map(float, bx[1])
            zlo_b, zhi_b, yz = map(float, bx[2])
            box = Box.from_dump_bounds(xlo_b, xhi_b, xy,
                                       ylo_b, yhi_b, xz,
                                       zlo_b, zhi_b, yz)
            last_df, last_box = df, box
    return last_df, last_box


def main():
    print(f"Reading final frame from {DUMP.name} ...")
    df, box = read_final_frame(DUMP)
    print(f"  atoms: {len(df)}, box: lx={box.lx:.3f}, ly={box.ly:.3f}, xy={box.xy:.3f}")

    print("Pairing atoms under PBC ...")
    pairs = pair_atoms(df, box)
    print(f"  pairs: {len(pairs)}")

    print("Labeling ...")
    labeled = label_stacking(pairs)
    raw_fracs = label_fractions(labeled)
    print(f"  raw    AA: {raw_fracs['AA']*100:.2f}%  AB: {raw_fracs['AB']*100:.2f}%  "
          f"BA: {raw_fracs['BA']*100:.2f}%")
    labeled = smooth_labels(labeled, box, k=15)
    labeled = add_spatial_blocks(labeled, box, n_blocks=(2, 2))
    print(f"  block_id distribution: {dict(labeled['block_id'].value_counts().sort_index())}")
    fracs = label_fractions(labeled)
    print(f"  smooth AA: {fracs['AA']*100:.2f}%  AB: {fracs['AB']*100:.2f}%  "
          f"BA: {fracs['BA']*100:.2f}%")
    flipped = (labeled['label'] != labeled['label_raw']).mean()
    print(f"  smoothing relabeled {flipped*100:.1f}% of atoms")
    print(f"  AB sectors: {labeled.attrs['ab_sectors']}, "
          f"BA sectors: {labeled.attrs['ba_sectors']}")
    print(f"  dz threshold: {labeled.attrs['dz_thresh']:.4f} Å")
    print(f"  confidence q25 (boundary cutoff): {labeled.attrs['conf_q25']:.4f}")
    print(f"  boundary atoms: {labeled['is_boundary'].sum()} "
          f"({labeled['is_boundary'].mean()*100:.1f}%)")

    print("Boundary vs core label distribution:")
    for region in ('core', 'boundary'):
        flag = 0 if region == 'core' else 1
        sub = labeled[labeled['is_boundary'] == flag]
        c = sub['label'].value_counts(normalize=True).to_dict()
        print(f"  {region}: AA {c.get('AA',0)*100:.1f}%, "
              f"AB {c.get('AB',0)*100:.1f}%, BA {c.get('BA',0)*100:.1f}%")

    out_pq = PROC / "pairs_labeled.parquet"
    out_csv = PROC / "pairs_labeled.csv"
    labeled.to_parquet(out_pq)
    labeled.to_csv(out_csv, index=False)
    print(f"saved -> {out_pq.name}, {out_csv.name}")

    colors = {'AA': 'tab:red', 'AB': 'tab:blue', 'BA': 'tab:green'}
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, c in colors.items():
        sel = labeled['label'] == k
        ax.scatter(labeled.loc[sel, 'dx'], labeled.loc[sel, 'dy'],
                   s=3, c=c, alpha=0.5, label=k)
    ax.set_aspect('equal'); ax.set_xlabel('dx (Å)'); ax.set_ylabel('dy (Å)')
    ax.set_title(f'Registry plane (AA={fracs["AA"]*100:.1f}%, '
                 f'AB={fracs["AB"]*100:.1f}%, BA={fracs["BA"]*100:.1f}%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "label_registry_plane.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    for k, c in colors.items():
        sel = labeled['label'] == k
        ax.scatter(labeled.loc[sel, 'x'], labeled.loc[sel, 'y'],
                   s=4, c=c, alpha=0.85, label=k)
    ax.set_aspect('equal')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_title('Stacking-domain map')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "label_spatial_map.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(labeled['x'], labeled['y'],
                         c=labeled['confidence'], s=4, cmap='viridis')
    axes[0].set_aspect('equal'); axes[0].set_title('confidence')
    plt.colorbar(sc, ax=axes[0])
    sc2 = axes[1].scatter(labeled['x'], labeled['y'],
                          c=labeled['is_boundary'], s=4, cmap='coolwarm')
    axes[1].set_aspect('equal'); axes[1].set_title('boundary (1) vs core (0)')
    plt.colorbar(sc2, ax=axes[1])
    plt.tight_layout()
    plt.savefig(FIG / "confidence_map.png", dpi=120)
    plt.close()
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
