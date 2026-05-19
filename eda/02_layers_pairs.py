"""
Phase 0 EDA — Step 2 (v2).
Strategy: for each bottom atom, find nearest top atom of ANY type
under periodic boundary of the tilted box.
Record (dx, dy, dz, dist_xy, bot_type, top_type).

Rationale: high-symmetry stackings (AA / AB / BA) appear as distinct
clusters in (dist_xy, type-pair) space, since BN/BN aligned bilayer with
twist places the bottom B/N either directly under top B (AA: B-B, N-N pairs),
directly under top N (AB: B-N pair), or vice versa (BA: N-B pair).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT_DIR = Path(__file__).resolve().parent / "out"

# True triclinic lattice vectors (after correcting LAMMPS BOX BOUNDS):
#   A = (LX, 0),  B = (XY, LY)
LX = 132.33893899756032
LY = 114.60888308176639
XY = 66.16946949878017


def wrap_dxy(dx, dy):
    """Minimum-image wrap in the (A, B) lattice."""
    nb = np.round(dy / LY)
    dy = dy - nb * LY
    dx = dx - nb * XY
    na = np.round(dx / LX)
    dx = dx - na * LX
    return dx, dy


def main():
    df = pd.read_parquet(OUT_DIR / "final_frame.parquet")
    df['layer'] = np.where(df['z'] < 19.0, 'bot', 'top')

    bot = df[df['layer'] == 'bot'].reset_index(drop=True)
    top = df[df['layer'] == 'top'].reset_index(drop=True)
    print(f"bot: {len(bot)}, top: {len(top)}")

    # Replicate top atoms in 3x3 periodic images for proper nearest-neighbor.
    t_xy = top[['x', 'y']].to_numpy()
    shifts = []
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            shift = ia * np.array([LX, 0.0]) + ib * np.array([XY, LY])
            shifts.append(t_xy + shift)
    t_xy_rep = np.vstack(shifts)
    t_idx_rep = np.tile(np.arange(len(top)), 9)

    tree = cKDTree(t_xy_rep)
    b_xy = bot[['x', 'y']].to_numpy()
    _, idx = tree.query(b_xy, k=1)
    top_match = t_idx_rep[idx]

    dx = t_xy[top_match, 0] - b_xy[:, 0]
    dy = t_xy[top_match, 1] - b_xy[:, 1]
    dx, dy = wrap_dxy(dx, dy)
    dist_xy = np.hypot(dx, dy)
    dz = top.loc[top_match, 'z'].to_numpy() - bot['z'].to_numpy()

    pairs = pd.DataFrame({
        'bot_id': bot['id'].to_numpy(),
        'top_id': top.loc[top_match, 'id'].to_numpy(),
        'bot_type': bot['type'].to_numpy(),
        'top_type': top.loc[top_match, 'type'].to_numpy(),
        'x': bot['x'].to_numpy(),
        'y': bot['y'].to_numpy(),
        'z_bot': bot['z'].to_numpy(),
        'z_top': top.loc[top_match, 'z'].to_numpy(),
        'dx': dx, 'dy': dy, 'dz': dz, 'dist_xy': dist_xy,
    })
    pairs['type_pair'] = (
        pairs['bot_type'].astype(str) + '->' + pairs['top_type'].astype(str)
    )

    print("\n--- type_pair counts (nearest top species for each bottom atom) ---")
    print(pairs['type_pair'].value_counts())
    print("\n--- dist_xy by type_pair ---")
    print(pairs.groupby('type_pair')['dist_xy'].describe()[
        ['count', 'mean', 'std', 'min', '50%', 'max']
    ])
    print("\n--- dz by type_pair ---")
    print(pairs.groupby('type_pair')['dz'].describe()[
        ['count', 'mean', 'std', 'min', '50%', 'max']
    ])
    print("\n--- dist_xy histogram (10 bins) ---")
    hist, edges = np.histogram(pairs['dist_xy'], bins=10)
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  [{e0:.3f}, {e1:.3f}): {h}")

    pairs.to_parquet(OUT_DIR / "pairs.parquet")
    print(f"\nsaved -> {OUT_DIR / 'pairs.parquet'}")


if __name__ == "__main__":
    main()
