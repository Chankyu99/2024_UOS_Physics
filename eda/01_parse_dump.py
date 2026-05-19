"""
Phase 0 EDA — Step 1.
Parse LAMMPS dump, extract final timestep, inspect basic structure.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DUMP = Path(__file__).resolve().parents[1] / "data" / "hbn_lammps_dump.dat"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def iter_frames(path: Path):
    """Yield (timestep, n_atoms, box, df) for each frame."""
    with open(path) as fp:
        while True:
            line = fp.readline()
            if not line:
                return
            assert line.startswith("ITEM: TIMESTEP"), line
            timestep = int(fp.readline().strip())
            assert fp.readline().startswith("ITEM: NUMBER OF ATOMS")
            n_atoms = int(fp.readline().strip())
            box_header = fp.readline()
            box = [fp.readline().split() for _ in range(3)]
            cols_line = fp.readline()
            assert cols_line.startswith("ITEM: ATOMS"), cols_line
            cols = cols_line.replace("ITEM: ATOMS", "").split()
            rows = [fp.readline().split() for _ in range(n_atoms)]
            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns:
                if c in ("id", "type", "ix", "iy", "iz"):
                    df[c] = df[c].astype(int)
                else:
                    df[c] = df[c].astype(float)
            yield timestep, n_atoms, box_header, box, df


def main():
    frames = list(iter_frames(DUMP))
    print(f"frames: {len(frames)}")
    print(f"timesteps: {[f[0] for f in frames]}")
    print(f"n_atoms (first): {frames[0][1]}, (last): {frames[-1][1]}")

    ts, n, bh, box, df = frames[-1]
    print(f"\n--- final frame (timestep={ts}) ---")
    print(f"box header: {bh.strip()}")
    print(f"box rows: {box}")
    print(f"type counts:\n{df['type'].value_counts().sort_index()}")
    print(f"z stats by type:")
    print(df.groupby('type')['z'].agg(['min', 'max', 'mean', 'std']))
    print(f"\nxy ranges: x [{df['x'].min():.2f}, {df['x'].max():.2f}], "
          f"y [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    print(f"z range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")

    df.to_parquet(OUT_DIR / "final_frame.parquet")
    print(f"\nsaved -> {OUT_DIR / 'final_frame.parquet'}")


if __name__ == "__main__":
    main()
