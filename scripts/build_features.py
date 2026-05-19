"""
Build three feature tracks and save to data/processed/features_<track>.parquet.

Usage: python3 scripts/build_features.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labeling import Box
from src.features import build_features
from scripts.run_labeling import read_final_frame

PROC = ROOT / "data" / "processed"
TRACKS = ('strict', 'species_aware', 'leakage_upper')


def main():
    df, box = read_final_frame(ROOT / "data" / "hbn_lammps_dump.dat")
    pairs = pd.read_parquet(PROC / "pairs_labeled.parquet")
    print(f"atoms: {len(df)}, pairs: {len(pairs)}, box lx={box.lx:.2f}, "
          f"ly={box.ly:.2f}, xy={box.xy:.2f}")

    for track in TRACKS:
        print(f"\n--- track: {track} ---")
        feats = build_features(df, pairs, box.lx, box.ly, box.xy, track=track)
        feat_cols = [c for c in feats.columns
                     if c not in ('bot_id', 'x', 'y', 'label',
                                  'is_boundary', 'block_id')]
        print(f"  {len(feat_cols)} feature columns")
        print(f"  examples: {feat_cols[:6]}")
        out = PROC / f"features_{track}.parquet"
        feats.to_parquet(out)
        print(f"  saved -> {out.name}")
        # quick sanity: distribution per label
        for lab in ('AA', 'AB', 'BA'):
            sub = feats[feats['label'] == lab]
            print(f"  {lab}: n={len(sub)}")


if __name__ == "__main__":
    main()
