"""
Feature engineering for stacking-domain classification.

Three feature tracks:
  - 'strict'        : geometry-only, NO species info. Hardest baseline.
                      Tests how far structural / lattice-relaxation signals
                      alone go for AA / stable / boundary detection.
  - 'species_aware' : strict + local species composition + species-specific
                      neighbor stats. Indirect registry information.
  - 'leakage_upper' : species_aware + raw dx/dy/dz/dist_xy/sector/type_pair.
                      Upper bound; used only as diagnostic, never as the
                      "deployed" model.

Excluded from all three (would be direct label leakage post-hoc):
  label, label_raw, confidence, is_boundary, dz_thresh, ang_deg,
  ang_margin, frac1, frac2, block_id.

PBC handled via the bottom-layer box (xy plane).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

A_HBN = 2.504


def _replicate_xy(xy: np.ndarray, lx: float, ly: float, xy_tilt: float):
    """3x3 periodic image replication of xy points. Returns (replicated_xy,
    index_map back to originals)."""
    shifts = []
    idx_map = []
    n = len(xy)
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            s = ia * np.array([lx, 0.0]) + ib * np.array([xy_tilt, ly])
            shifts.append(xy + s)
            idx_map.append(np.arange(n))
    return np.vstack(shifts), np.concatenate(idx_map)


def _knn_distances(query_xy, ref_xy, ref_idx_map, k, drop_self=False,
                   self_idx=None):
    """k nearest distances from query points to a (replicated) reference set."""
    tree = cKDTree(ref_xy)
    if drop_self:
        # Query k+1, then drop the first if it's the atom itself (distance 0).
        d, idx = tree.query(query_xy, k=k + 1)
        # if self_idx provided, only drop where matching
        keep = np.ones_like(d, dtype=bool)
        if self_idx is not None:
            # ref_idx_map[idx] gives original index. Match where == self_idx.
            mapped = ref_idx_map[idx]
            for i in range(len(query_xy)):
                # first occurrence of self atom
                hit = np.where(mapped[i] == self_idx[i])[0]
                if len(hit):
                    keep[i, hit[0]] = False
        # collapse: take first k surviving entries per row
        d_out = np.full((len(query_xy), k), np.nan)
        idx_out = np.full((len(query_xy), k), -1, dtype=int)
        for i in range(len(query_xy)):
            sel = np.where(keep[i])[0][:k]
            d_out[i] = d[i, sel]
            idx_out[i] = ref_idx_map[idx[i, sel]]
        return d_out, idx_out
    else:
        d, idx = tree.query(query_xy, k=k)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        return d, ref_idx_map[idx]


def _count_within_radius(query_xy, ref_xy_rep, radii):
    """Count points within each radius (PBC handled via replication)."""
    tree = cKDTree(ref_xy_rep)
    counts = np.zeros((len(query_xy), len(radii)), dtype=int)
    for j, r in enumerate(radii):
        c = tree.query_ball_point(query_xy, r=r, return_length=True)
        counts[:, j] = c
    return counts


def _angle_spread(query_xy, ref_xy_rep, ref_idx_map, k=6):
    """Std of angles to k nearest neighbors. Higher = lower symmetry."""
    tree = cKDTree(ref_xy_rep)
    d, idx = tree.query(query_xy, k=k + 1)
    # drop nearest (likely self if query is subset of ref) — but here we query
    # bot atoms against top atoms, so nearest is not self. Use full k.
    nb = ref_xy_rep[idx[:, :k]]
    rel = nb - query_xy[:, None, :]
    ang = np.arctan2(rel[..., 1], rel[..., 0])
    # sort, take consecutive differences, std measures non-uniformity
    ang_sorted = np.sort(ang, axis=1)
    gaps = np.diff(ang_sorted, axis=1, append=ang_sorted[:, :1] + 2 * np.pi)
    return gaps.std(axis=1)


def build_features(df_atoms: pd.DataFrame,
                   pairs_labeled: pd.DataFrame,
                   lx: float, ly: float, xy_tilt: float,
                   z_split: float = 19.0,
                   track: str = 'strict') -> pd.DataFrame:
    """Compute per-bottom-atom feature vector.

    df_atoms: full frame (bot + top).
    pairs_labeled: output of label_stacking (+ smoothing if applied). Order
                   defines the row order of the output.
    track: 'strict' | 'species_aware' | 'leakage_upper'.

    Returns DataFrame with columns:
      [feature columns] + label + is_boundary + block_id + x + y
    """
    df = df_atoms.copy()
    df['layer'] = np.where(df['z'] < z_split, 'bot', 'top')
    bot = df[df['layer'] == 'bot'].set_index('id', drop=False)
    top = df[df['layer'] == 'top'].set_index('id', drop=False)

    # Align to pairs_labeled row order via bot_id
    bot_ids = pairs_labeled['bot_id'].to_numpy()
    bot = bot.loc[bot_ids].reset_index(drop=True)

    bot_xy = bot[['x', 'y']].to_numpy()
    bot_z = bot['z'].to_numpy()
    top_xy = top[['x', 'y']].to_numpy()
    top_z = top['z'].to_numpy()
    top_ids = top['id'].to_numpy()
    top_types = top['type'].to_numpy()
    bot_types = bot['type'].to_numpy()

    bot_xy_rep, bot_idx_map = _replicate_xy(bot_xy, lx, ly, xy_tilt)
    top_xy_rep, top_idx_map = _replicate_xy(top_xy, lx, ly, xy_tilt)

    out = pd.DataFrame({'bot_id': bot_ids,
                        'x': bot_xy[:, 0], 'y': bot_xy[:, 1],
                        'z_bot': bot_z})

    # ------------- TRACK: strict (geometry-only) -------------
    # In-plane bottom neighbor distances (drop self).
    self_idx_arr = bot_ids - bot_ids.min()  # mapping into bot index (rough)
    # Actually rebuild: bot was sorted to match pairs, so index i corresponds
    # to row i of bot. self_idx is i.
    self_idx_arr = np.arange(len(bot))
    d_bot, _ = _knn_distances(bot_xy, bot_xy_rep, bot_idx_map, k=6,
                              drop_self=True, self_idx=self_idx_arr)
    out['bot_nn1'] = d_bot[:, 0]
    out['bot_nn3_mean'] = d_bot[:, :3].mean(axis=1)
    out['bot_nn6_mean'] = d_bot.mean(axis=1)
    out['bot_nn6_std'] = d_bot.std(axis=1)
    out['bot_strain'] = d_bot[:, 0] - A_HBN / np.sqrt(3)

    # Interlayer top neighbor stats — skip nearest (used by label).
    d_top, top_match_idx = _knn_distances(bot_xy, top_xy_rep, top_idx_map,
                                          k=5, drop_self=False)
    # d_top[:, 0] = nearest; for strict we drop it.
    out['top_dist_2'] = d_top[:, 1]
    out['top_dist_3'] = d_top[:, 2]
    out['top_dist_4'] = d_top[:, 3]
    out['top_dist_5'] = d_top[:, 4]
    out['top_dist_2to5_mean'] = d_top[:, 1:].mean(axis=1)
    out['top_dist_2to5_std'] = d_top[:, 1:].std(axis=1)

    # Mean / std of top z within neighborhood (different stat from per-atom dz).
    top_z_nb = top_z[top_match_idx]  # (N, 5)
    out['top_z_mean_5nn'] = top_z_nb.mean(axis=1)
    out['top_z_std_5nn'] = top_z_nb.std(axis=1)
    out['top_zminus_bot_mean'] = top_z_nb.mean(axis=1) - bot_z

    # Count top atoms within radii.
    radii = [1.0, 1.5, 2.0, 2.5]
    cnt = _count_within_radius(bot_xy, top_xy_rep, radii)
    for r, c in zip(radii, cnt.T):
        out[f'top_count_R{r}'] = c

    # 6-fold symmetry indicator (std of angular gaps to 6 NN bot atoms).
    out['bot_ang_gap_std'] = _angle_spread(bot_xy, bot_xy_rep, bot_idx_map, k=6)

    if track == 'strict':
        return _finalize(out, pairs_labeled)

    # ------------- TRACK: species_aware -------------
    # Local species composition: count of bot-B, bot-N, top-B, top-N within R.
    # Replicate species labels alongside xy.
    bot_type_rep = np.tile(bot_types, 9)
    top_type_rep = np.tile(top_types, 9)
    tree_bot = cKDTree(bot_xy_rep)
    tree_top = cKDTree(top_xy_rep)
    R = 2.5
    for label, ttree, ttypes in [('bot', tree_bot, bot_type_rep),
                                 ('top', tree_top, top_type_rep)]:
        nbr = ttree.query_ball_point(bot_xy, r=R)
        cnt_t1 = np.array([np.sum(ttypes[ix] == (1 if label == 'bot' else 3))
                           for ix in nbr])
        cnt_t2 = np.array([np.sum(ttypes[ix] == (2 if label == 'bot' else 4))
                           for ix in nbr])
        out[f'{label}_count_A_R{R}'] = cnt_t1
        out[f'{label}_count_B_R{R}'] = cnt_t2

    # Mean dz to top-B and top-N separately (different from "dz to nearest").
    # Use k=3 nearest top atoms of each species.
    for sp_lab, sp_types in [('A', (1, 3)), ('B', (2, 4))]:
        sel_top_sp = np.isin(top_types, sp_types)
        if sel_top_sp.sum() == 0:
            continue
        sub_xy = top_xy[sel_top_sp]
        sub_z = top_z[sel_top_sp]
        sub_xy_rep, sub_idx_map = _replicate_xy(sub_xy, lx, ly, xy_tilt)
        tree_sp = cKDTree(sub_xy_rep)
        d_sp, ix_sp = tree_sp.query(bot_xy, k=3)
        out[f'top{sp_lab}_d_mean_3nn'] = d_sp.mean(axis=1)
        z_sp = sub_z[sub_idx_map[ix_sp]]
        out[f'top{sp_lab}_z_mean_3nn'] = z_sp.mean(axis=1)
        out[f'top{sp_lab}_dz_mean_3nn'] = z_sp.mean(axis=1) - bot_z

    if track == 'species_aware':
        return _finalize(out, pairs_labeled)

    # ------------- TRACK: leakage_upper -------------
    out['raw_dx'] = pairs_labeled['dx'].to_numpy()
    out['raw_dy'] = pairs_labeled['dy'].to_numpy()
    out['raw_dz'] = pairs_labeled['dz'].to_numpy()
    out['raw_dist_xy'] = pairs_labeled['dist_xy'].to_numpy()
    out['raw_sector6'] = pairs_labeled['sector6'].to_numpy()
    # Encode type_pair categorically as integer
    tp = pairs_labeled['type_pair'].astype('category').cat.codes.to_numpy()
    out['raw_type_pair_code'] = tp
    out['raw_bot_type'] = bot_types

    return _finalize(out, pairs_labeled)


def _finalize(out: pd.DataFrame, pairs_labeled: pd.DataFrame) -> pd.DataFrame:
    out['label'] = pairs_labeled['label'].to_numpy()
    if 'is_boundary' in pairs_labeled.columns:
        out['is_boundary'] = pairs_labeled['is_boundary'].to_numpy()
    if 'block_id' in pairs_labeled.columns:
        out['block_id'] = pairs_labeled['block_id'].to_numpy()
    return out
