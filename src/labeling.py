"""
Stacking-domain labeling for twisted bilayer h-BN.

Approach (validated for θ=1.08° BN/BN; matches Li et al. 2024 Fig 4(i) area
fractions AA~10%, AB~45%, BA~45%, without reproducing the exact polygon rule
of Supplemental Fig S.5):

  1. Pair each bottom atom with its nearest top atom (any type) in xy under
     periodic boundary of the tilted box.
  2. AA label  := same-sublattice pair (B-B or N-N) AND dz > q90.
     AB label  := cross-sublattice pair 2->3 (bot-N, top-B above) OR
                  same-sublattice low-dz atom whose registry (dx,dy) lies in
                  AB sectors (learned from the cross-sub set).
     BA label  := mirror of AB, using 1->4 (bot-B, top-N above) and BA sectors.
  3. Confidence:
       AA     := normalized (dz - dz_thresh) / (dz_max - dz_thresh)  in [0, 1]
       AB/BA  := angular margin to nearest sector boundary times |d| / R_corner

A 25th-percentile cut on confidence defines boundary atoms; the rest are
core. The boundary set captures soliton / domain-wall atoms.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

A_HBN = 2.504
R_CORNER = A_HBN / np.sqrt(3)


@dataclass
class Box:
    """LAMMPS triclinic box (xy plane only). Use after correcting BOX BOUNDS."""
    lx: float
    ly: float
    xy: float

    @classmethod
    def from_dump_bounds(cls, xlo_b, xhi_b, xy, ylo_b, yhi_b, xz, zlo_b, zhi_b, yz):
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        return cls(xhi - xlo, yhi - ylo, xy)

    def wrap_dxy(self, dx, dy):
        nb = np.round(dy / self.ly)
        dy = dy - nb * self.ly
        dx = dx - nb * self.xy
        na = np.round(dx / self.lx)
        dx = dx - na * self.lx
        return dx, dy


def pair_atoms(df_atoms: pd.DataFrame, box: Box,
               z_split: float = 19.0) -> pd.DataFrame:
    """For each bottom atom, find nearest top atom under PBC.

    df_atoms columns: id, type, x, y, z.
    Returns: DataFrame indexed by bottom atom, with columns
      bot_id, top_id, bot_type, top_type, x, y, z_bot, z_top,
      dx, dy, dz, dist_xy, type_pair.
    """
    df = df_atoms.copy()
    df['layer'] = np.where(df['z'] < z_split, 'bot', 'top')
    bot = df[df['layer'] == 'bot'].reset_index(drop=True)
    top = df[df['layer'] == 'top'].reset_index(drop=True)

    t_xy = top[['x', 'y']].to_numpy()
    shifts = []
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            shifts.append(t_xy + ia * np.array([box.lx, 0.0])
                          + ib * np.array([box.xy, box.ly]))
    t_xy_rep = np.vstack(shifts)
    t_idx_rep = np.tile(np.arange(len(top)), 9)

    tree = cKDTree(t_xy_rep)
    b_xy = bot[['x', 'y']].to_numpy()
    _, idx = tree.query(b_xy, k=1)
    top_match = t_idx_rep[idx]

    dx = t_xy[top_match, 0] - b_xy[:, 0]
    dy = t_xy[top_match, 1] - b_xy[:, 1]
    dx, dy = box.wrap_dxy(dx, dy)

    pairs = pd.DataFrame({
        'bot_id': bot['id'].to_numpy(),
        'top_id': top.loc[top_match, 'id'].to_numpy(),
        'bot_type': bot['type'].to_numpy(),
        'top_type': top.loc[top_match, 'type'].to_numpy(),
        'x': bot['x'].to_numpy(),
        'y': bot['y'].to_numpy(),
        'z_bot': bot['z'].to_numpy(),
        'z_top': top.loc[top_match, 'z'].to_numpy(),
        'dx': dx, 'dy': dy,
        'dz': top.loc[top_match, 'z'].to_numpy() - bot['z'].to_numpy(),
        'dist_xy': np.hypot(dx, dy),
    })
    pairs['type_pair'] = (pairs['bot_type'].astype(str) + '->'
                         + pairs['top_type'].astype(str))
    return pairs


def label_stacking(pairs: pd.DataFrame,
                   aa_dz_quantile: float = 0.90,
                   same_sub_pairs=('1->3', '2->4'),
                   ab_pair: str = '2->3',
                   ba_pair: str = '1->4') -> pd.DataFrame:
    """Assign AA/AB/BA labels and confidences. Returns enriched DataFrame.

    Adds columns: label, sector6, ang_deg, confidence, is_boundary.
    """
    out = pairs.copy()
    dz = out['dz'].to_numpy()
    type_pair = out['type_pair'].to_numpy()

    same_sub = np.isin(type_pair, same_sub_pairs)
    is_pair_ab = type_pair == ab_pair
    is_pair_ba = type_pair == ba_pair

    dz_thresh = np.quantile(dz, aa_dz_quantile)
    is_aa = same_sub & (dz > dz_thresh)

    ang = np.degrees(np.arctan2(out['dy'].to_numpy(),
                                out['dx'].to_numpy())) % 360.0
    sector = ((ang + 30) // 60).astype(int) % 6

    # Learn AB / BA sectors from cross-sub atoms.
    s_ab = np.bincount(sector[is_pair_ab], minlength=6)
    s_ba = np.bincount(sector[is_pair_ba], minlength=6)
    ab_sectors = set(np.argsort(-s_ab)[:3].tolist())
    ba_sectors = set(np.argsort(-s_ba)[:3].tolist())

    label = np.full(len(out), 'AA', dtype='<U2')
    label[is_pair_ab] = 'AB'
    label[is_pair_ba] = 'BA'
    same_sub_lowdz = same_sub & ~is_aa
    in_ab = np.isin(sector, list(ab_sectors))
    in_ba = np.isin(sector, list(ba_sectors))
    label[same_sub_lowdz & in_ab] = 'AB'
    label[same_sub_lowdz & in_ba] = 'BA'
    label[is_aa] = 'AA'  # overwrite same-sub high-dz with AA

    # Sector center for each sector (degrees)
    sector_center = sector * 60.0
    # angle from sector center, wrapped to [-30, 30]
    delta = ((ang - sector_center + 180) % 360) - 180
    # for sectors defined so atom angle is in [center-30, center+30)
    delta_norm = ((delta + 30) % 60) - 30
    ang_margin = 30.0 - np.abs(delta_norm)  # 0..30

    d_mag = np.hypot(out['dx'].to_numpy(), out['dy'].to_numpy())

    # Confidence per label (0..1)
    conf = np.zeros(len(out))
    # AA: (dz - dz_thresh) normalized
    dz_max = dz.max()
    if dz_max > dz_thresh:
        conf_aa = (dz - dz_thresh) / (dz_max - dz_thresh)
    else:
        conf_aa = np.zeros_like(dz)
    conf_aa = np.clip(conf_aa, 0.0, 1.0)

    # AB / BA: angular margin (0..30 deg) × |d| / R_corner, both normalized.
    conf_ang = (ang_margin / 30.0) * np.clip(d_mag / R_CORNER, 0.0, 1.0)

    is_aa_label = label == 'AA'
    conf[is_aa_label] = conf_aa[is_aa_label]
    conf[~is_aa_label] = conf_ang[~is_aa_label]

    out['label'] = label
    out['sector6'] = sector
    out['ang_deg'] = ang
    out['ang_margin'] = ang_margin
    out['dz_thresh'] = dz_thresh
    out['confidence'] = conf

    q25 = np.quantile(conf, 0.25)
    out['is_boundary'] = (conf < q25).astype(int)
    out.attrs['ab_sectors'] = sorted(ab_sectors)
    out.attrs['ba_sectors'] = sorted(ba_sectors)
    out.attrs['dz_thresh'] = float(dz_thresh)
    out.attrs['conf_q25'] = float(q25)
    return out


def add_spatial_blocks(labeled: pd.DataFrame, box: Box,
                       n_blocks: tuple[int, int] = (2, 2)) -> pd.DataFrame:
    """Tag each atom with a fractional-cell block index for spatial CV.

    Adds columns: frac1, frac2, block_id. block_id ranges over n_blocks[0]*n_blocks[1].
    """
    out = labeled.copy()
    xy = out[['x', 'y']].to_numpy()
    M = np.array([[box.lx, box.xy], [0.0, box.ly]])
    Minv = np.linalg.inv(M)
    frac = xy @ Minv.T
    frac = frac - np.floor(frac)
    out['frac1'] = frac[:, 0]
    out['frac2'] = frac[:, 1]
    n1, n2 = n_blocks
    b1 = np.clip((frac[:, 0] * n1).astype(int), 0, n1 - 1)
    b2 = np.clip((frac[:, 1] * n2).astype(int), 0, n2 - 1)
    out['block_id'] = b1 * n2 + b2
    return out


def smooth_labels(labeled: pd.DataFrame, box: Box, k: int = 7) -> pd.DataFrame:
    """Spatial majority vote over k nearest atoms (PBC-aware).

    Rationale: at AB core, bot-B has two top atoms tied; cKDTree picks one
    arbitrarily, so half of bot-B atoms in AB get labeled BA. Same for bot-N
    in BA. Aggregate fractions still match paper but pixel-scale noise
    appears. A k-NN majority vote removes this sublattice noise while
    preserving domain-wall sharpness.
    """
    out = labeled.copy()
    xy = out[['x', 'y']].to_numpy()
    shifts = []
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            shifts.append(xy + ia * np.array([box.lx, 0.0])
                          + ib * np.array([box.xy, box.ly]))
    xy_rep = np.vstack(shifts)
    idx_rep = np.tile(np.arange(len(out)), 9)
    tree = cKDTree(xy_rep)
    _, nb_idx = tree.query(xy, k=k + 1)  # includes self
    nb_idx = idx_rep[nb_idx]

    label_arr = out['label'].to_numpy()
    smoothed = label_arr.copy()
    for i in range(len(out)):
        vals, counts = np.unique(label_arr[nb_idx[i]], return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    out['label_raw'] = label_arr
    out['label'] = smoothed
    return out


def label_fractions(df: pd.DataFrame) -> dict:
    counts = df['label'].value_counts().to_dict()
    n = len(df)
    return {k: counts.get(k, 0) / n for k in ('AA', 'AB', 'BA')}
