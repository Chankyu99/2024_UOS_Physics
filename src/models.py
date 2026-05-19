"""
Modeling: Random Forest baseline + rule-based baselines + evaluation.

Splits:
  - random 5-fold StratifiedKFold
  - spatial 4-block hold-out using block_id from add_spatial_blocks

Regions: overall, core (is_boundary == 0), boundary (is_boundary == 1)

Metrics: accuracy, macro F1, per-class precision/recall, confusion matrix.
"""
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix)


CLASSES = ('AA', 'AB', 'BA')


def make_rf(seed: int = 0):
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=seed,
    )


def _region_mask(df: pd.DataFrame, region: str):
    if region == 'overall':
        return np.ones(len(df), dtype=bool)
    if region == 'core':
        return df['is_boundary'].to_numpy() == 0
    if region == 'boundary':
        return df['is_boundary'].to_numpy() == 1
    raise ValueError(region)


def per_region_metrics(y_true, y_pred, is_boundary, classes=CLASSES) -> dict:
    out = {}
    for region in ('overall', 'core', 'boundary'):
        if region == 'overall':
            mask = np.ones(len(y_true), dtype=bool)
        elif region == 'core':
            mask = is_boundary == 0
        else:
            mask = is_boundary == 1
        yt = y_true[mask]; yp = y_pred[mask]
        if len(yt) == 0:
            continue
        out[f'{region}_acc'] = accuracy_score(yt, yp)
        out[f'{region}_f1_macro'] = f1_score(yt, yp, labels=list(classes),
                                             average='macro', zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(
            yt, yp, labels=list(classes), zero_division=0)
        for cls, p, r in zip(classes, prec, rec):
            out[f'{region}_{cls}_prec'] = p
            out[f'{region}_{cls}_rec'] = r
        out[f'{region}_n'] = int(mask.sum())
    return out


def random_kfold(features: pd.DataFrame, model_fn, n_splits=5, seed=0,
                 feature_cols=None):
    """Stratified random k-fold. Returns (per-fold metrics, importances,
    oof predictions array same length as features)."""
    if feature_cols is None:
        feature_cols = _default_feature_cols(features)
    X = features[feature_cols].to_numpy()
    y = features['label'].to_numpy()
    is_b = features['is_boundary'].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    importances = []
    oof = np.empty(len(features), dtype='<U2')
    oof[:] = ''
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        m = model_fn(seed=seed + fold)
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        oof[te] = yp
        met = per_region_metrics(y[te], yp, is_b[te])
        met['fold'] = fold
        results.append(met)
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
    return results, np.array(importances) if importances else None, oof


def spatial_block_cv(features: pd.DataFrame, model_fn, seed=0,
                     feature_cols=None):
    """Leave-one-block-out CV. Returns (metrics, importances, oof predictions)."""
    if feature_cols is None:
        feature_cols = _default_feature_cols(features)
    X = features[feature_cols].to_numpy()
    y = features['label'].to_numpy()
    is_b = features['is_boundary'].to_numpy()
    blocks = features['block_id'].to_numpy()
    results = []
    importances = []
    oof = np.empty(len(features), dtype='<U2')
    oof[:] = ''
    for b in np.unique(blocks):
        tr = blocks != b
        te = blocks == b
        m = model_fn(seed=seed + int(b))
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        oof[te] = yp
        met = per_region_metrics(y[te], yp, is_b[te])
        met['fold'] = int(b)
        results.append(met)
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
    return results, np.array(importances) if importances else None, oof


def area_ratio(y_true, y_pred, classes=CLASSES):
    """Return per-class area ratio for true and predicted, plus delta."""
    n = len(y_true)
    rows = []
    for c in classes:
        t = (y_true == c).mean()
        p = (y_pred == c).mean()
        rows.append({'class': c, 'true_frac': t, 'pred_frac': p,
                     'delta_pct_pt': (p - t) * 100})
    return pd.DataFrame(rows)


def make_hgb(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=None,
        min_samples_leaf=20, random_state=seed,
    )


def _default_feature_cols(features: pd.DataFrame):
    return [c for c in features.columns
            if c not in ('bot_id', 'x', 'y', 'label',
                         'is_boundary', 'block_id')]


def rule_dz_then_majority(features: pd.DataFrame) -> np.ndarray:
    """Rule: predict AA where raw_dz > dz threshold (90th percentile),
    else predict 'AB' (largest non-AA class)."""
    dz = features['raw_dz'].to_numpy()
    thr = np.quantile(dz, 0.90)
    pred = np.where(dz > thr, 'AA', 'AB')
    return pred


def rule_typepair(features: pd.DataFrame) -> np.ndarray:
    """Rule: AA if same-sub pair and high dz; AB if type_pair_code=='2->3';
    BA if '1->4'. Falls back via sector."""
    # type_pair_code is categorical encoded; reconstruct by mapping.
    tp = features['raw_type_pair_code'].to_numpy()
    # categories were sorted: 1->3, 1->4, 2->3, 2->4 -> codes 0, 1, 2, 3.
    # Verify by inspecting one row of features (best to pass mapping in).
    sector = features['raw_sector6'].to_numpy()
    dz = features['raw_dz'].to_numpy()
    thr = np.quantile(dz, 0.90)
    pred = np.empty(len(features), dtype='<U2')
    pred[:] = 'AB'
    is_same_sub = np.isin(tp, [0, 3])  # 1->3 and 2->4
    is_ab_pair = tp == 2  # 2->3
    is_ba_pair = tp == 1  # 1->4
    pred[is_ab_pair] = 'AB'
    pred[is_ba_pair] = 'BA'
    same_lowdz = is_same_sub & (dz <= thr)
    pred[same_lowdz & np.isin(sector, [0, 2, 4])] = 'AB'
    pred[same_lowdz & np.isin(sector, [1, 3, 5])] = 'BA'
    pred[is_same_sub & (dz > thr)] = 'AA'
    return pred


def rule_dz_only_aa(features: pd.DataFrame) -> np.ndarray:
    """Rule: dz threshold for AA, then split AB/BA randomly. Worst baseline."""
    dz = features['raw_dz'].to_numpy()
    thr = np.quantile(dz, 0.90)
    pred = np.where(dz > thr, 'AA', 'AB')
    return pred
