"""
Train + evaluate. Outputs:
  data/processed/model_metrics.csv     overall summary
  data/processed/oof_preds_<track>_<split>.parquet   out-of-fold predictions
  data/processed/fold_metrics_<track>_<split>.csv    per-fold breakdown
  data/processed/area_ratio_<track>_<split>.csv      label vs prediction area ratios
  data/processed/importance_<track>_<split>.csv      feature importance
  img/visualization/cm_oof_<model>_<track>_<split>.png   OOF confusion matrix
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import (make_rf, make_hgb, random_kfold, spatial_block_cv,
                        per_region_metrics, _default_feature_cols,
                        rule_typepair, rule_dz_only_aa, area_ratio, CLASSES)

PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "visualization"
FIG.mkdir(parents=True, exist_ok=True)


def plot_cm(cm, labels, title, outpath):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Pred'); ax.set_ylabel('True'); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black',
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def run_one(features: pd.DataFrame, model_name: str, model_fn,
            split: str, track: str):
    fcols = _default_feature_cols(features)
    y = features['label'].to_numpy()
    if split == 'random_5fold':
        results, imp, oof = random_kfold(features, model_fn, n_splits=5,
                                         seed=0, feature_cols=fcols)
    elif split == 'spatial_4block':
        results, imp, oof = spatial_block_cv(features, model_fn, seed=0,
                                             feature_cols=fcols)
    else:
        raise ValueError(split)

    # Per-fold dataframe
    fold_df = pd.DataFrame(results)
    fold_df.to_csv(PROC / f"fold_metrics_{model_name}_{track}_{split}.csv",
                   index=False)

    # OOF predictions
    oof_df = features[['bot_id', 'x', 'y', 'label',
                       'is_boundary', 'block_id']].copy()
    oof_df['pred'] = oof
    oof_df.to_parquet(PROC / f"oof_{model_name}_{track}_{split}.parquet")

    # Area ratio (OOF)
    ar = area_ratio(y, oof)
    ar.to_csv(PROC / f"area_ratio_{model_name}_{track}_{split}.csv",
              index=False)

    # OOF confusion matrix
    cm = confusion_matrix(y, oof, labels=list(CLASSES))
    plot_cm(cm, list(CLASSES),
            title=f"OOF {model_name} {track} {split}\n"
                  f"acc={(oof==y).mean():.4f}",
            outpath=FIG / f"cm_oof_{model_name}_{track}_{split}.png")

    # Feature importance (if available)
    if imp is not None:
        imp_mean = imp.mean(axis=0)
        imp_std = imp.std(axis=0)
        pd.DataFrame({'feature': fcols, 'importance_mean': imp_mean,
                      'importance_std': imp_std}) \
            .sort_values('importance_mean', ascending=False) \
            .to_csv(PROC / f"importance_{model_name}_{track}_{split}.csv",
                    index=False)

    # Headline row (mean across folds) + fold std on key metrics
    mean_row = fold_df.mean(numeric_only=True).to_dict()
    std_row = {f"{k}_std": fold_df[k].std() for k in
               ('overall_acc', 'overall_f1_macro',
                'core_acc', 'boundary_acc') if k in fold_df.columns}
    mean_row.update(std_row)
    mean_row.update({'track': track, 'model': model_name, 'split': split})
    return mean_row


def run_rules() -> list:
    feats = pd.read_parquet(PROC / "features_leakage_upper.parquet")
    y = feats['label'].to_numpy()
    is_b = feats['is_boundary'].to_numpy()
    rows = []
    for name, fn in [('rule_typepair', rule_typepair),
                     ('rule_dz_only_aa', rule_dz_only_aa)]:
        yp = fn(feats)
        met = per_region_metrics(y, yp, is_b)
        met.update({'track': 'diagnostic', 'model': name, 'split': 'none'})
        rows.append(met)
        cm = confusion_matrix(y, yp, labels=list(CLASSES))
        plot_cm(cm, list(CLASSES),
                title=f"{name}  acc={(yp==y).mean():.3f}",
                outpath=FIG / f"cm_{name}.png")
        ar = area_ratio(y, yp)
        ar.to_csv(PROC / f"area_ratio_{name}.csv", index=False)
    return rows


def main():
    rows = []
    for track in ('strict', 'species_aware', 'leakage_upper'):
        feats = pd.read_parquet(PROC / f"features_{track}.parquet")
        # RF on both splits
        for split in ('random_5fold', 'spatial_4block'):
            r = run_one(feats, 'RF', make_rf, split, track)
            rows.append(r)
            print(f"RF | {track:14s} | {split:14s} | "
                  f"acc={r['overall_acc']:.4f} ± {r.get('overall_acc_std', 0):.4f}  "
                  f"f1={r['overall_f1_macro']:.4f}  "
                  f"core={r['core_acc']:.4f}  bnd={r['boundary_acc']:.4f}")
        # HGB on strict and species_aware spatial only (per recommendation)
        if track in ('strict', 'species_aware'):
            r = run_one(feats, 'HGB', make_hgb, 'spatial_4block', track)
            rows.append(r)
            print(f"HGB| {track:14s} | spatial_4block | "
                  f"acc={r['overall_acc']:.4f} ± {r.get('overall_acc_std', 0):.4f}  "
                  f"f1={r['overall_f1_macro']:.4f}  "
                  f"core={r['core_acc']:.4f}  bnd={r['boundary_acc']:.4f}")

    rows.extend(run_rules())
    summary = pd.DataFrame(rows)
    summary.to_csv(PROC / "model_metrics.csv", index=False)
    print(f"\nsaved -> model_metrics.csv")

    headline_cols = ['track', 'model', 'split',
                     'overall_acc', 'overall_acc_std',
                     'overall_f1_macro', 'core_acc', 'boundary_acc']
    headline = summary[[c for c in headline_cols if c in summary.columns]]
    print("\n=== HEADLINE (mean ± std across folds) ===")
    with pd.option_context('display.width', 160,
                           'display.float_format', '{:.4f}'.format):
        print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
