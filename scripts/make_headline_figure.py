"""
Single-glance headline figure for README:
  (1) Stacking-domain spatial map (smoothed ground truth)
  (2) OOF confusion matrix (HGB strict spatial 4-block)
  (3) Area-ratio table: true vs predicted vs paper θ=1.08° BN/BN reference
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "img" / "result"
FIG.mkdir(parents=True, exist_ok=True)

CLASSES = ('AA', 'AB', 'BA')
COLORS = {'AA': 'tab:red', 'AB': 'tab:blue', 'BA': 'tab:green'}

# Paper reference values for BN/BN relaxed at θ ≈ 1.08° (Li et al. 2024 Fig 4(i))
PAPER_REF = {'AA': 0.10, 'AB': 0.45, 'BA': 0.45}


def main():
    labeled = pd.read_parquet(PROC / "pairs_labeled.parquet")
    oof = pd.read_parquet(PROC / "oof_HGB_strict_spatial_4block.parquet")
    metrics = pd.read_csv(PROC / "model_metrics.csv")
    headline = metrics[(metrics['model'] == 'HGB')
                       & (metrics['track'] == 'strict')
                       & (metrics['split'] == 'spatial_4block')].iloc[0]
    acc = headline['overall_acc']
    acc_std = headline.get('overall_acc_std', np.nan)
    core_acc = headline['core_acc']
    bnd_acc = headline['boundary_acc']

    y = oof['label'].to_numpy()
    yp = oof['pred'].to_numpy()
    cm = confusion_matrix(y, yp, labels=list(CLASSES))

    true_frac = {c: (y == c).mean() for c in CLASSES}
    pred_frac = {c: (yp == c).mean() for c in CLASSES}

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0])
    for k, c in COLORS.items():
        sel = labeled['label'] == k
        ax0.scatter(labeled.loc[sel, 'x'], labeled.loc[sel, 'y'],
                    s=4, c=c, alpha=0.85, label=k)
    ax0.set_aspect('equal')
    ax0.set_title('Stacking domain map (ground-truth labels)\n'
                  'θ ≈ 1.08° BN/BN, one moiré supercell')
    ax0.set_xlabel('x (Å)'); ax0.set_ylabel('y (Å)')
    ax0.legend(loc='upper left', fontsize=9)

    ax1 = fig.add_subplot(gs[1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax1.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
    ax1.set_xticklabels(CLASSES); ax1.set_yticklabels(CLASSES)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    title = (f'OOF confusion matrix\nHGB strict spatial 4-block CV\n'
             f'acc = {acc:.4f} ± {acc_std:.4f}')
    ax1.set_title(title)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                     ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.5 else 'black',
                     fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[2])
    ax2.axis('off')
    rows = [['Class', 'True', 'Pred (OOF)', 'Paper (θ=1.08°)']]
    for c in CLASSES:
        rows.append([c, f'{true_frac[c]*100:.2f}%',
                     f'{pred_frac[c]*100:.2f}%',
                     f'{PAPER_REF[c]*100:.0f}%'])
    rows.append(['—', '—', '—', '—'])
    rows.append(['core acc', '', f'{core_acc:.4f}', ''])
    rows.append(['boundary acc', '', f'{bnd_acc:.4f}', ''])
    tbl = ax2.table(cellText=rows, loc='center', cellLoc='center',
                    colWidths=[0.25, 0.22, 0.27, 0.27])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)
    for j in range(4):
        tbl[(0, j)].set_facecolor('#dde6f0')
        tbl[(0, j)].set_text_props(weight='bold')
    ax2.set_title('Area ratio (OOF) vs paper reference')

    plt.tight_layout()
    out = FIG / "headline.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
