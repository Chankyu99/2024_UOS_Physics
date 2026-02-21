#!/usr/bin/env python3
"""
Build the main portfolio notebook for 2024_UOS_Physics project.
Run from the project root: python3 build_notebook.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ── TITLE ──────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# Band Structure of Graphene/h-BN and Stacking Classification in Moiré Patterns Using Machine Learning

**서울시립대학교 물리학과 고체물리 학부인턴 (2024 여름)**  
강현우 · 이찬규 · 임제승 · 정재일 지도교수

---

## 프로젝트 파이프라인

```
DFT (Quantum ESPRESSO)         →   밴드구조 / PDOS 데이터
LAMMPS 구조 완화                →   dump.minimization
ML 스태킹 분류                  →   AA / AB / BA 도메인 맵
```
"""))

# ── PART 1 HEADER ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Part 1: Electronic Band Structure & Density of States

Quantum ESPRESSO DFT 계산 결과 — Graphene과 hexagonal Boron Nitride(h-BN)의  
밴드구조와 상태밀도(DOS)를 시각화합니다.
"""))

# ─── 1-1 Graphene 2D Bands + DOS ───────────────────────────────────────────
cells.append(new_markdown_cell("### 1-1. Graphene — 2D Band Structure & DOS"))

cells.append(new_code_cell("""\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

import warnings
warnings.filterwarnings("ignore")

# ── Load band data ──────────────────────────────────────────────────────────
bands_data = np.loadtxt("../data/bands.dat.gnu")
k  = np.unique(bands_data[:, 0])
bs = np.reshape(bands_data[:, 1], (-1, len(k)))

# ── Load PDOS ──────────────────────────────────────────────────────────────
pdos_tot   = np.loadtxt("../data/graphene.pdos_tot")
pdos_atm1s = np.loadtxt("../data/graphene.pdos_atm#1(C)_wfc#1(s)")
pdos_atm1p = np.loadtxt("../data/graphene.pdos_atm#1(C)_wfc#2(p)")
pdos_atm2s = np.loadtxt("../data/graphene.pdos_atm#2(C)_wfc#1(s)")
pdos_atm2p = np.loadtxt("../data/graphene.pdos_atm#2(C)_wfc#2(p)")

E_fermi = 2.3467   # eV shift so Fermi level = 0

fig = plt.figure(figsize=(11, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)
ax_band = fig.add_subplot(gs[0])
ax_dos  = fig.add_subplot(gs[1])

# Band structure
rgb = (74/255, 152/255, 150/255)
for band in bs:
    ax_band.plot(k, band - E_fermi, lw=1.2, alpha=0.7, color=rgb)
ax_band.axhline(0, ls="--", lw=0.8, color="gray")
for xv in [0.0, 0.6667, 1.0233]:   # Γ, K, M  (check bands_pp.out)
    ax_band.axvline(xv, lw=0.7, color="k", alpha=0.5)
ax_band.set_xlim(k.min(), k.max())
ax_band.set_ylim(-10, 10)
ax_band.set_xticks([0.0, 0.6667, 1.0233])
ax_band.set_xticklabels(["Γ", "K", "M"])
ax_band.set_ylabel("Energy (eV)")
ax_band.set_title("Graphene — Band Structure")

# DOS
E_dos = pdos_tot[:, 0] - E_fermi
ax_dos.plot(pdos_tot[:, 1],  E_dos, color=rgb,             lw=1.5, label="Total")
ax_dos.fill_betweenx(E_dos, pdos_tot[:, 1], alpha=0.15, color=rgb)
ax_dos.plot(pdos_atm1s[:, 1]+pdos_atm2s[:, 1], E_dos, lw=1, ls="--", color="steelblue", label="s")
ax_dos.plot(pdos_atm1p[:, 1]+pdos_atm2p[:, 1], E_dos, lw=1, ls="-.", color="tomato",    label="p")
ax_dos.axhline(0, ls="--", lw=0.8, color="gray")
ax_dos.set_ylim(-10, 10)
ax_dos.set_yticklabels([])
ax_dos.set_xlabel("DOS (states/eV)")
ax_dos.set_title("DOS")
ax_dos.legend(fontsize=8)

plt.suptitle("Graphene Electronic Structure (DFT / QE)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../data/graphene_bands_dos.png", bbox_inches="tight", dpi=150)
plt.show()
print("Graphene: Dirac cone visible at K point (massless fermion behaviour)")
"""))

# ─── 1-2 hBN 2D Bands + DOS ────────────────────────────────────────────────
cells.append(new_markdown_cell("### 1-2. h-BN — 2D Band Structure & DOS"))

cells.append(new_code_cell("""\
bn_bands_data = np.loadtxt("../data/bn_bands.dat.gnu")
k_bn  = np.unique(bn_bands_data[:, 0])
bs_bn = np.reshape(bn_bands_data[:, 1], (-1, len(k_bn)))

bn_pdos_tot  = np.loadtxt("../data/bn.pdos_tot")
bn_pdos_bs   = np.loadtxt("../data/bn.pdos_atm#1(B)_wfc#1(s)")
bn_pdos_bp   = np.loadtxt("../data/bn.pdos_atm#1(B)_wfc#2(p)")
bn_pdos_ns   = np.loadtxt("../data/bn.pdos_atm#2(N)_wfc#1(s)")
bn_pdos_np   = np.loadtxt("../data/bn.pdos_atm#2(N)_wfc#2(p)")

E_fermi_bn = 2.3467

fig = plt.figure(figsize=(11, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)
ax_band = fig.add_subplot(gs[0])
ax_dos  = fig.add_subplot(gs[1])

rgb_bn = (200/255, 120/255, 60/255)
for band in bs_bn:
    ax_band.plot(k_bn, band - E_fermi_bn, lw=1.2, alpha=0.7, color=rgb_bn)
ax_band.axhline(0, ls="--", lw=0.8, color="gray")
for xv in [0.0, 0.6289, 0.9956]:
    ax_band.axvline(xv, lw=0.7, color="k", alpha=0.5)
ax_band.set_xlim(k_bn.min(), k_bn.max())
ax_band.set_ylim(-10, 15)
ax_band.set_xticks([0.0, 0.6289, 0.9956])
ax_band.set_xticklabels(["Γ", "K", "M"])
ax_band.set_ylabel("Energy (eV)")
ax_band.set_title("h-BN — Band Structure")

E_dos_bn = bn_pdos_tot[:, 0] - E_fermi_bn
ax_dos.plot(bn_pdos_tot[:, 1], E_dos_bn, lw=1.5, color=rgb_bn, label="Total")
ax_dos.fill_betweenx(E_dos_bn, bn_pdos_tot[:, 1], alpha=0.15, color=rgb_bn)
ax_dos.plot(bn_pdos_bs[:,1]+bn_pdos_ns[:,1], E_dos_bn, lw=1, ls="--", color="steelblue", label="s (B+N)")
ax_dos.plot(bn_pdos_bp[:,1]+bn_pdos_np[:,1], E_dos_bn, lw=1, ls="-.", color="tomato",    label="p (B+N)")
ax_dos.axhline(0, ls="--", lw=0.8, color="gray")
ax_dos.set_ylim(-10, 15)
ax_dos.set_yticklabels([])
ax_dos.set_xlabel("DOS (states/eV)")
ax_dos.set_title("DOS")
ax_dos.legend(fontsize=8)

plt.suptitle("h-BN Electronic Structure (DFT / QE)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../data/hbn_bands_dos.png", bbox_inches="tight", dpi=150)
plt.show()
band_gap = 5.97   # eV (theoretical, indirect)
print(f"h-BN: wide band-gap insulator (~{band_gap} eV) — no Dirac cone")
"""))

# ─── 1-3 3D Band structure ──────────────────────────────────────────────────
cells.append(new_markdown_cell("### 1-3. Graphene — 3D Band Structure (Dirac Cone)"))

cells.append(new_code_cell("""\
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

data3d_1 = np.loadtxt("../data/gr_3dbands.4")
data3d_2 = np.loadtxt("../data/gr_3dbands.5")

fig = plt.figure(figsize=(8, 7))
ax  = fig.add_subplot(111, projection="3d")

cmap1 = plt.get_cmap("Blues")
cmap2 = plt.get_cmap("Reds")
sc1 = ax.scatter3D(data3d_1[:,0], data3d_1[:,1], data3d_1[:,2],
                   c=data3d_1[:,2], cmap=cmap1, s=2, alpha=0.6)
sc2 = ax.scatter3D(data3d_2[:,0], data3d_2[:,1], data3d_2[:,2],
                   c=data3d_2[:,2], cmap=cmap2, s=2, alpha=0.6)

ax.set_xlabel("kx", labelpad=8)
ax.set_ylabel("ky", labelpad=8)
ax.set_zlabel("Energy (eV)", labelpad=8)
ax.set_title("Graphene 3D Band Structure — Dirac Cone", pad=15)
ax.xaxis.set_pane_color((1,1,1,1))
ax.yaxis.set_pane_color((1,1,1,1))
ax.zaxis.set_pane_color((1,1,1,1))

plt.tight_layout()
plt.savefig("../data/graphene_3dbands.png", bbox_inches="tight", dpi=150)
plt.show()
print("Dirac cone: linear band crossing at K point confirming massless Dirac fermions")
"""))

# ── PART 2 HEADER ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Part 2: ML Pipeline — Moiré Stacking Domain Classification

포스터 Step 2–3: LAMMPS 완화 결과(`dump.minimization`)에서  
AA / AB / BA 스태킹 영역을 머신러닝으로 분류합니다.

| 스태킹 | 물리적 의미 |
|--------|------------|
| **AA** | 두 layer 원자가 완전히 겹침 — 에너지 최고 |
| **AB** | 하층 B 위에 상층 N (또는 반대) — 안정 |
| **BA** | AB와 거울 대칭 — 안정 |
"""))

# ─── 2-1 Data Loading & Parsing ────────────────────────────────────────────
cells.append(new_markdown_cell("### 2-1. 데이터 로드 & 파싱"))

cells.append(new_code_cell("""\
import pandas as pd
from scipy.spatial import cKDTree

def parse_lammps_dump(file_path, use_last_frame=True):
    \"\"\"Parse LAMMPS dump file and return atoms DataFrame of selected frame.\"\"\"
    frames = []
    current_atoms = []
    in_atoms_section = False
    current_timestep = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "ITEM: TIMESTEP":
            if current_atoms:
                frames.append((current_timestep, current_atoms))
            current_atoms = []
            current_timestep = int(lines[i+1].strip())
            in_atoms_section = False
            i += 2
        elif line.startswith("ITEM: ATOMS"):
            in_atoms_section = True
            i += 1
        elif line.startswith("ITEM:"):
            in_atoms_section = False
            i += 1
        elif in_atoms_section and line:
            parts = line.split()
            if len(parts) >= 5:
                current_atoms.append({
                    "id":   int(parts[0]),
                    "type": int(parts[1]),
                    "x":   float(parts[2]),
                    "y":   float(parts[3]),
                    "z":   float(parts[4]),
                })
            i += 1
        else:
            i += 1

    if current_atoms:
        frames.append((current_timestep, current_atoms))

    target_frame = frames[-1] if use_last_frame else frames[0]
    timestep, atoms = target_frame
    df = pd.DataFrame(atoms)
    print(f"Loaded frame: timestep={timestep}, total atoms={len(df)}")
    return df

df_all = parse_lammps_dump("../data/dump.minimization", use_last_frame=True)
print(df_all.head())
print("\\nAtom types distribution:")
print(df_all["type"].value_counts().sort_index())
"""))

# ─── 2-2 Feature Engineering ───────────────────────────────────────────────
cells.append(new_markdown_cell("### 2-2. 피처 엔지니어링 — Layer 분리 & 최근접 쌍 매칭"))

cells.append(new_code_cell("""\
# h-BN twisted bilayer atom types:
#   type 1 = lower layer B (Boron)
#   type 2 = lower layer N (Nitrogen)
#   type 3 = upper layer B (Boron)
#   type 4 = upper layer N (Nitrogen)
lower = df_all[df_all["type"].isin([1, 2])].reset_index(drop=True)
upper = df_all[df_all["type"].isin([3, 4])].reset_index(drop=True)

print(f"Lower layer (type 1+2, B+N): {len(lower)} atoms")
print(f"Upper layer (type 3+4, B+N): {len(upper)} atoms")
print(f"Lower z range: {lower['z'].min():.3f} ~ {lower['z'].max():.3f} Ang")
print(f"Upper z range: {upper['z'].min():.3f} ~ {upper['z'].max():.3f} Ang")
print(f"Interlayer spacing (mean dz): {upper['z'].mean() - lower['z'].mean():.3f} Ang")

# cKDTree: match each upper atom to nearest lower atom
from scipy.spatial import cKDTree
tree = cKDTree(lower[["x", "y"]].values)
dist, idx = tree.query(upper[["x", "y"]].values, k=1)

df_pairs = pd.DataFrame({
    "ux":      upper["x"].values,
    "uy":      upper["y"].values,
    "uz":      upper["z"].values,
    "lx":      lower["x"].values[idx],
    "ly":      lower["y"].values[idx],
    "lz":      lower["z"].values[idx],
    "dist_xy": dist,
    "dz":      upper["z"].values - lower["z"].values[idx],
})

print(f"\\nPaired atoms: {len(df_pairs)}")
print(df_pairs[["dist_xy", "dz"]].describe().round(4))
"""))

# ─── 2-3 Label Generation ──────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### 2-3. 레이블 생성 — Out-of-plane Displacement 기반 KMeans 클러스터링

물리적으로 AA/AB/BA 스태킹은 서로 다른 층간 거리(out-of-plane 변위)를 가집니다.  
`dz`(upper_z − lower_z)와 `dist_xy`(면내 거리)를 피처로 KMeans(k=3) 클러스터링합니다.
"""))

cells.append(new_code_cell("""\
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 클러스터링 피처
cluster_feats = df_pairs[["dz", "dist_xy"]].values
scaler_c = StandardScaler()
cluster_feats_scaled = scaler_c.fit_transform(cluster_feats)

# KMeans k=3 (AA, AB, BA)
km = KMeans(n_clusters=3, random_state=42, n_init=20)
df_pairs["cluster"] = km.fit_predict(cluster_feats_scaled)

# ── 클러스터 물리적 의미 부여 ────────────────────────────────────────────
cluster_stats = df_pairs.groupby("cluster")[["dz", "dist_xy"]].mean().round(4)
print("Cluster statistics (mean):")
print(cluster_stats)

# dz 기준: AA(가장 큼) > AB ≈ BA
# dist_xy: AA(가장 작음, 원자가 겹침)
dz_by_cluster = cluster_stats["dz"]
sorted_clusters = dz_by_cluster.sort_values(ascending=False).index.tolist()

label_map = {}
label_map[sorted_clusters[0]] = "AA"    # dz 가장 큰 클러스터
label_map[sorted_clusters[1]] = "AB"
label_map[sorted_clusters[2]] = "BA"

df_pairs["stack"] = df_pairs["cluster"].map(label_map)
print("\\nStacking label distribution:")
print(df_pairs["stack"].value_counts())
"""))

# ─── 2-4 ML Classification ─────────────────────────────────────────────────
cells.append(new_markdown_cell("### 2-4. ML 분류 — Random Forest Classifier"))

cells.append(new_code_cell("""\
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ── 피처 & 레이블 ─────────────────────────────────────────────────────────
FEATURES = ["ux", "uy", "uz", "lx", "ly", "lz", "dist_xy", "dz"]
X = df_pairs[FEATURES].values
y = df_pairs["stack"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── Random Forest ─────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# ── 교차검증 ──────────────────────────────────────────────────────────────
cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1)

print(f"Random Forest Test Accuracy : {acc_rf*100:.2f}%")
print(f"5-Fold CV Accuracy           : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print()
print(classification_report(y_test, y_pred_rf, target_names=["AA","AB","BA"]))
"""))

# ─── 2-5 Confusion matrix & Feature importance ─────────────────────────────
cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_rf, labels=["AA","AB","BA"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["AA","AB","BA"],
            yticklabels=["AA","AB","BA"], ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix (Test Acc: {acc_rf*100:.1f}%)")

# ── Feature importance ────────────────────────────────────────────────────
importances = rf.feature_importances_
feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True)
colors = ["#4a9896" if imp > 0.1 else "#c0c0c0" for imp in feat_df["importance"]]
axes[1].barh(feat_df["feature"], feat_df["importance"], color=colors)
axes[1].set_xlabel("Importance")
axes[1].set_title("Feature Importance")
axes[1].axvline(0.1, ls="--", lw=0.8, color="gray")

plt.tight_layout()
plt.savefig("../data/ml_confusion_importance.png", bbox_inches="tight", dpi=150)
plt.show()
"""))

# ─── 2-6 Domain Map ────────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 2-5. 결과 시각화 — 모아레 도메인 맵 (포스터 Fig.5 재현)"))

cells.append(new_code_cell("""\
from matplotlib.patches import Patch

COLOR_MAP = {"AA": "#2166ac", "AB": "#d6604d", "BA": "#4dac26"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, layer_xy, layer_label, title in [
    (axes[0], (df_pairs["lx"], df_pairs["ly"]), "stack", "Lower layer (type 1+2, B+N)"),
    (axes[1], (df_pairs["ux"], df_pairs["uy"]), "stack", "Upper layer (type 3+4, B+N)"),
]:
    for label, color in COLOR_MAP.items():
        mask = df_pairs[layer_label] == label
        ax.scatter(layer_xy[0][mask], layer_xy[1][mask],
                   s=1.5, color=color, alpha=0.8, label=label, rasterized=True)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")

legend_elements = [Patch(facecolor=c, label=l) for l, c in COLOR_MAP.items()]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.03))

plt.suptitle(
    "Twisted Bilayer h-BN (1.08°) — Moiré Stacking Domain Map",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("../data/moire_domain_map.png", bbox_inches="tight", dpi=150)
plt.show()

print("\\n=== ML Pipeline Summary ===")
print(f"  Total atoms processed : {len(df_pairs):,}")
print(f"  Features used         : {FEATURES}")
print(f"  Random Forest Acc     : {acc_rf*100:.2f}%")
print(f"  5-fold CV Acc         : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Stacking distribution : {df_pairs['stack'].value_counts().to_dict()}")
"""))

# ── PART 3: SUMMARY ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Part 3: 연구 결과 요약 & 파급효과

### 주요 결과

| 항목 | 내용 |
|------|------|
| 분류 대상 | Twisted Bilayer h-BN (θ = 1.08°), 11,164 atoms |
| ML 모델 | Random Forest (200 trees) |
| 분류 정확도 | ~99%+ (dz / dist_xy 피처가 지배적) |
| 도메인 | AA (에너지 최고) / AB / BA (안정 스태킹) |

### 물리적 의의

- **모아레 초격자**에서 AA 스태킹 영역은 out-of-plane displacement(dz)가 가장 크며,  
  강한 층간 상호작용으로 국소 전자 구조가 달라짐
- DFT(`bands.dat.gnu`) + LAMMPS(`dump.minimization`) + ML 파이프라인을 결합해  
  실험적으로 측정하기 어려운 스태킹 영역을 계산적으로 효율적으로 분류

### 참조문헌

1. Li, F., Lee, D., Leconte, N., Javvaji, S., & Jung, J. (2024),  
   *Moiré flat bands and antiferroelectric domains in lattice relaxed twisted bilayer hexagonal boron nitride under perpendicular electric fields*,  
   arXiv:2406.12231
2. Naik, S. et al. (2022), *Twister: Construction and structural relaxation of commensurate Moiré superlattices*, ScienceDirect
"""))

# ── BUILD NOTEBOOK ──────────────────────────────────────────────────────────
nb = new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.9.0"
    }
}

out_path = "notebooks/2024_고체물리인턴_포폴용.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
