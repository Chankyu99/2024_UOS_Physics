# 2024 UOS Physics Internship: Twisted Bilayer h-BN ML Study

![2024 고체물리 인턴 연구 포스터](poster.png)

> **서울시립대학교 물리학과 하계 인턴십 (2024)**  
> 뒤틀린 이중층 헥사고날 보론 나이트라이드(Twisted Bilayer h-BN)의 모아레 패턴을 DFT + LAMMPS + ML로 분류

---

## 📌 Research Background

뒤틀린 이중층 그래핀은 특정 "매직 각도(~1.1°)"에서 초전도성을 나타내는 물질로, 2018년 Nature에 발표된 이후 응집물질물리학의 핵심 연구 주제가 되었습니다. 본 프로젝트는 DFT(Density Functional Theory) 시뮬레이션으로 생성된 대규모 전자구조 데이터를 기반으로, **모아레 초격자의 도메인 분류 문제를 머신러닝으로 접근**한 연구입니다.

---

## 🧠 ML Pipeline

```
DFT Simulation (Quantum ESPRESSO)
        ↓
Raw band structure / PDOS data
        ↓  [data/]
Data Engineering (Python)
        ↓  [src/]
Feature Extraction + Preprocessing
        ↓  [notebooks/]
Domain Classification (ML)
        ↓
Stacking Region Identification (AA / AB / BA)
```

### Key Components

| Module | 역할 |
|--------|------|
| `src/twister.py` | 뒤틀림 각도별 초격자 좌표 생성 |
| `src/funcs.py` | 밴드구조·PDOS 파싱 및 피처 추출 유틸 |
| `src/tovasp.py` | VASP 입력 포맷 변환 |
| `src/plot.py` | 시각화 헬퍼 함수 |
| `notebooks/2024_고체물리인턴_포폴용.ipynb` | **메인 포트폴리오 노트북** (EDA → 분류) |
| `notebooks/Twisted_layerplot_DomainClassification.ipynb` | 도메인 분류 실험 |

---

## 🗂️ Directory Structure

```
2024_UOS_Physics/
├── data/
│   ├── graphene_bands.dat.gnu          # Graphene 밴드구조 (QE 출력)
│   ├── graphene_pdos_tot.dat           # Graphene 전체 PDOS
│   ├── graphene_pdos_C1_s/p.dat        # C 원자별 궤도 PDOS
│   ├── graphene_pdos_C2_s/p.dat
│   ├── graphene_3dbands_4/5.dat        # 3D 밴드구조 데이터
│   ├── hbn_bands.dat.gnu               # h-BN 밴드구조 (QE 출력)
│   ├── hbn_pdos_tot.dat                # h-BN 전체 PDOS
│   ├── hbn_pdos_B/N_s/p.dat           # B, N 원자별 궤도 PDOS
│   ├── hbn_lammps_dump.dat            # LAMMPS 완화 결과 (ML 입력, ~8MB)
│   ├── hbn_superlattice.dat           # 초격자 좌표
│   ├── hbn_twist.inp                  # Twister 입력 파라미터 (θ = 1.08°)
│   ├── hbn_input_data.txt             # 메인 입력 데이터
│   └── *.png                          # 시각화 결과 이미지
├── notebooks/
│   ├── 2024_고체물리인턴_포폴용.ipynb   ← 메인 포트폴리오
│   ├── domain_classification.ipynb    # 도메인 분류 실험
│   ├── 2024_summer_solid_state.ipynb  # 여름방학 스터디 노트
│   ├── tutorial_dft_qe.ipynb          # QE DFT 튜토리얼
│   ├── 2024_08_07_dft_notes.ipynb
│   ├── 2024_08_12_notes.ipynb
│   └── archive/                       # 임시 노트북 보관
├── src/
│   ├── __init__.py
│   ├── twister.py                     # 초격자 좌표 생성
│   ├── funcs.py                       # 파싱·피처 추출 유틸
│   ├── tovasp.py                      # VASP 포맷 변환
│   └── plot.py                        # 시각화 CLI 모듈
├── 2024 고체물리 인턴 연구 포스터.pdf
├── 참고논문.pdf
├── build_notebook.py                  # 포트폴리오 노트북 빌드 스크립트
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Key Results

### ML Classification (Twisted Bilayer h-BN θ = 1.08°)

| 항목 | 값 |
|------|-----|
| 데이터 | LAMMPS dump, 최종 프레임 (timestep 855), **11,164 atoms** |
| 레이어 분리 | lower B/N (type 1+2): 5,582개 / upper B/N (type 3+4): 5,582개 |
| 층간 거리 (interlayer Δz) | **3.273 Å** |
| ML 모델 | Random Forest (n_estimators=200, 5-fold CV) |
| **Test Accuracy** | **100.00%** |
| **5-fold CV** | **99.98% ± 0.04%** |

### Stacking Domain Distribution

| 도메인 | 원자 쌍 수 | 비율 | 물리적 의미 |
|--------|-----------|------|------------|
| **AA** | 571 | 10.2% | 두 layer 원자 완전 겹침 — 층간 반발 최고 |
| **AB** | 2,551 | 45.7% | 하층 B 위에 상층 N — 안정 스태킹 |
| **BA** | 2,460 | 44.1% | AB 거울 대칭 — 안정 스태킹 |

> AA 영역이 가장 좁고(10%), 안정 상태인 AB/BA가 90%를 차지하며  
> 이는 참고논문(Li et al. 2024)의 lattice relaxation 이론과 일치합니다.

---

## ⚙️ Setup

```bash
# 가상환경 생성 및 패키지 설치
python -m venv venv
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Jupyter 실행
jupyter notebook notebooks/
```

---

## 📎 References

1. Li, F., Lee, D., Leconte, N., Javvaji, S., & Jung, J. (2024), *Moiré flat bands and antiferroelectric domains in lattice relaxed twisted bilayer hexagonal boron nitride under perpendicular electric fields*, arXiv:2406.12231
2. Naik, S. et al. (2022). *Twister: Construction and structural relaxation of commensurate Moiré superlattices*, ScienceDirect
3. Cao, Y. et al. (2018). *Unconventional superconductivity in magic-angle graphene superlattices.* **Nature** 556, 43–50
4. Quantum ESPRESSO: [quantum-espresso.org](https://www.quantum-espresso.org)
5. LAMMPS: [lammps.org](https://www.lammps.org)
