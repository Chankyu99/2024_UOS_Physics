# 2024 UOS Physics Internship

![2024 고체물리 인턴 연구 포스터](img/result/poster.png)

---

## Architecture

![Architecture](img/result/architecture.png)
---

## Directory Structure

```
2024_UOS_Physics/
├── data/
│   ├── graphene_band_structure.gnu       # Graphene 밴드구조 (QE 출력)
│   ├── graphene_total_pdos.dat           # Graphene 전체 PDOS
│   ├── graphene_pdos_C1_s.dat            # C1 원자 궤도 s PDOS
│   ├── graphene_pdos_C1_p.dat            # C1 원자 궤도 p PDOS
│   ├── graphene_pdos_C2_s.dat            # C2 원자 궤도 s PDOS
│   ├── graphene_pdos_C2_p.dat            # C2 원자 궤도 p PDOS
│   ├── graphene_3d_bands_layer4.dat      # 3D 밴드구조 데이터 (layer 4)
│   ├── graphene_3d_bands_layer5.dat      # 3D 밴드구조 데이터 (layer 5)
│   ├── graphene_3dbands_sm_4.dat         # Smoothed 3D 밴드구조 (layer 4)
│   ├── graphene_3dbands_sm_5.dat         # Smoothed 3D 밴드구조 (layer 5)
│   ├── hbn_band_structure.gnu            # h-BN 밴드구조 (QE 출력)
│   ├── hbn_total_pdos.dat                # h-BN 전체 PDOS
│   ├── hbn_pdos_tot_v2.dat               # h-BN 전체 PDOS v2
│   ├── hbn_pdos_B_s.dat                  # B 원자 궤도 s PDOS
│   ├── hbn_pdos_B_p.dat                  # B 원자 궤도 p PDOS
│   ├── hbn_pdos_N_s.dat                  # N 원자 궤도 s PDOS
│   ├── hbn_pdos_N_p.dat                  # N 원자 궤도 p PDOS
│   ├── hbn_3d_bands_layer4.dat           # h-BN 3D 밴드구조 (layer 4)
│   ├── hbn_3d_bands_layer5.dat           # h-BN 3D 밴드구조 (layer 5)
│   ├── hbn2_3dbands_4.dat                # h-BN 3D 밴드 (System 2, layer 4)
│   ├── hbn2_3dbands_5.dat                # h-BN 3D 밴드 (System 2, layer 5)
│   ├── hbn_proj_bands.dat                # h-BN Projected 반드 구조
│   ├── hbn_lammps_dump.dat               # LAMMPS 완화 결과 (ML 입력)
│   ├── hbn_superlattice_coordinates.dat  # 초격자 좌표
│   ├── hbn_twist_input.inp               # Twister 입력 파라미터 (θ = 1.08°)
│   └── hbn_input_parameters.txt          # 메인 입력 파라미터 데이터
├── img/
│   ├── result/                           # 완성된 시각화 디렉토리
│   └── visualization/                    # 시각화 분석 디렉토리
├── src/
│   ├── __init__.py                       # 패키지 구성을 위한 파일
│   ├── twister.py                        # 초격자 좌표 생성
│   ├── funcs.py                          # 파싱·피처 추출 유틸
│   ├── tovasp.py                         # VASP 포맷 변환
│   └── plot.py                           # 시각화 CLI 모듈
├── Notebook.ipynb                        # 메인 머신러닝 분석 노트북
├── graphene_band_structure.ipynb         # Graphene 밴드구조 스터디 노트 (메인 노트북 참고용)
├── requirements.txt                      # 환경 패키지 목록
└── README.md                             # 메인 README.md
```

---

## 문제 정의

- 뒤틀림 격자(Moiré Superlattice) 단면 시뮬레이션(LAMMPS relaxation) 데이터에는 수만 개에 달하는 원자의 3D 좌표만 비정형하게 나열되어 있어, 어느 영역이 안정 상태(AB, BA)이고 불안정 상태(AA)인지 분석하려면 육안이나 수작업 병목이 심함
- 기존 브루트포스(Brute-force) 방식의 원자 쌍 거리 계산 시 공간 복잡도가 $O(N^2)$에 달해, 대규모 초격자 시스템 연구에 있어 연산 과부하가 발생

## 프로젝트 목표

- 방대한 원자 시뮬레이션 데이터를 전처리부터 모델 학습, 성능 평가, 그리고 벌집 맵(Honeycomb Map) 시각화까지 처리하는 End-to-End 예측 자동화 파이프라인을 구축

## 선행 연구 및 사례 분석

### 선행 연구 검토
- Li et al. (2024): 층간 완화(Lattice relaxation) 현상에 의하여 반발력이 높은 AA 도메인은 좁게 수축하고, 에너지적으로 안정된 AB/BA 도메인이 넓게 확장된다는 이론적 배경을 제시하며, 본 예측 결과 역시 이와 일치해야함

### 유사 프로젝트 사례
- 자성/비자성 물질의 결정 구조 데이터에 차원 축소 맵핑이나 딥러닝(GNN 등)을 적용해 스태킹 결함을 탐지하는 다양한 최신 소재 정보학사례들이 존재

## 데이터 수집 및 준비

### 데이터 소스
- Twister 도구를 통해 초기 초격자를 구축하고, 이를 LAMMPS로 구조 완화시킨 텍스트 형태의 덤프 파일 (hbn_lammps_dump.dat, 약 1.1만 개 원자 행, 8MB).

### 데이터 전처리
  - 원시 텍스트 파일 구조(ITEM: ATOMS)를 프로그램이 인식하여 마지막 프레임(Timestep 855)의 원자 ID, 타입, x,y,z 좌표만 동적으로 자동 파싱.
  - 이를 Pandas DataFrame으로 즉각 변환하여 상/하층 레이어를 분리한 뒤, cKDTree 모델로 상하층의 가장 인접한 원자 쌍 정보를 결합합.

## 모델링 방법론 (머신러닝 파이프라인)

### 모델링 방법 검토
- 라벨이 없는 비지도 상황이므로 K-Means 클러스터링(k=3) 을 통해 도메인의 Pseudo-Label을 생성
- K-Means 결과를 검증하고, 나아가 좌표계 자체와의 비선형적 관계를 모델링해 내기 위해 Random Forest 분류기 선택

### 모델링 계획
- 전처리: 스케일 이상치를 막기 위한 `StandardScaler` 적용.
- 피처 엔지니어링: 물리적 분류의 핵심인 층간 수직 거리(`dz`)와 수평 최단 거리(`dist_xy`) 변수 도출.
- 모델 학습 및 진단: K-Means로 기반 잡은 라벨을 이용해 RF 모델을 훈련 -> 이때, `dz` 와 `dist_xy` 변수가 피처 특성상 강력한 정보 Leakage를 유발한다는 것을 진단
- 고도화: 정보 누수를 일으키는 파생 변수를 모델에서 과감하게 배제하고, 가장 엄격한 조건의 공간 좌표(`ux`, `uy` 주변 등) 피처만으로 Base 모델 재설계 수행. 5-Fold 결측 검증 시도.

## 주요 리스크 요인 

- K-Means 클러스터링을 할 때 `dz`와 `dist_xy`를 정답화의 핵심 기준으로 사용했는데, 이를 평가 과정인 RF 훈련 피처에 그대로 투입하면 정확도 100%의 Overfitting이 발생
- Feature Importance 분석을 활용하여 어떤 변수가 누수를 유발했는지 원인을 파악하고, 이를 삭제한 Baseline 피처 로직 설계로 전환해 객관성과 공정성을 확보

## 결과

### ML Classification (Twisted Bilayer h-BN θ = 1.08°)

| 항목 | 값 |
|------|-----|
| 데이터 | LAMMPS dump, 최종 프레임 (timestep 855), **11,164 atoms** |
| 레이어 분리 | lower B/N (type 1+2): 5,582개 / upper B/N (type 3+4): 5,582개 |
| 층간 거리 (interlayer Δz) | **3.273 Å** |
| ML 모델 | Random Forest (n_estimators=200, 5-fold CV) |
| **Test Accuracy** | **91.94%** (Data Leakage 배제 기준) |
| **5-fold CV** | **26.24% ± 4.89%** |

### Stacking Domain Distribution

| 도메인 | 원자 쌍 수 | 비율 | 물리적 의미 |
|--------|-----------|------|------------|
| **AA** | 571 | 10.2% | 두 layer 원자 완전 겹침 — 층간 반발 최고 |
| **AB** | 2,551 | 45.7% | 하층 B 위에 상층 N — 안정 스태킹 |
| **BA** | 2,460 | 44.1% | AB 거울 대칭 — 안정 스태킹 |

> AA 영역이 가장 좁고(10%), 안정 상태인 AB/BA가 90%를 차지하며  
> 이는 참고논문(Li et al. 2024)의 lattice relaxation 이론과 일치함을 확인

---

## 참고문헌

1. Li, F., Lee, D., Leconte, N., Javvaji, S., & Jung, J. (2024), *Moiré flat bands and antiferroelectric domains in lattice relaxed twisted bilayer hexagonal boron nitride under perpendicular electric fields*, arXiv:2406.12231
2. Naik, S. et al. (2022). *Twister: Construction and structural relaxation of commensurate Moiré superlattices*, ScienceDirect
3. Quantum ESPRESSO: [quantum-espresso.org](https://www.quantum-espresso.org)
4. LAMMPS: [lammps.org](https://www.lammps.org)
