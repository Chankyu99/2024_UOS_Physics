# 인터뷰 준비 가이드 for `2024_UOS_Physics`

## 1️⃣ 프로젝트 개요
- **목표**: 2차원 물질인 h‑BN(헥사고날 보론 나이트라이드)의 트위스티드 빌레이어를 Quantum ESPRESSO와 LAMMPS 시뮬레이션으로 연구.
- **ML 파트**: 원자 데이터를 기반으로 AA, AB, BA 세 가지 스택 도메인을 Random Forest 로 분류.
- **핵심 산출물**:
  - 깔끔하게 정리된 레포지토리와 재현 가능한 Jupyter Notebook.
  - `src/` 패키지에 시각화 유틸리티 제공.
  - 자동으로 포트폴리오 Notebook을 생성하는 `build_notebook.py`.
  - 상세 README와 문서.

---

## 2️⃣ 디렉터리 구조 (리팩터링 후)
```
2024_UOS_Physics/
│
├─ data/                     # 시뮬레이션 결과 (snake_case, material prefix)
│   ├─ graphene_3dbands_4.dat
│   ├─ graphene_3dbands_5.dat
│   ├─ graphene_bands.dat.gnu
│   ├─ graphene_band_structure.png
│   ├─ graphene_band_structure_dos.png
│   ├─ hbn_bands.dat.gnu
│   ├─ hbn_band_structure.png
│   ├─ hbn_band_structure_dos.png
│   ├─ hbn_lammps_dump.dat          # LAMMPS dump (원래 dump.minimization)
│   ├─ hbn_superlattice.dat
│   ├─ hbn_twist.inp
│   └─ hbn_input_data.txt
│
├─ notebooks/                 # Jupyter Notebook (영문 snake_case)
│   ├─ 2024_08_07_dft_notes.ipynb
│   ├─ 2024_08_12_notes.ipynb
│   ├─ 2024_summer_solid_state.ipynb
│   ├─ domain_classification.ipynb   # 핵심 분석 Notebook (스크립트가 자동 생성)
│   └─ archive/                     # 이전 Untitled*.ipynb 보관
│
├─ src/                       # 파이썬 패키지
│   ├─ __init__.py
│   └─ plot.py                # 밴드, PDOS 시각화 유틸리티
│
├─ build_notebook.py          # 포트폴리오 Notebook 자동 생성 스크립트
├─ requirements.txt           # 파이썬 의존성 (seaborn, nbformat 추가)
├─ .gitignore                # 캐시, 대용량 파일, IDE 파일 무시
├─ README.md                 # 최신 디렉터리 구조와 결과 정리
└─ INTERVIEW_PREP.md         # **이 파일** – 면접 대비 요약
```

---

## 3️⃣ 핵심 스크립트와 역할
| 스크립트 | 목적 | 주요 내용 |
|----------|------|-----------|
| `src/plot.py` | 밴드와 PDOS 시각화 | `plot_bands`, `plot_pdos` 함수 제공, CLI 지원, 오류 처리 강화 |
| `build_notebook.py` | 포트폴리오 Notebook 자동 생성 | LAMMPS dump 파싱 → 원자 쌍 매칭 → `dz`, `dist_xy` 로 KMeans 라벨링 → **데이터 누수 진단** 섹션 추가 |
| `requirements.txt` | 의존성 관리 | `seaborn`(시각화)와 `nbformat`(Notebook 생성) 추가 |
| `.gitignore` | 불필요 파일 제외 | 대용량 dump 파일·캐시·IDE 파일 무시 |

---

## 4️⃣ 데이터 처리 파이프라인 (전체 흐름)
1. **LAMMPS dump**(`hbn_lammps_dump.dat`) 읽기.
2. 원자 타입에 따라 **하부(1,2)**와 **상부(3,4)** 레이어 구분.
3. `cKDTree` 로 가장 가까운 원자 쌍 매칭 → `dx, dy, dz` 계산.
4. **피처 생성**:
   - `dz` = 상부 z – 하부 z (층간 변위)
   - `dist_xy` = xy 평면 거리
5. **Ground‑truth 라벨**: `dz`와 `dist_xy` 로 KMeans(k=3) 수행 → AA/AB/BA 라벨 부여.
6. **ML 분류** (Random Forest) – 3가지 실험:
   - **Full (leakage)**: 모든 8개 피처 사용 (`dz`, `dist_xy` 포함).
   - **Physics‑aware**: 위치 피처만 사용 (`ux, uy, uz, lx, ly, lz`).
   - **Spatial‑only**: xy 좌표만 사용 (`ux, uy, lx, ly`).
7. **평가**: 테스트 정확도, 5‑fold CV, 피처 중요도, 혼동 행렬.
8. **시각화**: 도메인 맵, 피처 중요도 바 차트, 혼동 행렬 히트맵.

---

## 5️⃣ 머신러닝 결과 요약
| 모델 | 테스트 정확도 | 5‑fold CV (평균 ± 표준편차) | 데이터 누수? |
|------|---------------|---------------------------|--------------|
| **Full (leakage)** | 100 % | 99.98 % ± 0.04 % | **있음** – 라벨링에 사용한 `dz`, `dist_xy` 를 그대로 학습에 사용 → 100 %는 KMeans 경계 암기 |
| **Physics‑aware** | **91.94 %** | 26.24 % ± 4.89 % | 없음 – 순수 3D 좌표만 사용 |
| **Spatial‑only** | 85.05 % | 12.84 % ± 1.07 % | 없음 – xy 좌표만 사용 |

- **피처 중요도**: Full 모델에서 `dz`(≈ 69 %)와 `dist_xy`(≈ 10 %)가 전체 중요도의 80 % 이상을 차지 → 명확한 누수 증거.
- **Physics‑aware** 모델도 약 92 % 정확도를 보이며, 순수 기하학 정보만으로도 도메인을 구분할 수 있음을 입증.
- **CV 점수가 낮은 이유**: 모아레 패턴이 **주기적**이라 무작위 폴드가 같은 영역을 섞어버려 일반화가 어려움.

---

## 6️⃣ 주요 설계 결정 및 이유
- **snake_case + material prefix**(`graphene_`, `hbn_`) 로 파일명統一 → OS 간 대소문자 차이와 검색 편의성 향상.
- **`build_notebook.py`** 를 분석 코드와 분리 → 재현성·CI 친화.
- **`.gitignore`** 추가 → 대용량 dump 파일이 레포에 올라가지 않음.
- **데이터 누수 진단** 섹션을 명시적으로 삽입 → ML 결과의 신뢰성을 보장.
- **`src/` 패키지** 로 플롯 함수 모듈화 → 다른 프로젝트에서도 재사용 가능.

---

## 7️⃣ 실행 방법
```bash
# 1️⃣ 레포 복제 및 환경 설정
git clone https://github.com/Chankyu99/2024_UOS_Physics.git
cd 2024_UOS_Physics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣ 포트폴리오 Notebook 자동 생성
python build_notebook.py   # notebooks/2024_고체물리인턴_포폴용.ipynb 생성

# 3️⃣ Notebook 열어 결과 확인
jupyter notebook notebooks/2024_고체물리인턴_포폴용.ipynb
```

---

## 8️⃣ 면접 시 활용 가능한 이야기 포인트
- **문제 정의**: 트위스티드 h‑BN에서 로컬 스택 도메인을 원자 수준에서 자동 분류.
- **데이터 누수 발견**: 라벨링 피처와 학습 피처를 동일하게 사용해 정확도가 100 %가 된 것을 발견하고, 누수 진단·수정 과정을 문서화.
- **물리 기반 피처**: `dz` 없이도 3D 좌표만으로 92 % 정확도 달성 → 물리적 구분 가능성을 입증.
- **재현성**: `build_notebook.py` 로 한 번의 명령으로 전체 파이프라인 재현 가능.
- **소프트웨어 엔지니어링**: 스크립트를 패키지화, 파일명 표준화, `.gitignore` 로 대용량 파일 관리, CI 친화 구조.
- **시각화**: `seaborn`·`matplotlib` 로 고품질 도메인 맵·피처 중요도 차트 제공.

---

## 9️⃣ 앞으로 할 수 있는 확장 아이디어
- **공간 기반 교차 검증**: 모아레 셀 단위로 leave‑one‑out CV 수행 → 일반화 성능 정확히 평가.
- **그래프 신경망(GNN)** 적용 – 원자 간 연결 정보를 활용해 도메인 분류.
- **전이 학습**: 학습된 모델을 다른 2D 물질(예: 트위스티드 그래핀)에도 적용.
- **Git LFS** 도입 – 데이터셋이 커질 경우 효율적인 버전 관리.

---

*Prepared by Antigravity – AI 코딩 어시스턴트*
