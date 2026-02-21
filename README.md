# 2024 UOS Physics Internship: Twisted Bilayer Graphene ML Study

> **서울시립대학교 물리학과 하계 인턴십 (2024)**  
> 뒤틀린 이중층 그래핀(Twisted Bilayer Graphene)의 모아레 패턴을 DFT 시뮬레이션 + ML로 분류

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
Stacking Region Identification (AA / AB)
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
├── data/                  # 원시 데이터 및 중간 산출물
│   ├── Input_data.txt         # 메인 입력 데이터 (~8.2MB)
│   ├── dump.minimization      # 구조 최적화 dump
│   ├── superlattice.dat       # 초격자 좌표 데이터
│   ├── twist.inp              # 뒤틀림 각도 입력 파라미터
│   ├── *.pdos_tot / *.pdos_atm# # 부분 상태밀도(PDOS) 데이터
│   ├── *_3Dbands.dat.*        # 3D 밴드구조 데이터
│   └── *.dat.gnu              # gnuplot 형식 밴드구조
├── src/                   # 데이터 엔지니어링 모듈
│   ├── twister.py
│   ├── funcs.py
│   ├── tovasp.py
│   └── plot.py
├── notebooks/             # EDA · 모델링 노트북
│   ├── 2024_고체물리인턴_포폴용.ipynb   ← 메인 포트폴리오
│   ├── Twisted_layerplot_DomainClassification.ipynb
│   ├── 2024summer_vacation_solid state physics.ipynb
│   ├── t_basic_dft_QE.ipynb
│   └── ...
├── 2024 고체물리 인턴 연구 포스터.pdf
├── requirements.txt
└── README.md
```

---

## 📊 Key Results

- 모아레 초격자 내 **AA / AB 스태킹 영역 분류** 성공
- DFT 전자구조 데이터에서 ML 피처 추출 파이프라인 구현
- 뒤틀림 각도에 따른 밴드구조 변화 시각화

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

## 📎 Reference

- Cao, Y. et al. *Unconventional superconductivity in magic-angle graphene superlattices.* **Nature** 556, 43–50 (2018)
- Quantum ESPRESSO: [https://www.quantum-espresso.org](https://www.quantum-espresso.org)
