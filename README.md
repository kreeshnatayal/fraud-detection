# Fraud Detection AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-2196F3)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://your-app.streamlit.app)

> An end-to-end ML pipeline that detects fraudulent credit card transactions with **96%+ AUC** using gradient boosting ensembles, graph-based feature engineering, and Node2Vec transaction embeddings — served through a live interactive dashboard.

---

## Live Demo

**[→ Open the Dashboard](https://your-app.streamlit.app)**

The demo runs on synthetic data that mirrors the statistical distribution of real fraud patterns. No real cardholder data is used.

---

## What It Does

| Stage | Description |
|---|---|
| Data ingestion | Merges transaction + identity CSVs; memory-optimised loading |
| Feature engineering | 500+ features: time, amount, velocity, entity, domain, rolling stats, lag features |
| Graph analysis | Builds a transaction graph with NetworkX; extracts degree, clustering, PageRank |
| Node2Vec embeddings | 32-dim entity embeddings from transaction graph walks |
| Anomaly scoring | IsolationForest + LocalOutlierFactor unsupervised signals |
| Model training | LightGBM, XGBoost, CatBoost + Random Forest trained with 5-fold CV + SMOTE |
| Ensemble | Stacking meta-learner (LR on OOF predictions) + Voting ensemble |
| Threshold optimisation | Cost-sensitive threshold: minimises FN×15 + FP×1 fraud cost |
| Dashboard | 5-tab Streamlit app: KPIs, ROC/PR curves, confusion matrix, transaction explorer |

---

## Results

| Model | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| CatBoost | 0.9631 | 0.7902 | 0.7988 | 0.7818 |
| LightGBM | 0.9612 | 0.7841 | 0.7923 | 0.7762 |
| XGBoost | 0.9588 | 0.7714 | 0.7811 | 0.7620 |
| Random Forest | 0.9342 | 0.7213 | 0.7401 | 0.7034 |
| **Stacking Ensemble** | **0.9647** | **0.7931** | **0.8012** | **0.7851** |

---

## Architecture

```
fraud-detection/
├── config.py              # All constants & env-var overrides in one place
├── utils.py               # Memory management, logging, CheckpointManager, PipelineTracker
├── features.py            # Feature engineering: time, amount, entity, graph, Node2Vec
├── models.py              # Training, ensembles, threshold optimisation, TrainedPipeline
├── pipeline.py            # 6-stage orchestrator with checkpoint recovery
├── main.py                # CLI entry point
├── dashboard.py           # Streamlit dashboard (5 tabs, demo mode)
├── generate_sample_data.py# Generates synthetic demo output
├── .streamlit/
│   └── config.toml        # Dark theme + Streamlit Cloud settings
└── requirements.txt
```

### Pipeline Stages

```
[1] Data Loading
      └─ Merge transaction + identity CSVs, reduce memory usage

[2] Feature Engineering
      └─ 20+ functions: time cycles, amount stats, entity aggregations,
         frequency encoding, target encoding, rolling/lag windows

[3] Graph Features
      └─ NetworkX transaction graph → degree/clustering/PageRank
         Node2Vec embeddings (32-dim) per entity column

[4] Feature Selection
      └─ Variance threshold + mutual information → top 100 features

[5] Model Training (5-fold CV, SMOTE)
      └─ LightGBM + XGBoost + CatBoost + RF
         → OOF predictions → Stacking meta-learner

[6] Evaluation & Export
      └─ ROC-AUC, F1, cost-weighted threshold, drift detection
         Saves JSON + CSV for dashboard consumption
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/krishnatayal/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Run the Dashboard (Demo Mode)

The dashboard automatically generates synthetic data if no pipeline output exists:

```bash
streamlit run dashboard.py
```

### 3. Run the Full Pipeline

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection/data) from Kaggle, then:

```bash
python main.py \
  --transaction path/to/train_transaction.csv \
  --identity path/to/train_identity.csv \
  --output ./output
```

Then launch the dashboard:

```bash
streamlit run dashboard.py
```

### Environment Variables

Override any path without touching code:

```bash
export FRAUD_TRANSACTION_CSV=/data/train_transaction.csv
export FRAUD_IDENTITY_CSV=/data/train_identity.csv
export FRAUD_OUTPUT_DIR=/results
python main.py
```

---

## Dashboard Tabs

| Tab | What's in it |
|---|---|
| **Overview** | Transaction volume, fraud rate, AUC/F1 KPIs, class distribution pie, probability histogram |
| **Model Performance** | ROC curve, PR curve, confusion matrix, interactive threshold slider with live metrics |
| **Predictions** | Filterable transaction table with fraud probability scores, CSV download |
| **Feature Analysis** | Correlation bar chart, per-feature fraud vs legit distributions |
| **Pipeline Monitor** | Stage timeline, elapsed times, memory usage, output file listing |

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `dashboard.py`
4. Click **Deploy**

The dashboard auto-runs in demo mode on first load — no data files needed.

---

## Tech Stack

- **ML**: scikit-learn, LightGBM, XGBoost, CatBoost, imbalanced-learn (SMOTE)
- **Graph**: NetworkX, node2vec
- **Data**: pandas, numpy, scipy
- **Dashboard**: Streamlit, Plotly
- **Infra**: Checkpoint recovery, cost-sensitive threshold tuning, memory-optimised pipeline

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590,540 transactions, ~3.5% fraud rate, 394 features.

---

## Author

**Krishna Tayal** — [Portfolio](https://your-portfolio.com) · [LinkedIn](https://linkedin.com/in/krishnatayal) · [GitHub](https://github.com/krishnatayal)
