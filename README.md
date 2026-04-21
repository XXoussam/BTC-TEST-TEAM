# Automated Cryptocurrency Trading — Project Overview

Two independent research directions, each in its own folder. Both tackle automated BTC/crypto trading using deep learning but take fundamentally different approaches.

---

## Folder Structure

```
├── normalization_experiments/   ← Deep RL agent, 3 normalization variants
│   ├── colab_option1_episode_norm.ipynb
│   ├── colab_option2_log_returns.ipynb
│   ├── colab_option3_rolling_zscore.ipynb
│   └── README.md
│
└── CAT_BTC_Prediction/          ← Transformer classifier, BTC direction prediction
    ├── CAT_BTC_Prediction.ipynb
    └── README.md
```

---

## 1. `normalization_experiments/` — Deep RL Trading Agent

**Approach:** Reinforcement Learning (Double Dueling DQN + Prioritized Experience Replay)  
**Framework:** PyTorch + Tianshou  
**Assets:** ADA, BTC, ETH, LTC — 1-hour bars (resampled from minute HDF5 files)  
**Data:** 4 HDF5 files (upload required), 80/20 chronological train/test split  
**Original design:** Nick Kaparinos (2022)

The agent manages a portfolio of 4 cryptocurrencies: at each timestep it chooses which single asset to hold (or go to cash). A **Dueling DQN** with an **LSTM encoder** (128 hidden units, one per asset) learns from a **Prioritized Experience Replay** buffer (500k transitions, α=0.7, β=0.5). Episodes are 1000 steps drawn randomly from training data; starting balance is $1,000 with a 0.1% trading fee.

The three notebooks are identical except for how raw prices are normalized before being fed to the LSTM — the goal is to compare which strategy leads to better training stability:

| Notebook | Normalization | How it works |
|---|---|---|
| `colab_option1_episode_norm.ipynb` | Episode-Start Price | OHLC ÷ open at episode start — values near 1.0, scale-invariant |
| `colab_option2_log_returns.ipynb` | Log Returns | `log(P_t / P_{t−1})` — stationary signal, standard in quant finance |
| `colab_option3_rolling_zscore.ipynb` | Rolling Z-Score | Z-scored over the 24-bar window — adapts to local price regime |

Training: Adam `lr=1e-4`, 15 epochs × 150k steps, batch 128, ε-greedy from 0.6 → 0.0 over ~6 epochs. Tracked via W&B.

---

## 2. `CAT_BTC_Prediction/` — Candle Analysis Tool (CAT)

**Approach:** Supervised Learning (Binary Classification)  
**Framework:** TensorFlow / Keras  
**Asset:** BTC/USDT — 1-minute bars  
**Data:** Auto-fetched from Binance public S3 (no API key), 51,773 candles ≈ 35 days, 85/15 chronological split

CAT trains a **Transformer Encoder** (~33M parameters, 6 blocks, 8 heads, d_model=512) to predict whether the next 1-minute BTC candle closes **UP or DOWN**. It uses 10 features per candle: OHLCV, Order Book Imbalance proxy, RSI-14, EMA diff, volume ratio, and price acceleration. The output is a `P(UP)` sigmoid probability that drives LONG / SHORT / HOLD signals using a configurable confidence threshold (default: signal only when `|prob − 0.5| > 0.15`).

Training: AdamW `lr=1e-4`, `weight_decay=1e-4`, `binary_crossentropy` loss, 80 max epochs with early stopping (patience=6) and `ReduceLROnPlateau`.

---

## Key Differences at a Glance

| | `normalization_experiments` | `CAT_BTC_Prediction` |
|---|---|---|
| **Learning paradigm** | Reinforcement Learning | Supervised Learning |
| **Task** | Portfolio management | Price direction classification |
| **Assets** | 4 cryptos (ADA, BTC, ETH, LTC) | BTC/USDT only |
| **Timeframe** | 1-hour bars | 1-minute bars |
| **Model** | Double Dueling DQN + LSTM encoder | Transformer Encoder (BERT-style, ~33M params) |
| **Output** | Action: which asset to hold | P(UP) → LONG / SHORT / HOLD |
| **Loss / reward** | Portfolio value change (reward) | binary_crossentropy |
| **Data source** | HDF5 files (upload required) | Auto-fetched from Binance S3 |
| **Train/Test split** | 80% / 20% chronological | 85% / 15% chronological |
| **Experiment tracking** | Weights & Biases (W&B) | Local files + eval_report.txt |
| **Framework** | PyTorch + Tianshou | TensorFlow / Keras |

---

## Getting Started

Each folder has its own `README.md` with full details on data, features, architecture, and all training hyperparameters. Both notebooks are designed for **Google Colab** or **Kaggle** with a GPU runtime — all dependencies are installed automatically by the first cell.
