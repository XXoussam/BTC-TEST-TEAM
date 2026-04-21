# Automated Cryptocurrency Trading — Project Overview

This repository contains two independent research directions, each in its own folder. Both tackle automated BTC/crypto trading using deep learning but take fundamentally different approaches.

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
**Assets:** ADA, BTC, ETH, LTC — 1-hour bars  
**Data:** 4 HDF5 files (upload required)

The agent learns to manage a portfolio of 4 cryptocurrencies: at each timestep it decides which asset to hold (or go to cash). The three notebooks are identical except for how raw prices are normalized before being fed to the LSTM encoder network. The goal is to compare which normalization strategy leads to better training stability and final performance.

| Notebook | Normalization | How it works |
|---|---|---|
| `colab_option1_episode_norm.ipynb` | Episode-Start Price | OHLC divided by the open price at episode start — values stay near 1.0, scale-invariant |
| `colab_option2_log_returns.ipynb` | Log Returns | `log(P_t / P_{t-1})` replaces absolute prices — stationary signal, standard in quant finance |
| `colab_option3_rolling_zscore.ipynb` | Rolling Z-Score | Z-scored over the 24-bar observation window — adapts to local price regime dynamically |

---

## 2. `CAT_BTC_Prediction/` — Candle Analysis Tool (CAT)

**Approach:** Supervised Learning (Binary Classification)  
**Framework:** TensorFlow / Keras  
**Asset:** BTC/USDT — 1-minute bars  
**Data:** Fetched automatically from Binance (no files needed)

CAT trains a **Transformer Encoder** (~33M parameters) to predict whether the next 1-minute BTC candle will close **UP or DOWN**. It uses 10 features per candle including OHLCV, Order Book Imbalance (proxy), RSI, EMA diff, volume ratio, and price acceleration. The output is a probability that drives a LONG / SHORT / HOLD signal with a configurable confidence threshold.

---

## Key Differences at a Glance

| | `normalization_experiments` | `CAT_BTC_Prediction` |
|---|---|---|
| **Learning paradigm** | Reinforcement Learning | Supervised Learning |
| **Task** | Portfolio management | Price direction classification |
| **Assets** | 4 cryptos (ADA, BTC, ETH, LTC) | BTC/USDT only |
| **Timeframe** | 1-hour bars | 1-minute bars |
| **Model** | Double Dueling DQN + LSTM | Transformer Encoder (BERT-style) |
| **Output** | Action: which asset to hold | P(UP) → LONG / SHORT / HOLD signal |
| **Data source** | HDF5 files (upload required) | Auto-fetched from Binance S3 |

---

## Getting Started

Each folder has its own `README.md` with detailed setup instructions, architecture descriptions, and dependency lists. Both notebooks are designed to run on **Google Colab** or **Kaggle** with a GPU runtime — all dependencies are installed automatically by the first cell.
# BTC-TEST-TEAM
