# Normalization Experiments — Automated Crypto Trading with Deep RL

Three variants of the same Deep RL trading agent, differing only in how raw prices are normalized before being fed to the network. Everything else is identical: **Double Dueling DQN with Prioritized Experience Replay**, LSTM encoder, 4 crypto assets (ADA, BTC, ETH, LTC), 1-hour bars, 1000-step episodes.

The goal is to compare how the choice of normalization affects training stability and final portfolio performance.

> Original algorithm design based on work by Nick Kaparinos (2022).

---

## Task Definition

- **Learning paradigm:** Reinforcement Learning (off-policy, value-based)
- **Agent objective:** At each 1-hour timestep, choose which single asset to hold (or go to cash), maximizing total portfolio value over a 1000-step episode
- **Action space:** 5 discrete actions — hold ADA | BTC | ETH | LTC | go to cash
- **Reward:** change in portfolio value between consecutive timesteps (after trading fees)
- **Benchmark:** compared against an equal-weighted Buy-and-Hold baseline

---

## Data

| Property | Value |
|---|---|
| **Assets** | ADA/USDT, BTC/USDT, ETH/USDT, LTC/USDT |
| **Raw timeframe** | Minute-level OHLC from HDF5 files |
| **Resampled to** | 1-hour bars (OHLC aggregation) |
| **Date range** | Common overlapping period across all 4 assets |
| **Train / Test split** | 80% / 20% — chronological |
| **Required files** | `ADAUSDT_minutes.hdf5`, `BTCUSDT_minutes.hdf5`, `ETHUSDT_minutes.hdf5`, `LTCUSDT_minutes.hdf5` |

---

## Observation Space

At each timestep the agent receives a flat vector built from:

- **24 timesteps** (current + 23 previous) × **6 features** per asset × **4 assets** = time-series block
- **5 binary flags** appended: which asset is currently held (4 flags) + 1 flag for "all cash"

**6 features per timestep per asset:** `high`, `low`, `close`, `month` (÷12), `day` (÷31), `hour` (÷24)

The open price is used as the normalization anchor (see variants below) and not included as a raw feature.

---

## Normalization Variants

### Option 1 — `colab_option1_episode_norm.ipynb` — Episode-Start Price

OHLC prices are divided by the open price at the **start of each episode**.

- Values stay near 1.0; the network always sees scale-invariant ratios
- The anchor resets every episode — the agent cannot infer absolute price level, only relative movement within the episode
- Works regardless of whether BTC is at $5k or $60k

### Option 2 — `colab_option2_log_returns.ipynb` — Log Returns

Each price value is replaced with `log(P_t / P_{t-1})`.

- Produces a stationary, approximately mean-reverting signal — much more stable than raw prices for neural networks
- Fully scale-invariant; small values near zero = stability, large values = strong moves
- Standard representation in quantitative finance
- Requires one extra historical row per window to compute the first return

### Option 3 — `colab_option3_rolling_zscore.ipynb` — Rolling Z-Score

Each price column is z-scored over the current 24-bar observation window: `(x − mean) / std`.

- Adapts dynamically to the local price regime within the observation window
- Robust to regime changes (bull/bear, high/low volatility)
- Always zero-centered and unit-variance within each window
- If window std ≈ 0 (flat market), values are set to 0 to avoid division instability

---

## Model Architecture

**Q-network and Value-network (Dueling DQN):**

```
Observation vector
  → split into 4 per-asset timeseries + portfolio state flags
  → 4 × LSTM(input=6, hidden=128, batch_first=True)    — one LSTM per asset
       takes last hidden state → 128-dim embedding per asset
  → concat 4 embeddings (512) + portfolio flags (5) → 517-dim
  → Q-head:  Linear(517→128) → ReLU → Linear(128→5)    — action advantages
  → V-head:  Linear(517→128) → ReLU → Linear(128→1)    — state value
  → Dueling combination: Q − mean(Q) + V
```

Alternative encoder options are also implemented (selectable via `encoder_type`):
- `'Attention'` — Transformer encoder with sinusoidal positional encoding
- `'CNN'` — 1D convolutional encoder
- `'MLP'` — flat MLP on concatenated timesteps

Default used in all three notebooks: **`'LSTM'`**

| Hyperparameter | Value |
|---|---|
| `encoder_type` | LSTM |
| `n_neurons` | 128 |
| `q_n_linear_layers` | 2 |
| `v_n_linear_layers` | 2 |
| `dueling` | True |

---

## RL Algorithm — Double Dueling DQN + PER

| Component | Configuration |
|---|---|
| Algorithm | DQN (Tianshou `DQNPolicy`) |
| Double DQN | `is_double=True` — reduces overestimation bias |
| Dueling network | Q and V heads combined — better value estimation |
| Replay buffer | Prioritized VectorReplayBuffer (PER) |
| Buffer capacity | 500,000 transitions |
| PER alpha | 0.7 (prioritization strength) |
| PER beta | 0.5 (importance sampling correction) |
| Discount factor (γ) | 0.99 |
| Target network update | Every 20 steps |
| n-step return | 1 |

---

## Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (`lr=1e-4`) |
| Max epochs | 15 |
| Steps per epoch | 150,000 |
| Steps per collect | 100 |
| Batch size | 128 |
| Update per step | 0.1 |
| Test episodes per epoch | 25 |
| Early stopping patience | 4 epochs |
| Episode length | 1,000 steps |

**Epsilon schedule (ε-greedy exploration):**
- Starts at `ε = 0.6`, decays linearly to `ε = 0.0` over ~6 epochs (~810,000 steps)
- After decay the agent acts greedily

---

## Environment Details

| Parameter | Value |
|---|---|
| Starting balance | $1,000 |
| Trading fee | 0.1% per trade |
| Episode start | Random position sampled from training data |
| Observation window | 24 × 1-hour bars (current + 23 previous) |
| Action on same asset | Hold (no trade, no fee) |
| Action on different asset | Sell all → buy new asset (fee applied) |
| Action "cash" | Sell all → hold USD (no fee on USD hold) |

---

## Experiment Tracking

Training metrics, test visualizations, and learning curves are logged to **Weights & Biases (W&B)**. Outputs per epoch include:
- Portfolio value evolution per test episode
- Actions chosen (buy/sell markers on price chart)
- Reward per timestep
- Boxplot distribution of episode-ending portfolio values vs. Buy-and-Hold

---

## Setup

All three notebooks are self-contained Google Colab / Kaggle notebooks.

1. Enable GPU: **Runtime → Change runtime type → T4 GPU**
2. Run cells in order
3. When prompted, upload your 4 HDF5 data files: `ADAUSDT_minutes.hdf5`, `BTCUSDT_minutes.hdf5`, `ETHUSDT_minutes.hdf5`, `LTCUSDT_minutes.hdf5`
4. Log in to Weights & Biases (W&B) when prompted

### Dependencies
Installed automatically by the first cell:

```
tianshou==0.4.11  gym==0.25.2  numpy<2.0  wandb  seaborn  matplotlib
pandas  tables  tqdm  ccxt  tensorboard  numba  h5py
```

PyTorch is pre-installed on Kaggle/Colab GPU runtimes and is not reinstalled (to preserve CUDA compatibility).
