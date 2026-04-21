# Normalization Experiments — Automated Crypto Trading with Deep RL

This folder contains three variants of the same Deep RL trading agent, each using a different normalization strategy for the price observations fed to the network. Everything else is identical across all three: **Double Dueling DQN with Prioritized Experience Replay**, LSTM encoder, 4 crypto assets (ADA, BTC, ETH, LTC), 1-hour bars, 1000-step episodes.

The goal is to compare how the choice of normalization affects training stability and final performance.

---

## Notebooks

### `colab_option1_episode_norm.ipynb` — Episode-Start Price Normalization
**How it works:** OHLC prices are divided by the open price at the start of each episode.

- Output values are ratios relative to a fixed anchor (episode start), so the network always sees numbers near 1.0.
- Scale-invariant across different price levels (e.g., BTC at $20k vs $60k).
- The anchor resets every episode, which means the agent cannot infer the absolute price level, only relative movements within the episode.

---

### `colab_option2_log_returns.ipynb` — Log Returns Normalization
**How it works:** Each price value is replaced with `log(P_t / P_{t-1})`, the log return from the previous timestep.

- Produces a stationary, mean-reverting signal — much better behaved for neural networks than raw prices.
- Fully scale-invariant and removes any dependency on absolute price levels.
- Log returns are a standard representation in quantitative finance; small values near zero indicate stability, large positive/negative values indicate strong moves.
- Requires fetching one extra historical row per observation window to compute the return at the first timestep.

---

### `colab_option3_rolling_zscore.ipynb` — Rolling Z-Score Normalization
**How it works:** Each price column is z-scored over the current 24-bar observation window: `(x - mean) / std`.

- Adapts dynamically to the local price regime within the observation window.
- Robust to regime changes (bull/bear markets, high/low volatility periods).
- Values are always zero-centered and unit-variance within each window, regardless of the absolute price level.
- If the window standard deviation is near zero (flat market), values are set to zero to avoid division instability.

---

## Setup

All three notebooks are self-contained Google Colab / Kaggle notebooks. To run any of them:

1. Enable GPU: **Runtime → Change runtime type → T4 GPU**
2. Run cells in order
3. When prompted, upload your 4 HDF5 data files: `ADAUSDT_minutes.hdf5`, `BTCUSDT_minutes.hdf5`, `ETHUSDT_minutes.hdf5`, `LTCUSDT_minutes.hdf5`
4. Log in to Weights & Biases (W&B) when prompted — training metrics and test visualizations are logged there

## Dependencies
Installed automatically by the first cell: `tianshou==0.4.11`, `gym==0.25.2`, `numpy<2.0`, `wandb`, `pytorch` (pre-installed on Kaggle/Colab GPU runtimes).
