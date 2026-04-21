# CAT — Candle Analysis Tool: BTC/USDT Price Direction Prediction

Supervised binary classification model that predicts whether the next 1-minute BTC/USDT candle will close **UP or DOWN**.

---

## Task Definition

- **Input:** 150 consecutive 1-minute candles, each described by 10 features
- **Target (y):** `1` if `close[t+1] > close[t]`, else `0` — a binary direction label
- **Output:** `P(UP)` — a sigmoid probability in [0, 1]
- **Loss:** `binary_crossentropy` (directly optimises direction accuracy)

---

## Data

| Property | Value |
|---|---|
| **Asset** | BTC/USDT |
| **Timeframe** | 1-minute bars |
| **Volume** | 51,773 candles ≈ 35 days |
| **Source** | Binance public S3 (`data.binance.vision/data/spot/monthly/klines`) — no API key required |
| **Format** | Monthly ZIP files, auto-downloaded and concatenated |
| **Date range** | Automatically uses the most recent ~35 days available |
| **Train / Val split** | 85% / 15% — chronological (no shuffle, no leakage) |

---

## Feature Engineering

10 features are computed per candle before building sliding windows:

| Feature | Formula | Purpose |
|---|---|---|
| `open`, `high`, `low`, `close` | Raw OHLC | Price structure |
| `volume` | Raw volume | Traded quantity |
| `ob_imbalance` | `(close − open) / (high − low)` clipped to [−1, 1] | Candle body as bid/ask pressure proxy; replaces live L2 order book |
| `rsi_14` | Standard RSI-14, scaled to [0, 1] | Momentum — overbought / oversold |
| `ema_diff` | `(EMA9 − EMA21) / close` | Trend direction |
| `vol_ratio` | `volume / 20-bar rolling mean`, clipped at 10, scaled to [0, 1] | Volume spike detector |
| `price_accel` | `Δclose[t] − Δclose[t−1]` / `close` | Second derivative of price (momentum acceleration) |

**Normalisation:** `MinMaxScaler(0, 1)` fitted **only on the training split** and applied to both splits — prevents data leakage.

---

## Tensor Construction

```
Sliding window: 150 candles of 10 features → predict candle 151
X shape: (N, 150, 10)   — float32
y shape: (N, 1)         — 0 or 1
```

Class balance is approximately 50/50 (UP vs DOWN), so no class weighting is needed.

---

## Model Architecture

Encoder-only Transformer (BERT-style) — reads the full 150-candle window via self-attention, then classifies direction.

```
Input (150 candles × 10 features)
  → Dense(512)                         — input projection to d_model
  → Sinusoidal Positional Encoding     — injects position into each token
  → 6 × TransformerBlock:
       MultiHeadAttention(8 heads, key_dim=64)
       → Dropout(0.2) → Add & LayerNorm
       → FFN: Dense(1024, relu) → Dense(512)
       → Dropout(0.2) → Add & LayerNorm
  → GlobalAveragePooling1D             — aggregate sequence to single vector
  → Dense(1024, relu) → Dropout(0.2)
  → Dense(512, relu)  → Dropout(0.2)
  → Dense(1, sigmoid)                  — P(UP)
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `num_heads` | 8 |
| `ff_dim` | 1024 |
| `n_blocks` | 6 |
| `dropout` | 0.2 |
| **Total parameters** | ~33M |

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-4`) |
| Loss | `binary_crossentropy` |
| Batch size | 64 |
| Max epochs | 80 |
| Early stopping | patience = 6 on `val_loss` |
| LR reduction | `ReduceLROnPlateau` — factor 0.5, patience 3, min_lr 1e-6 |
| Best model | Saved via `ModelCheckpoint` (monitors `val_loss`) |

---

## Signal Generation

The output probability drives a LONG / SHORT / HOLD signal via a configurable confidence threshold (default `CONFIDENCE_THRESHOLD = 0.15`):

| Condition | Signal |
|---|---|
| `prob > 0.65` | **LONG** |
| `prob < 0.35` | **SHORT** |
| Otherwise | **HOLD** (model not confident enough) |

Lowering the threshold → more signals, potentially lower accuracy.  
Raising the threshold → fewer signals, potentially higher accuracy.

---

## Outputs

After training the notebook saves to `saved_model/`:
- `cat_model_nb.keras` — best weights
- `scaler_nb.joblib` — fitted MinMaxScaler
- `training_curves.png` — loss & accuracy curves
- `eval_report.txt` — full evaluation report (accuracy, signal backtest, probability stats)

---

## Setup

The notebook is self-contained and runs on Google Colab or Kaggle.

1. Enable GPU: **Runtime → Change runtime type → T4 GPU**
2. Run cells in order — data is fetched automatically from Binance (no API key required)
3. No external data files needed

### Dependencies
Installed automatically by the first cell:

```
ccxt  tensorflow  scikit-learn  pandas  numpy  matplotlib  tqdm  joblib
```
