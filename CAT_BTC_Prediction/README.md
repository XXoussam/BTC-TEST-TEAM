# CAT — Candle Analysis Tool: BTC/USDT Price Direction Prediction

This folder contains a **completely different approach** to cryptocurrency prediction compared to the normalization experiments. Instead of a reinforcement learning trading agent, this is a **supervised learning model** that predicts the direction of the next 1-minute BTC/USDT candle.

---

## What This Notebook Does

`CAT_BTC_Prediction.ipynb` trains a **Transformer Encoder** (encoder-only, similar to BERT — not a GPT/decoder) to classify whether the next 1-minute candle will close **UP or DOWN** relative to the current candle.

### Pipeline
1. **Data collection** — pulls 51,773 × 1-minute OHLCV candles directly from Binance public S3 (~35 days)
2. **Feature engineering** — constructs 10 input features per candle:
   - `open`, `high`, `low`, `close`, `volume` — raw OHLCV
   - `ob_imbalance` — Order Book Imbalance proxy: `(close - open) / (high - low)`, captures bid/ask pressure from candle shape
   - `rsi_14` — RSI momentum indicator (scaled 0→1)
   - `ema_diff` — (EMA9 − EMA21) / close, trend direction signal
   - `vol_ratio` — volume / 20-bar rolling mean, volume spike detector
   - `price_accel` — second derivative of close price, momentum acceleration
3. **Tensor construction** — sliding windows of 150 candles → predict direction of candle 151
4. **Training** — Transformer Encoder (~33M parameters) with binary cross-entropy loss
5. **Evaluation** — direction accuracy, confidence-filtered signal backtest, probability distribution analysis

### Model Architecture
```
Input (150 candles × 10 features)
  → Dense projection → d_model (512)
  → Sinusoidal Positional Encoding
  → 6 × [MultiHeadAttention (8 heads) → LayerNorm → FFN → LayerNorm]
  → GlobalAveragePooling1D
  → Dense(1024, relu) → Dense(512, relu)
  → Dense(1, sigmoid)   ← outputs P(UP)
```

### Output & Signal Logic
- Output is a probability between 0 and 1: **P(next candle closes higher)**
- `prob > 0.65` → **LONG signal**
- `prob < 0.35` → **SHORT signal**
- Otherwise → **HOLD** (model is not confident enough)
- The confidence threshold (default `0.15`) is configurable — lower = more signals, higher = fewer but more selective signals

---

## Key Differences vs the Normalization Experiments

| | Normalization Experiments | CAT BTC Prediction |
|---|---|---|
| **Approach** | Reinforcement Learning | Supervised Learning |
| **Framework** | PyTorch + Tianshou | TensorFlow / Keras |
| **Task** | Portfolio management (buy/hold/sell) | Binary direction classification |
| **Data** | 4 crypto assets, 1-hour bars, HDF5 files | BTC only, 1-minute bars, fetched from Binance |
| **Output** | Trading action (which asset to hold) | P(UP) probability + trading signal |
| **Model** | Double Dueling DQN + LSTM encoder | Transformer Encoder (BERT-style) |

---

## Setup

The notebook is self-contained and runs on Google Colab or Kaggle.

1. Enable GPU: **Runtime → Change runtime type → T4 GPU**
2. Run cells in order — data is fetched automatically from Binance (no API key required)
3. No external data files needed

## Dependencies
Installed automatically by the first cell: `ccxt`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `joblib`.
