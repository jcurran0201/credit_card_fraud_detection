# Credit Card Fraud Detection

A machine learning system to detect fraudulent credit card transactions using the [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — a synthetic dataset simulating ~1,000 cardholders with realistic, heavily imbalanced transaction data.

---

## Features Engineered

- **Behavioral:** Transaction frequency and spend totals over the past hour, day, and week
- **Temporal:** Weekend and off-peak (10 PM–6 AM) transaction flags
- **Demographic:** Cardholder age, job-based spending proxy
- **Spatial:** Haversine distance between cardholder residence and merchant location

---

## Models Trained

| Model | Precision | Recall | F1 | PR AUC |
|---|---|---|---|---|
| Decision Tree (baseline) | 0.27 | 0.83 | — | 0.69 |
| Random Forest | 0.82 | 0.64 | 0.72 | 0.76 |
| Random Forest + Tuning | 0.73 | 0.69 | 0.71 | 0.78 |
| XGBoost | 0.65 | 0.78 | 0.71 | 0.80 |
| XGBoost + Tuning | 0.66 | 0.77 | 0.71 | 0.80 |

Class imbalance was handled via class weighting and threshold tuning rather than SMOTE, to preserve natural transaction patterns.

---

## Key Findings

- Fraud transactions were mostly under $500; high-value fraud was rare
- Top fraudulent merchants: **Kozey-Boehm, Kuhic LLC, Terry-Huel, Boyer PLC**
- Most important features across models: transaction amount, 24-hour spend total, off-peak flag, weekly card activity
- **Tuned Random Forest** recovered the most fraud value (~$1M) with fewest missed fraud cases (~$131K), at the cost of more false positives (~$384K)
- **Tuned XGBoost** offered a better precision-recall balance with fewer customer disruptions

---

## FraudSentinel — Deployment

A FastAPI backend + HTML frontend for real-time fraud detection, serving both tuned models via REST API.

### Setup

**1. Place your data files** — put `fraudTrain.csv` and `fraudTest.csv` in the same directory as `main.py`.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the API**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
> ⚠️ On first startup, both models are trained. This takes a few minutes.

**4. Open the frontend** — open `index.html` in your browser. It connects to `http://localhost:8000`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + loaded models |
| GET | `/health` | Simple status |
| POST | `/predict/xgb` | XGBoost prediction |
| POST | `/predict/rf` | Random Forest prediction |
| GET | `/models/info` | Model params + thresholds |

### Model Thresholds

| Model | Tuning | Threshold |
|-------|--------|-----------|
| XGBoost | RandomizedSearchCV (25 iter, 3-fold) | 0.94 |
| Random Forest | RandomizedSearchCV (25 iter, 3-fold) | 0.9325 |

Both models optimize for **recall** on the fraud class.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_TRAIN_CSV` | `fraudTrain.csv` | Path to training CSV |
| `FRAUD_TEST_CSV` | `fraudTest.csv` | Path to test CSV |

---

## Future Improvements

- Systematic probability threshold search (0.9–1.0 range) instead of manual tuning

