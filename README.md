# Credit Card Fraud Detection

A machine learning system to detect fraudulent credit card transactions using the [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) a synthetic dataset simulating ~1,000 cardholders with realistic, heavily imbalanced transaction data. This project implements a production-style fraud detection system with:

   - **Real-time transaction scoring**
    
   - **Behavioral cardholder profiling**
     
   - **FastAPI that deploys the model**
     
   - **Custom frontend dashboard**
     
   - **3-tier fraud decision engine (approve/review/block)**

---

## Feature Engineering

| Category | Features |
|----------|----------|
| **Behavioral** | Transaction frequency and spend totals over the past hour, day, and week |
| **Temporal** | Weekend and off-peak (10 PM–6 AM) transaction flags |
| **Demographic** | Cardholder age, job-based spending proxy |
| **Spatial** | Haversine distance between cardholder residence and merchant location |

---

## Model Comparison

| Model | Precision | Recall | F1 | PR AUC |
|-------|-----------|--------|----|--------|
| Decision Tree (baseline) | 0.27 | 0.83 | — | 0.69 |
| Decision Tree (tuned) | 0.22 | 0.84 | — | 0.64 |
| Random Forest | 0.82 | 0.64 | 0.72 | 0.76 |
| **Random Forest + Tuning ✅** | **0.73** | **0.69** | **0.68** | **0.78** |
| XGBoost | 0.65 | 0.78 | 0.71 | 0.80 |
| XGBoost + Tuning | 0.66 | 0.77 | 0.71 | 0.80 |

Class imbalance was handled via class weighting and threshold tuning rather than SMOTE in order to preserve natural transaction patterns.

---

## Financial Impact Comparison 
| Metric | RF Untuned | RF Tuned | XGB Untuned | XGB Tuned |
|--------|-----------|----------|-------------|-----------|
| ✅ Fraud Caught (TP) | $882,333 | $1,002,299 | $996,279 | $968,317 |
| ❌ Fraud Missed (FN) | $250,991 | $131,025 | $137,046 | $165,008 |
| ⚠️ Legit Blocked (FP) | $108,884 | $384,410 | $445,262 | $356,053 |
| ✔️ Legit Passed (TN) | $37,320,695 | $37,045,169 | $36,984,316 | $37,073,525 |
| **Net Fraud Recovery** | **$631,342** | **$617,890** | **$551,017** | **$612,264** |

---

## Key Findings from testing phase

- Fraud transactions were mostly **under $500**; high-value fraud was uncommon
- Top fraudulent merchants: **Kozey-Boehm, Kuhic LLC, Terry-Huel, Boyer PLC**
- Most important features: transaction amount, 24-hour spend total, off-peak flag, weekly card activity
- **Tuned Random Forest** recovered the most fraud value (1M) with the fewest missed cases ($131K lost), at the cost of more false positives (~$384K blocked)
- **Tuned XGBoost** offered a better precision-recall balance with fewer customer disruptions. Tuned XGBoost did not recover as much money as the Tuned Random Forest did

---

## Deployment

A **FastAPI backend & HTML frontend** for real-time fraud scoring, serving the tuned Random Forest model via REST API.

### Setup

**1. Place your data files** — put `fraudTrain.csv` and `fraudTest.csv` in the same directory as `main.py`.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the API**
```bash
uvicorn Deployment_code:app --reload --host 0.0.0.0 --port 8000
```

**4. Open the frontend** — open `index.html` in your browser. It connects to `http://localhost:8000` by default.

---

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check + loaded model status |
| `GET` | `/health` | Simple status ping |
| `POST` | `/predict` | Submit a transaction for fraud scoring |
| `GET` | `/model/info` | Model params + decision thresholds |
| `GET` | `/card/{cc_num}/profile` | View behavioral history for a card |
| `DELETE` | `/card/{cc_num}/profile` | Clear a card's stored history |

### Decision Thresholds

| Risk Tier | Probability | Action |
|-----------|-------------|--------|
| AUTO_APPROVE | < 0.667 | Transaction approved |
| REVIEW | 0.667 – 0.9325 | Flagged for manual review |
| AUTO_BLOCK | ≥ 0.9325 | Transaction blocked |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_TRAIN_CSV` | `fraudTrain.csv` | Path to training CSV |
| `FRAUD_TEST_CSV` | `fraudTest.csv` | Path to test CSV |
| `FRAUD_MODEL_PATH` | `fraud_rf_model.joblib` | Path to save/load trained model |

---

## Future Improvements

- Systematic probability threshold search (0.9–1.0 range) instead of manual tuning
- Persist card behavioral profiles across server restarts 
