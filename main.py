"""
Fraud Detection API
- First run: trains Random Forest with tuned hyperparameters and saves to disk
- Subsequent runs: loads saved model instantly from disk
- Card profiles: maintains live behavioral history per card for real feature computation

Fixes applied:
- is_weekend / is_night_txn: derived automatically from trans_timestamp (not caller-supplied)
- income_proxy: derived from job title via a lookup table built from training data
- age: derived from dob + trans_timestamp exactly as in training (not caller-supplied)
"""

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import reverse_geocoder as rg

# ── App ───────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load or train model
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}...")
        saved = joblib.load(MODEL_PATH)
        models['rf']               = saved['model']
        models['feature_names']    = saved['feature_names']
        models['job_income_proxy'] = saved.get('job_income_proxy', {})
        print("✅ Model loaded from disk.")
        # If an old model file is missing the lookup, rebuild it now
        if not models['job_income_proxy'] and os.path.exists(TRAIN_PATH):
            print("Rebuilding job_income_proxy lookup (missing from saved model)...")
            models['job_income_proxy'] = _build_job_income_proxy(TRAIN_PATH)
            saved['job_income_proxy'] = models['job_income_proxy']
            joblib.dump(saved, MODEL_PATH)
            print(f"✅ Rebuilt and saved ({len(models['job_income_proxy'])} job titles).")
    else:
        print("No saved model found — training from scratch (this will take a while)...")
        X_train, y_train, _, _, _job_proxy = full_pipeline(TRAIN_PATH, TEST_PATH)
        models['feature_names']    = list(X_train.columns)
        models['rf']               = train_rf(X_train, y_train)
        models['job_income_proxy'] = _job_proxy
        joblib.dump({
            'model':            models['rf'],
            'feature_names':    models['feature_names'],
            'job_income_proxy': models['job_income_proxy'],
        }, MODEL_PATH)
        print(f"✅ Model trained and saved to {MODEL_PATH}.")

    # Bootstrap card profiles from training data if available
    if os.path.exists(TRAIN_PATH):
        print("Bootstrapping card profiles from training data...")
        _bootstrap_card_profiles(TRAIN_PATH)
        print(f"✅ Card profiles loaded for {len(card_profiles)} cards.")

    yield
    models.clear()
    card_profiles.clear()
    print("🛑 Models and profiles cleared.")

app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global stores ─────────────────────────────────────────────────────────────
models: dict = {}
# card_profiles[cc_num] = list of dicts, each dict is one past transaction
# Each entry: {"ts": datetime, "amt": float, "merchant": str, "state": str}
card_profiles: dict = defaultdict(list)

TRAIN_PATH  = os.getenv("FRAUD_TRAIN_CSV",  "fraudTrain.csv")
TEST_PATH   = os.getenv("FRAUD_TEST_CSV",   "fraudTest.csv")
MODEL_PATH  = os.getenv("FRAUD_MODEL_PATH", "fraud_rf_model.joblib")
THRESHOLD_BLOCK  = 0.9325          # auto block
THRESHOLD_REVIEW = 2/3             # flag for manual review (below this = auto approve)

# Global fallback income_proxy — median of all training transaction amounts
# Used only when a job title is not found in the lookup table
INCOME_PROXY_FALLBACK = 50.0


def _build_job_income_proxy(train_path: str) -> dict:
    """Build a job title → median transaction amount lookup from training data.
    This is the same computation as process_job_features() in the notebook,
    but keyed on the raw job string rather than an arbitrary category code so
    it can be used at inference time without re-encoding."""
    df = pd.read_csv(train_path, usecols=['job', 'amt'])
    lookup = df.groupby('job')['amt'].median().to_dict()
    print(f"  Built income_proxy lookup for {len(lookup)} job titles.")
    return lookup

# ── Card profile helpers ───────────────────────────────────────────────────────

def _bootstrap_card_profiles(train_path: str):
    """Load training data and build per-card history so deployed predictions
    have the same behavioral context the model was trained on."""
    df = pd.read_csv(train_path, usecols=['cc_num', 'trans_date_trans_time', 'amt', 'merchant', 'state'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], utc=True, format='mixed')
    df = df.sort_values('trans_date_trans_time')
    for _, row in df.iterrows():
        card_profiles[str(row['cc_num'])].append({
            "ts":       row['trans_date_trans_time'].to_pydatetime(),
            "amt":      float(row['amt']),
            "merchant": str(row['merchant']),
            "state":    str(row['state']),
        })


def _compute_behavioral_features(cc_num: str, now: datetime, current_amt: float) -> dict:
    """Compute all rolling behavioral features for a card at prediction time,
    using the card's stored history. This mirrors exactly what the notebook did
    over the full dataset — but now done live per transaction."""
    history = card_profiles.get(str(cc_num), [])

    if not history:
        # Brand new card — same defaults as a first transaction in training
        return {
            "is_first_txn":              1,
            "time_since_last_card_txn":  -1,
            "card_txn_count_last_1h":    0,
            "card_txn_count_last_24h":   0,
            "card_txn_count_last_7d":    0,
            "card_total_spend_last_24h": 0,
            "card_total_spend_last_7d":  0,
            "card_avg_amt":              0,
            "num_unique_merchants_card": 0,
        }

    # Sort history just in case
    history_sorted = sorted(history, key=lambda x: x["ts"])
    last_txn = history_sorted[-1]

    # Time since last transaction (seconds)
    time_since_last = (now - last_txn["ts"]).total_seconds()

    # Rolling windows — only look at PAST transactions (exclude current)
    one_hour_ago  = now - pd.Timedelta("1h")
    one_day_ago   = now - pd.Timedelta("24h")
    seven_days_ago = now - pd.Timedelta("7D")

    txns_1h  = [t for t in history_sorted if t["ts"] >= one_hour_ago]
    txns_24h = [t for t in history_sorted if t["ts"] >= one_day_ago]
    txns_7d  = [t for t in history_sorted if t["ts"] >= seven_days_ago]

    count_1h  = len(txns_1h)
    count_24h = len(txns_24h)
    count_7d  = len(txns_7d)

    spend_24h = sum(t["amt"] for t in txns_24h)
    spend_7d  = sum(t["amt"] for t in txns_7d)

    # Cumulative average amount across ALL past transactions
    all_amts = [t["amt"] for t in history_sorted]
    avg_amt  = float(np.mean(all_amts)) if all_amts else 0.0

    # Unique merchants seen historically
    unique_merchants = len(set(t["merchant"] for t in history_sorted))

    return {
        "is_first_txn":              0,
        "time_since_last_card_txn":  time_since_last,
        "card_txn_count_last_1h":    count_1h,
        "card_txn_count_last_24h":   count_24h,
        "card_txn_count_last_7d":    count_7d,
        "card_total_spend_last_24h": spend_24h,
        "card_total_spend_last_7d":  spend_7d,
        "card_avg_amt":              avg_amt,
        "num_unique_merchants_card": unique_merchants,
    }


def _update_card_profile(cc_num: str, ts: datetime, amt: float, merchant: str, state: str):
    """After prediction, add this transaction to the card's history so future
    predictions have up-to-date context. Keeps only last 90 days to cap memory."""
    cutoff = ts - pd.Timedelta("90D")
    profile = card_profiles[str(cc_num)]
    # Trim old entries
    card_profiles[str(cc_num)] = [t for t in profile if t["ts"] >= cutoff]
    card_profiles[str(cc_num)].append({
        "ts": ts, "amt": amt, "merchant": merchant, "state": state
    })


# ── Full training pipeline (mirrors notebook) ─────────────────────────────────

def add_card_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], utc=True, format='mixed')
    df["dob"] = pd.to_datetime(df["dob"])
    df['merchant_code'] = df['merchant'].astype('category').cat.codes
    df['state_code']    = df['state'].astype('category').cat.codes
    df = df.sort_values(['cc_num', 'trans_date_trans_time'])
    df['is_first_txn'] = df.groupby('cc_num').cumcount().eq(0).astype(int)
    df['time_since_last_card_txn'] = (
        df.groupby('cc_num')['trans_date_trans_time'].diff()
          .dt.total_seconds().fillna(-1))
    df = df.set_index('trans_date_trans_time')
    amt_shifted = df.groupby('cc_num')['amt'].shift(1)
    for window, col in [('1h', 'card_txn_count_last_1h'), ('24h', 'card_txn_count_last_24h'), ('7D', 'card_txn_count_last_7d')]:
        df[col] = (amt_shifted.groupby(df['cc_num']).rolling(window).count()
                   .reset_index(level=0, drop=True).fillna(0))
    for window, col in [('24h', 'card_total_spend_last_24h'), ('7D', 'card_total_spend_last_7d')]:
        df[col] = (amt_shifted.groupby(df['cc_num']).rolling(window).sum()
                   .reset_index(level=0, drop=True).fillna(0))
    df['card_avg_amt'] = (amt_shifted.groupby(df['cc_num']).expanding().mean()
                          .reset_index(level=0, drop=True).fillna(0))
    df['num_unique_merchants_card'] = (
        df.groupby('cc_num')['merchant_code'].expanding()
          .apply(lambda x: x.nunique(), raw=False)
          .reset_index(level=0, drop=True).fillna(0))
    df['num_unique_states_card'] = (
        df.groupby('cc_num')['state_code'].expanding()
          .apply(lambda x: x.nunique(), raw=False)
          .reset_index(level=0, drop=True).fillna(0))
    df = df.reset_index()
    df.drop(columns=['merchant_code', 'state_code'], inplace=True)
    return df


def add_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_weekend']  = (df['trans_date_trans_time'].dt.weekday >= 5).astype(int)
    hour = df['trans_date_trans_time'].dt.hour
    df['is_night_txn'] = ((hour >= 22) | (hour < 6)).astype(int)
    return df


def process_job_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    train, test = train.copy(), test.copy()
    train['_job_code'] = train['job'].astype('category').cat.codes
    test['_job_code']  = test['job'].astype('category').cat.codes

    # Build lookup: raw job title → income_proxy value
    job_to_code   = dict(zip(train['job'], train['_job_code']))
    job_income_proxy = train.groupby('_job_code')['amt'].median()
    # Map back to job title strings for use at inference time
    job_title_proxy  = {job: float(job_income_proxy[code])
                        for job, code in job_to_code.items()}

    train['income_proxy'] = train['_job_code'].map(job_income_proxy)
    test['income_proxy']  = test['_job_code'].map(job_income_proxy)
    job_target_mean = train.groupby('_job_code')['is_fraud'].mean()
    train['job_target_enc'] = train['_job_code'].map(job_target_mean)
    test['job_target_enc']  = test['_job_code'].map(job_target_mean).fillna(train['is_fraud'].mean())
    train.drop(columns=['_job_code'], inplace=True)
    test.drop(columns=['_job_code'], inplace=True)
    return train, test, job_title_proxy


def compute_exact_age(df: pd.DataFrame) -> pd.DataFrame:
    t = df['trans_date_trans_time'].dt.tz_localize(None)
    b = df['dob']
    df['age'] = (t.dt.year - b.dt.year
                 - ((t.dt.month < b.dt.month)
                    | ((t.dt.month == b.dt.month) & (t.dt.day < b.dt.day))))
    return df


def add_city_from_coords(df: pd.DataFrame) -> pd.DataFrame:
    unique_coords = df[['merch_lat', 'merch_long']].drop_duplicates()
    coords_list   = list(zip(unique_coords['merch_lat'], unique_coords['merch_long']))
    results       = rg.search(coords_list)
    geo_df = pd.DataFrame({
        'merch_lat':  unique_coords['merch_lat'].values,
        'merch_long': unique_coords['merch_long'].values,
        'merch_city': [r['name'] for r in results]})
    return df.merge(geo_df, on=['merch_lat', 'merch_long'], how='left')


def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def move_col_to_end(df: pd.DataFrame, col: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != col] + [col]
    return df[cols]


def drop_string(df: pd.DataFrame, col: str, s: str) -> pd.DataFrame:
    df[col] = df[col].str.replace(s, "")
    return df


COLS_TO_DROP_FIRST  = ['first','last','unix_time','dob','gender','city','street','long','lat','trans_num','job']
COLS_TO_DROP_SECOND = ['job_target_enc','merch_lat','merch_long','city_pop','state','cc_num','zip','merchant',
                       'num_unique_states_card','merch_city','category','trans_date_trans_time']


def full_pipeline(train_path: str, test_path: str):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
    drop_string(train, 'merchant', 'fraud_')
    drop_string(test,  'merchant', 'fraud_')
    print("Engineering features...")
    train = add_card_behavioral_features(train)
    test  = add_card_behavioral_features(test)
    train = add_time_flags(train)
    test  = add_time_flags(test)
    train, test, job_income_proxy = process_job_features(train, test)
    train = move_col_to_end(train, 'is_fraud')
    test  = move_col_to_end(test,  'is_fraud')
    train = compute_exact_age(train)
    test  = compute_exact_age(test)
    train = add_city_from_coords(train)
    test  = add_city_from_coords(test)
    train['miles_apart'] = haversine_vectorized(train['lat'], train['long'], train['merch_lat'], train['merch_long'])
    test['miles_apart']  = haversine_vectorized(test['lat'],  test['long'],  test['merch_lat'],  test['merch_long'])
    train = move_col_to_end(train, 'is_fraud')
    test  = move_col_to_end(test,  'is_fraud')
    for df in [train, test]:
        if 'merch_city_x' in df.columns:
            df.rename(columns={'merch_city_x': 'merch_city'}, inplace=True)
    for col in COLS_TO_DROP_FIRST:
        if col in train.columns: train.drop(columns=[col], inplace=True)
        if col in test.columns:  test.drop(columns=[col], inplace=True)
    for col in COLS_TO_DROP_SECOND:
        if col in train.columns: train.drop(columns=[col], inplace=True)
        if col in test.columns:  test.drop(columns=[col], inplace=True)
    X_train = train.drop(columns=['is_fraud'])
    y_train = train['is_fraud']
    X_test  = test.drop(columns=['is_fraud'])
    y_test  = test['is_fraud']
    return X_train, y_train, X_test, y_test, job_income_proxy


def train_rf(X_train, y_train):
    print("Training Random Forest with tuned hyperparameters...")
    rf = RandomForestClassifier(
        n_estimators=400,
        criterion='gini',
        max_depth=16,
        min_samples_leaf=200,
        min_samples_split=100,
        max_features=0.5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf


# ── Request / Response schemas ────────────────────────────────────────────────

class Transaction(BaseModel):
    # Card identifier — used to look up and update the card's behavioral profile
    cc_num: str

    # Raw transaction fields used to compute static features
    amt:         float
    merchant:    str   = "unknown"
    state:       str   = "unknown"
    merch_lat:   float = 0.0
    merch_long:  float = 0.0
    lat:         float = 0.0
    long:        float = 0.0

    # FIX: derived automatically from trans_timestamp (was caller-supplied, defaulted to 0)
    # is_weekend and is_night_txn are now computed inside /predict

    # income_proxy_override: supply a direct dollar amount to bypass the job lookup
    income_proxy_override: Optional[float] = None

    # job title used to look up income_proxy from training data lookup table
    job: Optional[str] = None

    # FIX: dob used to compute exact age at transaction time, matching training pipeline exactly
    # (was an int defaulting to 35 for everyone)
    dob: Optional[str] = None  # ISO format: "YYYY-MM-DD"

    # Optional: caller can supply a specific timestamp; defaults to now
    trans_timestamp: Optional[str] = None


class PredictionResponse(BaseModel):
    model:             str
    is_fraud:          bool
    review_required:   bool
    risk_tier:         str   # "AUTO_APPROVE" | "REVIEW" | "AUTO_BLOCK"
    fraud_probability: float
    decision:          str
    financial_impact:  dict
    card_context:      dict   # shows what behavioral features were computed


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":        "running",
        "model_loaded":  "rf" in models,
        "cards_profiled": len(card_profiles),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "random_forest", "cards_profiled": len(card_profiles)}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if 'rf' not in models:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    model    = models['rf']
    features = models['feature_names']

    # ── Resolve transaction timestamp ─────────────────────────────────────────
    if transaction.trans_timestamp:
        now = pd.to_datetime(transaction.trans_timestamp, utc=True).to_pydatetime()
    else:
        now = datetime.now(timezone.utc)

    # ── FIX: derive is_weekend and is_night_txn from timestamp (not caller-supplied) ──
    is_weekend  = int(now.weekday() >= 5)
    hour        = now.hour
    is_night_txn = int(hour >= 22 or hour < 6)

    # ── derive income_proxy: direct override > job lookup > global median ───────
    job_income_proxy = models.get('job_income_proxy', {})
    if transaction.income_proxy_override is not None:
        income_proxy = transaction.income_proxy_override
    elif transaction.job and transaction.job in job_income_proxy:
        income_proxy = job_income_proxy[transaction.job]
    else:
        # Fallback: global median across all jobs in training data
        income_proxy = float(np.median(list(job_income_proxy.values()))) if job_income_proxy else 50.0

    # ── FIX: compute exact age from dob + transaction timestamp ───────────────
    if transaction.dob:
        dob = pd.to_datetime(transaction.dob)
        t   = pd.Timestamp(now).tz_localize(None) if now.tzinfo else pd.Timestamp(now)
        t   = t.tz_localize(None)
        age = int(t.year - dob.year - ((t.month < dob.month) or
              (t.month == dob.month and t.day < dob.day)))
    else:
        # Fallback: median age from training data (~35 is reasonable but flagged)
        age = 35

    # ── Compute behavioral features from card's live profile ──────────────────
    behavioral = _compute_behavioral_features(transaction.cc_num, now, transaction.amt)

    # ── Compute static / derived features ────────────────────────────────────
    miles_apart = float(haversine_vectorized(
        transaction.lat, transaction.long,
        transaction.merch_lat, transaction.merch_long
    )) if (transaction.lat or transaction.merch_lat) else 5.0

    # Assemble full feature row matching training column order
    row = {
        "amt":                        transaction.amt,
        "is_first_txn":               behavioral["is_first_txn"],
        "time_since_last_card_txn":   behavioral["time_since_last_card_txn"],
        "card_txn_count_last_1h":     behavioral["card_txn_count_last_1h"],
        "card_txn_count_last_24h":    behavioral["card_txn_count_last_24h"],
        "card_txn_count_last_7d":     behavioral["card_txn_count_last_7d"],
        "card_total_spend_last_24h":  behavioral["card_total_spend_last_24h"],
        "card_total_spend_last_7d":   behavioral["card_total_spend_last_7d"],
        "card_avg_amt":               behavioral["card_avg_amt"],
        "num_unique_merchants_card":  behavioral["num_unique_merchants_card"],
        "is_weekend":                 is_weekend,
        "is_night_txn":               is_night_txn,
        "income_proxy":               income_proxy,
        "age":                        age,
        "miles_apart":                miles_apart,
    }

    # Align to exact feature order model was trained on
    X = pd.DataFrame([{f: row.get(f, 0) for f in features}])

    proba    = float(model.predict_proba(X)[:, 1][0])

    # ── Tiered decision logic ─────────────────────────────────────────────────
    if proba >= THRESHOLD_BLOCK:
        risk_tier       = "AUTO_BLOCK"
        is_fraud        = True
        review_required = False
    elif proba >= THRESHOLD_REVIEW:
        risk_tier       = "REVIEW"
        is_fraud        = False   # not hard-blocked, but needs human review
        review_required = True
    else:
        risk_tier       = "AUTO_APPROVE"
        is_fraud        = False
        review_required = False

    # ── Update card profile AFTER prediction (not before) ────────────────────
    _update_card_profile(
        cc_num=transaction.cc_num,
        ts=now,
        amt=transaction.amt,
        merchant=transaction.merchant,
        state=transaction.state,
    )

    fi = {
        "transaction_amount":  round(transaction.amt, 2),
        "fraud_caught_saved":  round(transaction.amt, 2) if is_fraud else 0.0,
        "legitimate_blocked":  round(transaction.amt, 2) if (is_fraud and proba < 0.99) else 0.0,
        "fraud_missed_lost":   round(transaction.amt, 2) if (not is_fraud and not review_required and proba > THRESHOLD_REVIEW) else 0.0,
    }

    # Surface the behavioral context so the caller can see what drove the decision
    card_context = {
        "card_history_size":          len(card_profiles.get(str(transaction.cc_num), [])),
        "card_avg_amt":               round(behavioral["card_avg_amt"], 2),
        "amt_vs_avg_ratio":           round(transaction.amt / behavioral["card_avg_amt"], 2) if behavioral["card_avg_amt"] > 0 else None,
        "txns_last_1h":               behavioral["card_txn_count_last_1h"],
        "txns_last_24h":              behavioral["card_txn_count_last_24h"],
        "spend_last_24h":             round(behavioral["card_total_spend_last_24h"], 2),
        "time_since_last_txn_secs":   behavioral["time_since_last_card_txn"],
        "unique_merchants_seen":      behavioral["num_unique_merchants_card"],
        "miles_from_home":            round(miles_apart, 1),
        # FIX: show derived values so callers can verify correctness
        "derived_is_weekend":         is_weekend,
        "derived_is_night_txn":       is_night_txn,
        "derived_income_proxy":       round(income_proxy, 2),
        "derived_age":                age,
        "job_lookup_hit":             bool(transaction.job and transaction.job in job_income_proxy),
        "dob_provided":               transaction.dob is not None,
    }

    decision_str = {
        "AUTO_BLOCK":   "🚨 FRAUD DETECTED — Auto Blocked",
        "REVIEW":       "⚠️ SUSPICIOUS — Flagged for Review",
        "AUTO_APPROVE": "✅ LEGITIMATE — Auto Approved",
    }[risk_tier]

    return PredictionResponse(
        model="Random Forest (tuned)",
        is_fraud=is_fraud,
        review_required=review_required,
        risk_tier=risk_tier,
        fraud_probability=round(proba, 4),
        decision=decision_str,
        financial_impact=fi,
        card_context=card_context,
    )


@app.get("/card/{cc_num}/profile")
def get_card_profile(cc_num: str):
    """Inspect the stored history for a given card."""
    history = card_profiles.get(cc_num, [])
    if not history:
        return {"cc_num": cc_num, "message": "No history found.", "txn_count": 0}
    amts = [t["amt"] for t in history]
    return {
        "cc_num":          cc_num,
        "txn_count":       len(history),
        "avg_amt":         round(float(np.mean(amts)), 2),
        "total_spend":     round(float(np.sum(amts)), 2),
        "first_seen":      min(t["ts"] for t in history).isoformat(),
        "last_seen":       max(t["ts"] for t in history).isoformat(),
        "unique_merchants": len(set(t["merchant"] for t in history)),
    }


@app.delete("/card/{cc_num}/profile")
def reset_card_profile(cc_num: str):
    """Clear a card's history (useful for testing)."""
    if cc_num in card_profiles:
        del card_profiles[cc_num]
    return {"cc_num": cc_num, "message": "Profile cleared."}


@app.get("/model/info")
def model_info():
    if 'rf' not in models:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    m = models['rf']
    return {
        "type":      type(m).__name__,
        "params":    m.get_params(),
        "threshold": THRESHOLD,
        "features":  models['feature_names'],
    }
