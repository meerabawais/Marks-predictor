#Streamlit dashboard
import os
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Load data
DATA_CSV = "clean_marks.csv"
DATA_XLS = "clean_marks.xls"
DATA_XLSX = "clean_marks.xlsx"

def load_data():
    if os.path.exists(DATA_CSV):
        return pd.read_csv(DATA_CSV)
    if os.path.exists(DATA_XLS):
        return pd.read_excel(DATA_XLS)
    if os.path.exists(DATA_XLSX):
        return pd.read_excel(DATA_XLSX)
    return None

df = load_data()
if df is None:
    st.error("Data file not found. Please upload clean_marks.csv (or .xls/.xlsx) in the repo next to app.py and redeploy.")
    st.stop()

# ---------- Page UI ----------
st.title("Student's Marks predictor")
st.sidebar.header("Choose RQ & settings")
rq = st.sidebar.selectbox("Research Question", ["RQ1: S-I", "RQ2: S-II", "RQ3: Final"])
n_boot = st.sidebar.slider("Bootstrap samples (train only)", min_value=100, max_value=500, value=200, step=50)

# ---------- Helper functions ----------
def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': _rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def make_pipe(model):
    return make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), model)

def bootstrap_mae(model_factory, X_tr, y_tr, n_boot):
    rng = np.random.RandomState(0)
    X_tr = X_tr.reset_index(drop=True); y_tr = y_tr.reset_index(drop=True)
    n = len(X_tr)
    maes = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        Xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]
        model = model_factory()
        model.fit(Xb, yb)
        oob_mask = np.ones(n, dtype=bool); oob_mask[np.unique(idx)] = False
        if oob_mask.sum() == 0:
            y_oob_true = y_tr; y_oob_pred = model.predict(X_tr)
        else:
            y_oob_true = y_tr.iloc[oob_mask]
            y_oob_pred = model.predict(X_tr.iloc[oob_mask])
        maes.append(mean_absolute_error(y_oob_true, y_oob_pred))
    return np.array(maes)

# ---------- Normalize and inspect columns ----------
# Trim whitespace from column names
orig_cols = list(df.columns)
df.columns = df.columns.astype(str).str.strip()

# Show available columns (useful for debugging; you can remove later)
st.write("Available columns (from uploaded file):", orig_cols)

# utility to "normalize" strings for matching
def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

# try to find column by candidate list
def find_col(candidates):
    norm_to_col = {norm(c): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        nc = norm(cand)
        if nc in norm_to_col:
            return norm_to_col[nc]
    # substring fallback
    for cand in candidates:
        for col in df.columns:
            if norm(cand) in norm(col):
                return col
    return None

# compute assign_sum / quiz_sum if missing by aggregating columns that start with As / Qz
as_cols = [c for c in df.columns if c.strip().lower().startswith("as")]
qz_cols = [c for c in df.columns if c.strip().lower().startswith("qz")]

if 'assign_sum' not in df.columns:
    if len(as_cols) > 0:
        df['assign_sum'] = df[as_cols].sum(axis=1, skipna=True)
    else:
        # try other heuristics (columns that contain 'assign' or 'assignment')
        alt_assign = [c for c in df.columns if 'assign' in c.lower()]
        if alt_assign:
            df['assign_sum'] = df[alt_assign].sum(axis=1, skipna=True)

if 'quiz_sum' not in df.columns:
    if len(qz_cols) > 0:
        df['quiz_sum'] = df[qz_cols].sum(axis=1, skipna=True)
    else:
        alt_quiz = [c for c in df.columns if 'quiz' in c.lower()]
        if alt_quiz:
            df['quiz_sum'] = df[alt_quiz].sum(axis=1, skipna=True)

# Candidate names (common variants)
si_cands = ["S-I", "S I", "SI", "s-i", "midterm1", "midterm_1", "midterm i"]
sii_cands = ["S-II", "S II", "SII", "s-ii", "midterm2", "midterm_2", "midterm ii"]
final_cands = ["Final", "final_exam", "final exam", "final"]

si_col = find_col(si_cands)
sii_col = find_col(sii_cands)
final_col = find_col(final_cands)

# ---------- Build X, y depending on RQ (with safety checks) ----------
base_features = []
if 'assign_sum' in df.columns:
    base_features.append('assign_sum')
if 'quiz_sum' in df.columns:
    base_features.append('quiz_sum')

# If neither aggregated feature exists, warn and stop
if len(base_features) == 0:
    st.error("Could not find assign_sum or quiz_sum in data. Please include assignment/quiz columns or precomputed aggregates.")
    st.stop()

if rq == "RQ1: S-I":
    if si_col is None:
        st.error(f"Could not find S-I column. Available columns: {', '.join(df.columns)}")
        st.stop()
    X = df[base_features].copy(); y = df[si_col].copy()
elif rq == "RQ2: S-II":
    if sii_col is None:
        st.error(f"Could not find S-II column. Available columns: {', '.join(df.columns)}")
        st.stop()
    if si_col is None:
        st.error(f"RQ2 requires S-I as a predictor but S-I column not found. Available columns: {', '.join(df.columns)}")
        st.stop()
    X = df[base_features + [si_col]].copy(); y = df[sii_col].copy()
else:  # RQ3 Final
    if final_col is None:
        st.error(f"Could not find Final column. Available columns: {', '.join(df.columns)}")
        st.stop()
    features = base_features.copy()
    if si_col is not None:
        features.append(si_col)
    else:
        st.warning("S-I not found; proceeding without it.")
    if sii_col is not None:
        features.append(sii_col)
    else:
        st.warning("S-II not found; proceeding without it.")
    X = df[features].copy(); y = df[final_col].copy()

st.write("Using features:", X.columns.tolist())
st.write("Target:", y.name)

# ---------- Main button: run evaluation ----------
if st.button("Run evaluation"):
    st.info("Splitting data (80/20) and training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    dummy = make_pipe(DummyRegressor(strategy='mean')); dummy.fit(X_train, y_train); y_pred_dummy = dummy.predict(X_test)
    lin = make_pipe(LinearRegression()); lin.fit(X_train, y_train); y_pred_lin = lin.predict(X_test)
    rf = make_pipe(RandomForestRegressor(n_estimators=200, random_state=42)); rf.fit(X_train, y_train); y_pred_rf = rf.predict(X_test)

    # Results table
    rows = [
        {'Model': 'Dummy', **metrics(y_test, y_pred_dummy)},
        {'Model': 'Linear', **metrics(y_test, y_pred_lin), 'Train_R2': round(lin.score(X_train, y_train), 4), 'Test_R2': round(lin.score(X_test, y_test), 4)},
        {'Model': 'RandomForest', **metrics(y_test, y_pred_rf), 'Train_R2': round(rf.score(X_train, y_train), 4), 'Test_R2': round(rf.score(X_test, y_test), 4)}
    ]
    df_res = pd.DataFrame(rows).set_index('Model')
    st.subheader("Test metrics")
    st.dataframe(df_res)

    # Best model
    non_dummy = df_res.drop(index='Dummy')
    best_name = non_dummy['MAE'].idxmin()
    st.write("**Best model on test MAE:**", best_name)
    st.write("Train R2:", non_dummy.loc[best_name, 'Train_R2'], " Test R2:", non_dummy.loc[best_name, 'Test_R2'])

    # Bootstrap (optional)
    if st.checkbox("Run bootstrap on training set (may take time)"):
        st.write(f"Running bootstrap (n={n_boot}) on training data (OOB evaluation) — this may take time.")
        lin_maes = bootstrap_mae(lambda: LinearRegression(), X_train, y_train, n_boot=n_boot)
        rf_maes = bootstrap_mae(lambda: RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, n_boot=n_boot)
        st.write("Linear bootstrap MAE mean and 95% CI:", lin_maes.mean(), np.percentile(lin_maes, [2.5, 97.5]))
        st.write("RF bootstrap MAE mean and 95% CI:", rf_maes.mean(), np.percentile(rf_maes, [2.5, 97.5]))

# Pipeline diagram in sidebar (optional)
if os.path.exists("pipeline_diagram.png"):
    st.sidebar.write("Pipeline diagram below")
    st.sidebar.image("pipeline_diagram.png", use_column_width=True)
