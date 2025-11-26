# app.py (Streamlit dashboard) 
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------- Load data ----------
DATA_PATH = "clean_marks.csv"
df = pd.read_csv(DATA_PATH)

# ---------- Page UI ----------
st.title("Student's Marks predictor")
st.sidebar.header("Choose RQ & settings")
rq = st.sidebar.selectbox("Research Question", ["RQ1: S-I", "RQ2: S-II", "RQ3: Final"])
n_boot = st.sidebar.slider("Bootstrap samples (train only)", min_value=100, max_value=500, value=200, step=50)

# ---------- Helper functions ----------
def _rmse(y_true, y_pred):
    """Compute RMSE (compatible with all sklearn versions)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred):
    """Return dictionary of MAE, RMSE, R2."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': _rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def make_pipe(model):
    """Pipeline: imputer -> scaler -> model."""
    return make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), model)

def bootstrap_mae(model_factory, X_tr, y_tr, n_boot):
    """Bootstrap (train-only) OOB MAE estimates for given model factory."""
    rng = np.random.RandomState(0)
    X_tr = X_tr.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    n = len(X_tr)
    maes = []
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)            # bootstrap sample indices
        Xb, yb = X_tr.iloc[idx], y_tr.iloc[idx]    # bootstrap sample
        model = model_factory()
        model.fit(Xb, yb)
        # OOB mask
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(idx)] = False
        if oob_mask.sum() == 0:
            # fallback if no OOB (rare)
            y_oob_true = y_tr
            y_oob_pred = model.predict(X_tr)
        else:
            y_oob_true = y_tr.iloc[oob_mask]
            y_oob_pred = model.predict(X_tr.iloc[oob_mask])
        maes.append(mean_absolute_error(y_oob_true, y_oob_pred))
    return np.array(maes)

# ---------- Select features / target ----------
base_features = ['assign_sum', 'quiz_sum']
if rq == "RQ1: S-I":
    X = df[base_features].copy(); y = df['S-I'].copy()
elif rq == "RQ2: S-II":
    X = df[base_features + ['S-I']].copy(); y = df['S-II'].copy()
else:
    X = df[base_features + ['S-I','S-II']].copy(); y = df['Final'].copy()

st.write("Selected features:", X.columns.tolist())
st.write("Target:", y.name)

# ---------- Main button: run evaluation ----------
if st.button("Run evaluation"):
    st.info("Splitting data (80/20) and training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models using pipelines (imputer + scaler inside pipeline avoids leakage)
    dummy = make_pipe(DummyRegressor(strategy='mean'))
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    lin = make_pipe(LinearRegression())
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    rf = make_pipe(RandomForestRegressor(n_estimators=200, random_state=42))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Display test metrics
    rows = [
        {'Model': 'Dummy', **metrics(y_test, y_pred_dummy)},
        {'Model': 'Linear', **metrics(y_test, y_pred_lin), 'Train_R2': round(lin.score(X_train, y_train), 4), 'Test_R2': round(lin.score(X_test, y_test), 4)},
        {'Model': 'RandomForest', **metrics(y_test, y_pred_rf), 'Train_R2': round(rf.score(X_train, y_train), 4), 'Test_R2': round(rf.score(X_test, y_test), 4)}
    ]
    df_res = pd.DataFrame(rows).set_index('Model')
    st.subheader("Test metrics")
    st.dataframe(df_res)

    # Indicate best model by MAE (ignore Dummy)
    non_dummy = df_res.drop(index='Dummy')
    best_name = non_dummy['MAE'].idxmin()
    st.write("**Best model on test MAE:**", best_name)
    st.write("Train R2:", non_dummy.loc[best_name, 'Train_R2'], " Test R2:", non_dummy.loc[best_name, 'Test_R2'])

    # Optional: bootstrap on training data (OOB)
    if st.checkbox("Run bootstrap on training set (may take time)"):
        st.write(f"Running bootstrap (n={n_boot}) on training data (OOB evaluation) — this may take time.")
        lin_maes = bootstrap_mae(lambda: LinearRegression(), X_train, y_train, n_boot=n_boot)
        rf_maes = bootstrap_mae(lambda: RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train, n_boot=n_boot)
        st.write("Linear bootstrap MAE mean and 95% CI:", lin_maes.mean(), np.percentile(lin_maes, [2.5, 97.5]))
        st.write("RF bootstrap MAE mean and 95% CI:", rf_maes.mean(), np.percentile(rf_maes, [2.5, 97.5]))

# Pipeline diagram in sidebar
st.sidebar.write("Pipeline diagram below")
st.sidebar.image("pipeline_diagram.png", use_column_width=True)
