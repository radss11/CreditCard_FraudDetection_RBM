import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch import nn

st.set_page_config(page_title="FraudLens AI", page_icon="🛡️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "rbm_model.pt"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
FEATURES_PATH = PROCESSED_DIR / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
DEFAULT_THRESHOLD = 0.019421609118580818

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#0b1020;color:#e8edf8;}
[data-testid="stSidebar"] {background:#10172a;border-right:1px solid #263653;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.03em;color:#f5f7ff;}
p,label {color:#a8b4cc;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#151f35,#111a2d);border:1px solid #273653;padding:16px;border-radius:15px;}
div[data-testid="stMetricLabel"] p {color:#9eacc6!important;}
div[data-testid="stMetricValue"] {color:#f4f7ff!important;font-family:'Space Grotesk',sans-serif;}
.stButton>button {border-radius:10px;border:1px solid #7868f6;color:white;background:linear-gradient(100deg,#6658e8,#8b5cf6);font-weight:700;}
.hero {background:radial-gradient(circle at 85% 20%,rgba(113,91,242,.22),transparent 35%),linear-gradient(120deg,#151f37,#11182b);border:1px solid #293754;border-radius:20px;padding:24px 26px;margin-bottom:18px;}
.kicker {color:#a99cff;font-size:.76rem;text-transform:uppercase;letter-spacing:.15em;font-weight:700;}
.hero-title {color:#f6f7ff;font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;margin:7px 0;}
.hero-sub {color:#aab7d0;font-size:.94rem;}
.pill {display:inline-block;border:1px solid #344464;color:#b8c7e5;background:#17223a;padding:5px 10px;border-radius:999px;font-size:.76rem;}
.card {background:#111a2d;border:1px solid #263653;border-radius:16px;padding:17px 18px;}
hr {border-color:#26334d;}
</style>
""", unsafe_allow_html=True)

class RBM(nn.Module):
    """Same parameter layout as the trained FraudLens RBM checkpoint."""
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        hidden_prob = torch.sigmoid(x @ self.W + self.hidden_bias)
        reconstructed = torch.sigmoid(hidden_prob @ self.W.T + self.visible_bias)
        return torch.mean((x - reconstructed) ** 2, dim=1)

@st.cache_resource
def load_model():
    needed = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Required model files missing: " + ", ".join(missing))
    device = torch.device("cpu")
    # This is the user's own locally trained checkpoint.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = RBM(int(checkpoint["n_visible"]), int(checkpoint["n_hidden"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = list(pickle.load(f))
    return model, scaler, feature_columns, device

def score_transactions(frame: pd.DataFrame, model, scaler, feature_columns, device):
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError("CSV is missing required model features: " + ", ".join(missing))
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError("Missing/non-numeric values found in: " + ", ".join(bad))
    X_scaled = scaler.transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        scores = model.reconstruction_error(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy()
    out = frame.copy()
    out["anomaly_score"] = scores
    if "transaction_id" not in out.columns:
        out.insert(0, "transaction_id", [f"TXN-{i:06d}" for i in range(1, len(out) + 1)])
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.Timestamp.now()
    if "Amount" in out.columns:
        out["amount"] = pd.to_numeric(out["Amount"], errors="coerce").fillna(0)
    elif "amount" not in out.columns:
        out["amount"] = 0.0
    if "channel" not in out.columns:
        out["channel"] = "Card"
    if "region" not in out.columns:
        out["region"] = "Unknown"
    return out

@st.cache_data(show_spinner="Preparing sample transactions…")
def load_demo_raw():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place creditcard.csv inside the data folder.")
    raw = pd.read_csv(DATA_PATH)
    # A manageable sample keeps the dashboard responsive while retaining real rows.
    n = min(2500, len(raw))
    demo = raw.sample(n=n, random_state=17).reset_index(drop=True)
    demo.insert(0, "transaction_id", [f"DEMO-{i:05d}" for i in range(1, len(demo) + 1)])
    demo["timestamp"] = pd.Timestamp.now().floor("min") - pd.to_timedelta(np.arange(len(demo)), unit="m")
    rng = np.random.default_rng(17)
    demo["channel"] = rng.choice(["E-commerce", "POS", "Mobile", "ATM"], len(demo))
    demo["region"] = rng.choice(["West", "South", "North", "East", "International"], len(demo))
    return demo

try:
    model, scaler, feature_columns, device = load_model()
    model_ready = True
    load_error = ""
except Exception as exc:
    model_ready = False
    load_error = str(exc)

with st.sidebar:
    st.markdown("## 🛡️ FraudLens AI")
    st.markdown('<span class="pill">RBM · Anomaly Intelligence</span>', unsafe_allow_html=True)
    st.markdown("---")
    source = st.radio("Data source", ["Demo dataset", "Upload CSV"])
    threshold = st.slider("Anomaly threshold", 0.0005, 0.08, float(DEFAULT_THRESHOLD), 0.0005, format="%.4f")
    min_amount = st.number_input("Minimum amount", min_value=0, value=0, step=500)
    status_filter = st.multiselect("Show status", ["Flagged", "Normal"], default=["Flagged", "Normal"])
    st.markdown("---")
    st.markdown("**Model status**")
    if model_ready:
        st.markdown('<span class="pill">RBM connected · CPU inference</span>', unsafe_allow_html=True)
        st.caption("Scores use your trained checkpoint and saved scaler.")
    else:
        st.error("RBM files not found / could not load.")
        st.caption(load_error)

st.markdown("""
<div class="hero">
 <div class="kicker">Transaction intelligence platform</div>
 <div class="hero-title">See the unusual. Investigate with confidence.</div>
 <div class="hero-sub">RBM-powered anomaly scoring for transaction review. High reconstruction error indicates a transaction that differs from patterns learned from normal transactions.</div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error("The dashboard cannot score transactions until the trained RBM files are available.")
    st.code("credit_fraud_detection/\n├── app.py\n├── artifacts/rbm_model.pt\n├── data/creditcard.csv\n└── data/processed/\n    ├── scaler.pkl\n    └── feature_columns.pkl")
    st.stop()

if source == "Upload CSV":
    upload = st.file_uploader("Upload transaction CSV", type=["csv"])
    st.caption("The CSV must contain all 30 model feature columns (Time, V1–V28, Amount). An optional Class column is used only to show evaluation metrics.")
    if upload is None:
        st.info("Upload your CSV, or choose Demo dataset from the sidebar.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
        with st.spinner("Scoring transactions with the trained RBM…"):
            data = score_transactions(raw, model, scaler, feature_columns, device)
    except Exception as exc:
        st.error(f"Could not score this file: {exc}")
        st.stop()
else:
    try:
        raw = load_demo_raw()
        with st.spinner("Scoring sample transactions with the trained RBM…"):
            data = score_transactions(raw, model, scaler, feature_columns, device)
    except Exception as exc:
        st.error(f"Could not load demo data: {exc}")
        st.stop()

# Preserve a clear distinction between anomaly flags and ground-truth labels.
data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0)
data["anomaly_score"] = pd.to_numeric(data["anomaly_score"], errors="coerce").fillna(0)
data["status"] = np.where(data["anomaly_score"] >= threshold, "Flagged", "Normal")
data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
view = data[(data["amount"] >= min_amount) & (data["status"].isin(status_filter))].copy()
flagged = view[view.status == "Flagged"]
count = len(view)
flag_rate = len(flagged) / count * 100 if count else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions reviewed", f"{count:,}")
c2.metric("Flagged for review", f"{len(flagged):,}", f"{flag_rate:.1f}% of view", delta_color="inverse")
c3.metric("Transaction value", f"₹{view.amount.sum():,.0f}")
c4.metric("Average anomaly score", f"{view.anomaly_score.mean():.4f}" if count else "—")

if "Class" in data.columns and count:
    labelled = view[view["Class"].isin([0, 1])]
    if len(labelled):
        actual = labelled["Class"].astype(int).to_numpy()
        predicted = (labelled["anomaly_score"].to_numpy() >= threshold).astype(int)
        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        st.caption(f"Demo sample with labels: precision {precision:.1%} · recall {recall:.1%}. Threshold and sample selection affect these values; flags are for review, not proof of fraud.")

left, right = st.columns([1.5, 1])
with left:
    st.markdown("### Risk activity")
    if count:
        trend_data = view.dropna(subset=["timestamp"]).copy()
        trend_data["period"] = trend_data.timestamp.dt.floor("6h")
        trend = trend_data.groupby("period", as_index=False).agg(transactions=("transaction_id", "count"), flagged=("status", lambda s: (s == "Flagged").sum()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.period, y=trend.transactions, name="Transactions", mode="lines", line=dict(color="#8b9cff", width=2.5), fill="tozeroy", fillcolor="rgba(139,156,255,.08)"))
        fig.add_trace(go.Scatter(x=trend.period, y=trend.flagged, name="Flagged", mode="lines+markers", line=dict(color="#fb7185", width=2.5), marker=dict(size=5)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=310, margin=dict(l=5,r=5,t=25,b=5), legend=dict(orientation="h",y=1.15,x=0), xaxis_title=None, yaxis_title="Count", xaxis=dict(gridcolor="#26334d"), yaxis=dict(gridcolor="#26334d"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions match the current filters.")
with right:
    st.markdown("### Anomaly score distribution")
    if count:
        fig2 = px.histogram(view, x="anomaly_score", nbins=24, color="status", barmode="overlay", opacity=.85, color_discrete_map={"Flagged":"#fb7185", "Normal":"#818cf8"})
        fig2.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24", annotation_text="Threshold")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=310, margin=dict(l=5,r=5,t=25,b=5), xaxis_title="Reconstruction error", yaxis_title="Transactions", xaxis=dict(gridcolor="#26334d"), yaxis=dict(gridcolor="#26334d"), legend=dict(orientation="h",y=1.15,x=0))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data to display.")

tab1, tab2, tab3 = st.tabs(["🚨 Investigation queue", "🧠 Model overview", "📁 Data & export"])
with tab1:
    st.markdown("### Transactions requiring attention")
    st.caption("Sorted by RBM reconstruction error. A flag indicates an anomaly for investigation, not confirmed fraud.")
    if len(flagged):
        q = flagged.sort_values("anomaly_score", ascending=False).copy()
        q["amount_display"] = q.amount.map(lambda x: f"₹{x:,.2f}")
        cols = [c for c in ["transaction_id", "timestamp", "amount_display", "channel", "region", "anomaly_score", "status"] if c in q]
        st.dataframe(q[cols].head(100), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download flagged transactions", flagged.to_csv(index=False).encode(), file_name="fraudlens_flagged_transactions.csv", mime="text/csv")
    else:
        st.success("No transactions flagged at the current threshold.")
    with st.expander("Inspect one transaction"):
        if len(view):
            chosen = st.selectbox("Transaction ID", view.transaction_id.astype(str).tolist())
            row = view[view.transaction_id.astype(str) == chosen].iloc[0]
            a, b, c = st.columns(3)
            a.metric("Amount", f"₹{row.amount:,.2f}")
            b.metric("Anomaly score", f"{row.anomaly_score:.5f}")
            c.metric("Status", row.status)
            show = {k: (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items() if k not in feature_columns}
            st.json(show)
        else:
            st.caption("No transaction to inspect.")
with tab2:
    st.markdown("### RBM model card")
    st.markdown("""
    <div class="card">
      <h3>Restricted Boltzmann Machine · Connected</h3>
      <p>The model was trained on normal transactions. The dashboard scales incoming feature vectors using the saved scaler and computes mean squared reconstruction error. Higher scores are flagged for review.</p>
      <span class="pill">30 visible features</span>
      <span class="pill">32 hidden units</span>
      <span class="pill">CD-1 training</span>
    </div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Model connected", "Yes")
    b.metric("Inference device", "CPU")
    c.metric("Current threshold", f"{threshold:.4f}")
    st.write("Validation metrics from the current run: ROC-AUC **0.9075**, PR-AUC **0.3374**, and validation F1 **0.4730** at the selected threshold.")
    st.info("This is an academic anomaly-detection prototype. A flagged transaction is not automatically confirmed as fraud.")
with tab3:
    a, b, c = st.columns(3)
    a.metric("Rows in source", f"{len(data):,}")
    a.metric("Columns", f"{len(data.columns):,}")
    a.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    st.download_button("⬇ Download scored dataset", data.to_csv(index=False).encode(), file_name="fraudlens_scored_transactions.csv", mime="text/csv")

st.markdown("---")
st.caption("FraudLens AI · Trained RBM anomaly detection · Academic prototype")
