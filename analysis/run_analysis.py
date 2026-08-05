"""
run_analysis.py
Reads the Telco churn CSV, runs descriptive analysis + a simple
logistic regression + k-means segmentation, and writes Plotly
figure JSONs + data JSONs to ../public/data/.

Usage:
    python analysis/run_analysis.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT  = ROOT / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

CHURN_COLOR  = "#d94f4f"
RETAIN_COLOR = "#3d9970"
ACCENT       = "#2c5f8a"

# ── helpers ───────────────────────────────────────────────────────────────────

def save_fig(name: str, fig: go.Figure) -> None:
    (OUT / name).write_text(fig.to_json())
    print(f"  {name}")

def save_json(name: str, data: dict | list) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2))
    print(f"  {name}")

def base_layout(**extra) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#666"),
        margin=dict(l=50, r=20, t=40, b=50),
        **extra,
    )

def grid(axis="xy") -> dict:
    opts = dict(showgrid=True, gridcolor="rgba(128,128,128,0.14)", zeroline=False)
    if axis == "x":
        return {"xaxis": opts}
    if axis == "y":
        return {"yaxis": opts}
    return {"xaxis": opts, "yaxis": opts}


# ── load & clean ──────────────────────────────────────────────────────────────

print(f"\nReading {CSV.name} ...")
df = pd.read_csv(CSV)
print(f"  {len(df)} rows, {df.shape[1]} columns")

# TotalCharges is stored as a string and contains a space character (not NaN)
# for the handful of customers with tenure=0 who haven't been billed yet.
# We convert to numeric (coercing the blanks to NaN) and fill with 0.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
tc_blanks = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(f"  {tc_blanks} blank TotalCharges values -> 0 (new customers, tenure=0)")

df["Churn"] = (df["Churn"] == "Yes").astype(int)

df["TenureBand"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    include_lowest=True,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────

kpis = {
    "total_customers":    int(len(df)),
    "churned":            int(df["Churn"].sum()),
    "churn_rate":         round(float(df["Churn"].mean() * 100), 1),
    "avg_tenure":         round(float(df["tenure"].mean()), 1),
    "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
    "tc_blanks_fixed":    tc_blanks,
}
save_json("kpis.json", kpis)
print(f"  Churn rate: {kpis['churn_rate']}%")


# ── overview charts ───────────────────────────────────────────────────────────

print("\nOverview charts ...")

churned_n  = int(df["Churn"].sum())
retained_n = len(df) - churned_n

fig = go.Figure(go.Pie(
    labels=["Retained", "Churned"],
    values=[retained_n, churned_n],
    hole=0.62,
    marker_colors=[RETAIN_COLOR, CHURN_COLOR],
    textinfo="percent+label",
    hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
))
fig.update_layout(**base_layout(), showlegend=False)
save_fig("overview_donut.json", fig)

cc     = df.groupby("Contract")["Churn"].mean() * 100
cc_n   = df.groupby("Contract")["Churn"].count()
colors = [CHURN_COLOR if v > 30 else ("#e8a838" if v > 15 else RETAIN_COLOR)
          for v in cc.values]
fig = go.Figure(go.Bar(
    x=cc.index, y=cc.round(1).values,
    marker_color=colors,
    text=[f"{v:.1f}%" for v in cc.values], textposition="outside",
    customdata=cc_n.values,
    hovertemplate="%{x}<br>Churn rate: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Contract Type"),
    yaxis=dict(title="Churn Rate (%)", range=[0, cc.max() * 1.3],
               **grid("y")["yaxis"]))
save_fig("overview_by_contract.json", fig)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df[df["Churn"] == 0]["tenure"], name="Retained",
    marker_color=RETAIN_COLOR, opacity=0.75, xbins=dict(size=4),
    hovertemplate="Tenure %{x}+ mo<br>Count: %{y}<extra></extra>",
))
fig.add_trace(go.Histogram(
    x=df[df["Churn"] == 1]["tenure"], name="Churned",
    marker_color=CHURN_COLOR, opacity=0.75, xbins=dict(size=4),
    hovertemplate="Tenure %{x}+ mo<br>Count: %{y}<extra></extra>",
))
fig.update_layout(**base_layout(), barmode="overlay",
    xaxis=dict(title="Tenure (months)", **grid("x")["xaxis"]),
    yaxis=dict(title="Customers", **grid("y")["yaxis"]),
    legend=dict(x=0.72, y=0.92))
save_fig("overview_tenure_hist.json", fig)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df[df["Churn"] == 0]["MonthlyCharges"], name="Retained",
    marker_color=RETAIN_COLOR, opacity=0.75, xbins=dict(size=5),
))
fig.add_trace(go.Histogram(
    x=df[df["Churn"] == 1]["MonthlyCharges"], name="Churned",
    marker_color=CHURN_COLOR, opacity=0.75, xbins=dict(size=5),
))
fig.update_layout(**base_layout(), barmode="overlay",
    xaxis=dict(title="Monthly Charges ($)", **grid("x")["xaxis"]),
    yaxis=dict(title="Customers", **grid("y")["yaxis"]),
    legend=dict(x=0.1, y=0.92))
save_fig("overview_charges_hist.json", fig)


# ── churn driver charts ───────────────────────────────────────────────────────

print("\nDriver charts ...")

pm = df.groupby("PaymentMethod").agg(churn_rate=("Churn", "mean"), n=("Churn", "count"))
pm["churn_rate"] *= 100
short = {
    "Bank transfer (automatic)": "Bank Transfer",
    "Credit card (automatic)":   "Credit Card",
    "Electronic check":          "E-Check",
    "Mailed check":              "Mailed Check",
}
fig = go.Figure(go.Bar(
    x=[short.get(l, l) for l in pm.index],
    y=pm["churn_rate"].round(1).values,
    marker_color=[CHURN_COLOR if v > 30 else RETAIN_COLOR for v in pm["churn_rate"]],
    text=[f"{v:.1f}%" for v in pm["churn_rate"]], textposition="outside",
    customdata=pm["n"].values,
    hovertemplate="%{x}<br>Churn: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Payment Method"),
    yaxis=dict(title="Churn Rate (%)", range=[0, pm["churn_rate"].max() * 1.35],
               **grid("y")["yaxis"]))
save_fig("drivers_payment.json", fig)

services = {
    "OnlineSecurity":  "Online Security",
    "TechSupport":     "Tech Support",
    "OnlineBackup":    "Online Backup",
    "DeviceProtection":"Device Protection",
    "StreamingTV":     "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "MultipleLines":   "Multiple Lines",
}
svc_labels, svc_yes, svc_no = [], [], []
for col, label in services.items():
    svc_labels.append(label)
    svc_yes.append(round(df[df[col] == "Yes"]["Churn"].mean() * 100, 1))
    svc_no.append(round(df[df[col] == "No"]["Churn"].mean() * 100, 1))

fig = go.Figure()
fig.add_trace(go.Bar(name="Without Service", y=svc_labels, x=svc_no,
    orientation="h", marker_color=CHURN_COLOR, opacity=0.85,
    hovertemplate="%{y} — without<br>Churn: %{x:.1f}%<extra></extra>"))
fig.add_trace(go.Bar(name="With Service", y=svc_labels, x=svc_yes,
    orientation="h", marker_color=RETAIN_COLOR, opacity=0.85,
    hovertemplate="%{y} — with<br>Churn: %{x:.1f}%<extra></extra>"))
fig.update_layout(**base_layout(), barmode="group",
    xaxis=dict(title="Churn Rate (%)", **grid("x")["xaxis"]),
    height=370, legend=dict(x=0.58, y=0.04))
save_fig("drivers_services.json", fig)

pivot = (df.groupby(["Contract", "InternetService"])["Churn"]
           .mean().unstack() * 100)
fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale=[[0, "#3d9970"], [0.5, "#e8d5a0"], [1, "#d94f4f"]],
    text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    hovertemplate="%{y} / %{x}<br>Churn: %{z:.1f}%<extra></extra>",
    colorbar=dict(title="Churn %", thickness=12),
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Internet Service"),
    yaxis=dict(title="Contract Type"))
save_fig("drivers_heatmap.json", fig)

fig = go.Figure()
fig.add_trace(go.Box(y=df[df["Churn"] == 0]["MonthlyCharges"],
    name="Retained", marker_color=RETAIN_COLOR, boxmean="sd"))
fig.add_trace(go.Box(y=df[df["Churn"] == 1]["MonthlyCharges"],
    name="Churned", marker_color=CHURN_COLOR, boxmean="sd"))
fig.update_layout(**base_layout(),
    yaxis=dict(title="Monthly Charges ($)", **grid("y")["yaxis"]))
save_fig("drivers_charges_box.json", fig)

tb   = df.groupby("TenureBand", observed=True)["Churn"].agg(rate="mean", n="count")
tb["rate"] *= 100
bar_colors = [CHURN_COLOR, "#e8a838", "#7dbea8", RETAIN_COLOR]
fig = go.Figure(go.Bar(
    x=tb.index.astype(str), y=tb["rate"].round(1).values,
    marker_color=bar_colors,
    text=[f"{v:.1f}%" for v in tb["rate"]], textposition="outside",
    customdata=tb["n"].values,
    hovertemplate="%{x}<br>Churn: %{y:.1f}%<br>n=%{customdata}<extra></extra>",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Customer Tenure"),
    yaxis=dict(title="Churn Rate (%)", range=[0, tb["rate"].max() * 1.3],
               **grid("y")["yaxis"]))
save_fig("drivers_tenure_band.json", fig)

# Statistical associations
findings = {}
for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"]:
    ct = pd.crosstab(df[col], df["Churn"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    findings[col] = {"chi2": round(float(chi2), 2), "p": round(float(p), 8),
                     "df": int(dof), "significant": bool(p < 0.001)}
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    r, p = stats.pointbiserialr(df["Churn"], df[col])
    findings[col] = {"r": round(float(r), 4), "p": round(float(p), 8),
                     "significant": bool(p < 0.001)}
save_json("statistical_findings.json", findings)


# ── logistic regression ───────────────────────────────────────────────────────

print("\nLogistic regression ...")

cat_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

encoded = pd.get_dummies(df[cat_cols], drop_first=True)
X = pd.concat([df[num_cols], encoded], axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# class_weight='balanced' handles the ~26% churn minority without oversampling
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_test_s)
y_prob = lr.predict_proba(X_test_s)[:, 1]

report  = classification_report(y_test, y_pred, output_dict=True)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

model_stats = {
    "accuracy":  round(report["accuracy"], 4),
    "precision": round(report["1"]["precision"], 4),
    "recall":    round(report["1"]["recall"], 4),
    "f1":        round(report["1"]["f1-score"], 4),
    "roc_auc":   round(roc_auc, 4),
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "class_balance_note": (
        f"Churn is {kpis['churn_rate']}% of data. "
        "class_weight='balanced' used to prevent the model from ignoring the minority class."
    ),
}
save_json("model_stats.json", model_stats)
print(f"  accuracy={model_stats['accuracy']}, AUC={model_stats['roc_auc']}, "
      f"recall={model_stats['recall']}")

fig = go.Figure(go.Heatmap(
    z=cm,
    x=["Predicted: No", "Predicted: Yes"],
    y=["Actual: No", "Actual: Yes"],
    colorscale=[[0, "#f2f2f2"], [1, ACCENT]],
    text=cm, texttemplate="%{text}",
    showscale=False,
    hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
))
fig.update_layout(**base_layout())
save_fig("model_cm.json", fig)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr.tolist(), y=tpr.tolist(), mode="lines",
    name=f"AUC = {roc_auc:.3f}",
    line=dict(color=ACCENT, width=2.5),
    hovertemplate="FPR %{x:.3f}  TPR %{y:.3f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(color="#bbb", dash="dash", width=1),
    showlegend=False, hoverinfo="skip",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="False Positive Rate", range=[0, 1], **grid("x")["xaxis"]),
    yaxis=dict(title="True Positive Rate",  range=[0, 1], **grid("y")["yaxis"]),
    legend=dict(x=0.55, y=0.08))
save_fig("model_roc.json", fig)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rec_vals.tolist(), y=prec_vals.tolist(), mode="lines",
    line=dict(color=CHURN_COLOR, width=2.5),
    hovertemplate="Recall %{x:.3f}  Precision %{y:.3f}<extra></extra>",
    showlegend=False,
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Recall",    range=[0, 1], **grid("x")["xaxis"]),
    yaxis=dict(title="Precision", range=[0, 1], **grid("y")["yaxis"]))
save_fig("model_pr.json", fig)

coefs    = lr.coef_[0]
feat_names = X.columns.tolist()
top15_idx  = np.argsort(np.abs(coefs))[::-1][:15]
top_names  = [feat_names[i] for i in top15_idx]
top_abs    = [float(np.abs(coefs[i])) for i in top15_idx]
top_signed = [float(coefs[i]) for i in top15_idx]

fig = go.Figure(go.Bar(
    y=top_names[::-1], x=top_abs[::-1], orientation="h",
    marker_color=[CHURN_COLOR if s > 0 else RETAIN_COLOR for s in top_signed[::-1]],
    text=[f"{v:.3f}" for v in top_abs[::-1]], textposition="outside",
    hovertemplate="%{y}<br>|Coefficient|: %{x:.4f}<extra></extra>",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="|Coefficient| (standardised)", **grid("x")["xaxis"]),
    height=480)
save_fig("model_importance.json", fig)

# Export model coefficients for the in-browser prediction tool
form_features = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "Contract_One year", "Contract_Two year",
    "InternetService_Fiber optic", "InternetService_No",
    "TechSupport_Yes", "OnlineSecurity_Yes",
    "PaperlessBilling_Yes", "MultipleLines_Yes",
    "StreamingTV_Yes", "StreamingMovies_Yes",
]
form_features = [f for f in form_features if f in X.columns]
form_idx      = [X.columns.tolist().index(f) for f in form_features]

save_json("model_coefficients.json", {
    "intercept":    float(lr.intercept_[0]),
    "features":     form_features,
    "coefficients": [float(coefs[i]) for i in form_idx],
    "scaler_mean":  [float(scaler.mean_[i]) for i in form_idx],
    "scaler_std":   [float(scaler.scale_[i]) for i in form_idx],
})


# ── customer segmentation ─────────────────────────────────────────────────────

print("\nClustering ...")

churn_prob_all = lr.predict_proba(scaler.transform(X))[:, 1]
df["churn_prob"] = churn_prob_all

cluster_raw = np.column_stack([
    df["tenure"].values,
    df["MonthlyCharges"].values,
    churn_prob_all,
])
cs = StandardScaler()
cluster_scaled = cs.fit_transform(cluster_raw)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["segment"] = km.fit_predict(cluster_scaled)

seg_stats = df.groupby("segment").agg(
    avg_tenure         = ("tenure",         "mean"),
    avg_charges        = ("MonthlyCharges",  "mean"),
    avg_churn_prob     = ("churn_prob",      "mean"),
    actual_churn_rate  = ("Churn",           "mean"),
    size               = ("Churn",           "count"),
).round(3)

median_charges = float(df["MonthlyCharges"].median())

seg_colors = {
    "Stable High-Value": "#3d9970",
    "Stable Budget":     "#4a9ead",
    "At-Risk Budget":    "#e8a838",
    "At-Risk VIPs":      "#d94f4f",
}
seg_strategies = {
    "Stable High-Value": (
        "Loyal, high-spending customers with low churn risk. Focus on rewarding "
        "loyalty and offering premium add-ons rather than discounts."
    ),
    "Stable Budget": (
        "Long-tenured, lower-spending customers who are unlikely to leave. "
        "Maintain satisfaction; gentle nudges toward higher-value plans can increase LTV."
    ),
    "At-Risk Budget": (
        "Lower-spending customers showing elevated churn signals. Targeted "
        "outreach with competitive pricing or contract incentives may retain this group."
    ),
    "At-Risk VIPs": (
        "High-value customers with the strongest churn risk — the highest priority "
        "for any retention team. Proactive intervention and personalised offers are warranted."
    ),
}

def label_segment(row):
    high_val  = row["avg_charges"] > median_charges
    high_risk = row["avg_churn_prob"] > 0.35
    if high_val and high_risk:      return "At-Risk VIPs"
    if high_val and not high_risk:  return "Stable High-Value"
    if not high_val and high_risk:  return "At-Risk Budget"
    return "Stable Budget"

seg_stats["name"]     = seg_stats.apply(label_segment, axis=1)
seg_stats["color"]    = seg_stats["name"].map(seg_colors)
seg_stats["strategy"] = seg_stats["name"].map(seg_strategies)

# 2D PCA scatter
pca    = PCA(n_components=2)
coords = pca.fit_transform(cluster_scaled)

fig = go.Figure()
for sid in range(4):
    mask  = km.labels_ == sid
    sname = seg_stats.loc[sid, "name"]
    col   = seg_stats.loc[sid, "color"]
    hover = [
        f"Tenure: {t:.0f}m | ${c:.0f}/mo | Risk: {p:.0%}"
        for t, c, p in zip(
            df.loc[mask, "tenure"],
            df.loc[mask, "MonthlyCharges"],
            df.loc[mask, "churn_prob"],
        )
    ]
    fig.add_trace(go.Scatter(
        x=coords[mask, 0].tolist(), y=coords[mask, 1].tolist(),
        mode="markers", name=sname,
        marker=dict(color=col, size=4, opacity=0.5),
        text=hover,
        hovertemplate="%{text}<extra>%{fullData.name}</extra>",
    ))
fig.update_layout(**base_layout(),
    xaxis=dict(title="PC1"),
    yaxis=dict(title="PC2"),
    legend=dict(x=0.62, y=0.97))
save_fig("segments_scatter.json", fig)

# Churn rate bar per segment
seg_order  = seg_stats.sort_values("avg_churn_prob").index.tolist()
ord_names  = [seg_stats.loc[i, "name"]                 for i in seg_order]
ord_colors = [seg_stats.loc[i, "color"]                for i in seg_order]
ord_churn  = [seg_stats.loc[i, "actual_churn_rate"]*100 for i in seg_order]
ord_n      = [seg_stats.loc[i, "size"]                 for i in seg_order]

fig = go.Figure(go.Bar(
    x=ord_names, y=[round(v, 1) for v in ord_churn],
    marker_color=ord_colors,
    text=[f"{v:.1f}%" for v in ord_churn], textposition="outside",
    customdata=ord_n,
    hovertemplate="%{x}<br>Churn rate: %{y:.1f}%<br>n=%{customdata:,}<extra></extra>",
))
fig.update_layout(**base_layout(),
    xaxis=dict(title="Segment"),
    yaxis=dict(title="Actual Churn Rate (%)", range=[0, max(ord_churn)*1.35],
               **grid("y")["yaxis"]))
save_fig("segments_churn_bar.json", fig)

# Segment profiles for the HTML page
seg_export = {}
for sid, row in seg_stats.iterrows():
    seg_export[str(sid)] = {
        "name":               row["name"],
        "color":              row["color"],
        "strategy":           row["strategy"],
        "avg_tenure":         round(float(row["avg_tenure"]), 1),
        "avg_charges":        round(float(row["avg_charges"]), 2),
        "avg_churn_prob":     round(float(row["avg_churn_prob"]) * 100, 1),
        "actual_churn_rate":  round(float(row["actual_churn_rate"]) * 100, 1),
        "size":               int(row["size"]),
    }
save_json("segments_profiles.json", seg_export)

print(f"\nDone. All files written to {OUT.relative_to(ROOT)}/")
print("\nModel summary:")
print(f"  Accuracy : {model_stats['accuracy']}")
print(f"  Precision: {model_stats['precision']}")
print(f"  Recall   : {model_stats['recall']}")
print(f"  F1       : {model_stats['f1']}")
print(f"  ROC-AUC  : {model_stats['roc_auc']}")
