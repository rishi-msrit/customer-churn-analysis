# Customer Churn Analysis & Retention Modeling

**Live demo:** [customer-churn-analysis.vercel.app](#) *(link added after deploy)*

Interactive analysis of telecom customer churn using the IBM Telco Customer Churn dataset (~7,000 customers). Exploratory data analysis, statistical driver identification, logistic regression churn prediction, and k-means customer segmentation — all visualised in a five-page static web app deployed on Vercel.

---

## What This Solves

Retention teams need to know *who* is about to leave and *why*. This project answers both:

- **Who** — a live prediction tool returns a churn probability for any customer profile
- **Why** — statistical analysis identifies which contract type, tenure band, and service combinations are most strongly associated with churn
- **How to act** — k-means clustering groups customers into retention-relevant segments with recommended strategies per segment

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & analysis | Python, pandas, scikit-learn, scipy |
| Visualisation | Plotly (all charts interactive) |
| Frontend | HTML, CSS, vanilla JS |
| Hosting | Vercel (static, always-on free tier) |

---

## Architecture

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
        │
        ▼
analysis/run_analysis.py   (run once locally)
        │
        ├─► public/data/kpis.json
        ├─► public/data/*.json          (Plotly chart specs)
        ├─► public/data/model_coefficients.json
        └─► public/data/segments_profiles.json
        │
        ▼
Static HTML/CSS/JS (public/)  →  Vercel
```

---

## Key Findings

| Finding | Value |
|---|---|
| Overall churn rate | **26.5%** (1,869 of 7,043 customers) |
| Month-to-month contract churn rate | **~43%** |
| Two-year contract churn rate | **~3%** |
| Churn rate without Online Security | significantly higher (χ²=850.0, p<0.001) |
| Churn rate without Tech Support | significantly higher (χ²=828.2, p<0.001) |
| Tenure vs churn (point-biserial r) | **−0.352** — longer tenure strongly predicts retention |
| Monthly charges vs churn (r) | **+0.193** — higher charges weakly predict churn |
| TotalCharges blanks fixed | **11 rows** (new customers, tenure=0, set to $0) |
| Logistic regression ROC-AUC | **0.8414** |
| Logistic regression recall (churn class) | **0.7861** |
| Logistic regression F1 (churn class) | **0.6164** |

**Segment breakdown (k=4):**

| Segment | Size | Avg Tenure | Avg Monthly | Actual Churn |
|---|---|---|---|---|
| Stable Budget | 1,362 | 50.0 mo | $33.63 | 4.2% |
| Stable High-Value | 1,860 | 58.9 mo | $92.02 | 12.6% |
| At-Risk Budget | 1,734 | 10.3 mo | $36.75 | 24.3% |
| At-Risk VIPs | 2,087 | 15.5 mo | $84.06 | **55.4%** |

---

## Pages

| Page | What it shows |
|---|---|
| Overview | Headline KPIs, churn donut, contract bar, tenure and charges histograms |
| Churn Drivers | Chi-square findings panel, 5 driver charts including heatmap |
| Prediction Tool | Live form — model runs client-side with no server request |
| Model Performance | Confusion matrix, ROC, PR curve, feature importance, limitations |
| Customer Segments | PCA scatter, segment churn rates, profile cards with retention strategies |

---

## Methodology & Assumptions

**Missing data:** The `TotalCharges` column contains a space character (not NaN) for customers with `tenure=0` who have not yet been billed. These are converted to numeric via `pd.to_numeric(errors='coerce')` and filled with `0`, representing $0 billed. The exact count is reported in `kpis.json`.

**Model:** Logistic regression with `class_weight='balanced'` to handle the class imbalance (churn is approximately 26% of the dataset). 80/20 stratified train-test split with `random_state=42`. All features standardised with `StandardScaler`.

**Clustering:** K-means (k=4) on standardised (tenure, monthly charges, predicted churn probability). Segment labels assigned programmatically based on whether a cluster's average charges and churn probability fall above or below dataset medians.

**In-browser prediction:** The exported `model_coefficients.json` contains the LR intercept, feature coefficients, and scaler parameters. The browser re-applies standardisation and runs the sigmoid — mathematically identical to sklearn's `predict_proba`.

---

## Local Setup

```bash
# Clone
git clone https://github.com/rishi-msrit/customer-churn-analysis.git
cd customer-churn-analysis

# Install Python dependencies
pip install -r analysis/requirements.txt

# Place dataset
# Copy WA_Fn-UseC_-Telco-Customer-Churn.csv → data/raw/

# Run analysis (generates all public/data/*.json files)
python analysis/run_analysis.py

# Serve locally
npx serve public
# → open http://localhost:3000
```

---

## What I'd Add With More Time

- Filters on the Drivers page (filter charts by contract type or internet service)
- Survival analysis (Kaplan-Meier) to model *time-to-churn* rather than binary outcome
- SHAP values for the prediction tool (per-customer contribution breakdown beyond top features)
- Monthly charge per-feature breakdown (the dataset bundles many services into one charge figure)
