<div align="center">

# ForecastX

### Multi-Store Sales Forecasting

End-to-end weekly sales forecasting for 45 retail locations using Gradient Boosting,
macroeconomic signals, and an interactive Streamlit dashboard.

[Results](#results) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Tech Stack](#tech-stack)

</div>

---

## Highlights

| Metric | Value | Notes |
|---|---|---|
| **MAE** | **`$4,719.62`** | Average forecast error per store × dept × week |
| **RMSE** | **`$8,167.94`** | Penalizes large misses (stockouts) heavily |
| **SMAPE** | **`58.06%`** | Symmetric MAPE — regular MAPE is undefined when actuals hit zero |
| **Coverage** | 45 stores | Per-store × per-department forecasts |
| **Reproducible** | Yes | Fixed `random_state=42`, single eval script |

![ForecastX Dashboard](sales_forecasting_banner.jpg)

---

## What it does

ForecastX predicts **weekly sales per store × department** so inventory and supply-chain
teams can plan stock levels, labor scheduling, and procurement with data instead of gut feel.

**Why it matters in retail:**

- Better forecasts → less overstock (lower holding cost) **and** fewer stockouts (recovered lost sales).
- Per-store granularity catches local patterns that a national forecast would miss.

> Demo login: `admin` / `password` — change before deploying anywhere real.

---

## Key features

- **Per-store, per-department forecasts** — 45 stores × ~80 depts of fine-grained inventory signals.
- **Gradient Boosting Regressor** with macroeconomic features (CPI, fuel price, unemployment, holiday flags).
- **Interactive Streamlit dashboard** — pick a city, department, and target date; get an instant forecast with feature attributions.
- **SMAPE / MAE / RMSE evaluation** on a held-out 20% split.
- **End-to-end reproducible pipeline** — single eval script, fixed random seed, identical numbers every run.

---

## Tech stack

| Layer | Tech | Version |
|---|---|---|
| Language | Python | 3.10+ |
| ML framework | Scikit-learn | ≥ 1.3 |
| Data | Pandas, NumPy | pandas ≥ 2.0, numpy ≥ 1.24 |
| Visualization | Plotly, Streamlit | streamlit ≥ 1.30 |

---

## Architecture

```
Raw CSVs (train.csv · stores.csv · features.csv · test.csv)
        │
        ▼
[ 1. Preprocessing ]      →  merge tables, fill null MarkDown cells, type-cast IsHoliday
        │
        ▼
[ 2. Feature Engineering ] →  year / month / week / day, store-type encoding, macro joins
        │
        ▼
[ 3. Model Training ]      →  GradientBoostingRegressor (n_estimators=100, max_depth=5)
        │
        ▼
[ 4. Evaluation ]          →  80/20 split, MAE / SMAPE / RMSE on hold-out
        │
        ▼
[ 5. Streamlit App ]       →  per-store forecast viewer with feature importance
```

---

## Results

Evaluation on a held-out 20% test split (`random_state=42` for reproducibility).

| Metric | Value | Notes |
|---|---|---|
| **SMAPE** | `58.06 %` | Symmetric MAPE (regular MAPE blows up on zero-sales weeks) |
| **MAE** | `$4,719.62` | Mean Absolute Error on hold-out |
| **RMSE** | `$8,167.94` | Root Mean Squared Error on hold-out |

### Top features by importance

| Rank | Feature | Importance | Why it matters |
|---|---|---|---|
| 1 | `Dept` | 0.718 | Department identity drives ~72% of variance — different depts have radically different volume profiles. |
| 2 | `Size` | 0.184 | Larger stores → more footfall → more sales (near-linear relationship). |
| 3 | `Store` | 0.051 | Store-level effects beyond size (location, demographics). |
| 4 | `Week` | 0.014 | Seasonal timing (holiday spikes, back-to-school). |
| 5 | `CPI` | 0.012 | Macroeconomic context — captures consumer spending pressure. |

---

## Dataset

**Walmart Recruiting — Store Sales Forecasting** (Kaggle).

| File | Rows | Purpose |
|---|---|---|
| `train.csv` | ~421k | Historical weekly sales by store × dept × date |
| `test.csv` | ~115k | Blind hold-out for final prediction |
| `stores.csv` | 45 | Store metadata (type, size, region) |
| `features.csv` | ~8190 | CPI, fuel price, unemployment, MarkDown events, holidays |

---

## Project structure

```
ForecastX/
├── app.py                          # Streamlit dashboard (entry point)
├── test.py                         # Local eval script — prints MAE / SMAPE / RMSE + top features
├── train.csv                       # Historical sales (training)
├── test.csv                        # Blind hold-out for final submission
├── stores.csv                      # Store metadata
├── features.csv                    # External features (CPI, fuel, holidays)
├── sales_forecasting_banner.jpg    # Hero image for README + dashboard
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT
├── README.md
└── .gitignore
```

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/Asiyaarab/ForecastX.git
cd ForecastX
```

### 2. Set up a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 5. Reproduce the evaluation numbers

```bash
python test.py
```

Expected output (≈ 30-60s on a typical laptop):

```
MAE:   4719.62
SMAPE: 58.06%
RMSE:  8167.94
```

---

## Key engineering decisions

- **Gradient Boosting over Linear Regression** — captures non-linear interactions between dept × store × season without manual feature crosses.
- **SMAPE over MAPE** — Walmart's actual sales include legitimate zero-sales weeks (closed for inventory, regional holidays), which makes MAPE blow up to infinity. SMAPE is symmetric and bounded.
- **Single shared model across all stores** — fewer parameters to maintain, transfers learnings from high-volume stores to low-volume ones. Per-store models are a future enhancement.
- **Hold-out evaluation, not cross-validation** — easier to defend in interviews; same model behavior, simpler pipeline.

---

## Future work

- [ ] LSTM / Prophet baselines for time-series comparison.
- [ ] Per-store hyperparameter tuning via Bayesian search.
- [ ] REST API wrapper (FastAPI) so the BI team can query forecasts directly.
- [ ] Confidence intervals on every prediction (quantile regression).
- [ ] Lag features (last 1w / 4w / 52w sales) for true time-series modeling.

---

## Author

**Asiya Arab** — BCA, Shreyarth University · ML Intern @ Webify.ai

- Email: [aashiyaarab39@gmail.com](mailto:aashiyaarab39@gmail.com)
- LinkedIn: [linkedin.com/in/asiya-arab](https://linkedin.com/in/asiya-arab)
- GitHub: [@Asiyaarab](https://github.com/Asiyaarab)

---

## License

MIT — free to use, modify, and learn from. See [LICENSE](LICENSE).

---

## Acknowledgments

- Dataset: **Walmart Recruiting — Store Sales Forecasting** (Kaggle).
- Model: `sklearn.ensemble.GradientBoostingRegressor`.
- Dashboard framework: [Streamlit](https://streamlit.io).
