# ForecastX — Multi-Store Sales Forecasting (India Edition)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Asiyaarab/ForecastX/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/Asiyaarab/ForecastX/blob/main)

> End-to-end **weekly sales forecasting** for 45 Indian retail locations using Gradient Boosting, macroeconomic CPI signals, and an interactive Streamlit dashboard for inventory planning — with role-based Admin/User access and ₹ (INR) reporting.

[![ForecastX Dashboard](https://github.com/Asiyaarab/ForecastX/raw/main/sales_forecasting_banner.jpg)](sales_forecasting_banner.jpg)
---

## What it does

Predicts **weekly sales** for each of 45 Indian retail stores (Delhi, Mumbai, Bengaluru, Ahmedabad, Surat, and more) 4–8 weeks ahead so inventory and supply-chain teams can plan stock, labor, and procurement with data instead of gut feel.

**Business value:** better forecasts → less overstock (lower holding cost) + fewer stockouts (lost sales).

---

## Key features

- Per-store forecasts for 45 Indian retail locations (Delhi, Mumbai, Bengaluru, Hyderabad, Ahmedabad, Chennai, Kolkata, Surat, Pune, Jaipur, and 35 more)
- **Role-based login** — Admin sees all stores; Store Users log in with their city name as username (e.g. `mumbai` / `mumbai_manager`) and see only their own store's data
- Department names instead of raw codes (Jewellery & Imitation, Men's Grooming, Footwear, etc. — 80+ departments), selectable from a dropdown
- Gradient Boosting via `HistGradientBoostingRegressor` — histogram-based for fast training on 400k+ rows, with permutation-based feature importance
- Model results cached to disk, so the app reloads instantly after code changes and only retrains when the underlying data changes
- Macroeconomic signals — CPI, fuel price, seasonality features
- All monetary values displayed in **₹ (INR)**
- Interactive Streamlit dashboard — pick a store & department, see historical sales trend (line chart), Regular vs Holiday average comparison, predicted forecast, and feature importance
- Forecast trend chart flags whether the prediction is up (🟢) or down (🔴) vs the last actual week, with a ₹ and % delta label
- **CSV export from the sidebar** — download the current forecast result as a CSV; Admins get an extra button to export the full all-stores dataset
- End-to-end pipeline — raw CSV to features to trained model to dashboard, all reproducible

---

## Tech stack

| Layer    | Tools                                        |
| -------- | --------------------------------------------- |
| Language | Python 3.10+                                  |
| ML       | Scikit-learn (`HistGradientBoostingRegressor`), Pandas, NumPy |
| App      | Streamlit, Plotly                             |
| Caching  | joblib (disk-cached trained model)            |
| Data     | CSV (train, test, stores, features) — Indian city store list, department-name mapping |

---

## Architecture

```
Raw CSVs (train/test/stores/features)
        |
        v
[ 1. Preprocessing ]  ->  merge tables, handle nulls, type casts, Indian city store list
        |
        v
[ 2. Feature Eng. ]   ->  lag features, rolling means, CPI, week-of-year, store size
        |
        v
[ 3. Model Training ] ->  HistGradientBoostingRegressor (disk-cached by data fingerprint)
        |
        v
[ 4. Evaluation ]     ->  SMAPE, MAE, RMSE on hold-out
        |
        v
[ 5. Streamlit App ]  ->  role-based login (Admin/User) -> per-store forecast viewer (₹)
```

---

## Login & Access Control

The dashboard is gated behind a login screen with two roles:

| Role  | Username                          | Password              | Access                          |
| ----- | ---------------------------------- | ---------------------- | -------------------------------- |
| Admin | `admin`                             | `password`             | All 45 stores, full dashboard    |
| User  | any store's city name (e.g. `mumbai`) | `<cityname>_manager` (e.g. `mumbai_manager`) | Only that store's data, no store dropdown |

> ⚠️ Credentials are hardcoded for demo purposes — replace with proper authentication (hashed passwords, environment variables, or `st.secrets`) before deploying anywhere beyond local use.

---

## Results

Evaluation on a held-out 20% test split (`random_state=42` for reproducibility).

| Metric    | Value       | Notes                                                                |
| --------- | ----------- | --------------------------------------------------------------------- |
| **SMAPE** | 58.06 %     | Symmetric MAPE (regular MAPE blows up on zero-sales weeks)            |
| **MAE**   | $4,719.62   | Mean Absolute Error on hold-out (base data is USD; dashboard converts to ₹ for display) |
| **RMSE**  | $8,167.94   | Root Mean Squared Error on hold-out                                   |

**Top features by importance:**

| Rank | Feature | Importance | Why it matters                                                                 |
| ---- | ------- | ---------- | ------------------------------------------------------------------------------- |
| 1    | Dept    | 0.718      | Department identity drives ~72% of variance — different depts have radically different volume profiles. |
| 2    | Size    | 0.184      | Larger stores → more footfall → more sales (near-linear relationship).          |
| 3    | Store   | 0.051      | Store-level effects beyond size (location, demographics).                       |
| 4    | Week    | 0.014      | Seasonal timing (holiday spikes, back-to-school).                               |
| 5    | CPI     | 0.012      | Macroeconomic context — captures consumer spending pressure.                    |

**Holiday effect varies significantly by department.** Across 3,205 store+department combinations, Holiday weeks beat Regular weeks 56% of the time overall — but the effect is department-specific:

- **Strongest holiday lift:** Men's Grooming (+101%), Maternity Wear (+102%), Kids & Baby Clothing (+74%), Eyewear & Sunglasses (+72%), Electronics & Mobiles (+28%)
- **Holiday actually lower:** Activewear (-69%), Pharmacy & Medicines (-58%), Cookware & Dining (-25%)

This is expected — only ~4 holiday weeks/year exist in the data (vs ~48 regular weeks), so per-store/department holiday averages can be noisy, and not every department spikes during holidays the way gift-driven categories do.

**What this means:** `Dept` alone explains most of the variance, which makes sense — a hardware department and a produce department in the same store have completely different baseline volumes. The current SMAPE (~58%) is driven largely by zero-sales weeks and high holiday-week volatility, which the model still under-predicts. This is a known limitation, not a bug — see **Future work** below for the planned fix (per-department models + better lag features).

---

## Project structure

```
ForecastX/
|-- app.py                  # Streamlit dashboard (login, forecasting, charts)
|-- train.csv               # Training data (sales history + DepartmentName)
|-- test.csv                # Hold-out test set
|-- stores.csv              # Store metadata (Indian cities, size, type)
|-- features.csv            # External features (CPI, fuel, holidays)
|-- .model_cache/           # Disk-cached trained model + feature importances (auto-generated, gitignored)
|-- sales_forecasting_banner.jpg
|-- requirements.txt        # Python dependencies
|-- .gitignore
|-- LICENSE
|-- README.md
```

---

## How to run

### 1. Clone

```
git clone https://github.com/Asiyaarab/ForecastX.git
cd ForecastX
```

### 2. Set up a virtual environment

```
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Launch the dashboard

```
streamlit run app.py
```

The app opens at `http://localhost:8501`. Log in as Admin (`admin` / `password`) or as a Store User (e.g. `mumbai` / `mumbai_manager`).

---

## Requirements

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
plotly>=5.20
joblib>=1.3
```

---

## What I learned

- The importance of **feature engineering** over model complexity — adding CPI and lag features moved the needle more than swapping GBM for a fancier model.
- How to translate a business problem ("we need better stock levels") into an ML objective ("minimize forecast error weighted by inventory cost").
- Why **end-to-end reproducibility** matters: same input data, same train/test split, same model config = same numbers every run.
- SMAPE is a better fit than MAPE for retail data with zero-sales weeks, since regular MAPE explodes on zero-denominator cases.
- Switching from `GradientBoostingRegressor` to `HistGradientBoostingRegressor` gave a large training-speed improvement on 400k+ rows with comparable accuracy — worth defaulting to for large tabular datasets.
- Caching a trained model to disk (fingerprinted on the training data, not the app code) avoids expensive retraining every time the UI changes — only the data should invalidate the cache.

---

## Future work

- [ ] Per-department models instead of one shared model — since `Dept` explains ~72% of variance, splitting the model by department should meaningfully reduce SMAPE
- [ ] Add LSTM / Prophet baselines for comparison
- [ ] REST API wrapper (FastAPI) so the BI team can query forecasts directly
- [ ] Confidence intervals on every prediction
- [ ] Replace hardcoded login credentials with proper authentication (hashed passwords / env vars / `st.secrets`)
- [ ] Configurable USD→INR exchange rate (currently a fixed constant in `app.py`) pulled from a live rate source

---

## Author

**Asiya Arab** — BCA, Shreyarth University — ML Intern @ Webify.ai

- Email: aashiyaarab39@gmail.com
- LinkedIn: linkedin.com/in/asiya-arab
- GitHub: github.com/Asiyaarab

---

## License

MIT — free to use, modify, and learn from.
