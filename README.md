# ForecastX — Multi-Store Sales Forecasting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

> End-to-end **weekly sales forecasting** for 45 retail locations using Gradient Boosting, macroeconomic CPI signals, and an interactive Streamlit dashboard for inventory planning.

![ForecastX Dashboard](sales_forecasting_banner.jpg)

---

## What it does

Predicts **weekly sales** for each of 45 retail stores 4-8 weeks ahead so inventory and supply-chain teams can plan stock, labor, and procurement with data instead of gut feel.

**Business value:** better forecasts -> less overstock (lower holding cost) + fewer stockouts (lost sales).

---

## Key features

- Per-store forecasts for 45 retail locations
- Gradient Boosting Regressor (GBM) with hyperparameter tuning
- Macroeconomic signals - CPI, fuel price, seasonality features
- Interactive Streamlit dashboard - pick a store, see historical + predicted sales, drill into feature importance
- Inventory decision support - converts forecast into recommended stock levels
- End-to-end pipeline - raw CSV to features to trained model to dashboard, all reproducible

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-learn, Pandas, NumPy |
| Model | Gradient Boosting Regressor (GBM) |
| App | Streamlit |
| Data | CSV (train, test, stores, features) |

---

## Architecture

```
Raw CSVs (train/test/stores/features)
        |
        v
[ 1. Preprocessing ]  ->  merge tables, handle nulls, type casts
        |
        v
[ 2. Feature Eng. ]   ->  lag features, rolling means, CPI, week-of-year, store size
        |
        v
[ 3. Model Training ] ->  Gradient Boosting + cross-validation
        |
        v
[ 4. Evaluation ]     ->  MAE, MAPE, RMSE on hold-out
        |
        v
[ 5. Streamlit App ]  ->  per-store forecast viewer
```

---

## Results

<!--
REPLACE WITH YOUR ACTUAL NUMBERS - re-run your eval script and paste the real metrics.
Hiring managers look for these numbers first.
-->

| Metric | Value | Notes |
|---|---|---|
| **MAPE** | `[FILL_IN] %` | Mean Absolute Percentage Error on hold-out |
| **MAE** | `[FILL_IN]` units | Mean Absolute Error |
| **RMSE** | `[FILL_IN]` units | Root Mean Squared Error |
| **Baseline beat** | `[FILL_IN] %` improvement vs naive last-week forecast |

**Top features by importance:**
1. `[FILL_IN - e.g. lag-4 sales]`
2. `[FILL_IN - e.g. CPI index]`
3. `[FILL_IN - e.g. store size]`

---

## Project structure

```
ForecastX/
|-- app.py                  # Streamlit dashboard
|-- train.csv               # Training data (sales history)
|-- test.csv                # Hold-out test set
|-- stores.csv              # Store metadata (size, type, region)
|-- features.csv            # External features (CPI, fuel, holidays)
|-- sales_forecasting_banner.jpg
|-- requirements.txt        # Python dependencies
|-- .gitignore
|-- LICENSE
|-- README.md
```

---

## How to run

### 1. Clone

```bash
git clone https://github.com/Asiyaarab/ForecastX.git
cd ForecastX
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows
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

---

## Requirements

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
```

---

## What I learned

- The importance of **feature engineering** over model complexity - adding CPI and lag features moved the needle more than swapping GBM for a fancier model.
- How to translate a business problem ("we need better stock levels") into an ML objective ("minimize forecast error weighted by inventory cost").
- Why **end-to-end reproducibility** matters: same input data, same train/test split, same model config = same numbers every run.

---

## Future work

- [ ] Add LSTM / Prophet baselines for comparison
- [ ] Per-store hyperparameter tuning (currently shared model)
- [ ] REST API wrapper (FastAPI) so the BI team can query forecasts directly
- [ ] Confidence intervals on every prediction

---

## Author

**Asiya Arab** - BCA, Shreyarth University - ML Intern @ Webify.ai
- Email: aashiyaarab39@gmail.com
- LinkedIn: linkedin.com/in/asiya-arab
- GitHub: github.com/Asiyaarab

---

## License

MIT - free to use, modify, and learn from.
