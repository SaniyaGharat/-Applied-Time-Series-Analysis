# Applied Time Series Analysis (ATSA)

An interactive Streamlit web application designed for end-to-end applied time series analysis, diagnostic checking, and forecasting.

---

## 🚀 Project Roadmap

The application is structured into the following sequential development phases:

- **Phase 1: Data Ingestion** `[Active]`
  - Robust file upload (CSV, XLSX, XLS) and reproducible demo dataset generator.
  - Column auto-detection (tightened date heuristics, numeric targets, integer-like Year parsing).
  - Day-first date parsing support (`dd/mm/yyyy`).
  - Preprocessing and regularization (duplicate aggregation by mean, frequency alignment, missing data filling).
  - Streamlit `session_state` persistence and genuine widget clearing.
  - Interactive preview and Plotly time series visualization.
- **Phase 2: EDA & Diagnostics (stationarity, ACF/PACF, decomposition)** `[Active]`
  - Augmented Dickey-Fuller (ADF) & KPSS tests with joint stationarity verdict.
  - Rolling statistics (mean and standard deviation).
  - Autocorrelation (ACF) & Partial Autocorrelation (PACF) stem plots with heuristic $(p, q)$ order hints.
  - Classical and STL time series decomposition with Hyndman trend and seasonal strength metrics.
  - Transformation and differencing pipeline (Log, Box-Cox, regular & seasonal differencing) with stationarity re-evaluation.
  - Supplemental diagnostics: seasonal distribution box plots, lag plots, distribution & Q-Q plots, Ljung-Box test.
- **Phase 3: Model Zoo (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, Holt-Winters)** `[Upcoming]`
- **Phase 4: Model Selector UI with per-model plots** `[Upcoming]`
- **Phase 5: Forecast & Metrics Comparison** `[Upcoming]`
- **Phase 6: Deployment** `[Upcoming]`
- **Phase 7: Polish** `[Upcoming]`

---

## 📁 Project Structure

```text
atsa/
├── app.py                   # Main Streamlit router with tab navigation
├── requirements.txt         # Project dependencies (Streamlit >= 1.60.0)
├── README.md                # Project documentation
├── modules/
│   ├── __init__.py          # Python package initializer
│   ├── data_loader.py       # Ingestion, date detection, frequency regularization
│   ├── diagnostics.py       # Pure statistical tests (ADF, KPSS, ACF, decomposition, transforms)
│   ├── plots.py             # Pure Plotly figure builders (template: plotly_white)
│   └── utils.py             # Shared helper functions
├── ui/
│   ├── __init__.py          # UI package initializer
│   ├── data_tab.py          # Data ingestion and preprocessing UI tab
│   └── eda_tab.py           # Exploratory data analysis & diagnostics UI tab
└── tests/
    ├── __init__.py          # Test suite initializer
    ├── test_a1.py           # Test for A1 frequency alignment resample fallback
    └── test_diagnostics.py  # Unit & smoke tests for Phase 2 diagnostics and plots
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd atsa
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   *Note: Requires Streamlit version >= 1.60.0.*
   ```bash
   pip install -r requirements.txt
   ```

4. **Run unit tests:**
   ```bash
   python tests/test_a1.py
   python tests/test_diagnostics.py
   ```

5. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
