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
  - Streamlit `session_state` persistence and data clearing.
  - Interactive preview and Plotly time series visualization.
- **Phase 2: EDA & Diagnostics (stationarity, ACF/PACF, decomposition)** `[Upcoming]`
- **Phase 3: Model Zoo (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, Holt-Winters)** `[Upcoming]`
- **Phase 4: Model Selector UI with per-model plots** `[Upcoming]`
- **Phase 5: Forecast & Metrics Comparison** `[Upcoming]`
- **Phase 6: Deployment** `[Upcoming]`
- **Phase 7: Polish** `[Upcoming]`

---

## 📁 Project Structure

```text
atsa/
├── app.py                   # Streamlit web application entry point
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── modules/
    ├── __init__.py          # Python package initializer
    ├── data_loader.py       # File ingestion, date detection, time-series regularization
    └── utils.py             # Shared helper functions (placeholder for future phases)
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
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
