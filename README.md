# Ethiopia Financial Inclusion Forecasting System

## Overview
This repository contains a comprehensive **Financial Inclusion Forecasting System for Ethiopia**. It provides data-driven insights into the drivers of financial inclusion and predicts future rates for **Access** (Account Ownership) and **Usage** (Digital Payment Adoption) for the period 2025-2030.

The system uses historical data from Global Findex, EthSwitch, and Telebirr, combined with evidence-based impact modeling of policy events (e.g., Telebirr launch, Fayda Digital ID, EthioPay interoperability).

## Key Features
*   **Data Enrichment:** Integrated NBE stability reports, operator data, and policy event logs.
*   **Impact Analysis:** Modeled the effect of key events on financial indicators using linear and log-linear approaches.
*   **Predictive Modeling:** Trend-augmented and event-augmented forecasting with 95% confidence intervals.
*   **Scenario Analysis:** Optimistic, Base, and Pessimistic projections based on policy implementation.
*   **Interactive Dashboard:** Streamlit-powered visualization tool for stakeholder exploration.
*   **CI/CD:** Automated testing suite via GitHub Actions.

## Project Structure
```text
ethiopia-fi-forecast/
├── dashboard/               # Interactive Streamlit Application
│   └── app.py               # Simplified single-file dashboard
├── data/
│   ├── raw/                 # Enriched unified dataset and reference codes
│   └── processed/           # Generated forecast results (CSV)
├── notebooks/               # Analysis and modeling notebooks
│   ├── data_exploration.ipynb
│   ├── financial_inclusion_analysis.ipynb
│   ├── event_indicator_impact_analysis.ipynb
│   └── financial_inclusion_forecast.ipynb
├── reports/                 # Analysis reports and visualizations
├── scripts/                 # Utility scripts for data extraction
├── src/                     # Core model and forecast logic
│   ├── impact_model_utils.py # Impact modeling functions
│   └── forecast_utils.py     # Forecasting and trend logic
├── tests/                   # Unit test suite
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zerubabelalpha/ethiopia-fi-forecast.git
   cd ethiopia-fi-forecast
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Interactive Dashboard
Launch the Streamlit dashboard to explore data, trends, and forecasts:
```bash
cd dashboard
streamlit run app.py
```

### 2. Analysis Notebooks
Explore the modeling process sequentially:
- Start with `notebooks/financial_inclusion_analysis.ipynb` for EDA.
- See `notebooks/event_indicator_impact_analysis.ipynb` for impact estimates.
- Run `notebooks/financial_inclusion_forecast.ipynb` to update projections.

### 3. Run Tests
Verify the code integrity:
```bash
python -m pytest tests/test_impact_model_utils.py -v
python -m pytest tests/test_forecast_utils.py -v
```

## Contributing
Professional contributions are welcome. Please fork the repository, create a branch, and ensure all unit tests pass before submitting a Pull Request.



