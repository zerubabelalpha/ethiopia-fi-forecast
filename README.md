# Ethiopia Financial Inclusion Forecasting System


## Overview
This repository contains the development and analysis of a **Financial Inclusion Forecasting System for Ethiopia**. This project aims to provide deep insights into the drivers of financial inclusion and predict future rates for **Access** (Account Ownership) and **Usage** (Digital Payment Adoption) for the period 2025-2027.


## Key Features
*   **Data Enrichment:** Integrates NBE stability reports, operator data, and policy event logs.
*   **EDA Notebook:** Detailed Exploratory Data Analysis mapping the "Access-Usage" gap.
*   **Automated Results Extraction:** Script to programmatically extract insights from Jupyter notebooks.
*   **Reporting:** Comprehensive blog-style analysis reports for stakeholders.
*   **CI/CD:** Automated testing via GitHub Actions.

##  Project Structure
```text
ethiopia-fi-forecast/
├── .github/workflows/       # CI/CD pipelines (Unittests)
├── data/
│   └── raw/                 # Enriched unified dataset (CSV)
├── notebooks/
│   └── financial_inclusion_analysis.ipynb  # Core EDA & Visualization
├── reports/
│   └── REPORT.md            # Comprehensive sector report
├── scripts/
│   ├── extract_notebook_results.py # Data extraction utility
│   └── simple_test.py       # Dependency & module verification
├── tests/
│   └── simple_test.py       # Unit testing suite
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zerubabelalpha/ethiopia-fi-forecast.git
   cd ethiopia-fi-forecast
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Interactive Dashboard
Launch the simplified Streamlit dashboard:
```bash
cd dashboard
streamlit run app.py
```

**Features:**
- **Overview**: Key metrics, P2P/ATM crossover, growth charts
- **Trends**: Interactive time series with filtering
- **Forecasts**: Predictions with confidence intervals
- **Projections**: Scenario analysis and target tracking

### 3. Run Tests
```bash
python -m pytest tests/test_forecast_utils.py -v
```

## Contributing
For contributions, please fork the repository, create a new branch, and submit a Pull Request.
 Ensure all tests pass before submission.



