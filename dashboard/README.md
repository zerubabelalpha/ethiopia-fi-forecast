# Financial Inclusion Interactive Dashboard

This directory contains the interactive dashboard for exploring Ethiopia's financial inclusion data and forecasts.

## Features

### 1. Overview
- **Key Metrics**: Real-time view of Account Ownership, Mobile Money penetration, and P2P transaction volumes.
- **P2P/ATM Ratio**: Track the historic crossover where digital P2P transactions surpass physical ATM cash-outs.
- **Growth Highlights**: Visual summaries of year-over-year progress.

### 2. Trends
- **Interactive Time Series**: Plot multiple indicators simultaneously.
- **Dynamic Filtering**: Select specific date ranges and indicators to explore.
- **Channel Comparisons**: Compare Telebirr vs. M-Pesa growth or Digital vs. Cash usage.

### 3. Forecasts
- **Confidence Intervals**: View projections with 95% uncertainty bands.
- **Scenario Comparison**: Choose between Baseline, Event-Augmented, and Target-based models.
- **Milestone Tracking**: Projected dates for reaching 60% and 70% inclusion targets.

### 4. Inclusion Projections
- **Scenario Selector**: Toggle between **Optimistic**, **Base**, and **Pessimistic** policy implementation scenarios.
- **Consortium Insights**: Data-driven answers to key regional and sectoral questions.

## Getting Started

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install streamlit plotly pandas
```

### Running the App
From the project root:
```bash
cd dashboard
streamlit run app.py
```

## Technical Details
- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Source**: `data/raw/ethiopia_fi_unified_data.csv` and `data/processed/account_ownership_forecast.csv`
- **Architecture**: Simplified single-file implementation for high maintainability.
