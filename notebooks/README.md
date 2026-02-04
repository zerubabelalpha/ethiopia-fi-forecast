# Analysis & Modeling Notebooks

This directory contains the Jupyter notebooks used for data exploration, impact modeling, and forecasting.

## Notebook Overview

### 1. [Data Exploration](file:///c:/Users/Acer/Documents/KAIM_PROJECT/TEST/ethiopia-fi-forecast/notebooks/data_exploration.ipynb)
- **Purpose**: Initial data validation and cleaning.
- **Key Activities**: Checking data types, identifying missing values, and validating column structures.

### 2. [Financial Inclusion Analysis (EDA)](file:///c:/Users/Acer/Documents/KAIM_PROJECT/TEST/ethiopia-fi-forecast/notebooks/financial_inclusion_analysis.ipynb)
- **Purpose**: Comprehensive Exploratory Data Analysis.
- **Key Activities**:
    - Mapping the "Access-Usage" gap.
    - Visualizing historical trends for core indicators (Account Ownership, Mobile Money).
    - Analyzing growth rates and temporal patterns.
- **Output**: Visualizations of historical trends and growth drivers.

### 3. [Event-Indicator Impact Analysis](file:///c:/Users/Acer/Documents/KAIM_PROJECT/TEST/ethiopia-fi-forecast/notebooks/event_indicator_impact_analysis.ipynb)
- **Purpose**: Modeling the relationship between policy events and indicators.
- **Key Activities**:
    - Building an Association Matrix between events and indicators.
    - Estimating impact magnitudes and lags.
    - Testing linear vs. log-linear impact models.
- **Output**: Impact coefficients and refined estimates used for forecasting.

### 4. [Financial Inclusion Forecast](file:///c:/Users/Acer/Documents/KAIM_PROJECT/TEST/ethiopia-fi-forecast/notebooks/financial_inclusion_forecast.ipynb)
- **Purpose**: Generating future projections for 2025-2030.
- **Key Activities**:
    - Baseline trend modeling.
    - Event-augmented forecasting (Trend + Policy Effects).
    - Scenario Analysis (Optimistic, Base, Pessimistic).
    - Uncertainty quantification with 95% confidence intervals.
- **Output**: `data/processed/account_ownership_forecast.csv`.

## Running the Notebooks
To run these notebooks locally:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`.
2. Start Jupyter Notebook: `jupyter notebook`.
3. Open the desired notebook file.
