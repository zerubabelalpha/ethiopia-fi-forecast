"""
Forecasting Utilities for Financial Inclusion Indicators

This module provides functions for trend-based and scenario-based forecasting
of financial inclusion indicators with uncertainty quantification.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from impact_model_utils import linear_impact_model, log_linear_impact_model


def prepare_time_series(df: pd.DataFrame, indicator_code: str, 
                        gender: str = 'all', location: str = 'national') -> pd.DataFrame:
    """
    Extract and prepare time series data for a specific indicator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Observations DataFrame
    indicator_code : str
        Indicator code to extract
    gender : str
        Gender filter (default: 'all')
    location : str
        Location filter (default: 'national')
        
    Returns:
    --------
    pd.DataFrame
        Time series with columns: year, value
    """
    filtered = df[
        (df['indicator_code'] == indicator_code) &
        (df['gender'] == gender) &
        (df['location'] == location)
    ].copy()
    
    filtered['year'] = pd.to_datetime(filtered['observation_date']).dt.year
    filtered = filtered.sort_values('year')
    
    return filtered[['year', 'value_numeric']].rename(columns={'value_numeric': 'value'})


def fit_linear_trend(ts_df: pd.DataFrame) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Fit linear trend: y = β₀ + β₁ × t
    
    Parameters:
    -----------
    ts_df : pd.DataFrame
        Time series with 'year' and 'value' columns
        
    Returns:
    --------
    Tuple[LinearRegression, Dict]
        Fitted model and metrics (R², RMSE, coefficients)
    """
    X = ts_df['year'].values.reshape(-1, 1)
    y = ts_df['value'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'intercept': model.intercept_,
        'slope': model.coef_[0],
        'n_obs': len(ts_df)
    }
    
    return model, metrics


def fit_loglinear_trend(ts_df: pd.DataFrame) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Fit log-linear trend: log(y) = β₀ + β₁ × t → y = exp(β₀ + β₁ × t)
    
    Parameters:
    -----------
    ts_df : pd.DataFrame
        Time series with 'year' and 'value' columns
        
    Returns:
    --------
    Tuple[LinearRegression, Dict]
        Fitted model and metrics
    """
    X = ts_df['year'].values.reshape(-1, 1)
    y = ts_df['value'].values
    
    # Log transform (add small constant to avoid log(0))
    y_log = np.log(y + 0.01)
    
    model = LinearRegression()
    model.fit(X, y_log)
    
    # Predict in original scale
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log) - 0.01
    
    # Calculate R² in original scale
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'intercept': model.intercept_,
        'slope': model.coef_[0],
        'n_obs': len(ts_df),
        'type': 'log-linear'
    }
    
    return model, metrics


def generate_trend_forecast(model: LinearRegression, 
                           forecast_years: List[int],
                           model_type: str = 'linear') -> pd.DataFrame:
    """
    Generate forecast using fitted trend model.
    
    Parameters:
    -----------
    model : LinearRegression
        Fitted trend model
    forecast_years : List[int]
        Years to forecast
    model_type : str
        'linear' or 'log-linear'
        
    Returns:
    --------
    pd.DataFrame
        Forecast with columns: year, forecast
    """
    X_future = np.array(forecast_years).reshape(-1, 1)
    
    if model_type == 'log-linear':
        y_pred_log = model.predict(X_future)
        y_pred = np.exp(y_pred_log) - 0.01
    else:
        y_pred = model.predict(X_future)
    
    # Clip to [0, 100] for percentage indicators
    y_pred = np.clip(y_pred, 0, 100)
    
    return pd.DataFrame({'year': forecast_years, 'forecast': y_pred})


def calculate_confidence_interval(ts_df: pd.DataFrame,
                                  model: LinearRegression,
                                  forecast_years: List[int],
                                  model_type: str = 'linear',
                                  confidence: float = 0.95) -> pd.DataFrame:
    """
    Calculate prediction confidence intervals.
    
    Parameters:
    -----------
    ts_df : pd.DataFrame
        Historical time series used for fitting
    model : LinearRegression
        Fitted model
    forecast_years : List[int]
        Years to forecast
    model_type : str
        'linear' or 'log-linear'
    confidence : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    pd.DataFrame
        Forecast with lower and upper bounds
    """
    X_hist = ts_df['year'].values.reshape(-1, 1)
    y_hist = ts_df['value'].values
    
    # Calculate residual standard error
    if model_type == 'log-linear':
        y_log = np.log(y_hist + 0.01)
        y_pred_log = model.predict(X_hist)
        residuals = y_log - y_pred_log
    else:
        y_pred = model.predict(X_hist)
        residuals = y_hist - y_pred
    
    n = len(ts_df)
    dof = n - 2  # degrees of freedom
    residual_std = np.std(residuals, ddof=2)
    
    # t-statistic for confidence level
    t_stat = stats.t.ppf((1 + confidence) / 2, dof)
    
    # Generate forecast
    X_future = np.array(forecast_years).reshape(-1, 1)
    
    if model_type == 'log-linear':
        y_pred_log = model.predict(X_future)
        y_pred = np.exp(y_pred_log) - 0.01
        
        # Uncertainty increases with distance from data
        X_mean = np.mean(X_hist)
        se = residual_std * np.sqrt(1 + 1/n + (X_future.flatten() - X_mean)**2 / np.sum((X_hist.flatten() - X_mean)**2))
        
        # Transform to original scale
        lower = np.exp(y_pred_log - t_stat * se) - 0.01
        upper = np.exp(y_pred_log + t_stat * se) - 0.01
    else:
        y_pred = model.predict(X_future)
        
        X_mean = np.mean(X_hist)
        se = residual_std * np.sqrt(1 + 1/n + (X_future.flatten() - X_mean)**2 / np.sum((X_hist.flatten() - X_mean)**2))
        
        lower = y_pred - t_stat * se
        upper = y_pred + t_stat * se
    
    # Clip to valid range
    y_pred = np.clip(y_pred, 0, 100)
    lower = np.clip(lower, 0, 100)
    upper = np.clip(upper, 0, 100)
    
    return pd.DataFrame({
        'year': forecast_years,
        'forecast': y_pred,
        'lower': lower,
        'upper': upper,
        'confidence': confidence
    })


def combine_trend_and_events(trend_forecast: pd.DataFrame,
                             events_df: pd.DataFrame,
                             impact_links_df: pd.DataFrame,
                             indicator_code: str,
                             baseline_year: int,
                             baseline_value: float,
                             model_type: str = 'log-linear') -> pd.DataFrame:
    """
    Combine trend forecast with event impacts.
    
    Parameters:
    -----------
    trend_forecast : pd.DataFrame
        Baseline trend forecast
    events_df : pd.DataFrame
        Events DataFrame
    impact_links_df : pd.DataFrame
        Impact links DataFrame
    indicator_code : str
        Indicator to forecast
    baseline_year : int
        Reference year for event impacts
    baseline_value : float
        Baseline value
    model_type : str
        'linear' or 'log-linear'
        
    Returns:
    --------
    pd.DataFrame
        Combined forecast
    """
    combined = trend_forecast.copy()
    combined['event_impact'] = 0.0
    combined['combined_forecast'] = combined['forecast']
    
    for idx, row in combined.iterrows():
        target_date = datetime(int(row['year']), 12, 31)
        
        if model_type == 'log-linear':
            event_pred = log_linear_impact_model(
                baseline_value, events_df, impact_links_df,
                indicator_code, target_date, 'gradual'
            )
        else:
            event_pred = linear_impact_model(
                baseline_value, events_df, impact_links_df,
                indicator_code, target_date, 'gradual'
            )
        
        event_impact = event_pred - baseline_value
        combined.at[idx, 'event_impact'] = event_impact
        combined.at[idx, 'combined_forecast'] = np.clip(row['forecast'] + event_impact, 0, 100)
    
    return combined


def generate_scenario_forecast(base_forecast: pd.DataFrame,
                               scenario_type: str,
                               impact_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Generate scenario-based forecast.
    
    Parameters:
    -----------
    base_forecast : pd.DataFrame
        Base forecast with event impacts
    scenario_type : str
        'optimistic', 'base', or 'pessimistic'
    impact_multiplier : float
        Multiplier for event impacts (optimistic: 1.3, base: 1.0, pessimistic: 0.7)
        
    Returns:
    --------
    pd.DataFrame
        Scenario forecast
    """
    scenario = base_forecast.copy()
    
    if 'event_impact' in scenario.columns:
        scenario['event_impact'] = scenario['event_impact'] * impact_multiplier
        scenario['scenario_forecast'] = np.clip(
            scenario['forecast'] + scenario['event_impact'], 0, 100
        )
    else:
        scenario['scenario_forecast'] = scenario['forecast']
    
    scenario['scenario'] = scenario_type
    
    return scenario


def backtest_model(ts_df: pd.DataFrame,
                  holdout_year: int,
                  model_type: str = 'linear') -> Dict[str, float]:
    """
    Backtest model by holding out recent data.
    
    Parameters:
    -----------
    ts_df : pd.DataFrame
        Full time series
    holdout_year : int
        Year to hold out for testing
    model_type : str
        'linear' or 'log-linear'
        
    Returns:
    --------
    Dict[str, float]
        Backtest metrics (actual, predicted, error, error_pct)
    """
    # Split data
    train = ts_df[ts_df['year'] < holdout_year]
    test = ts_df[ts_df['year'] == holdout_year]
    
    if len(test) == 0:
        return {'error': 'No test data available'}
    
    # Fit model on training data
    if model_type == 'log-linear':
        model, _ = fit_loglinear_trend(train)
    else:
        model, _ = fit_linear_trend(train)
    
    # Predict holdout year
    forecast = generate_trend_forecast(model, [holdout_year], model_type)
    
    actual = test['value'].values[0]
    predicted = forecast['forecast'].values[0]
    error = predicted - actual
    error_pct = (error / actual) * 100
    
    return {
        'holdout_year': holdout_year,
        'actual': actual,
        'predicted': predicted,
        'error': error,
        'error_pct': error_pct,
        'model_type': model_type
    }


def calculate_scenario_range(scenarios: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate range across scenarios.
    
    Parameters:
    -----------
    scenarios : List[pd.DataFrame]
        List of scenario forecasts
        
    Returns:
    --------
    pd.DataFrame
        Combined with min, max, range
    """
    combined = pd.concat(scenarios, ignore_index=True)
    
    summary = combined.groupby('year').agg({
        'scenario_forecast': ['min', 'max', 'mean']
    }).reset_index()
    
    summary.columns = ['year', 'min', 'max', 'mean']
    summary['range'] = summary['max'] - summary['min']
    
    return summary
