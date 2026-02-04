"""
Impact Model Utilities for Event-Indicator Analysis

This module provides utility functions for modeling the impact of events
on financial inclusion indicators using both linear and log-linear approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def parse_impact_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and structure impact_link records from unified data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified data containing all record types
        
    Returns:
    --------
    pd.DataFrame
        Structured DataFrame with impact link information
    """
    impact_links = df[df['record_type'] == 'impact_link'].copy()
    
    # Parse parent_id to get event reference
    impact_links['event_id'] = impact_links['parent_id']
    
    # Convert impact_estimate to numeric
    impact_links['impact_estimate'] = pd.to_numeric(impact_links['impact_estimate'], errors='coerce')
    impact_links['lag_months'] = pd.to_numeric(impact_links['lag_months'], errors='coerce')
    
    return impact_links[['record_id', 'event_id', 'related_indicator', 'impact_direction', 
                         'impact_magnitude', 'impact_estimate', 'lag_months', 
                         'evidence_basis', 'comparable_country', 'notes']]


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all event records with metadata.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified data containing all record types
        
    Returns:
    --------
    pd.DataFrame
        Events with id, name, date, category, description
    """
    events = df[df['record_type'] == 'event'].copy()
    
    # Convert observation_date to datetime
    events['event_date'] = pd.to_datetime(events['observation_date'])
    
    return events[['record_id', 'indicator', 'event_date', 'category', 
                   'value_text', 'notes', 'original_text']]


def extract_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all indicators with their historical observations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified data containing all record types
        
    Returns:
    --------
    pd.DataFrame
        Indicators with observations over time
    """
    observations = df[df['record_type'] == 'observation'].copy()
    
    # Convert observation_date to datetime
    observations['observation_date'] = pd.to_datetime(observations['observation_date'])
    observations['value_numeric'] = pd.to_numeric(observations['value_numeric'], errors='coerce')
    
    return observations[['record_id', 'indicator_code', 'indicator', 'pillar', 
                         'value_numeric', 'observation_date', 'unit', 'value_type',
                         'gender', 'location', 'source_name', 'confidence']]


def build_association_matrix(events_df: pd.DataFrame, 
                             impact_links_df: pd.DataFrame,
                             indicators: List[str]) -> pd.DataFrame:
    """
    Build event × indicator association matrix showing impact estimates.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Events DataFrame from extract_events()
    impact_links_df : pd.DataFrame
        Impact links DataFrame from parse_impact_links()
    indicators : List[str]
        List of indicator codes to include as columns
        
    Returns:
    --------
    pd.DataFrame
        Matrix with events as rows, indicators as columns, impact estimates as values
    """
    # Create empty matrix
    event_ids = events_df['record_id'].unique()
    matrix = pd.DataFrame(index=event_ids, columns=indicators, dtype=float)
    
    # Fill in documented impacts
    for _, link in impact_links_df.iterrows():
        event_id = link['event_id']
        indicator = link['related_indicator']
        impact = link['impact_estimate']
        
        if event_id in matrix.index and indicator in matrix.columns:
            # Handle direction: negative for decrease
            if link['impact_direction'] == 'decrease':
                impact = -abs(impact)
            matrix.loc[event_id, indicator] = impact
    
    return matrix


def decay_immediate(t: float, lag: float) -> float:
    """
    Immediate impact: full effect after lag period, constant thereafter.
    
    Parameters:
    -----------
    t : float
        Time since event (in months)
    lag : float
        Lag period (in months)
        
    Returns:
    --------
    float
        Decay factor (0 to 1)
    """
    return 1.0 if t >= lag else 0.0


def decay_gradual(t: float, lag: float) -> float:
    """
    Gradual impact: linear ramp-up over lag period.
    
    Parameters:
    -----------
    t : float
        Time since event (in months)
    lag : float
        Lag period (in months)
        
    Returns:
    --------
    float
        Decay factor (0 to 1)
    """
    if t <= 0:
        return 0.0
    elif t >= lag:
        return 1.0
    else:
        return t / lag


def decay_exponential(t: float, lag: float, rate: float = 0.5) -> float:
    """
    Exponential approach: asymptotic approach to full impact.
    
    Parameters:
    -----------
    t : float
        Time since event (in months)
    lag : float
        Lag period (in months) - time to reach ~63% of full impact
    rate : float
        Decay rate parameter (default 0.5)
        
    Returns:
    --------
    float
        Decay factor (0 to 1)
    """
    if t <= 0:
        return 0.0
    return 1.0 - np.exp(-rate * t / lag)


def combine_impacts_additive(impacts: List[float]) -> float:
    """
    Combine multiple event impacts additively (for linear model).
    
    Parameters:
    -----------
    impacts : List[float]
        List of impact values from different events
        
    Returns:
    --------
    float
        Combined impact
    """
    return sum(impacts)


def combine_impacts_multiplicative(impacts: List[float]) -> float:
    """
    Combine multiple event impacts multiplicatively (for log-linear model).
    
    Parameters:
    -----------
    impacts : List[float]
        List of impact percentages from different events
        
    Returns:
    --------
    float
        Combined impact percentage
    """
    # Convert percentages to multipliers, combine, convert back
    multiplier = 1.0
    for impact in impacts:
        multiplier *= (1.0 + impact / 100.0)
    return (multiplier - 1.0) * 100.0


def linear_impact_model(baseline: float,
                        events_df: pd.DataFrame,
                        impact_links_df: pd.DataFrame,
                        indicator_code: str,
                        target_date: datetime,
                        decay_function: str = 'gradual') -> float:
    """
    Apply linear impact model to predict indicator value.
    
    Parameters:
    -----------
    baseline : float
        Baseline indicator value (before events)
    events_df : pd.DataFrame
        Events DataFrame with event_date
    impact_links_df : pd.DataFrame
        Impact links for this indicator
    indicator_code : str
        Indicator code to predict
    target_date : datetime
        Date to predict value for
    decay_function : str
        Decay function type: 'immediate', 'gradual', or 'exponential'
        
    Returns:
    --------
    float
        Predicted indicator value
    """
    # Filter impact links for this indicator
    relevant_links = impact_links_df[impact_links_df['related_indicator'] == indicator_code]
    
    # Calculate total impact
    total_impact = 0.0
    
    for _, link in relevant_links.iterrows():
        event_id = link['event_id']
        event_row = events_df[events_df['record_id'] == event_id]
        
        if event_row.empty:
            continue
            
        event_date = event_row.iloc[0]['event_date']
        
        # Calculate time since event in months
        months_since = (target_date - event_date).days / 30.44
        
        if months_since < 0:
            continue  # Event hasn't happened yet
        
        # Get impact estimate and lag
        impact_estimate = link['impact_estimate']
        lag = link['lag_months']
        
        # Apply decay function
        if decay_function == 'immediate':
            decay = decay_immediate(months_since, lag)
        elif decay_function == 'gradual':
            decay = decay_gradual(months_since, lag)
        elif decay_function == 'exponential':
            decay = decay_exponential(months_since, lag)
        else:
            decay = 1.0
        
        # Add to total impact (percentage points for linear model)
        total_impact += impact_estimate * decay
    
    # Apply impact to baseline (additive for linear model)
    return baseline + total_impact


def log_linear_impact_model(baseline: float,
                            events_df: pd.DataFrame,
                            impact_links_df: pd.DataFrame,
                            indicator_code: str,
                            target_date: datetime,
                            decay_function: str = 'gradual') -> float:
    """
    Apply log-linear impact model to predict indicator value.
    
    Parameters:
    -----------
    baseline : float
        Baseline indicator value (before events)
    events_df : pd.DataFrame
        Events DataFrame with event_date
    impact_links_df : pd.DataFrame
        Impact links for this indicator
    indicator_code : str
        Indicator code to predict
    target_date : datetime
        Date to predict value for
    decay_function : str
        Decay function type: 'immediate', 'gradual', or 'exponential'
        
    Returns:
    --------
    float
        Predicted indicator value
    """
    if baseline <= 0:
        return 0.0  # Log-linear requires positive baseline
    
    # Filter impact links for this indicator
    relevant_links = impact_links_df[impact_links_df['related_indicator'] == indicator_code]
    
    # Calculate cumulative multiplier
    cumulative_multiplier = 1.0
    
    for _, link in relevant_links.iterrows():
        event_id = link['event_id']
        event_row = events_df[events_df['record_id'] == event_id]
        
        if event_row.empty:
            continue
            
        event_date = event_row.iloc[0]['event_date']
        
        # Calculate time since event in months
        months_since = (target_date - event_date).days / 30.44
        
        if months_since < 0:
            continue  # Event hasn't happened yet
        
        # Get impact estimate and lag
        impact_estimate = link['impact_estimate']
        lag = link['lag_months']
        
        # Apply decay function
        if decay_function == 'immediate':
            decay = decay_immediate(months_since, lag)
        elif decay_function == 'gradual':
            decay = decay_gradual(months_since, lag)
        elif decay_function == 'exponential':
            decay = decay_exponential(months_since, lag)
        else:
            decay = 1.0
        
        # Convert percentage to multiplier and apply decay
        event_multiplier = 1.0 + (impact_estimate / 100.0) * decay
        cumulative_multiplier *= event_multiplier
    
    # Apply cumulative multiplier to baseline
    return baseline * cumulative_multiplier


def validate_predictions(predictions: pd.Series, 
                        actuals: pd.Series) -> Dict[str, float]:
    """
    Calculate validation metrics comparing predictions to actual values.
    
    Parameters:
    -----------
    predictions : pd.Series
        Predicted values
    actuals : pd.Series
        Actual observed values
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with MAE, RMSE, MAPE metrics
    """
    # Align series
    aligned = pd.DataFrame({'pred': predictions, 'actual': actuals}).dropna()
    
    if len(aligned) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    
    pred = aligned['pred'].values
    actual = aligned['actual'].values
    
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    
    # MAPE (avoid division by zero)
    mape_values = np.abs((actual - pred) / actual) * 100
    mape_values = mape_values[np.isfinite(mape_values)]
    mape = np.mean(mape_values) if len(mape_values) > 0 else np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'N': len(aligned)
    }


def calculate_confidence_score(evidence_basis: str, 
                               validation_error: Optional[float] = None) -> str:
    """
    Calculate confidence level based on evidence basis and validation results.
    
    Parameters:
    -----------
    evidence_basis : str
        Type of evidence: 'empirical', 'literature', 'theoretical', 'expert'
    validation_error : Optional[float]
        MAPE from validation (if available)
        
    Returns:
    --------
    str
        Confidence level: 'high', 'medium', 'low'
    """
    # Base confidence from evidence type
    base_confidence = {
        'empirical': 'high',
        'literature': 'medium',
        'theoretical': 'low',
        'expert': 'medium'
    }.get(evidence_basis, 'low')
    
    # Adjust based on validation error if available
    if validation_error is not None:
        if validation_error < 10:
            return 'high'
        elif validation_error < 25:
            return 'medium'
        else:
            return 'low'
    
    return base_confidence
