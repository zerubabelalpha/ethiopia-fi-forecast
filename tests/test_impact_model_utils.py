
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from impact_model_utils import *


class TestDecayFunctions:
    """Test decay function behavior"""
    
    def test_decay_immediate(self):
        """Test immediate decay function"""
        lag = 12
        assert decay_immediate(0, lag) == 0.0
        assert decay_immediate(6, lag) == 0.0
        assert decay_immediate(12, lag) == 1.0
        assert decay_immediate(24, lag) == 1.0
    
    def test_decay_gradual(self):
        """Test gradual decay function"""
        lag = 12
        assert decay_gradual(0, lag) == 0.0
        assert abs(decay_gradual(6, lag) - 0.5) < 0.01
        assert decay_gradual(12, lag) == 1.0
        assert decay_gradual(24, lag) == 1.0
    
    def test_decay_exponential(self):
        """Test exponential decay function"""
        lag = 12
        assert decay_exponential(0, lag) == 0.0
        assert 0 < decay_exponential(6, lag) < 1.0
        assert decay_exponential(100, lag) > 0.99


class TestImpactCombination:
    """Test impact combination functions"""
    
    def test_combine_impacts_additive(self):
        """Test additive combination"""
        impacts = [5.0, 10.0, -3.0]
        result = combine_impacts_additive(impacts)
        assert result == 12.0
    
    def test_combine_impacts_multiplicative(self):
        """Test multiplicative combination"""
        impacts = [10.0, 20.0]  # +10% and +20%
        result = combine_impacts_multiplicative(impacts)
        # (1.1 * 1.2 - 1) * 100 = 32%
        assert abs(result - 32.0) < 0.01


class TestAssociationMatrix:
    """Test association matrix construction"""
    
    def test_build_association_matrix(self):
        """Test matrix construction with known inputs"""
        # Create mock data
        events_df = pd.DataFrame({
            'record_id': ['EVT_001', 'EVT_002'],
            'indicator': ['Event 1', 'Event 2'],
            'event_date': [datetime(2021, 1, 1), datetime(2022, 1, 1)]
        })
        
        impact_links_df = pd.DataFrame({
            'event_id': ['EVT_001', 'EVT_001', 'EVT_002'],
            'related_indicator': ['IND_A', 'IND_B', 'IND_A'],
            'impact_estimate': [15.0, 10.0, 20.0],
            'impact_direction': ['increase', 'increase', 'decrease']
        })
        
        indicators = ['IND_A', 'IND_B', 'IND_C']
        
        matrix = build_association_matrix(events_df, impact_links_df, indicators)
        
        assert matrix.shape == (2, 3)
        assert matrix.loc['EVT_001', 'IND_A'] == 15.0
        assert matrix.loc['EVT_001', 'IND_B'] == 10.0
        assert matrix.loc['EVT_002', 'IND_A'] == -20.0  # Negative for decrease
        assert pd.isna(matrix.loc['EVT_001', 'IND_C'])


class TestLinearModel:
    """Test linear impact model"""
    
    def test_linear_model_single_event(self):
        """Test linear model with single event"""
        baseline = 10.0
        
        events_df = pd.DataFrame({
            'record_id': ['EVT_001'],
            'event_date': [datetime(2021, 1, 1)]
        })
        
        impact_links_df = pd.DataFrame({
            'event_id': ['EVT_001'],
            'related_indicator': ['IND_A'],
            'impact_estimate': [5.0],
            'lag_months': [12.0]
        })
        
        # Predict 12 months after event (full impact)
        target_date = datetime(2022, 1, 1)
        result = linear_impact_model(baseline, events_df, impact_links_df, 
                                     'IND_A', target_date, 'gradual')
        
        assert result == 15.0  # baseline + impact
    
    def test_linear_model_before_event(self):
        """Test linear model before event occurs"""
        baseline = 10.0
        
        events_df = pd.DataFrame({
            'record_id': ['EVT_001'],
            'event_date': [datetime(2021, 1, 1)]
        })
        
        impact_links_df = pd.DataFrame({
            'event_id': ['EVT_001'],
            'related_indicator': ['IND_A'],
            'impact_estimate': [5.0],
            'lag_months': [12.0]
        })
        
        # Predict before event
        target_date = datetime(2020, 1, 1)
        result = linear_impact_model(baseline, events_df, impact_links_df, 
                                     'IND_A', target_date, 'gradual')
        
        assert result == baseline  # No impact yet


class TestLogLinearModel:
    """Test log-linear impact model"""
    
    def test_loglinear_model_single_event(self):
        """Test log-linear model with single event"""
        baseline = 10.0
        
        events_df = pd.DataFrame({
            'record_id': ['EVT_001'],
            'event_date': [datetime(2021, 1, 1)]
        })
        
        impact_links_df = pd.DataFrame({
            'event_id': ['EVT_001'],
            'related_indicator': ['IND_A'],
            'impact_estimate': [20.0],  # +20%
            'lag_months': [12.0]
        })
        
        # Predict 12 months after event (full impact)
        target_date = datetime(2022, 1, 1)
        result = log_linear_impact_model(baseline, events_df, impact_links_df, 
                                         'IND_A', target_date, 'gradual')
        
        assert abs(result - 12.0) < 0.01  # baseline * 1.2


class TestValidation:
    """Test validation metrics"""
    
    def test_validate_predictions(self):
        """Test validation metric calculation"""
        predictions = pd.Series([10, 20, 30])
        actuals = pd.Series([12, 18, 32])
        
        metrics = validate_predictions(predictions, actuals)
        
        assert metrics['N'] == 3
        assert metrics['MAE'] == 2.0  # Mean of [2, 2, 2]
        assert abs(metrics['RMSE'] - 2.0) < 0.01


class TestConfidenceScore:
    """Test confidence scoring"""
    
    def test_confidence_from_evidence(self):
        """Test confidence based on evidence type"""
        assert calculate_confidence_score('empirical') == 'high'
        assert calculate_confidence_score('literature') == 'medium'
        assert calculate_confidence_score('theoretical') == 'low'
    
    def test_confidence_from_validation(self):
        """Test confidence adjusted by validation error"""
        assert calculate_confidence_score('theoretical', validation_error=5.0) == 'high'
        assert calculate_confidence_score('empirical', validation_error=30.0) == 'low'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
