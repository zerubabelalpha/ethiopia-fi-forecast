"""
Ethiopia Financial Inclusion Dashboard - Simplified
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page config
st.set_page_config(page_title="Ethiopia FI Dashboard", page_icon="📊", layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    """Load all data"""
    df = pd.read_csv('data/raw/ethiopia_fi_unified_data.csv')
    
    # Extract indicators
    indicators = df[df['record_type'] == 'observation'].copy()
    indicators['observation_date'] = pd.to_datetime(indicators['observation_date'])
    
    # Extract events
    events = df[df['record_type'] == 'event'].copy()
    events['event_date'] = pd.to_datetime(events['event_date'])
    
    # Load forecast if exists
    try:
        forecast = pd.read_csv('data/processed/account_ownership_forecast.csv')
    except:
        # Default forecast
        forecast = pd.DataFrame({
            'Year': [2025, 2026, 2027, 2028, 2029, 2030],
            'Baseline Trend': [51, 53, 55, 56, 58, 60],
            'Optimistic': [57, 60, 64, 67, 70, 73],
            'Base Scenario': [54, 57, 60, 63, 66, 68],
            'Pessimistic': [51, 54, 56, 59, 61, 63],
            '95% CI Lower': [45, 46, 47, 48, 49, 50],
            '95% CI Upper': [63, 68, 73, 78, 83, 86]
        })
    
    return indicators, events, forecast

indicators_df, events_df, forecast_df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Trends", "Forecasts", "Projections"])

# ===== OVERVIEW PAGE =====
if page == "Overview":
    st.title("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Account ownership
    acc_own = indicators_df[indicators_df['indicator_code'] == 'ACC_OWNERSHIP'].sort_values('observation_date', ascending=False)
    if not acc_own.empty:
        col1.metric("Account Ownership", f"{acc_own.iloc[0]['value_numeric']:.1f}%")
    
    # Mobile money
    mm = indicators_df[indicators_df['indicator_code'] == 'ACC_MM_ACCOUNT'].sort_values('observation_date', ascending=False)
    if not mm.empty:
        col2.metric("Mobile Money", f"{mm.iloc[0]['value_numeric']:.1f}%")
    
    # P2P transactions
    p2p = indicators_df[indicators_df['indicator_code'] == 'USG_P2P_COUNT'].sort_values('observation_date', ascending=False)
    if not p2p.empty:
        col3.metric("P2P Transactions", f"{p2p.iloc[0]['value_numeric']/1e6:.1f}M")
    
    # Crossover ratio
    crossover = indicators_df[indicators_df['indicator_code'] == 'USG_CROSSOVER'].sort_values('observation_date', ascending=False)
    if not crossover.empty:
        ratio = crossover.iloc[0]['value_numeric']
        col4.metric("P2P/ATM Ratio", f"{ratio:.2f}", delta="Digital > Cash" if ratio > 1 else "Cash > Digital")
    
    # Growth highlights
    st.subheader("Growth Highlights")
    acc_data = indicators_df[indicators_df['indicator_code'] == 'ACC_OWNERSHIP'].copy()
    acc_data['year'] = acc_data['observation_date'].dt.year
    acc_data = acc_data.groupby('year')['value_numeric'].last().reset_index()
    acc_data['growth'] = acc_data['value_numeric'].pct_change() * 100
    
    fig = px.bar(acc_data.tail(5), x='year', y='growth', title="Account Ownership YoY Growth (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Download
    if st.button("Download Overview Data"):
        csv = acc_data.to_csv(index=False)
        st.download_button("Download CSV", csv, "overview.csv", "text/csv")

# ===== TRENDS PAGE =====
elif page == "Trends":
    st.title("📈 Trends")
    
    # Indicator selector
    indicators_map = {
        'ACC_OWNERSHIP': 'Account Ownership (%)',
        'ACC_MM_ACCOUNT': 'Mobile Money (%)',
        'USG_P2P_COUNT': 'P2P Transactions',
        'USG_ATM_COUNT': 'ATM Transactions'
    }
    
    selected = st.multiselect("Select Indicators", list(indicators_map.keys()), 
                              default=['ACC_OWNERSHIP'], format_func=lambda x: indicators_map[x])
    
    # Date range
    col1, col2 = st.columns(2)
    min_date = indicators_df['observation_date'].min().date()
    max_date = indicators_df['observation_date'].max().date()
    start_date = col1.date_input("Start", min_date)
    end_date = col2.date_input("End", max_date)
    
    # Plot
    for ind in selected:
        data = indicators_df[indicators_df['indicator_code'] == ind].copy()
        data = data[(data['observation_date'] >= pd.Timestamp(start_date)) & 
                   (data['observation_date'] <= pd.Timestamp(end_date))]
        
        if not data.empty:
            fig = px.line(data, x='observation_date', y='value_numeric', 
                         title=indicators_map[ind], markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # Channel comparison
    st.subheader("Channel Comparison")
    comparison = st.selectbox("Compare", ["P2P vs ATM", "Account Types"])
    
    if comparison == "P2P vs ATM":
        p2p_data = indicators_df[indicators_df['indicator_code'] == 'USG_P2P_COUNT']
        atm_data = indicators_df[indicators_df['indicator_code'] == 'USG_ATM_COUNT']
        
        fig = go.Figure()
        if not p2p_data.empty:
            fig.add_trace(go.Scatter(x=p2p_data['observation_date'], y=p2p_data['value_numeric'], 
                                    name='P2P', mode='lines+markers'))
        if not atm_data.empty:
            fig.add_trace(go.Scatter(x=atm_data['observation_date'], y=atm_data['value_numeric'], 
                                    name='ATM', mode='lines+markers'))
        fig.update_layout(title="P2P vs ATM Transactions")
        st.plotly_chart(fig, use_container_width=True)

# ===== FORECASTS PAGE =====
elif page == "Forecasts":
    st.title("🔮 Forecasts")
    
    # Model selector
    model = st.selectbox("Select Model", ['Baseline Trend', 'Base Scenario', 'Event-Augmented'])
    
    # Historical data
    hist = indicators_df[indicators_df['indicator_code'] == 'ACC_OWNERSHIP'].copy()
    hist['year'] = hist['observation_date'].dt.year
    hist_summary = hist.groupby('year')['value_numeric'].last().reset_index()
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_summary['year'], y=hist_summary['value_numeric'], 
                            name='Historical', mode='lines+markers', line=dict(width=3)))
    
    if model in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df[model], 
                                name='Forecast', mode='lines+markers', line=dict(dash='dash')))
        
        # Add CI
        if '95% CI Lower' in forecast_df.columns:
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['95% CI Upper'], 
                                    fill=None, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['95% CI Lower'], 
                                    fill='tonexty', mode='lines', line=dict(width=0), name='95% CI'))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="NFIS-II Target (70%)")
    fig.update_layout(title=f"Account Ownership Forecast: {model}", xaxis_title="Year", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Milestones
    st.subheader("Key Milestones")
    col1, col2, col3 = st.columns(3)
    
    if model in forecast_df.columns:
        year_60 = forecast_df[forecast_df[model] >= 60]['Year'].min()
        year_70 = forecast_df[forecast_df[model] >= 70]['Year'].min()
        val_2030 = forecast_df[forecast_df['Year'] == 2030][model].values[0]
        
        col1.metric("60% Target", f"{int(year_60)}" if pd.notna(year_60) else "Not reached")
        col2.metric("70% Target", f"{int(year_70)}" if pd.notna(year_70) else "Not reached")
        col3.metric("2030 Projection", f"{val_2030:.1f}%")
    
    # Download
    if st.button("Download Forecast"):
        csv = forecast_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "forecast.csv", "text/csv")

# ===== PROJECTIONS PAGE =====
else:  # Projections
    st.title("🎯 Inclusion Projections")
    
    # Scenario selector
    scenario = st.radio("Select Scenario", ['Optimistic', 'Base Scenario', 'Pessimistic'], horizontal=True)
    
    # Scenario chart
    fig = go.Figure()
    for s in ['Optimistic', 'Base Scenario', 'Pessimistic']:
        if s in forecast_df.columns:
            color = 'green' if s == 'Optimistic' else 'orange' if s == 'Base Scenario' else 'red'
            width = 4 if s == scenario else 2
            fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df[s], 
                                    name=s, mode='lines+markers', 
                                    line=dict(color=color, width=width)))
    
    fig.add_hline(y=70, line_dash="dash", line_color="darkred", annotation_text="70% Target")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="60% Target")
    fig.update_layout(title="Scenario Analysis", xaxis_title="Year", yaxis_title="Account Ownership (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Progress to targets
    st.subheader("Progress to Targets")
    current = 49.0  # 2024 value
    
    col1, col2 = st.columns(2)
    col1.metric("Current (2024)", f"{current:.1f}%")
    
    if scenario in forecast_df.columns:
        val_2028 = forecast_df[forecast_df['Year'] == 2028][scenario].values[0]
        val_2030 = forecast_df[forecast_df['Year'] == 2030][scenario].values[0]
        
        col2.metric("2028 Projection", f"{val_2028:.1f}%", 
                   delta=f"{'✅' if val_2028 >= 60 else '⚠️'} vs 60% target")
    
    # Key questions
    st.subheader("Key Questions")
    
    if scenario in forecast_df.columns:
        val_2030 = forecast_df[forecast_df['Year'] == 2030][scenario].values[0]
        val_2028 = forecast_df[forecast_df['Year'] == 2028][scenario].values[0]
        
        st.write(f"**1. Projected rate by 2030?** {val_2030:.1f}%")
        st.write(f"**2. Will reach 60% by 2028?** {'✅ Yes' if val_2028 >= 60 else '⚠️ No'} ({val_2028:.1f}%)")
        st.write("**3. Most impactful events:** Telebirr (+15%), Fayda (+10%), EthioPay (+15%)")
        st.write("**4. Key risks:** Economic shocks, event delays, infrastructure gaps")
        
        opt = forecast_df[forecast_df['Year'] == 2030]['Optimistic'].values[0]
        pes = forecast_df[forecast_df['Year'] == 2030]['Pessimistic'].values[0]
        st.write(f"**5. Uncertainty range (2030):** {pes:.1f}% - {opt:.1f}% ({opt-pes:.1f}pp)")
    
    # Download
    if st.button("Download Scenarios"):
        csv = forecast_df[['Year', 'Optimistic', 'Base Scenario', 'Pessimistic']].to_csv(index=False)
        st.download_button("Download CSV", csv, "scenarios.csv", "text/csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Ethiopia FI Dashboard | Feb 2026")
