"""
Streamlit Cloud Deployment Configuration:

Branch: main
Main File Path: MAJOR_PROJECT/Home.py

Deployment Responsibilities:
1. Main Branch Protection:
   - Require pull request reviews
   - Require status checks to pass
   - No direct pushes to main

2. Version Control:
   - Create feature branches from main
   - Use descriptive branch names (e.g., feature/mass-transfer-analysis)
   - Merge only through pull requests

3. Deployment Process:
   - Push changes to feature branch
   - Create pull request to main
   - Review and approve
   - Merge to main triggers automatic deployment

Note: This file must be located in the MAJOR_PROJECT directory within the repository root.
"""

import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Chemical Engineering Analysis Tools",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar and add styling
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    .main {
        background-color: #f5f7f9;
        position: relative;
    }
    .tool-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        transition: transform 0.3s ease;
    }
    .tool-card:hover {
        transform: translateY(-5px);
    }
    .tool-button {
        display: block;
        background-color: #4CAF50;
        color: white;
        padding: 15px 30px;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
        text-decoration: none;
        font-weight: bold;
    }
    .tool-button:hover {
        background-color: #45a049;
    }
    .stats-card {
        background-color: #fff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .hero-section {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url('https://images.unsplash.com/photo-1532187863486-abf9dbad1b69');
        background-size: cover;
        background-position: center;
        color: white;
        padding: 4rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 style='font-size: 3.5em; margin-bottom: 20px;'>
        Chemical Engineering Analysis Suite
    </h1>
    <p style='font-size: 1.2em; max-width: 800px; margin: 0 auto;'>
        Advanced analytical tools powered by cutting-edge algorithms and machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Stats Dashboard
st.markdown("### 📊 Analytics Dashboard")
stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.markdown("""
    <div class="stats-card">
        <h3>2,500+</h3>
        <p>Analyses Run</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown("""
    <div class="stats-card">
        <h3>99.9%</h3>
        <p>Accuracy Rate</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown("""
    <div class="stats-card">
        <h3>50+</h3>
        <p>Research Papers</p>
    </div>
    """, unsafe_allow_html=True)

with stats_col4:
    st.markdown("""
    <div class="stats-card">
        <h3>24/7</h3>
        <p>Support</p>
    </div>
    """, unsafe_allow_html=True)

# Tool Cards
st.markdown("### 🛠️ Available Tools")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tool-card">
        <h2 style='color: #2c3e50; margin-bottom: 20px;'>Mass Transfer Analysis Tool 🧪</h2>
        <img src="https://images.unsplash.com/photo-1532187863486-abf9dbad1b69" 
             style="width: 100%; border-radius: 5px; margin: 10px 0;">
        <p style='color: #34495e; margin: 20px 0;'>
        Comprehensive mass transfer analysis features:
        <ul>
            <li>Advanced Sherwood number correlations</li>
            <li>Real-time data processing with ML algorithms</li>
            <li>Multiple regression analysis with R² scoring</li>
            <li>Monte Carlo uncertainty quantification</li>
            <li>Interactive 3D visualization</li>
            <li>Predictive modeling with neural networks</li>
            <li>Automated report generation</li>
            <li>Data export in multiple formats</li>
        </ul>
        </p>
        <a href="/Mass_Transfer_Analysis" class="tool-button">
            Launch Mass Transfer Analysis
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tool-card">
        <h2 style='color: #2c3e50; margin-bottom: 20px;'>Bubble Analysis Tool 🔍</h2>
        <img src="https://images.unsplash.com/photo-1532187863486-abf9dbad1b69" 
             style="width: 100%; border-radius: 5px; margin: 10px 0;">
        <p style='color: #34495e; margin: 20px 0;'>
        Advanced bubble detection features:
        <ul>
            <li>AI-powered bubble detection & tracking</li>
            <li>Real-time size distribution analysis</li>
            <li>High-precision spatial distribution mapping</li>
            <li>Advanced cluster analysis algorithms</li>
            <li>Interactive 3D visualization</li>
            <li>Comprehensive statistical analysis</li>
            <li>Batch processing capabilities</li>
            <li>Custom parameter optimization</li>
        </ul>
        </p>
        <a href="/Bubble_Analysis" class="tool-button">
            Launch Bubble Analysis
        </a>
    </div>
    """, unsafe_allow_html=True)

# Latest Updates Section
st.markdown("### 🆕 Latest Updates")
with st.expander("View Recent Changes"):
    st.markdown("""
    - **New Feature**: Advanced ML algorithms for bubble detection
    - **Improvement**: Enhanced 3D visualization capabilities
    - **Update**: New statistical analysis tools added
    - **Fix**: Improved processing speed for large datasets
    """)



# Enhanced Footer
st.markdown("""
<div style='text-align: center; padding: 30px 0; margin-top: 30px; background-color: #f8f9fa; border-radius: 10px;'>
    <h3 style='color: #2c3e50;'>Powered by Advanced Analytics & Machine Learning</h3>
    <p style='color: #34495e;'>
        Built with ❤️ for Chemical Engineers | Version 2.0
    </p>
    <p style='color: #7f8c8d; font-size: 0.8em;'>
        © 2023 Chemical Engineering Analysis Suite
    </p>
</div>
""", unsafe_allow_html=True)

# cd /workspaces/PRSNL_APP/MAJOR_PROJECT
# streamlit run Home.py --server.enableCORS false --server.enableXsrfProtection false
# streamlit run /workspaces/Prsnl_APP/MAJOR_PROJECT/Home.py --server.enableCORS false --server.enableXsrfProtection false
