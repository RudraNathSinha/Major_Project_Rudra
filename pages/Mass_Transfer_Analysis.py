# ...existing code from t_1.py...
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import random
import time
import pickle
import os
from streamlit_lottie import st_lottie
import json
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mass Transfer Analysis Tool",
    page_icon="ðŸ§ª",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        color: #1e1e1e;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .dark-mode {
        background-color: #1e1e1e;
        color: #f5f7f9;
    }
    .dark-mode h1, .dark-mode h2, .dark-mode h3 {
        color: #f5f7f9;
    }
    .dark-mode .highlight {
        background-color: #2c3e50;
        border-left: 5px solid #45a049;
    }
    .dark-mode .result-card {
        background-color: #2c3e50;
        color: #f5f7f9;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Function to load lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_x62chJ.json")
lottie_success = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_jbrw3hcz.json")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'selected_model_data' not in st.session_state:
    st.session_state.selected_model_data = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for theme toggle and history
with st.sidebar:
    st.title("Settings")
    
    # Create tabs for settings and history
    tab1, tab2 = st.tabs(["Settings", "History"])
    
    with tab1:
        theme = st.radio("Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == 'light' else 1)
        st.session_state.theme = theme.lower()
        
        if st.session_state.theme == 'dark':
            st.markdown("""
            <script>
                document.body.classList.add('dark-mode');
            </script>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <script>
                document.body.classList.remove('dark-mode');
            </script>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Analysis History")
        if len(st.session_state.history) > 0:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
        else:
            st.info("No analysis history available yet.")

# Main app
def main():
    st.title("Advanced Mass Transfer Analysis Tool")
    
    # Description and options
    st.markdown("""
    ## Project Description
    
    This advanced tool is designed for comprehensive mass transfer analysis in chemical engineering applications. 
    It utilizes cutting-edge machine learning algorithms to analyze experimental data and develop accurate predictive models 
    for Sherwood number correlations. The tool can handle various dimensionless parameters including Reynolds number, 
    Schmidt number, Weber number, and Eotvos number to predict mass transfer coefficients.
    
    The application employs multiple regression techniques, neural networks, and ensemble methods to provide 
    robust analysis and visualization of mass transfer phenomena.
    """)
    
    # Display lottie animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st_lottie(lottie_analysis, height=200, key="analysis_animation")
    
    # Start mass transfer analysis directly
    mass_transfer_analysis()

def mass_transfer_analysis():
    st.header("Mass Transfer Analysis")
    
    st.markdown("""
    <div class="highlight">
    <h3>About Mass Transfer Analysis</h3>
    <p>Mass transfer analysis is crucial in chemical engineering for understanding the transport of chemical species within a system. 
    This module uses dimensionless correlations to predict mass transfer coefficients based on experimental data.</p>
    
    <p>The analysis employs four different models based on the Sherwood number (Sh) correlation with Reynolds number (Re), 
    Schmidt number (Sc), Weber number (We), and Eotvos number (Eg). Through regression analysis, the tool identifies 
    the optimal model parameters that best fit your experimental data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.subheader("Model Selection")
    
    model_descriptions = {
        "Model 1": "Sh = a(Re^X1)(Sc^X2)(We^X3)(Eg^X4)",
        "Model 2": "Sh = a(Re^X1)(Sc^X2)(We^X3)",
        "Model 3": "Sh = a(Re^X1)(Sc^X2)(Eg^X4)",
        "Model 4": "Sh = a(Re^X1)(Sc^X2)"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox("Select Model", list(model_descriptions.keys()))
        st.markdown(f"**Selected Model Equation**: {model_descriptions[selected_model]}")
        
        st.markdown("""
        **Parameter Ranges:**
        - X1: 0.65 to 0.75 (step size depends on iterations)
        - X2: fixed at 0.33
        - X3: -0.5 to -0.2 (step size depends on iterations) - for Models 1 & 2
        - X4: 0.1 to 0.15 (step size depends on iterations) - for Models 1 & 3
        """)
    
    with col2:
        # Display model visualization
        fig = go.Figure()
        
        x = np.linspace(0, 10000, 100)
        
        # Sample visualization of how different models might look
        fig.add_trace(go.Scatter(x=x, y=10*x**0.7*0.33**0.33, name="Model 4", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=12*x**0.75*0.33**0.33*x**0.12, name="Model 3", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=x, y=8*x**0.68*0.33**0.33*x**(-0.3), name="Model 2", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=x, y=15*x**0.72*0.33**0.33*x**(-0.25)*x**0.13, name="Model 1", line=dict(color='purple')))
        
        fig.update_layout(
            title="Visualization of Model Behavior",
            xaxis_title="Reynolds Number (Re)",
            yaxis_title="Sherwood Number (Sh)",
            legend_title="Models",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data input section
    st.subheader("Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_data_points = st.number_input("Number of Experimental Data Points", min_value=3, max_value=100, value=10)
    
    with col2:
        num_iterations = st.number_input("Number of Iterations for Analysis", min_value=1, max_value=1000, value=100)
        if num_iterations > 1000:
            st.error("Number of iterations cannot exceed 1000")
    
    # Data input method
    data_input_method = st.radio("Select Data Input Method", ["Upload Excel File", "Enter Data Manually", "Load Previous Data"])
    
    if data_input_method == "Upload Excel File":
        uploaded_file = st.file_uploader("Upload Excel file with experimental data", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Check if the dataframe has the required columns
                required_cols = ["Sh", "Re", "Sc"]
                if selected_model in ["Model 1", "Model 2"]:
                    required_cols.append("We")
                if selected_model in ["Model 1", "Model 3"]:
                    required_cols.append("Eg")
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your Excel file has columns for all parameters in the selected model.")
                    
                    # Show example format
                    st.markdown("**Example format:**")
                    example_df = pd.DataFrame({
                        "Sh": [10.5, 15.2, 20.1],
                        "Re": [1000, 2000, 3000],
                        "Sc": [0.33, 0.33, 0.33],
                        "We": [2.5, 3.5, 4.5],
                        "Eg": [0.12, 0.15, 0.18]
                    })
                    st.dataframe(example_df)
                else:
                    st.session_state.data = df
                    st.write("Preview of uploaded data:")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    elif data_input_method == "Enter Data Manually":
        st.write("Enter your experimental data:")
        
        # Create empty dataframe with required columns
        cols = ["Sh", "Re", "Sc"]
        if selected_model in ["Model 1", "Model 2"]:
            cols.append("We")
        if selected_model in ["Model 1", "Model 3"]:
            cols.append("Eg")
        
        # Initialize empty dataframe
        if st.session_state.data is None or st.session_state.data.shape[0] != num_data_points or not all(col in st.session_state.data.columns for col in cols):
            data = {col: [0.0] * num_data_points for col in cols}
            st.session_state.data = pd.DataFrame(data)
        
        # Create a form for data entry
        with st.form("data_input_form"):
            edited_df = st.experimental_data_editor(st.session_state.data, use_container_width=True)
            submit_button = st.form_submit_button("Save Data")
            
            if submit_button:
                st.session_state.data = edited_df
                st.success("Data saved successfully!")
    
    elif data_input_method == "Load Previous Data":
        if os.path.exists("previous_data.pkl"):
            try:
                with open("previous_data.pkl", "rb") as f:
                    previous_data_dict = pickle.load(f)
                
                # Show available datasets
                dataset_names = list(previous_data_dict.keys())
                selected_dataset = st.selectbox("Select a previously saved dataset", dataset_names)
                
                if selected_dataset:
                    st.session_state.data = previous_data_dict[selected_dataset]
                    st.write("Preview of loaded data:")
                    st.dataframe(st.session_state.data)
                    st.success(f"Dataset '{selected_dataset}' loaded successfully!")
            except Exception as e:
                st.error(f"Error loading previous data: {e}")
        else:
            st.info("No previous data found. Please upload a file or enter data manually first.")
    
    # Save current data for future use
    if st.session_state.data is not None and st.button("Save Current Data for Future Use"):
        dataset_name = st.text_input("Enter a name for this dataset", "Dataset_" + time.strftime("%Y%m%d_%H%M%S"))
        
        if dataset_name:
            try:
                # Load existing data if available
                previous_data_dict = {}
                if os.path.exists("previous_data.pkl"):
                    with open("previous_data.pkl", "rb") as f:
                        previous_data_dict = pickle.load(f)
                
                # Add current data
                previous_data_dict[dataset_name] = st.session_state.data
                
                # Save updated dictionary
                with open("previous_data.pkl", "wb") as f:
                    pickle.dump(previous_data_dict, f)
                
                st.success(f"Dataset '{dataset_name}' saved successfully!")
            except Exception as e:
                st.error(f"Error saving data: {e}")
    
    # Run analysis if data is available
    if st.session_state.data is not None and st.button("Run Regression Analysis"):
        with st.spinner("Running regression analysis..."):
            # Display loading animation
            st_lottie(lottie_loading, height=200, key="loading_animation")
            
            # Check if data has required columns
            required_cols = ["Sh", "Re", "Sc"]
            if selected_model in ["Model 1", "Model 2"]:
                required_cols.append("We")
            if selected_model in ["Model 1", "Model 3"]:
                required_cols.append("Eg")
            
            missing_cols = [col for col in required_cols if col not in st.session_state.data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Run regression analysis
                model_results = run_regression_analysis(st.session_state.data, selected_model, num_iterations)
                st.session_state.model_results = model_results
                
                # Show success animation
                st_lottie(lottie_success, height=200, key="success_animation")
                st.success("Analysis completed successfully!")
    
    # Display results if available
    if st.session_state.model_results is not None:
        display_regression_results(st.session_state.data, st.session_state.model_results, selected_model, num_iterations)

def run_regression_analysis(data, selected_model, num_iterations):
    """Run regression analysis for the selected model using scipy.optimize.minimize"""
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Extract data
    Sh = data['Sh'].values
    Re = data['Re'].values
    Sc = data['Sc'].values
    We = data['We'].values if 'We' in data.columns else None
    Eg = data['Eg'].values if 'Eg' in data.columns else None
    
    # Define parameter ranges
    x1_range = np.linspace(0.65, 0.75, 20)  # Range for X1
    x2_value = 0.33  # Fixed value for X2
    x3_range = np.linspace(-0.5, -0.2, 20)  # Range for X3
    x4_range = np.linspace(0.1, 0.15, 20)  # Range for X4
    
    # Define model function based on selected model
    def model_function(params, model_type):
        A, X1, X2, X3, X4 = params
        result = A * (Re**X1) * (Sc**X2)
        
        if model_type in [1, 2] and We is not None:
            result *= We**X3
            
        if model_type in [1, 3] and Eg is not None:
            result *= Eg**X4
            
        return result
    
    # Define objective function (sum of squared errors)
    def objective_function(params, model_type):
        predicted = model_function(params, model_type)
        return np.sum((Sh - predicted)**2)
    
    # Convert model name to model type (1-4)
    model_type = int(selected_model.split(" ")[1])
    
    # Initialize results storage
    results = []
    
    # Process iterations in chunks to update progress bar
    chunk_size = max(1, num_iterations // 20)  # Update progress bar ~20 times
    
    for chunk_start in range(0, num_iterations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_iterations)
        chunk_iterations = chunk_end - chunk_start
        
        # Run iterations in this chunk
        for _ in range(chunk_iterations):
            # Randomly select parameters from ranges
            x1 = np.random.choice(x1_range)
            x3 = np.random.choice(x3_range) if model_type in [1, 2] else 0
            x4 = np.random.choice(x4_range) if model_type in [1, 3] else 0
            
            # Initial parameter guess
            x0 = [1.0, x1, x2_value, x3, x4]
            
            # Set bounds for optimization
            bounds = [(0.1, 10.0), (x1, x1), (x2_value, x2_value)]
            
            if model_type in [1, 2]:
                bounds.append((x3, x3))
            else:
                bounds.append((0, 0))
                
            if model_type in [1, 3]:
                bounds.append((x4, x4))
            else:
                bounds.append((0, 0))
            
            # Run optimization
            result = minimize(
                lambda params: objective_function(params, model_type),
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            # Calculate predicted values and RÂ²
            predicted = model_function(result.x, model_type)
            ss_total = np.sum((Sh - np.mean(Sh))**2)
            ss_residual = np.sum((Sh - predicted)**2)
            r2 = 1 - (ss_residual / ss_total)
            
            # Create model equation string
            if model_type == 1:  # Sh=a(Re^X1)*(Sc^X2)*(We^X3)*(Eg^x4)
                model_eq = f"Sh = {result.x[0]:.4f}(Re^{result.x[1]:.4f})(Sc^{result.x[2]:.4f})(We^{result.x[3]:.4f})(Eg^{result.x[4]:.4f})"
            elif model_type == 2:  # Sh=a(Re^X1)*(Sc^X2)*(We^X3)
                model_eq = f"Sh = {result.x[0]:.4f}(Re^{result.x[1]:.4f})(Sc^{result.x[2]:.4f})(We^{result.x[3]:.4f})"
            elif model_type == 3:  # Sh=a(Re^X1)*(Sc^X2)*(Eg^x4)
                model_eq = f"Sh = {result.x[0]:.4f}(Re^{result.x[1]:.4f})(Sc^{result.x[2]:.4f})(Eg^{result.x[4]:.4f})"
            else:  # Sh=a(Re^X1)*(Sc^X2)
                model_eq = f"Sh = {result.x[0]:.4f}(Re^{result.x[1]:.4f})(Sc^{result.x[2]:.4f})"
            
            # Store results
            results.append({
                'model': model_eq,
                'a': result.x[0],
                'x1': result.x[1],
                'x2': result.x[2],
                'x3': result.x[3] if model_type in [1, 2] else None,
                'x4': result.x[4] if model_type in [1, 3] else None,
                'r2': r2
            })
        
        # Update progress bar
        progress_bar.progress(chunk_end / num_iterations)
    
    # Sort results by RÂ² (descending)
    results = sorted(results, key=lambda x: x['r2'], reverse=True)
    
    # Add rank to results
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results

def display_regression_results(data, model_results, selected_model, num_iterations):
    """Display regression analysis results"""
    st.header("Regression Analysis Results")
    
    # Display input data
    st.subheader("Input Data")
    st.dataframe(data)
    
    # Display model parameters
    st.subheader("Model Parameters")
    
    # Create a table of parameters used in the model
    model_type = int(selected_model.split(" ")[1])
    
    params_df = pd.DataFrame({
        'Parameter': ['X1', 'X2', 'X3', 'X4'],
        'Range/Value': [
            '0.65 to 0.75', 
            'Fixed at 0.33', 
            '-0.5 to -0.2' if model_type in [1, 2] else 'Not used',
            '0.1 to 0.15' if model_type in [1, 3] else 'Not used'
        ],
        'Description': [
            'Reynolds number exponent', 
            'Schmidt number exponent', 
            'Weber number exponent' if model_type in [1, 2] else 'Not applicable',
            'Eotvos number exponent' if model_type in [1, 3] else 'Not applicable'
        ]
    })
    
    st.table(params_df)
    
    # Display all regression results
    st.subheader("All Regression Results")
    
    # Create a dataframe for results
    results_df = pd.DataFrame([
        {'Rank': r['rank'], 'Regression Model': r['model'], 'RÂ²': f"{r['r2']:.6f}"}
        for r in model_results
    ])
    
    st.dataframe(results_df)
    
    # Plot RÂ² distribution
    fig = px.histogram(
        results_df, 
        x='RÂ²', 
        nbins=20, 
        title='Distribution of RÂ² Values',
        labels={'RÂ²': 'RÂ² Value', 'count': 'Frequency'},
        color_discrete_sequence=['#4CAF50']
    )
    
    fig.update_layout(
        xaxis_title='RÂ² Value',
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Select model for further analysis
    st.subheader("Further Analysis")
    
    selected_rank = st.number_input(
        "Enter the rank of the model you want to analyze further:", 
        min_value=1, 
        max_value=len(model_results), 
        value=1
    )
    
    selected_model_data = model_results[selected_rank - 1]
    st.session_state.selected_model_data = selected_model_data
    
    st.write(f"Selected Model: **{selected_model_data['model']}**")
    st.write(f"RÂ² Value: **{selected_model_data['r2']:.6f}**")
    
    # Input for further analysis
    st.subheader("Mass Transfer Coefficient Calculation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        char_length = st.number_input("Characteristic Length (m):", min_value=1e-6, max_value=1.0, value=0.01, format="%.6f")
    
    with col2:
        diffusivity = st.number_input("Diffusivity (mÂ²/s):", min_value=1e-12, max_value=1e-5, value=1e-9, format="%.2e")
    
    col1, col2 = st.columns(2)
    
    with col1:
        w_min = st.number_input("Minimum W (rpm):", min_value=0, max_value=10000, value=100)
        w_max = st.number_input("Maximum W (rpm):", min_value=0, max_value=10000, value=1000)
    
    with col2:
        i_min = st.number_input("Minimum I (A):", min_value=0.0, max_value=1000.0, value=1.0)
        i_max = st.number_input("Maximum I (A):", min_value=0.0, max_value=1000.0, value=10.0)
    
    if st.button("Proceed with Analysis"):
        # Save to history
        history_entry = {
            'Date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Model': selected_model,
            'Data Points': len(data),
            'Iterations': num_iterations,
            'Selected Model': selected_model_data['model'],
            'RÂ²': selected_model_data['r2'],
            'Char Length': char_length,
            'Diffusivity': diffusivity,
            'W Range': f"{w_min}-{w_max}",
            'I Range': f"{i_min}-{i_max}"
        }
        st.session_state.history.append(history_entry)
        
        perform_detailed_analysis(data, selected_model_data, char_length, diffusivity, w_min, w_max, i_min, i_max, len(data), model_type)

def perform_detailed_analysis(data, model_data, char_length, diffusivity, w_min, w_max, i_min, i_max, num_points, model_type):
    """Perform detailed analysis for the selected model"""
    st.header("Detailed Analysis Results")
    
    # Calculate step sizes
    w_step = (w_max - w_min) / (num_points - 1) if num_points > 1 else 0
    i_step = (i_max - i_min) / (num_points - 1) if num_points > 1 else 0
    
    # Generate W and I values
    w_values = np.linspace(w_min, w_max, num_points)
    i_values = np.linspace(i_min, i_max, num_points)
    
    # Extract model parameters
    a = model_data['a']
    x1 = model_data['x1']
    x2 = model_data['x2']
    x3 = model_data['x3']
    x4 = model_data['x4']
    
    # Calculate observed Sh and mass transfer coefficients
    observed_sh = []
    observed_mtc = []
    
    for i, row in data.iterrows():
        # Calculate Sh based on model
        sh_val = a * (row['Re'] ** x1) * (row['Sc'] ** x2)
        
        if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
            sh_val *= row['We'] ** x3
            
        if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
            sh_val *= row['Eg'] ** x4
        
        observed_sh.append(sh_val)
        
        # Calculate mass transfer coefficient
        mtc = sh_val * diffusivity / char_length
        observed_mtc.append(mtc)
    
    # Calculate percentage error - corrected formula: ((exp/model)Sh - 1)*100
    exp_sh = data['Sh'].values
    percent_error = [(exp / obs - 1) * 100 for exp, obs in zip(exp_sh, observed_sh)]
    
    # Calculate experimental mass transfer coefficient
    exp_mtc = [sh * diffusivity / char_length for sh in exp_sh]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Data Point': range(1, len(data) + 1),
        'Experimental Sh': exp_sh,
        'Experimental MTC (m/s)': exp_mtc,
        'Observed Sh': observed_sh,
        'Observed MTC (m/s)': observed_mtc,
        'Percentage Error (%)': percent_error,
        'W (rpm)': w_values,
        'I (A)': i_values
    })
    
    # Display results table
    st.subheader("Comparison of Experimental and Model Results")
    st.dataframe(results_df)
    
    # Calculate summary statistics
    mean_error = np.mean(np.abs(percent_error))
    max_error = np.max(np.abs(percent_error))
    
    st.write(f"Mean Absolute Percentage Error: **{mean_error:.2f}%**")
    st.write(f"Maximum Absolute Percentage Error: **{max_error:.2f}%**")
    
    # Check if errors are within acceptable range
    if max_error > 10:
        st.warning("Some percentage errors exceed 10%. Consider selecting a different model or refining your data.")
    else:
        st.success("All percentage errors are within the acceptable range (â‰¤10%).")
    
    # Visualization options
    st.subheader("Visualization Options")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "Sh Comparison", 
        "MTC Comparison", 
        "Sh vs Data Points", 
        "MTC vs Data Points",
        "Sh & MTC Relationships",
        "Current & RPM Effects",
        "Error Analysis"
    ])
    
    # Sh Comparison
    with viz_tabs[0]:
        fig = px.scatter(
            results_df, 
            x='Experimental Sh', 
            y='Observed Sh',
            title='Comparison of Experimental vs. Model Sherwood Number',
            labels={'Experimental Sh': 'Experimental Sh', 'Observed Sh': 'Model Sh'},
            color='Percentage Error (%)',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Data Point', 'Percentage Error (%)']
        )
        
        # Add diagonal line (perfect prediction)
        min_val = min(results_df['Experimental Sh'].min(), results_df['Observed Sh'].min())
        max_val = max(results_df['Experimental Sh'].max(), results_df['Observed Sh'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        # Add Â±10% error bands
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val*0.9, max_val*0.9],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='-10% Error'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val*1.1, max_val*1.1],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='+10% Error'
            )
        )
        
        fig.update_layout(
            xaxis_title='Experimental Sh',
            yaxis_title='Model Sh',
            legend_title='',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
            # MTC Comparison
    with viz_tabs[1]:
        fig = px.scatter(
            results_df, 
            x='Experimental MTC (m/s)', 
            y='Observed MTC (m/s)',
            title='Comparison of Experimental vs. Model Mass Transfer Coefficient',
            labels={'Experimental MTC (m/s)': 'Experimental MTC (m/s)', 'Observed MTC (m/s)': 'Model MTC (m/s)'},
            color='Percentage Error (%)',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Data Point', 'Percentage Error (%)']
        )
        
        # Add diagonal line (perfect prediction)
        min_val = min(results_df['Experimental MTC (m/s)'].min(), results_df['Observed MTC (m/s)'].min())
        max_val = max(results_df['Experimental MTC (m/s)'].max(), results_df['Observed MTC (m/s)'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        # Add Â±10% error bands
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val*0.9, max_val*0.9],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='-10% Error'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val*1.1, max_val*1.1],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='+10% Error'
            )
        )
        
        fig.update_layout(
            xaxis_title='Experimental MTC (m/s)',
            yaxis_title='Model MTC (m/s)',
            legend_title='',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sh vs Data Points
    with viz_tabs[2]:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=results_df['Experimental Sh'],
                mode='lines+markers',
                name='Experimental Sh',
                line=dict(color='blue', width=2),
                marker=dict(size=10, symbol='circle')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=results_df['Observed Sh'],
                mode='lines+markers',
                name='Model Sh',
                line=dict(color='green', width=2),
                marker=dict(size=10, symbol='diamond')
            )
        )
        
        # Add error bands
        upper_bound = [exp * 1.1 for exp in results_df['Experimental Sh']]
        lower_bound = [exp * 0.9 for exp in results_df['Experimental Sh']]
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=upper_bound,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                name='+10% Error Band',
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=lower_bound,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                name='-10% Error Band',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title='Comparison of Sherwood Number Across Data Points',
            xaxis_title='Data Point',
            yaxis_title='Sherwood Number (Sh)',
            legend_title='',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # MTC vs Data Points
    with viz_tabs[3]:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=results_df['Experimental MTC (m/s)'],
                mode='lines+markers',
                name='Experimental MTC',
                line=dict(color='purple', width=2),
                marker=dict(size=10, symbol='circle')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=results_df['Observed MTC (m/s)'],
                mode='lines+markers',
                name='Model MTC',
                line=dict(color='orange', width=2),
                marker=dict(size=10, symbol='diamond')
            )
        )
        
        # Add error bands
        upper_bound = [exp * 1.1 for exp in results_df['Experimental MTC (m/s)']]
        lower_bound = [exp * 0.9 for exp in results_df['Experimental MTC (m/s)']]
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=upper_bound,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                name='+10% Error Band',
                showlegend=True
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=results_df['Data Point'],
                y=lower_bound,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.2)'),
                name='-10% Error Band',
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title='Comparison of Mass Transfer Coefficient Across Data Points',
            xaxis_title='Data Point',
            yaxis_title='Mass Transfer Coefficient (m/s)',
            legend_title='',
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sh & MTC Relationships
    with viz_tabs[4]:
        # Create subplot with 2 rows and 1 column
        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=('Experimental Sh vs MTC', 'Model Sh vs MTC'),
            vertical_spacing=0.15
        )
        
        # Experimental values
        fig.add_trace(
            go.Scatter(
                x=results_df['Experimental Sh'],
                y=results_df['Experimental MTC (m/s)'],
                mode='markers',
                name='Experimental Values',
                marker=dict(
                    size=12,
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Data Point')
                ),
                text=[f"Data Point {i}" for i in results_df['Data Point']],
                hovertemplate='<b>%{text}</b><br>Sh: %{x:.2f}<br>MTC: %{y:.2e} m/s<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Model values
        fig.add_trace(
            go.Scatter(
                x=results_df['Observed Sh'],
                y=results_df['Observed MTC (m/s)'],
                mode='markers',
                name='Model Values',
                marker=dict(
                    size=12,
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"Data Point {i}" for i in results_df['Data Point']],
                hovertemplate='<b>%{text}</b><br>Sh: %{x:.2f}<br>MTC: %{y:.2e} m/s<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add trend lines
        x_exp = results_df['Experimental Sh']
        y_exp = results_df['Experimental MTC (m/s)']
        z_exp = np.polyfit(x_exp, y_exp, 1)
        p_exp = np.poly1d(z_exp)
        
        x_mod = results_df['Observed Sh']
        y_mod = results_df['Observed MTC (m/s)']
        z_mod = np.polyfit(x_mod, y_mod, 1)
        p_mod = np.poly1d(z_mod)
        
        # Add equation text
        exp_eq = f"MTC = {z_exp[0]:.2e} Ã— Sh + {z_exp[1]:.2e}"
        mod_eq = f"MTC = {z_mod[0]:.2e} Ã— Sh + {z_mod[1]:.2e}"
        
        fig.add_trace(
            go.Scatter(
                x=[min(x_exp), max(x_exp)],
                y=[p_exp(min(x_exp)), p_exp(max(x_exp))],
                mode='lines',
                name='Experimental Trend',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[min(x_mod), max(x_mod)],
                y=[p_mod(min(x_mod)), p_mod(max(x_mod))],
                mode='lines',
                name='Model Trend',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        
        # Add annotations for equations
        fig.add_annotation(
            x=0.95,
            y=0.15,
            xref="paper",
            yref="paper",
            text=exp_eq,
            showarrow=False,
            font=dict(size=12, color="red"),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="red",
            borderwidth=1,
            borderpad=4,
            row=1,
            col=1
        )
        
        fig.add_annotation(
            x=0.95,
            y=0.15,
            xref="paper",
            yref="paper",
            text=mod_eq,
            showarrow=False,
            font=dict(size=12, color="red"),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="red",
            borderwidth=1,
            borderpad=4,
            row=2,
            col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='Relationship Between Sherwood Number and Mass Transfer Coefficient',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update xaxis properties
        fig.update_xaxes(title_text='Sherwood Number (Sh)', row=1, col=1)
        fig.update_xaxes(title_text='Sherwood Number (Sh)', row=2, col=1)
        
        # Update yaxis properties
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=1, col=1)
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add theoretical explanation
        st.markdown("""
        ### Theoretical Relationship
        
        The relationship between Sherwood number (Sh) and Mass Transfer Coefficient (MTC) is given by:
        
        $$ MTC = \frac{Sh \times D}{L} $$
        
        Where:
        - MTC is the Mass Transfer Coefficient (m/s)
        - Sh is the Sherwood number (dimensionless)
        - D is the Diffusivity (mÂ²/s)
        - L is the Characteristic Length (m)
        
        The linear relationship observed in the plots confirms this theoretical relationship with the slope approximately equal to D/L.
        """)
    
    # Current & RPM Effects
    with viz_tabs[5]:
        # Create subplot with 2 rows and 2 columns
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=(
                'Experimental MTC vs Current', 
                'Model MTC vs Current',
                'Experimental MTC vs RPM', 
                'Model MTC vs RPM'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Experimental MTC vs Current
        fig.add_trace(
            go.Scatter(
                x=results_df['I (A)'],
                y=results_df['Experimental MTC (m/s)'],
                mode='markers+lines',
                name='Experimental MTC vs I',
                marker=dict(
                    size=10, 
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=False
                ),
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Model MTC vs Current
        fig.add_trace(
            go.Scatter(
                x=results_df['I (A)'],
                y=results_df['Observed MTC (m/s)'],
                mode='markers+lines',
                name='Model MTC vs I',
                marker=dict(
                    size=10, 
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=False
                ),
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Experimental MTC vs RPM
        fig.add_trace(
            go.Scatter(
                x=results_df['W (rpm)'],
                y=results_df['Experimental MTC (m/s)'],
                mode='markers+lines',
                name='Experimental MTC vs W',
                marker=dict(
                    size=10, 
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=False
                ),
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Model MTC vs RPM
        fig.add_trace(
            go.Scatter(
                x=results_df['W (rpm)'],
                y=results_df['Observed MTC (m/s)'],
                mode='markers+lines',
                name='Model MTC vs W',
                marker=dict(
                    size=10, 
                    color=results_df['Data Point'],
                    colorscale='Viridis',
                    showscale=False
                ),
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        # Add trend lines
        # Current vs Experimental MTC
        x = results_df['I (A)']
        y = results_df['Experimental MTC (m/s)']
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        
        x_range = np.linspace(min(x), max(x), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend (Exp MTC vs I)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Current vs Model MTC
        y = results_df['Observed MTC (m/s)']
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend (Model MTC vs I)',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # RPM vs Experimental MTC
        x = results_df['W (rpm)']
        y = results_df['Experimental MTC (m/s)']
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        
        x_range = np.linspace(min(x), max(x), 100)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend (Exp MTC vs W)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # RPM vs Model MTC
        y = results_df['Observed MTC (m/s)']
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trend (Model MTC vs W)',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text='Effect of Current and RPM on Mass Transfer Coefficient',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update xaxis properties
        fig.update_xaxes(title_text='Current (A)', row=1, col=1)
        fig.update_xaxes(title_text='Current (A)', row=1, col=2)
        fig.update_xaxes(title_text='RPM (W)', row=2, col=1)
        fig.update_xaxes(title_text='RPM (W)', row=2, col=2)
        
        # Update yaxis properties
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=1, col=1)
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=1, col=2)
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=2, col=1)
        fig.update_yaxes(title_text='Mass Transfer Coefficient (m/s)', row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive 3D plot
        st.subheader("Interactive 3D Visualization of Current, RPM, and MTC")
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=results_df['I (A)'],
                y=results_df['W (rpm)'],
                z=results_df['Experimental MTC (m/s)'],
                mode='markers',
                name='Experimental MTC',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.8
                ),
                text=[f"Data Point {i}" for i in results_df['Data Point']],
                hovertemplate='<b>%{text}</b><br>Current: %{x} A<br>RPM: %{y}<br>MTC: %{z:.2e} m/s<extra></extra>'
            ),
            go.Scatter3d(
                x=results_df['I (A)'],
                y=results_df['W (rpm)'],
                z=results_df['Observed MTC (m/s)'],
                mode='markers',
                name='Model MTC',
                marker=dict(
                    size=8,
                    color='green',
                    opacity=0.8
                ),
                text=[f"Data Point {i}" for i in results_df['Data Point']],
                hovertemplate='<b>%{text}</b><br>Current: %{x} A<br>RPM: %{y}<br>MTC: %{z:.2e} m/s<extra></extra>'
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Current (A)',
                yaxis_title='RPM (W)',
                zaxis_title='Mass Transfer Coefficient (m/s)'
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Effect of Operating Parameters
        
        The plots above show how the mass transfer coefficient is affected by:
        
        1. **Current (I)**: Higher current generally leads to increased mass transfer rates due to enhanced electrochemical reactions.
        
        2. **Rotational Speed (W)**: Higher RPM typically improves mass transfer by reducing boundary layer thickness and enhancing mixing.
        
        The 3D visualization helps identify optimal operating conditions where both parameters can be adjusted to achieve desired mass transfer rates.
        """)
    
    # Error Analysis
    with viz_tabs[6]:
        # Create subplot with 2 rows and 1 column
        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=('Percentage Error Across Data Points', 'Error Distribution'),
            vertical_spacing=0.15
        )
        
        # Percentage error across data points
        fig.add_trace(
            go.Bar(
                x=results_df['Data Point'],
                y=results_df['Percentage Error (%)'],
                name='Percentage Error',
                marker=dict(
                    color=results_df['Percentage Error (%)'],
                    colorscale='RdBu_r',
                    cmin=-max(abs(results_df['Percentage Error (%)'])),
                    cmax=max(abs(results_df['Percentage Error (%)'])),
                    colorbar=dict(title='Error (%)')
                ),
                text=results_df['Percentage Error (%)'].round(2).astype(str) + '%',
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[0, len(results_df) + 1],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add Â±10% error lines
        fig.add_trace(
            go.Scatter(
                x=[0, len(results_df) + 1],
                y=[10, 10],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='+10% Error Limit',
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(results_df) + 1],
                y=[-10, -10],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='-10% Error Limit',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(
                x=results_df['Percentage Error (%)'],
                name='Error Distribution',
                marker=dict(
                    color='rgba(0, 128, 255, 0.7)',
                    line=dict(color='rgba(0, 128, 255, 1)', width=1)
                ),
                nbinsx=20,
                histnorm='probability density'
            ),
            row=2, col=1
        )
        
        # Add vertical line at zero
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, 1],  # Will be adjusted with update_yaxes
                mode='lines',
                line=dict(color='black', dash='dash'),
                name='Zero Error',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add vertical lines at Â±10%
        fig.add_trace(
            go.Scatter(
                x=[10, 10],
                y=[0, 1],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='+10% Error',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[-10, -10],
                y=[0, 1],
                mode='lines',
                line=dict(color='red', dash='dot'),
                name='-10% Error',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Fit normal distribution to errors
        from scipy import stats
        
        errors = results_df['Percentage Error (%)']
        mu, sigma = stats.norm.fit(errors)
        
        x = np.linspace(min(errors) - 5, max(errors) + 5, 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='green', width=2),
                name=f'Normal Distribution (Î¼={mu:.2f}, Ïƒ={sigma:.2f})'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='Error Analysis',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update xaxis properties
        fig.update_xaxes(title_text='Data Point', row=1, col=1)
        fig.update_xaxes(title_text='Percentage Error (%)', row=2, col=1)
        
        # Update yaxis properties
        fig.update_yaxes(title_text='Percentage Error (%)', row=1, col=1)
        fig.update_yaxes(title_text='Probability Density', row=2, col=1, autorange=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error statistics
        st.subheader("Error Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Error (%)", f"{np.mean(errors):.2f}")
        
        with col2:
            st.metric("Standard Deviation (%)", f"{np.std(errors):.2f}")
        
        with col3:
            # Calculate percentage of points within Â±10% error
            within_limit = sum(abs(errors) <= 10)
            percentage_within = (within_limit / len(errors)) * 100
            st.metric("Points Within Â±10% Error", f"{percentage_within:.1f}%")
        
        # Error analysis explanation
        st.markdown("""
        ### Error Analysis Interpretation
        
        The error analysis provides insights into the model's accuracy:
        
        - **Error Distribution**: Shows how errors are distributed around zero. A normal distribution centered near zero indicates a well-balanced model.
        
        - **Points Within Â±10% Error**: Industry standard often requires at least 90% of predictions to fall within Â±10% error.
        
        - **Systematic Bias**: If errors consistently trend positive or negative, it suggests a systematic bias in the model.
        
        - **Outliers**: Data points with errors significantly outside the normal range may indicate experimental anomalies or limitations of the model.
        """)
    
    # Advanced AI Analysis
    st.subheader("Advanced AI Analysis")
    
    # Create tabs for advanced analysis
    ai_tabs = st.tabs([
        "Parameter Sensitivity", 
        "Optimization Suggestions", 
        "3D Visualization",
        "Neural Network Prediction",
        "Uncertainty Analysis"
    ])
    
    # Parameter Sensitivity Analysis
    with ai_tabs[0]:
        st.write("### Parameter Sensitivity Analysis")
        st.write("This analysis shows how sensitive the model is to changes in each parameter.")
        
        # Create sensitivity data
        sensitivity_data = []
        base_sh = a * (data['Re'].mean() ** x1) * (data['Sc'].mean() ** x2)
        
        if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
            base_sh *= data['We'].mean() ** x3
            
        if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
            base_sh *= data['Eg'].mean() ** x4
        
        # Sensitivity to Re
        re_values = np.linspace(data['Re'].min() * 0.5, data['Re'].max() * 1.5, 100)
        re_sh_values = []
        
        for re in re_values:
            sh_val = a * (re ** x1) * (data['Sc'].mean() ** x2)
            
            if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
                sh_val *= data['We'].mean() ** x3
                
            if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
                sh_val *= data['Eg'].mean() ** x4
                
            re_sh_values.append(sh_val)
        
        # Sensitivity to Sc (if it varies in the data)
        sc_values = np.linspace(data['Sc'].min() * 0.5, data['Sc'].max() * 1.5, 100)
        sc_sh_values = []
        
        for sc in sc_values:
            sh_val = a * (data['Re'].mean() ** x1) * (sc ** x2)
            
            if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
                sh_val *= data['We'].mean() ** x3
                
            if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
                sh_val *= data['Eg'].mean() ** x4
                
            sc_sh_values.append(sh_val)
        
        # Create sensitivity plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=re_values / data['Re'].mean(),
                y=[val / base_sh for val in re_sh_values],
                mode='lines',
                name='Sensitivity to Re',
                line=dict(color='blue', width=3)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sc_values / data['Sc'].mean(),
                y=[val / base_sh for val in sc_sh_values],
                mode='lines',
                name='Sensitivity to Sc',
                line=dict(color='green', width=3)
            )
        )
        
        # Add We sensitivity if applicable
        if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
            we_values = np.linspace(data['We'].min() * 0.5, data['We'].max() * 1.5, 100)
            we_sh_values = []
            
            for we in we_values:
                sh_val = a * (data['Re'].mean() ** x1) * (data['Sc'].mean() ** x2) * (we ** x3)
                
                if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
                    sh_val *= data['Eg'].mean() ** x4
                    
                we_sh_values.append(sh_val)
                
            fig.add_trace(
                go.Scatter(
                    x=we_values / data['We'].mean(),
                    y=[val / base_sh for val in we_sh_values],
                    mode='lines',
                    name='Sensitivity to We',
                    line=dict(color='red', width=3)
                )
            )
        
        # Add Eg sensitivity if applicable
        if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
            eg_values = np.linspace(data['Eg'].min() * 0.5, data['Eg'].max() * 1.5, 100)
            eg_sh_values = []
            
            for eg in eg_values:
                sh_val = a * (data['Re'].mean() ** x1) * (data['Sc'].mean() ** x2)
                
                if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
                    sh_val *= data['We'].mean() ** x3
                    
                sh_val *= eg ** x4
                eg_sh_values.append(sh_val)
                
            fig.add_trace(
                go.Scatter(
                    x=eg_values / data['Eg'].mean(),
                    y=[val / base_sh for val in eg_sh_values],
                    mode='lines',
                    name='Sensitivity to Eg',
                    line=dict(color='purple', width=3)
                )
            )
        
        fig.update_layout(
            title='Parameter Sensitivity Analysis',
            xaxis_title='Normalized Parameter Value (Parameter/Mean)',
            yaxis_title='Normalized Sh (Sh/Base Sh)',
            legend_title='Parameter',
            height=600
        )
        
        # Add reference line
        fig.add_shape(
            type="line",
            x0=0.5,
            y0=1,
            x1=1.5,
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dash",
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
                # Parameter importance
        st.write("### Parameter Importance")
        
        # Calculate parameter importance based on exponents
        importance = {}
        importance['Re'] = abs(x1)
        importance['Sc'] = abs(x2)
        
        if x3 is not None and model_type in [1, 2]:
            importance['We'] = abs(x3)
            
        if x4 is not None and model_type in [1, 3]:
            importance['Eg'] = abs(x4)
            
        # Normalize importance
        total = sum(importance.values())
        for key in importance:
            importance[key] = importance[key] / total * 100
            
        # Create bar chart
        fig = px.bar(
            x=list(importance.keys()),
            y=list(importance.values()),
            labels={'x': 'Parameter', 'y': 'Relative Importance (%)'},
            title='Parameter Importance in the Model',
            color=list(importance.values()),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Parameter',
            yaxis_title='Relative Importance (%)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter recommendations
        st.write("### Parameter Recommendations")
        st.write("""
        Based on the sensitivity analysis, here are some recommendations for optimizing your mass transfer process:
        """)
        
        # Generate recommendations based on model parameters
        recommendations = []
        
        if 'Re' in importance and importance['Re'] > 20:
            recommendations.append("- **Reynolds Number (Re)**: This parameter has significant influence on mass transfer. Consider optimizing fluid velocity or characteristic length to achieve optimal Reynolds numbers.")
        
        if 'Sc' in importance and importance['Sc'] > 20:
            recommendations.append("- **Schmidt Number (Sc)**: This parameter is important for mass transfer. Consider adjusting fluid properties to optimize diffusion.")
        
        if 'We' in importance and x3 is not None and x3 < -0.3:
            recommendations.append("- **Weber Number (We)**: The negative exponent suggests that lower Weber numbers improve mass transfer. Consider reducing interfacial tension or adjusting fluid velocity.")
        
        if 'Eg' in importance and x4 is not None and x4 > 0.12:
            recommendations.append("- **Eotvos Number (Eg)**: The positive exponent indicates that increasing Eotvos number enhances mass transfer. Consider adjusting density difference or characteristic length.")
        
        for rec in recommendations:
            st.markdown(rec)
    
    # Optimization Suggestions
    with ai_tabs[1]:
        st.write("### Model Optimization Suggestions")
        
        # Create a pipeline for optimization
        X = data.drop('Sh', axis=1)
        y = data['Sh']
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with preprocessing and multiple models
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate RÂ²
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Machine Learning Model RÂ²: **{r2:.6f}**")
        
        # Compare with regression model
        st.write(f"Current Regression Model RÂ²: **{model_data['r2']:.6f}**")
        
        if r2 > model_data['r2']:
            st.success("The machine learning model outperforms the current regression model. Consider using more advanced modeling techniques for better predictions.")
        else:
            st.info("The current regression model performs well. The simpler model is preferred for interpretability.")
        
        # Feature importance from Random Forest
        if isinstance(pipeline['model'], RandomForestRegressor):
            importances = pipeline['model'].feature_importances_
            feature_names = X.columns
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title='Feature Importance from Random Forest',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title='Feature',
                yaxis_title='Importance',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on feature importance
            st.write("### Optimization Recommendations")
            st.write("Based on the machine learning analysis, here are recommendations for optimizing your model:")
            
            for i, row in feature_importance.iterrows():
                if row['Importance'] > 0.2:
                    st.markdown(f"- **{row['Feature']}**: High importance ({row['Importance']:.2f}). Focus on accurate measurement and control of this parameter.")
                elif row['Importance'] > 0.1:
                    st.markdown(f"- **{row['Feature']}**: Moderate importance ({row['Importance']:.2f}). Consider including this parameter in your model.")
                else:
                    st.markdown(f"- **{row['Feature']}**: Low importance ({row['Importance']:.2f}). This parameter has less impact on the model.")
        
        # Optimal operating conditions
        st.write("### Optimal Operating Conditions")
        
        # Create a grid of possible operating conditions
        if 'Re' in data.columns and 'Sc' in data.columns:
            re_range = np.linspace(data['Re'].min(), data['Re'].max(), 20)
            sc_range = np.linspace(data['Sc'].min(), data['Sc'].max(), 20)
            
            grid_points = []
            for re in re_range:
                for sc in sc_range:
                    point = {'Re': re, 'Sc': sc}
                    
                    if 'We' in data.columns:
                        point['We'] = data['We'].mean()
                    
                    if 'Eg' in data.columns:
                        point['Eg'] = data['Eg'].mean()
                    
                    grid_points.append(point)
            
            grid_df = pd.DataFrame(grid_points)
            
            # Predict Sh for each point using the model
            grid_df['Predicted_Sh'] = np.zeros(len(grid_df))
            
            for i, row in grid_df.iterrows():
                sh_val = a * (row['Re'] ** x1) * (row['Sc'] ** x2)
                
                if x3 is not None and model_type in [1, 2] and 'We' in row:
                    sh_val *= row['We'] ** x3
                    
                if x4 is not None and model_type in [1, 3] and 'Eg' in row:
                    sh_val *= row['Eg'] ** x4
                
                grid_df.at[i, 'Predicted_Sh'] = sh_val
            
            # Find optimal conditions
            optimal_point = grid_df.loc[grid_df['Predicted_Sh'].idxmax()]
            
            st.write("Based on the model, the optimal operating conditions are:")
            
            for col in optimal_point.index:
                if col != 'Predicted_Sh':
                    st.write(f"- **{col}**: {optimal_point[col]:.2f}")
            
            st.write(f"- **Predicted Sherwood Number**: {optimal_point['Predicted_Sh']:.2f}")
            
            # Create contour plot
            if 'Re' in data.columns and 'Sc' in data.columns:
                pivot_table = grid_df.pivot_table(
                    values='Predicted_Sh', 
                    index='Re', 
                    columns='Sc'
                )
                
                fig = go.Figure(data=
                    go.Contour(
                        z=pivot_table.values,
                        x=pivot_table.columns.tolist(),  # Sc values
                        y=pivot_table.index.tolist(),    # Re values
                        colorscale='Viridis',
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=12, color='white')
                        ),
                        colorbar=dict(title='Predicted Sh')
                    )
                )
                
                # Mark optimal point
                fig.add_trace(
                    go.Scatter(
                        x=[optimal_point['Sc']],
                        y=[optimal_point['Re']],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=15,
                            color='red',
                            line=dict(width=2, color='black')
                        ),
                        name='Optimal Conditions'
                    )
                )
                
                # Mark experimental points
                fig.add_trace(
                    go.Scatter(
                        x=data['Sc'],
                        y=data['Re'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='white',
                            line=dict(width=1, color='black')
                        ),
                        name='Experimental Points'
                    )
                )
                
                fig.update_layout(
                    title='Predicted Sherwood Number Contour Map',
                    xaxis_title='Schmidt Number (Sc)',
                    yaxis_title='Reynolds Number (Re)',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # 3D Visualization
    with ai_tabs[2]:
        st.write("### 3D Visualization of Parameter Relationships")
        
        # Select parameters for 3D plot
        available_params = list(data.columns)
        available_params.remove('Sh')  # Remove target variable
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("Select X-axis parameter", available_params, index=0)
        
        with col2:
            y_param = st.selectbox("Select Y-axis parameter", available_params, index=min(1, len(available_params)-1))
        
        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=data[x_param],
                y=data[y_param],
                z=data['Sh'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=data['Sh'],
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Sh')
                ),
                text=[f"Data Point {i+1}" for i in range(len(data))],
                hovertemplate=
                    f"<b>%{{text}}</b><br>" +
                    f"{x_param}: %{{x}}<br>" +
                    f"{y_param}: %{{y}}<br>" +
                    "Sh: %{z}<br>" +
                    "<extra></extra>"
            )
        ])
        
        # Create surface plot if enough data points
        if len(data) >= 10:
            try:
                # Create a grid for the surface
                x_range = np.linspace(data[x_param].min(), data[x_param].max(), 20)
                y_range = np.linspace(data[y_param].min(), data[y_param].max(), 20)
                x_grid, y_grid = np.meshgrid(x_range, y_range)
                
                # Prepare data for prediction
                grid_points = pd.DataFrame({
                    x_param: x_grid.flatten(),
                    y_param: y_grid.flatten()
                })
                
                # Fill in other columns with mean values
                for col in available_params:
                    if col not in [x_param, y_param]:
                        grid_points[col] = data[col].mean()
                
                # Make predictions using the model
                z_pred = []
                for _, row in grid_points.iterrows():
                    # Calculate Sh based on model
                    sh_val = a * (row['Re'] ** x1) * (row['Sc'] ** x2)
                    
                    if x3 is not None and model_type in [1, 2] and 'We' in data.columns:
                        sh_val *= row['We'] ** x3
                        
                    if x4 is not None and model_type in [1, 3] and 'Eg' in data.columns:
                        sh_val *= row['Eg'] ** x4
                    
                    z_pred.append(sh_val)
                
                z_grid = np.array(z_pred).reshape(x_grid.shape)
                
                # Add surface plot
                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=z_grid,
                        opacity=0.7,
                        colorscale='Viridis',
                        showscale=False
                    )
                )
            except Exception as e:
                st.warning(f"Could not create surface plot: {e}")
        
        fig.update_layout(
            title=f'3D Visualization of Sh vs {x_param} and {y_param}',
            scene=dict(
                xaxis_title=x_param,
                yaxis_title=y_param,
                zaxis_title='Sh'
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.write("""
        This 3D visualization shows the relationship between Sherwood number and two selected parameters. 
        The colored points represent actual experimental data, while the surface (if shown) represents the model prediction.
        
        You can:
        - Rotate the plot by clicking and dragging
        - Zoom in/out using the scroll wheel
        - Select different parameters for the X and Y axes
        - Hover over points to see detailed information
        """)
        
        # Add interactive sliders for exploring parameter space
        st.write("### Interactive Parameter Space Explorer")
        
        # Create sliders for each parameter
        slider_values = {}
        
        for param in available_params:
            if param not in [x_param, y_param]:
                min_val = float(data[param].min())
                max_val = float(data[param].max())
                mean_val = float(data[param].mean())
                
                step = (max_val - min_val) / 100
                slider_values[param] = st.slider(
                    f"{param} Value", 
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step
                )
        
        # Calculate Sh for the current parameter values
        if slider_values:
            # Create a point with the current slider values
            point = {param: value for param, value in slider_values.items()}
            point[x_param] = data[x_param].mean()
            point[y_param] = data[y_param].mean()
            
            # Calculate Sh based on model
            sh_val = a * (point['Re'] ** x1) * (point['Sc'] ** x2)
            
            if x3 is not None and model_type in [1, 2] and 'We' in point:
                sh_val *= point['We'] ** x3
                
            if x4 is not None and model_type in [1, 3] and 'Eg' in point:
                sh_val *= point['Eg'] ** x4
            
            # Display the calculated Sh
            st.metric("Predicted Sherwood Number", f"{sh_val:.2f}")
            
            # Calculate MTC
            mtc = sh_val * diffusivity / char_length
            st.metric("Predicted Mass Transfer Coefficient", f"{mtc:.2e} m/s")
    
    # Neural Network Prediction
    with ai_tabs[3]:
        st.write("### Neural Network Prediction")
        
        # Create and train a neural network
        if st.button("Train Neural Network Model"):
            with st.spinner("Training neural network..."):
                # Prepare data
                X = data.drop('Sh', axis=1)
                y = data['Sh']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create neural network model
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                
                # Compile model
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=8,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                nn_r2 = r2_score(y_test, y_pred)
                
                # Plot training history
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['loss']) + 1)),
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue', width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['val_loss']) + 1)),
                        y=history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red', width=2)
                    )
                )
                
                fig.update_layout(
                    title='Neural Network Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    legend_title='',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare with regression model
                st.write(f"Neural Network Model RÂ²: **{nn_r2:.6f}**")
                st.write(f"Current Regression Model RÂ²: **{model_data['r2']:.6f}**")
                
                if nn_r2 > model_data['r2']:
                    st.success("The neural network outperforms the current regression model. Consider using neural networks for more accurate predictions.")
                else:
                    st.info("The current regression model performs well. The simpler model is preferred for interpretability.")
                
                # Make predictions on all data
                X_all_scaled = scaler.transform(X)
                y_all_pred = model.predict(X_all_scaled)
                
                # Create comparison plot
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Sh'],
                        mode='lines+markers',
                        name='Experimental Sh',
                        line=dict(color='blue'),
                        marker=dict(size=10)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=y_all_pred.flatten(),
                        mode='lines+markers',
                        name='Neural Network Prediction',
                        line=dict(color='green'),
                        marker=dict(size=10)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=observed_sh,
                        mode='lines+markers',
                        name='Regression Model Prediction',
                        line=dict(color='red'),
                        marker=dict(size=10)
                    )
                )
                
                fig.update_layout(
                    title='Comparison of Experimental Data with Model Predictions',
                    xaxis_title='Data Point',
                    yaxis_title='Sherwood Number (Sh)',
                    legend_title='',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance using permutation importance
                from sklearn.inspection import permutation_importance
                
                # Create a pipeline with the neural network
                nn_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('nn', MLPRegressor(
                        hidden_layer_sizes=(64, 32, 16),
                        activation='relu',
                        solver='adam',
                        random_state=42,
                        max_iter=1000
                    ))
                ])
                
                # Fit the pipeline
                nn_pipeline.fit(X_train, y_train)
                
                # Calculate permutation importance
                result = permutation_importance(
                    nn_pipeline, X_test, y_test,
                    n_repeats=10,
                    random_state=42
                )
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': result.importances_mean
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Neural Network Feature Importance',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title='Feature',
                    yaxis_title='Importance',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Uncertainty Analysis
    with ai_tabs[4]:
        st.write("### Uncertainty Analysis")
        
        # Bootstrap analysis for parameter uncertainty
        st.write("#### Parameter Uncertainty Analysis")
        
        if st.button("Run Bootstrap Analysis"):
            with st.spinner("Running bootstrap analysis..."):
                # Number of bootstrap samples
                n_bootstrap = 100
                
                # Storage for bootstrap results
                bootstrap_params = []
                
                # Progress bar
                progress_bar = st.progress(0)
                
                # Run bootstrap
                for i in range(n_bootstrap):
                    # Sample with replacement
                    indices = np.random.choice(len(data), len(data), replace=True)
                    bootstrap_sample = data.iloc[indices]
                    
                    # Extract data
                    Sh_boot = bootstrap_sample['Sh'].values
                    Re_boot = bootstrap_sample['Re'].values
                    Sc_boot = bootstrap_sample['Sc'].values
                    We_boot = bootstrap_sample['We'].values if 'We' in bootstrap_sample.columns else None
                    Eg_boot = bootstrap_sample['Eg'].values if 'Eg' in bootstrap_sample.columns else None
                    
                    # Define model function for this bootstrap sample
                    def model_function_boot(params):
                        A, X1, X2, X3, X4 = params
                        result = A * (Re_boot**X1) * (Sc_boot**X2)
                        
                        if model_type in [1, 2] and We_boot is not None:
                            result *= We_boot**X3
                            
                        if model_type in [1, 3] and Eg_boot is not None:
                            result *= Eg_boot**X4
                            
                        return result
                    
                    # Define objective function
                    def objective_function_boot(params):
                        predicted = model_function_boot(params)
                        return np.sum((Sh_boot - predicted)**2)
                    
                    # Initial parameter guess (use the best model parameters)
                    x0 = [a, x1, x2]
                    if x3 is not None:
                        x0.append(x3)
                    else:
                        x0.append(0)
                        
                    if x4 is not None:
                        x0.append(x4)
                    else:
                        x0.append(0)
                    
                    # Set bounds for optimization
                    bounds = [(0.1, 10.0), (0.65, 0.75), (0.33, 0.33)]
                    
                    if model_type in [1, 2]:
                        bounds.append((-0.5, -0.2))
                    else:
                        bounds.append((0, 0))
                        
                    if model_type in [1, 3]:
                        bounds.append((0.1, 0.15))
                    else:
                        bounds.append((0, 0))
                    
                    # Run optimization
                    try:
                        result = minimize(
                            objective_function_boot,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds
                        )
                        
                        # Calculate predicted values and RÂ²
                        predicted = model_function_boot(result.x)
                        ss_total = np.sum((Sh_boot - np.mean(Sh_boot))**2)
                        ss_residual = np.sum((Sh_boot - predicted)**2)
                        r2 = 1 - (ss_residual / ss_total)
                        
                        # Store results
                        bootstrap_params.append({
                            'a': result.x[0],
                            'x1': result.x[1],
                            'x2': result.x[2],
                            'x3': result.x[3] if model_type in [1, 2] else None,
                            'x4': result.x[4] if model_type in [1, 3] else None,
                            'r2': r2
                        })
                    except Exception as e:
                        st.warning(f"Bootstrap iteration {i} failed: {e}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / n_bootstrap)
                
                # Convert to dataframe
                bootstrap_df = pd.DataFrame(bootstrap_params)
                
                # Calculate confidence intervals
                a_mean = bootstrap_df['a'].mean()
                a_std = bootstrap_df['a'].std()
                a_ci_lower = np.percentile(bootstrap_df['a'], 2.5)
                a_ci_upper = np.percentile(bootstrap_df['a'], 97.5)
                
                x1_mean = bootstrap_df['x1'].mean()
                x1_std = bootstrap_df['x1'].std()
                x1_ci_lower = np.percentile(bootstrap_df['x1'], 2.5)
                x1_ci_upper = np.percentile(bootstrap_df['x1'], 97.5)
                
                r2_mean = bootstrap_df['r2'].mean()
                r2_std = bootstrap_df['r2'].std()
                r2_ci_lower = np.percentile(bootstrap_df['r2'], 2.5)
                r2_ci_upper = np.percentile(bootstrap_df['r2'], 97.5)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean 'a' Value", f"{a_mean:.4f}")
                    st.metric("'a' Standard Deviation", f"{a_std:.4f}")
                    st.write(f"95% Confidence Interval for 'a': [{a_ci_lower:.4f}, {a_ci_upper:.4f}]")
                
                with col2:
                    st.metric("Mean 'x1' Value", f"{x1_mean:.4f}")
                    st.metric("'x1' Standard Deviation", f"{x1_std:.4f}")
                    st.write(f"95% Confidence Interval for 'x1': [{x1_ci_lower:.4f}, {x1_ci_upper:.4f}]")
                
                with col3:
                    st.metric("Mean RÂ² Value", f"{r2_mean:.4f}")
                    st.metric("RÂ² Standard Deviation", f"{r2_std:.4f}")
                    st.write(f"95% Confidence Interval for RÂ²: [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
                
                # Plot parameter distributions
                fig = make_subplots(
                    rows=1, 
                    cols=3,
                    subplot_titles=("Distribution of 'a' Parameter", "Distribution of 'x1' Parameter", "Distribution of RÂ² Values")
                )
                
                # 'a' distribution
                fig.add_trace(
                    go.Histogram(
                        x=bootstrap_df['a'],
                        name="'a' Parameter",
                        marker=dict(color='blue'),
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=1
                )
                
                # Add vertical line for mean and CI
                fig.add_trace(
                    go.Scatter(
                        x=[a_mean, a_mean],
                        y=[0, bootstrap_df['a'].value_counts().max()],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Mean',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # 'x1' distribution
                fig.add_trace(
                    go.Histogram(
                        x=bootstrap_df['x1'],
                        name="'x1' Parameter",
                        marker=dict(color='green'),
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=2
                )
                
                # Add vertical line for mean and CI
                fig.add_trace(
                    go.Scatter(
                        x=[x1_mean, x1_mean],
                        y=[0, bootstrap_df['x1'].value_counts().max()],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Mean',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # RÂ² distribution
                fig.add_trace(
                    go.Histogram(
                        x=bootstrap_df['r2'],
                        name="RÂ² Value",
                        marker=dict(color='purple'),
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=1, col=3
                )
                
                # Add vertical line for mean and CI
                fig.add_trace(
                    go.Scatter(
                        x=[r2_mean, r2_mean],
                        y=[0, bootstrap_df['r2'].value_counts().max()],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Mean',
                        showlegend=False
                    ),
                    row=1, col=3
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("""
                ### Uncertainty Analysis Interpretation
                
                The bootstrap analysis provides insights into the uncertainty of the model parameters:
                
                - **Parameter 'a'**: The coefficient in the Sherwood number correlation. The 95% confidence interval shows the range of likely values.
                
                - **Parameter 'x1'**: The Reynolds number exponent. The confidence interval indicates the stability of this parameter.
                
                - **RÂ² Value**: The goodness of fit measure. The confidence interval indicates the stability of the model's predictive power.
                
                A narrow confidence interval suggests high confidence in the parameter estimates, while a wide interval indicates greater uncertainty.
                """)
        
        # Prediction uncertainty
        st.write("#### Prediction Uncertainty Analysis")
        
        # Create sliders for input parameters
        st.write("Enter parameter values to analyze prediction uncertainty:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            re_value = st.number_input("Reynolds Number (Re):", min_value=0.1, max_value=100000.0, value=float(data['Re'].mean()))
        
        with col2:
            sc_value = st.number_input("Schmidt Number (Sc):", min_value=0.1, max_value=10000.0, value=float(data['Sc'].mean()))
        
        extra_params = {}
        
        if model_type in [1, 2] and 'We' in data.columns:
            extra_params['We'] = st.number_input("Weber Number (We):", min_value=0.01, max_value=1000.0, value=float(data['We'].mean()))
        
        if model_type in [1, 3] and 'Eg' in data.columns:
            extra_params['Eg'] = st.number_input("Eotvos Number (Eg):", min_value=0.01, max_value=1000.0, value=float(data['Eg'].mean()))
        
        if st.button("Calculate Prediction Uncertainty"):
            # Calculate predicted Sh
            sh_pred = a * (re_value ** x1) * (sc_value ** x2)
            
            if x3 is not None and model_type in [1, 2] and 'We' in extra_params:
                sh_pred *= extra_params['We'] ** x3
                
            if x4 is not None and model_type in [1, 3] and 'Eg' in extra_params:
                sh_pred *= extra_params['Eg'] ** x4
            
            # Calculate MTC
            mtc_pred = sh_pred * diffusivity / char_length
            
            # Estimate prediction uncertainty (simplified approach)
            # Using error propagation formula
            rel_error_a = a_std / a_mean if 'a_std' in locals() else 0.05  # Default 5% if bootstrap not run
            rel_error_x1 = x1_std / x1_mean if 'x1_std' in locals() else 0.02  # Default 2% if bootstrap not run
            
            # Simplified error propagation
            rel_error_sh = np.sqrt(rel_error_a**2 + (rel_error_x1 * np.log(re_value))**2)
            
            sh_uncertainty = sh_pred * rel_error_sh
            mtc_uncertainty = mtc_pred * rel_error_sh
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Sherwood Number", f"{sh_pred:.2f} Â± {sh_uncertainty:.2f}")
            
            with col2:
                st.metric("Predicted Mass Transfer Coefficient", f"{mtc_pred:.2e} Â± {mtc_uncertainty:.2e} m/s")
            
            # Create visualization of uncertainty
            fig = go.Figure()
            
            # Sh prediction with uncertainty
            fig.add_trace(
                go.Scatter(
                    x=['Sherwood Number', 'Mass Transfer Coefficient'],
                    y=[sh_pred, mtc_pred],
                    mode='markers',
                    marker=dict(size=15, color='blue'),
                    name='Predicted Value'
                )
            )
            
            # Error bars
            fig.add_trace(
                go.Scatter(
                    x=['Sherwood Number', 'Mass Transfer Coefficient'],
                    y=[sh_pred, mtc_pred],
                    error_y=dict(
                        type='data',
                        array=[sh_uncertainty, mtc_uncertainty],
                        visible=True,
                        color='red',
                        thickness=1.5,
                        width=5
                    ),
                    mode='markers',
                    marker=dict(size=15, color='blue'),
                    name='Uncertainty'
                )
            )
            
            fig.update_layout(
                title='Prediction with Uncertainty',
                xaxis_title='Parameter',
                yaxis_title='Value',
                height=500,
                yaxis=dict(type='log')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("""
            ### Prediction Uncertainty Interpretation
            
            The prediction uncertainty represents the range within which the true value is likely to fall. 
            Factors contributing to uncertainty include:
            
            - Parameter estimation uncertainty
            - Model structural uncertainty
            - Measurement errors in the experimental data
            
            For engineering applications, it's important to consider this uncertainty when making design decisions.
            """)

# Run the app
if __name__ == "__main__":
    main()

#streamlit run /workspaces/Algorithmic_Trading_K25/MAJOR_PROJECT/t_1.py --server.enableCORS false --server.enableXsrfProtection false
