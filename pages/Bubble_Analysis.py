import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
import plotly.express as px
from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageEnhance, ImageFilter
import plotly.figure_factory as ff
from skimage import io as skio  # Rename skimage.io import to avoid conflict
from skimage import color, filters, feature, measure
from skimage.transform import resize
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO  # Add this import

# Create directory for saved images
if not os.path.exists('saved_images'):
    os.makedirs('saved_images')

def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes."""
    img_bytes = BytesIO()  # Use BytesIO from io module
    image.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

@st.cache_data
def analyze_image(image_bytes: bytes, bubble_params: dict, scale_factor: float):
    """Analyze image to detect bubbles."""
    image = Image.open(BytesIO(image_bytes))  # Use BytesIO directly
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    orig_height, orig_width = img.shape[:2]

    scale = 0.5 if bubble_params['speed_mode'] else 1.0
    processed_img = cv2.resize(img, (int(orig_width * scale), int(orig_height * scale)))

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=bubble_params['dp'],
        minDist=bubble_params['minDist'],
        param1=bubble_params['param1'],
        param2=bubble_params['param2'],
        minRadius=bubble_params['minRadius'],
        maxRadius=bubble_params['maxRadius']
    )

    return circles, processed_img, scale, img.shape[:2]

@st.cache_data
def analyze_image_with_skimage(image_bytes: bytes, bubble_params: dict, scale_factor: float):
    """Analyze image to detect bubbles using scikit-image."""
    image = Image.open(BytesIO(image_bytes))  # Use BytesIO directly
    img = np.array(image)
    orig_height, orig_width = img.shape[:2]

    # Convert to grayscale
    gray = color.rgb2gray(img)

    # Apply Gaussian blur
    blurred = filters.gaussian(gray, sigma=2)

    # Detect edges using Canny edge detection
    edges = feature.canny(blurred, sigma=2)

    # Detect circles using Hough Transform
    hough_radii = np.arange(bubble_params['minRadius'], bubble_params['maxRadius'], 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=100
    )

    # Create a DataFrame for bubble metrics
    bubble_data = []
    for x, y, r in zip(cx, cy, radii):
        bubble_data.append({
            'x': x,
            'y': y,
            'radius': r,
            'diameter_px': 2 * r,
            'diameter_mm': (2 * r) * (10 / scale_factor),
            'diameter_cm': (2 * r) / scale_factor,
        })

    df_metrics = pd.DataFrame(bubble_data)
    df_metrics['Area (cm²)'] = np.pi * (df_metrics['diameter_cm'] / 2) ** 2
    df_metrics = df_metrics.sort_values(by='diameter_px', ascending=False)
    df_metrics['Rank'] = range(1, len(df_metrics) + 1)

    # Create marked image
    marked_image = img.copy()
    for x, y, r in zip(cx, cy, radii):
        rr, cc = circle_perimeter(y, x, r)
        marked_image[rr, cc] = (255, 0, 0)  # Mark circles in red

    return df_metrics, marked_image

def get_saved_images():
    """Retrieve list of saved images."""
    return [f for f in os.listdir('saved_images') if os.path.isfile(os.path.join('saved_images', f))]

def main():
    st.title("🔍 Bubble Analyzer Pro")

    # Initialize session states
    states = ['confirmed', 'analyze', 'selected_rank']
    for state in states:
        if (state not in st.session_state):
            st.session_state[state] = False if state != 'selected_rank' else 1

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    # Option to load previously saved images
    saved_images = get_saved_images()
    selected_saved_image = st.selectbox("Or select a previously uploaded image", ["None"] + saved_images)

    if selected_saved_image != "None":
        uploaded_file = open(os.path.join('saved_images', selected_saved_image), "rb")
        st.success(f"Loaded saved image: {selected_saved_image}")

    if uploaded_file:
        # Confirm image section
        if not st.session_state.confirmed:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Confirm Image"):
                    st.session_state.confirmed = True
                    st.rerun()
        
        if st.session_state.confirmed:
            # Image saving option
            save_image = st.checkbox("Save image for future analysis")
            if save_image and selected_saved_image == "None":
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = f"saved_images/{timestamp}_{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Image saved to: {save_path}")

            # Detection parameters
            with st.expander("⚙️ Detection Parameters", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    dp = st.slider("Detection Precision", 1.0, 3.0, 1.2)
                    min_dist = st.slider("Minimum Bubble Distance (px)", 10, 100, 20)
                    param1 = st.slider("Edge Detection Threshold", 30, 300, 50)
                with col2:
                    param2 = st.slider("Circle Accumulator Threshold", 10, 100, 30)
                    min_radius = st.slider("Minimum Radius", 0, 100, 0)
                    max_radius = st.slider("Maximum Radius", 10, 500, 100)
                
                scale_factor = st.number_input("Pixels per cm", min_value=1.0, value=100.0)
                speed_mode = st.checkbox("Fast Processing Mode", True)

            # Analysis trigger
            if st.button("🔍 Start Image Analysis"):
                st.session_state.analyze = True
            
            if st.session_state.analyze:
                image = Image.open(uploaded_file)
                image_bytes = image_to_bytes(image)

                bubble_params = {
                    'dp': dp,
                    'minDist': min_dist,
                    'param1': param1,
                    'param2': param2,
                    'minRadius': min_radius,
                    'maxRadius': max_radius,
                    'speed_mode': speed_mode
                }

                circles, processed_img, proc_scale, orig_dims = analyze_image(
                    image_bytes, bubble_params, scale_factor
                )

                # Create output tabs
                tabs = ["📤 Original", "📊 Overview", "📋 Metrics", "🔴 Marked Bubbles", "📌 Rank Analysis" ,"3D Analysis"]
                tab1, tab2, tab3, tab4, tab5 ,tab6 = st.tabs(tabs)
                if circles is not None:
                    # Convert processed image coordinates to original dimensions
                    orig_width, orig_height = orig_dims
                    circles_np = np.uint16(np.around(circles))
                    
                    # Create dataframe with scaled coordinates
                    bubble_data = []
                    for circle in circles_np[0, :]:
                        x_proc, y_proc, r_proc = circle
                        x_orig = int((x_proc / proc_scale) * (orig_width / (orig_width * proc_scale)))
                        y_orig = int((y_proc / proc_scale) * (orig_height / (orig_height * proc_scale)))
                        radius_orig = int(r_proc / proc_scale)
                        
                        bubble_data.append({
                            'x': x_orig,
                            'y': y_orig,
                            'radius': radius_orig,
                            'diameter_px': 2 * radius_orig,
                            'diameter_mm': (2 * radius_orig) * (10 / scale_factor),
                            'diameter_cm': (2 * radius_orig) / scale_factor,
                        })

                    df_metrics = pd.DataFrame(bubble_data)
                    df_metrics['Area (cm²)'] = np.pi * (df_metrics['diameter_cm']/2)**2
                    df_metrics = df_metrics.sort_values(by='diameter_px', ascending=False)
                    df_metrics['Rank'] = range(1, len(df_metrics)+1)
                    df_metrics = df_metrics[['Rank', 'x', 'y', 'diameter_px', 'diameter_mm', 
                                           'diameter_cm', 'Area (cm²)']]

                    # Create marked image
                    original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    for idx, row in df_metrics.iterrows():
                        x, y, r = int(row['x']), int(row['y']), int(row['diameter_px']/2)
                        cv2.rectangle(original_image, 
                                    (x - r, y - r), 
                                    (x + r, y + r), 
                                    (0, 0, 255), 2)
                        cv2.putText(original_image, str(row['Rank']), 
                                  (x + r + 5, y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    marked_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                    with tab4:
                        st.image(marked_image, caption="Ranked Bubbles", use_container_width=True)

                    with tab5:
                        max_rank = len(df_metrics)
                        selected_rank = st.number_input("Enter Bubble Rank", 
                                                       min_value=1, max_value=max_rank, 
                                                       value=1)
                        if selected_rank:
                            bubble = df_metrics[df_metrics['Rank'] == selected_rank].iloc[0]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Diameter (cm)", f"{bubble['diameter_cm']:.4f}")
                                st.metric("Position X", f"{bubble['x']} px")
                            with col2:
                                st.metric("Area (cm²)", f"{bubble['Area (cm²)']:.4f}")
                                st.metric("Position Y", f"{bubble['y']} px")
                            
                            # Highlight selected bubble
                            highlighted = original_image.copy()
                            x, y, r = int(bubble['x']), int(bubble['y']), int(bubble['diameter_px']/2)
                            cv2.rectangle(highlighted, 
                                        (x - r, y - r), 
                                        (x + r, y + r), 
                                        (0, 255, 0), 3)
                            st.image(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB), 
                                   caption=f"Bubble Rank {selected_rank}", 
                                   use_container_width=True)
                    with tab6:
                        st.header("3D Analysis")
                        subtabs = ["Size Distribution", "Spatial Distribution", "Density Map", "Bubble Clusters", "Time Evolution"]
                        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(subtabs)

                        with subtab1:
                            fig = px.scatter_3d(df_metrics, x='x', y='y', z='diameter_cm', 
                                                color='diameter_cm', size='diameter_cm',
                                                title="3D Bubble Size Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Size Distribution Summary")
                            size_distribution_table = df_metrics['diameter_cm'].describe().reset_index()
                            size_distribution_table.columns = ['Metric', 'Value']
                            st.table(size_distribution_table.round(4))
                            
                            st.write("Conclusion: This 3D plot shows the spatial distribution of bubbles with their sizes. "
                                    "Larger bubbles are represented by bigger and more intensely colored points. "
                                    "The table provides a statistical summary of bubble sizes, helping to quantify "
                                    "the distribution we observe in the plot.")

                        with subtab2:
                            fig = go.Figure(data=[go.Scatter3d(x=df_metrics['x'], y=df_metrics['y'], z=df_metrics['diameter_cm'],
                                                            mode='markers', marker=dict(size=5, color=df_metrics['diameter_cm'], 
                                                            colorscale='Viridis', opacity=0.8))])
                            fig.update_layout(title="3D Spatial Distribution of Bubbles")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Spatial Distribution Summary")
                            spatial_distribution_table = df_metrics[['x', 'y', 'diameter_cm']].describe().reset_index()
                            spatial_distribution_table.columns = ['Metric'] + list(spatial_distribution_table.columns)[1:]
                            st.table(spatial_distribution_table.round(4))
                            
                            st.write("Conclusion: This plot reveals the spatial patterns of bubbles across the image. "
                                    "The accompanying table summarizes the distribution of x and y coordinates, as well as "
                                    "bubble diameters, providing insights into the overall spatial characteristics of the bubbles.")

                        with subtab3:
                            x = df_metrics['x']
                            y = df_metrics['y']
                            z = df_metrics['diameter_cm']
                            
                            xi = np.linspace(x.min(), x.max(), 100)
                            yi = np.linspace(y.min(), y.max(), 100)
                            zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
                            
                            fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Viridis')])
                            fig.update_layout(title="3D Density Map of Bubbles")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Density Map Summary")
                            density_map_table = pd.DataFrame({
                                'Metric': ['Min Density', 'Max Density', 'Mean Density', 'Median Density'],
                                'Value': [zi.min(), zi.max(), zi.mean(), np.median(zi[~np.isnan(zi)])]
                            })
                            st.table(density_map_table.round(4))
                            
                            st.write("Conclusion: The 3D density map visualizes the concentration of bubbles in different "
                                    "areas of the image. Brighter and more elevated regions indicate higher bubble density. "
                                    "The table provides key statistics about the density distribution, helping to quantify "
                                    "the variations in bubble concentration across the image.")

                        with subtab4:
                            kmeans = KMeans(n_clusters=3)
                            df_metrics['Cluster'] = kmeans.fit_predict(df_metrics[['x', 'y', 'diameter_cm']])
                            fig = px.scatter_3d(df_metrics, x='x', y='y', z='diameter_cm', color='Cluster',
                                                title="3D Bubble Clusters")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Cluster Analysis Summary")
                            cluster_summary = df_metrics.groupby('Cluster').agg({
                                'x': 'mean',
                                'y': 'mean',
                                'diameter_cm': 'mean',
                                'Cluster': 'count'
                            }).rename(columns={'Cluster': 'Count'}).reset_index()
                            st.table(cluster_summary.round(4))
                            
                            st.write("Conclusion: This clustering analysis groups bubbles based on their spatial location "
                                    "and size. The table summarizes the characteristics of each cluster, including the "
                                    "average x, y coordinates, mean diameter, and number of bubbles in each cluster. "
                                    "This helps identify distinct populations of bubbles, which might correspond to "
                                    "different formation processes or environmental conditions in the sample.")

                        with subtab5:
                            st.subheader("Single Bubble Time Evolution")
                            # Let user select bubble rank for analysis
                            selected_bubble_rank = st.number_input("Select Bubble Rank for Time Evolution", 
                                                                 min_value=1, 
                                                                 max_value=len(df_metrics),
                                                                 value=1)
                            
                            # Get selected bubble data
                            selected_bubble = df_metrics[df_metrics['Rank'] == selected_bubble_rank].iloc[0]
                            
                            # Generate time series data for selected bubble
                            time_points = np.linspace(0, 10, 50)
                            diameter_evolution = selected_bubble['diameter_cm'] * (1 + 0.1 * np.sin(time_points))
                            
                            # Create time evolution DataFrame for the selected bubble
                            bubble_evolution = pd.DataFrame({
                                'Time': time_points,
                                'x': selected_bubble['x'],
                                'y': selected_bubble['y'],
                                'diameter_cm': diameter_evolution
                            })
                            
                            # Create 3D animation
                            fig = px.scatter_3d(bubble_evolution, 
                                              x='x', y='y', z='Time',
                                              color='diameter_cm',
                                              size='diameter_cm',
                                              animation_frame='Time',
                                              title=f"Time Evolution of Bubble Rank {selected_bubble_rank}")
                            
                            fig.update_traces(marker=dict(symbol='circle'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show evolution statistics
                            st.subheader("Evolution Statistics")
                            evolution_stats = pd.DataFrame({
                                'Metric': ['Initial Diameter', 'Max Diameter', 'Min Diameter', 'Mean Diameter'],
                                'Value (cm)': [
                                    f"{bubble_evolution['diameter_cm'].iloc[0]:.4f}",
                                    f"{bubble_evolution['diameter_cm'].max():.4f}",
                                    f"{bubble_evolution['diameter_cm'].min():.4f}",
                                    f"{bubble_evolution['diameter_cm'].mean():.4f}"
                                ]
                            })
                            st.table(evolution_stats)
                            
                            st.write("Note: This simulation shows a hypothetical time evolution of the selected bubble's "
                                   "diameter using a sinusoidal pattern. In real applications, this could be replaced "
                                   "with actual time series data if available.")
                                                                    
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_container_width=True)
                    with col2:
                        output_img = processed_img.copy()
                        circles_np = np.uint16(np.around(circles))
                        for circle in circles_np[0]:
                            cv2.circle(output_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                            cv2.circle(output_img, (circle[0], circle[1]), 2, (0, 0, 255), 3)
                        st.image(output_img, caption="Processed Image", use_container_width=True)

                if circles is not None:
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Key Metrics")
                            st.metric("Total Bubbles", len(df_metrics))
                            st.metric("Avg Diameter (cm)", f"{df_metrics['diameter_cm'].mean():.4f}")
                        with col2:
                            st.subheader("Size Distribution")
                            fig = go.Figure([go.Histogram(x=df_metrics['diameter_cm'], nbinsx=20)])
                            st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        st.dataframe(df_metrics.style.format({
                            'diameter_cm': '{:.4f}',
                            'Area (cm²)': '{:.4f}'
                        }))

if __name__ == "__main__":
    st.set_page_config(
        page_title="Bubble Analyzer Pro",
        layout="wide"
    )
    main()

# streamlit run /workspaces/Prsnl_APP/MAJOR_PROJECT/pages/Bubble_Analysis.py --server.enableCORS false --server.enableXsrfProtection false
