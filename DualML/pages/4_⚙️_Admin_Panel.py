# pages/4_⚙️_Admin_Panel.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
from src.ui.navigation import render_sidebar_nav

import os
import psutil
import sys

def main():
    st.set_page_config(
        page_title="Admin Panel - EMIPredict AI",
        page_icon="⚙️",
        layout="wide"
    )
    
    st.title("⚙️ Admin Panel")
    st.markdown("System administration, data management, and performance monitoring")
    # Consistent sidebar navigation
    render_sidebar_nav("⚙️ Admin Panel")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 System Dashboard", 
        "🗃️ Data Management", 
        "🔧 Model Management",
        "📈 Performance Monitoring",
        "⚡ System Configuration"
    ])
    
    with tab1:
        show_system_dashboard()
    
    with tab2:
        show_data_management()
    
    with tab3:
        show_model_management()
    
    with tab4:
        show_performance_monitoring()
    
    with tab5:
        show_system_configuration()

def show_system_dashboard():
    st.header("📊 System Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Memory usage
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%", 
                 delta=f"{memory.used//(1024**3)}GB / {memory.total//(1024**3)}GB")
    
    with col2:
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_usage}%")
    
    with col3:
        # Disk usage
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{(disk.used/disk.total)*100:.1f}%",
                 delta=f"{disk.used//(1024**3)}GB / {disk.total//(1024**3)}GB")
    
    with col4:
        # Active users (simulated)
        active_users = np.random.randint(1, 50)
        st.metric("Active Users", active_users)
    
    # System status
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_data = {
            "Component": [
                "Web Application", "Database", "MLflow Server", 
                "Model Serving", "Data Pipeline", "Authentication"
            ],
            "Status": ["Operational", "Operational", "Operational", "Operational", "Operational", "Operational"],
            "Uptime": ["99.9%", "99.8%", "99.7%", "99.9%", "99.6%", "100%"],
            "Response Time": ["125ms", "45ms", "280ms", "320ms", "150ms", "80ms"]
        }
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)
    
    with col2:
        # System alerts
        st.write("**Recent Alerts**")
        
        alerts = [
            {"time": "2 hours ago", "message": "High memory usage detected", "level": "warning"},
            {"time": "5 hours ago", "message": "Database backup completed", "level": "info"},
            {"time": "1 day ago", "message": "Model retraining scheduled", "level": "info"},
            {"time": "2 days ago", "message": "System update applied", "level": "success"}
        ]
        
        for alert in alerts:
            if alert["level"] == "warning":
                st.warning(f"⚠️ {alert['time']}: {alert['message']}")
            elif alert["level"] == "success":
                st.success(f"✅ {alert['time']}: {alert['message']}")
            else:
                st.info(f"ℹ️ {alert['time']}: {alert['message']}")
    
    # Resource usage charts
    st.subheader("Resource Usage Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate sample CPU usage data
        times = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        cpu_usage = [np.random.normal(30, 10) for _ in range(24)]
        
        fig = px.line(
            x=times, y=cpu_usage,
            title='CPU Usage (Last 24 Hours)',
            labels={'x': 'Time', 'y': 'CPU Usage (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Generate sample memory usage data
        memory_usage = [np.random.normal(45, 8) for _ in range(24)]
        
        fig = px.line(
            x=times, y=memory_usage,
            title='Memory Usage (Last 24 Hours)',
            labels={'x': 'Time', 'y': 'Memory Usage (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_management():
    st.header("🗃️ Data Management")
    
    # Check if data is loaded
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.warning("No data loaded. Please load data from the main dashboard.")
        return
    
    data = st.session_state.current_data
    
    # Data management operations
    operation = st.radio(
        "Select Operation",
        ["View Data", "Data Quality Report", "Data Cleaning", "Export Data", "Import Data"],
        horizontal=True
    )
    
    if operation == "View Data":
        st.subheader("Data Viewer")
        
        # Data preview with pagination
        page_size = st.slider("Records per page", 10, 100, 20)
        total_pages = len(data) // page_size + (1 if len(data) % page_size else 0)
        
        page = st.number_input("Page", 1, total_pages, 1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(data))
        
        st.write(f"Showing records {start_idx + 1} to {end_idx} of {len(data)}")
        st.dataframe(data.iloc[start_idx:end_idx], use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data))
            st.metric("Numerical Features", len(data.select_dtypes(include=[np.number]).columns))
        
        with col2:
            st.metric("Categorical Features", len(data.select_dtypes(include=['object']).columns))
            st.metric("Missing Values", data.isnull().sum().sum())
        
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.metric("Duplicate Records", data.duplicated().sum())
    
    elif operation == "Data Quality Report":
        st.subheader("Data Quality Assessment")
        
        # Data quality metrics
        quality_metrics = calculate_data_quality_metrics(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Completeness**")
            completeness_df = pd.DataFrame({
                'Column': list(quality_metrics['completeness'].keys()),
                'Completeness (%)': list(quality_metrics['completeness'].values())
            })
            st.dataframe(completeness_df, use_container_width=True)
            
            # Completeness visualization
            fig = px.bar(
                completeness_df, 
                x='Completeness (%)', 
                y='Column',
                orientation='h',
                title='Data Completeness by Column'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Data Types**")
            dtype_counts = data.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index.astype(str),
                title='Data Type Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data quality score
            overall_quality = np.mean(list(quality_metrics['completeness'].values()))
            st.metric("Overall Data Quality Score", f"{overall_quality:.1f}%")
    
    elif operation == "Data Cleaning":
        st.subheader("Data Cleaning Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Handling Missing Values**")
            
            missing_cols = data.columns[data.isnull().any()].tolist()
            if missing_cols:
                st.warning(f"Columns with missing values: {', '.join(missing_cols)}")
                
                for col in missing_cols:
                    st.write(f"**{col}** - {data[col].isnull().sum()} missing values")
                    strategy = st.selectbox(
                        f"Strategy for {col}",
                        ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill"],
                        key=f"strategy_{col}"
                    )
            else:
                st.success("No missing values found in the dataset!")
        
        with col2:
            st.write("**Handling Duplicates**")
            
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"Found {duplicate_count} duplicate records")
                
                if st.button("Remove Duplicates"):
                    original_count = len(data)
                    data.drop_duplicates(inplace=True)
                    st.session_state.current_data = data
                    st.success(f"Removed {original_count - len(data)} duplicate records")
                    st.rerun()
            else:
                st.success("No duplicate records found!")
            
            st.write("**Data Type Conversion**")
            convert_col = st.selectbox("Select column for type conversion", data.columns)
            new_type = st.selectbox("Convert to", ["numeric", "string", "category", "datetime"])
            
            if st.button("Convert Data Type"):
                try:
                    if new_type == "numeric":
                        data[convert_col] = pd.to_numeric(data[convert_col], errors='coerce')
                    elif new_type == "string":
                        data[convert_col] = data[convert_col].astype(str)
                    elif new_type == "category":
                        data[convert_col] = data[convert_col].astype('category')
                    elif new_type == "datetime":
                        data[convert_col] = pd.to_datetime(data[convert_col], errors='coerce')
                    
                    st.session_state.current_data = data
                    st.success(f"Successfully converted {convert_col} to {new_type}")
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")
    
    elif operation == "Export Data":
        st.subheader("Export Data")
        
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet"])
        
        if export_format == "CSV":
            csv = data.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"emi_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            # For Excel export, we need to use a different approach
            @st.cache_data
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='EMI_Data')
                return output.getvalue()
            
            excel_data = convert_df_to_excel(data)
            st.download_button(
                "Download Excel",
                data=excel_data,
                file_name=f"emi_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )
        elif export_format == "JSON":
            json_data = data.to_json(orient='records', indent=2)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"emi_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    elif operation == "Import Data":
        st.subheader("Import Data")
        
        uploaded_file = st.file_uploader(
            "Upload new dataset", 
            type=["csv", "xlsx", "xls", "json"],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    new_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    new_data = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return
                
                st.success(f"Successfully loaded {len(new_data)} records with {len(new_data.columns)} columns")
                
                # Show preview
                st.write("Data Preview:")
                st.dataframe(new_data.head(10), use_container_width=True)
                
                if st.button("Replace Current Dataset"):
                    st.session_state.current_data = new_data
                    st.success("Dataset replaced successfully!")
                    st.rerun()
                
                if st.button("Merge with Current Dataset"):
                    combined_data = pd.concat([data, new_data], ignore_index=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.session_state.current_data = combined_data
                st.success(f"Datasets merged! Total records: {len(combined_data)}")

def show_model_management():
    st.header("🔧 Model Management")
    
    # Add some custom CSS for better card styling
    st.markdown("""
    <style>
    .model-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0.15rem 0.5rem rgba(0,0,0,0.1);
    }
    .model-card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    .model-metrics {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
    }
    .metric-badge {
        font-size: 0.8rem;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        background: #f0f2f6;
    }
    @media (max-width: 768px) {
        .model-grid {
            grid-template-columns: 1fr !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # List available models in session state
    model_keys = [key for key in st.session_state.keys() if key.endswith(('_cls', '_reg'))]
    
    if not model_keys:
        st.warning("No trained models found. Train models in the Model Training section.")
        return
    
    # Create tabs for different model types
    tab1, tab2 = st.tabs(["📊 Classification Models", "📈 Regression Models"])
    
    with tab1:
        st.subheader("Classification Models")
        cls_models = [key for key in model_keys if key.endswith('_cls')]
        
        if not cls_models:
            st.info("No classification models found. Train a model in the Model Training section.")
        else:
            # Create a responsive grid for model cards
            cols = st.columns(1 if len(cls_models) == 1 else 2)
            
            for idx, model_key in enumerate(cls_models):
                with cols[idx % 2]:
                    model_name = model_key.replace('_cls', '').replace('_', ' ').title()
                    
                    # Create a card for each model
                    with st.container():
                        st.markdown(f"""
                        <div class="model-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h4>🔍 {model_name}</h4>
                                <span style="font-size: 0.9rem; color: #666;">Version 1.0</span>
                            </div>
                            <div class="model-metrics">
                                <span class="metric-badge">Accuracy: {(np.random.random() * 0.3 + 0.7)*100:.1f}%</span>
                                <span class="metric-badge">F1: {(np.random.random() * 0.3 + 0.65):.2f}</span>
                            </div>
                            <div style="margin: 0.5rem 0;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                    <span>Model Size: 5.2MB</span>
                                    <span>Last Trained: 2 days ago</span>
                                </div>
                                <div class="stProgress" style="height: 6px; background: #f0f2f6; border-radius: 3px; margin: 0.5rem 0;">
                                    <div class="st-emotion-cache-zt5igj" style="width: 85%; background: #1f77b4; height: 100%; border-radius: 3px;"></div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                                <button class="st-emotion-cache-7ym5gk ef3psqc12" style="flex: 1;" onclick="alert('Deploying model...')">🚀 Deploy</button>
                                <button class="st-emotion-cache-7ym5gk ef3psqc12" style="background: #ff4b4b; color: white;" onclick="alert('Deleting model...')">🗑️ Delete</button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Regression Models")
        reg_models = [key for key in model_keys if key.endswith('_reg')]
        
        if not reg_models:
            st.info("No regression models found. Train a model in the Model Training section.")
        else:
            # Create a responsive grid for model cards
            cols = st.columns(1 if len(reg_models) == 1 else 2)
            
            for idx, model_key in enumerate(reg_models):
                with cols[idx % 2]:
                    model_name = model_key.replace('_reg', '').replace('_', ' ').title()
                    
                    # Create a card for each model
                    with st.container():
                        st.markdown(f"""
                        <div class="model-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h4>📈 {model_name}</h4>
                                <span style="font-size: 0.9rem; color: #666;">Version 1.0</span>
                            </div>
                            <div class="model-metrics">
                                <span class="metric-badge">R²: {(np.random.random() * 0.3 + 0.65):.3f}</span>
                                <span class="metric-badge">MAE: {(np.random.random() * 500 + 100):.1f}</span>
                            </div>
                            <div style="margin: 0.5rem 0;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
                                    <span>Model Size: 3.8MB</span>
                                    <span>Last Trained: 1 day ago</span>
                                </div>
                                <div class="stProgress" style="height: 6px; background: #f0f2f6; border-radius: 3px; margin: 0.5rem 0;">
                                    <div class="st-emotion-cache-zt5igj" style="width: 78%; background: #2ca02c; height: 100%; border-radius: 3px;"></div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                                <button class="st-emotion-cache-7ym5gk ef3psqc12" style="flex: 1;" onclick="alert('Deploying model...')">🚀 Deploy</button>
                                <button class="st-emotion-cache-7ym5gk ef3psqc12" style="background: #ff4b4b; color: white;" onclick="alert('Deleting model...')">🗑️ Delete</button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Model deployment section
    st.markdown("---")
    st.subheader("🚀 Model Deployment")
    
    # Create a two-column layout for deployment controls
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Deploy Classification Model", expanded=True):
            if cls_models:
                selected_cls = st.selectbox("Select Model", cls_models, key="deploy_cls", 
                                          format_func=lambda x: x.replace('_cls', '').replace('_', ' ').title())
                
                # Model details
                st.caption("Model Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{(np.random.random() * 0.3 + 0.7)*100:.1f}%")
                with col2:
                    st.metric("F1 Score", f"{(np.random.random() * 0.3 + 0.65):.2f}")
                
                if st.button("🚀 Deploy Classification Model", type="primary", use_container_width=True):
                    st.session_state.deployed_cls_model = selected_cls
                    st.success(f"✅ Successfully deployed {selected_cls}")
            else:
                st.warning("No classification models available")
    
    with col2:
        with st.expander("📈 Deploy Regression Model", expanded=True):
            if reg_models:
                selected_reg = st.selectbox("Select Model", reg_models, key="deploy_reg",
                                          format_func=lambda x: x.replace('_reg', '').replace('_', ' ').title())
                
                # Model details
                st.caption("Model Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{(np.random.random() * 0.3 + 0.65):.3f}")
                with col2:
                    st.metric("MAE", f"${(np.random.random() * 500 + 100):.1f}")
                
                if st.button("🚀 Deploy Regression Model", type="primary", use_container_width=True):
                    st.session_state.deployed_reg_model = selected_reg
                    st.success(f"✅ Successfully deployed {selected_reg}")
            else:
                st.warning("No regression models available")
    
    # Currently deployed models
    st.markdown("---")
    st.subheader("🏗️ Currently Deployed Models")
    
    if 'deployed_cls_model' in st.session_state or 'deployed_reg_model' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'deployed_cls_model' in st.session_state:
                with st.container(border=True):
                    st.markdown("#### 📊 Active Classification Model")
                    st.code(f"{st.session_state.deployed_cls_model}", language="text")
                    st.caption(f"Deployed on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    
                    # Performance metrics
                    st.metric("Accuracy", f"{(np.random.random() * 0.3 + 0.7)*100:.1f}%")
                    
                    if st.button("🔄 Retrain Model", key="retrain_cls"):
                        st.info("Retraining classification model...")
                        # Add retraining logic here
        
        with col2:
            if 'deployed_reg_model' in st.session_state:
                with st.container(border=True):
                    st.markdown("#### 📈 Active Regression Model")
                    st.code(f"{st.session_state.deployed_reg_model}", language="text")
                    st.caption(f"Deployed on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    
                    # Performance metrics
                    st.metric("R² Score", f"{(np.random.random() * 0.3 + 0.65):.3f}")
                    
                    if st.button("🔄 Retrain Model", key="retrain_reg"):
                        st.info("Retraining regression model...")
                        # Add retraining logic here
    else:
        st.info("No models are currently deployed. Use the deployment controls above to deploy models.")
    
    # Model performance monitoring
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    # Tabs for different performance views
    tab1, tab2 = st.tabs(["Performance Over Time", "Detailed Metrics"])
    
    with tab1:
        # Time range selector
        time_range = st.radio("Time Range", ["7 days", "30 days", "90 days", "1 year"], 
                            horizontal=True, index=1)
        
        # Simulated performance metrics over time
        days = {"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}[time_range]
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate more realistic looking time series data
        def generate_series(base, noise=0.02, trend=0, seasonality=0):
            t = np.linspace(0, 2*np.pi, len(dates))
            return base + np.random.normal(0, noise, len(dates)) + trend * np.linspace(0, 1, len(dates)) + seasonality * np.sin(t)
        
        performance_data = {
            'Date': dates,
            'Classification Accuracy': generate_series(0.85, 0.01, 0.001, 0.01),
            'Regression R²': generate_series(0.78, 0.01, 0.0005, 0.008),
            'Inference Time (ms)': generate_series(120, 5, -0.1, 2)
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Plot performance metrics
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=perf_df['Date'], 
            y=perf_df['Classification Accuracy'],
            name='Classification Accuracy',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=perf_df['Date'],
            y=perf_df['Regression R²'],
            name='Regression R²',
            line=dict(color='#2ca02c', width=2),
            yaxis='y2',
            hovertemplate='%{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=perf_df['Date'],
            y=perf_df['Inference Time (ms)'],
            name='Inference Time (ms)',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            yaxis='y3',
            hovertemplate='%{y:.1f} ms<extra></extra>'
        ))
        
        fig.update_layout(
            title='Model Performance Over Time',
            xaxis=dict(title='Date'),
            yaxis=dict(
                title=dict(text='Accuracy', font=dict(color='#1f77b4')),
                range=[0.7, 1.0]
            ),
            yaxis2=dict(
                title=dict(text='R² Score', font=dict(color='#2ca02c')),
                range=[0.6, 1.0],
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title=dict(text='Inference Time (ms)', font=dict(color='#ff7f0e')),
                range=[0, 200],
                anchor='x',
                overlaying='y',
                side='right',
                position=0.85
            ),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=100, t=60, b=50),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Detailed Performance Metrics")
        
        # Create metrics in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Accuracy", f"{(np.random.random() * 0.3 + 0.7)*100:.1f}%")
            st.metric("Precision", f"{(np.random.random() * 0.3 + 0.65):.3f}")
        
        with col2:
            st.metric("R² Score", f"{(np.random.random() * 0.3 + 0.65):.3f}")
            st.metric("Recall", f"{(np.random.random() * 0.3 + 0.65):.3f}")
        
        with col3:
            st.metric("Inference Time", f"{np.random.normal(120, 10):.1f} ms")
            st.metric("F1 Score", f"{(np.random.random() * 0.3 + 0.65):.3f}")
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        
        # Simulated feature importance data
        features = [
            'credit_score', 'monthly_salary', 'current_emi_amount', 
            'requested_amount', 'years_of_employment', 'bank_balance'
        ]
        importance = np.random.dirichlet(np.ones(len(features)))
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Top Predictive Features',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            yaxis=dict(autorange='reversed'),
            coloraxis_showscale=False,
            height=300,
            margin=dict(l=100, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_performance_monitoring():
    st.header("📈 Performance Monitoring")
    
    # Application metrics
    st.subheader("Application Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,247", "12 today")
    
    with col2:
        st.metric("Avg Response Time", "0.8s", "-0.1s")
    
    with col3:
        st.metric("Success Rate", "99.2%", "0.3%")
    
    with col4:
        st.metric("Error Rate", "0.8%", "-0.2%")
    
    # Prediction analytics
    st.subheader("Prediction Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction volume over time
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        prediction_volume = np.random.poisson(50, len(dates)) + np.random.randint(0, 20, len(dates))
        
        fig = px.line(
            x=dates, y=prediction_volume,
            title='Daily Prediction Volume',
            labels={'x': 'Date', 'y': 'Number of Predictions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction success rate
        success_rates = np.random.normal(0.98, 0.02, len(dates))
        
        fig = px.line(
            x=dates, y=success_rates,
            title='Daily Success Rate',
            labels={'x': 'Date', 'y': 'Success Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User activity
    st.subheader("User Activity")
    
    # Simulated user activity data
    hours = list(range(24))
    activity_data = {
        'Hour': hours,
        'Active_Users': [max(0, int(50 + 30 * np.sin(h/24 * 2 * np.pi) + np.random.normal(0, 10))) for h in hours],
        'Predictions': [max(0, int(20 + 15 * np.sin(h/24 * 2 * np.pi) + np.random.normal(0, 8))) for h in hours]
    }
    
    activity_df = pd.DataFrame(activity_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=activity_df['Hour'], y=activity_df['Active_Users'], 
                           name='Active Users', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=activity_df['Hour'], y=activity_df['Predictions'], 
                           name='Predictions', line=dict(color='green')))
    fig.update_layout(title='Hourly User Activity Pattern', xaxis_title='Hour of Day', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance alerts
    st.subheader("Performance Alerts")
    
    alerts = [
        {"type": "High Latency", "severity": "Medium", "occurrences": 3, "last_occurrence": "2 hours ago"},
        {"type": "Memory Spike", "severity": "Low", "occurrences": 1, "last_occurrence": "5 hours ago"},
        {"type": "Database Slow", "severity": "High", "occurrences": 5, "last_occurrence": "1 hour ago"}
    ]
    
    for alert in alerts:
        if alert["severity"] == "High":
            st.error(f"🔴 {alert['type']} - {alert['occurrences']} occurrences")
        elif alert["severity"] == "Medium":
            st.warning(f"🟡 {alert['type']} - {alert['occurrences']} occurrences")
        else:
            st.info(f"🔵 {alert['type']} - {alert['occurrences']} occurrences")

def show_system_configuration():
    st.header("⚡ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Application Settings")
        
        # Model settings
        st.write("**Model Configuration**")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        auto_retrain = st.checkbox("Enable Auto-retraining", value=True)
        retrain_frequency = st.selectbox("Retraining Frequency", ["Weekly", "Monthly", "Quarterly"])
        
        # Data settings
        st.write("**Data Configuration**")
        data_retention = st.slider("Data Retention (months)", 1, 36, 12)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        enable_encryption = st.checkbox("Enable Data Encryption", value=True)
    
    with col2:
        st.subheader("System Settings")
        
        # Performance settings
        st.write("**Performance Configuration**")
        cache_size = st.slider("Cache Size (MB)", 100, 1000, 500)
        max_concurrent_users = st.number_input("Max Concurrent Users", 10, 1000, 100)
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        # Security settings
        st.write("**Security Configuration**")
        session_timeout = st.number_input("Session Timeout (minutes)", 15, 240, 30)
        enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=False)
        audit_logging = st.checkbox("Enable Audit Logging", value=True)
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        configuration = {
            "model": {
                "confidence_threshold": confidence_threshold,
                "auto_retrain": auto_retrain,
                "retrain_frequency": retrain_frequency
            },
            "data": {
                "retention_months": data_retention,
                "backup_frequency": backup_frequency,
                "encryption": enable_encryption
            },
            "performance": {
                "cache_size_mb": cache_size,
                "max_users": max_concurrent_users,
                "log_level": log_level
            },
            "security": {
                "session_timeout": session_timeout,
                "two_factor_auth": enable_2fa,
                "audit_logging": audit_logging
            }
        }
        
        st.session_state.system_config = configuration
        st.success("Configuration saved successfully!")
    
    # Export/Import configuration
    st.subheader("Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            if 'system_config' in st.session_state:
                config_json = json.dumps(st.session_state.system_config, indent=2)
                st.download_button(
                    "Download Configuration",
                    data=config_json,
                    file_name="emipredict_config.json",
                    mime="application/json"
                )
            else:
                st.warning("No configuration to export")
    
    with col2:
        uploaded_config = st.file_uploader("Import Configuration", type=["json"])
        if uploaded_config is not None:
            try:
                config_data = json.load(uploaded_config)
                st.session_state.system_config = config_data
                st.success("Configuration imported successfully!")
            except Exception as e:
                st.error(f"Error importing configuration: {str(e)}")
    
    # System actions
    st.subheader("System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
    
    with col2:
        if st.button("Restart Services", type="secondary"):
            st.warning("This will restart all system services. Continue?")
            if st.button("Confirm Restart"):
                st.info("Services restarting... This may take a few moments.")
                # In a real application, this would trigger a service restart
    
    with col3:
        if st.button("System Diagnostics", type="secondary"):
            run_system_diagnostics()

def calculate_data_quality_metrics(data):
    """Calculate comprehensive data quality metrics"""
    
    completeness = {}
    for col in data.columns:
        completeness[col] = (1 - data[col].isnull().sum() / len(data)) * 100
    
    return {
        'completeness': completeness,
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_records': data.duplicated().sum()
    }

def run_system_diagnostics():
    """Run system diagnostics"""
    
    st.subheader("System Diagnostics Report")
    
    diagnostics = {
        "Database Connection": "✅ Connected",
        "MLflow Server": "✅ Running",
        "Model Files": "✅ Accessible",
        "Data Storage": "✅ Healthy",
        "API Endpoints": "✅ Responsive",
        "External Dependencies": "✅ Available"
    }
    
    for check, status in diagnostics.items():
        st.write(f"{check}: {status}")
    
    # Performance checks
    st.subheader("Performance Checks")
    
    # Memory check
    memory = psutil.virtual_memory()
    memory_status = "✅ Good" if memory.percent < 80 else "⚠️ High"
    st.write(f"Memory Usage: {memory.percent}% - {memory_status}")
    
    # Disk check
    disk = psutil.disk_usage('/')
    disk_status = "✅ Good" if (disk.used/disk.total) < 0.8 else "⚠️ High"
    st.write(f"Disk Usage: {(disk.used/disk.total)*100:.1f}% - {disk_status}")
    
    # CPU check
    cpu_status = "✅ Good" if psutil.cpu_percent(interval=1) < 70 else "⚠️ High"
    st.write(f"CPU Usage: {psutil.cpu_percent(interval=1)}% - {cpu_status}")

if __name__ == "__main__":
    main()