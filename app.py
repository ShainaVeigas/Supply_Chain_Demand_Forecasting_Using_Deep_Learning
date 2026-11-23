import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import pickle
import os
from datetime import timedelta

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
    }
    .status-ok { color: #2e7d32; font-weight: bold; }
    .status-warning { color: #f57f17; font-weight: bold; }
    .status-critical { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINEERED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "full_engineered.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "supply_chain_dataset1.csv")
SCALER_X_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler_y.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "data", "processed", "encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "top_features.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
TEST_TRUE_PATH = os.path.join(BASE_DIR, "data", "processed", "y_true_test.npy")

MODEL_OPTIONS = {
    "LSTM Model": "lstm_model.keras",
    "Hybrid CNN-BiLSTM-GRU": "hybrid_cnn_lstm_gru.keras",
    "GRU Model": "gru_model.keras"
}

# --- Caching Data & Artifacts ---
@st.cache_data
def load_datasets():
    eng_df = pd.read_csv(ENGINEERED_DATA_PATH, parse_dates=['Date']) if os.path.exists(ENGINEERED_DATA_PATH) else None
    raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=['Date']) if os.path.exists(RAW_DATA_PATH) else None
    return eng_df, raw_df

@st.cache_resource
def load_artifacts():
    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_y = pickle.load(f)
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    return scaler_X, scaler_y, encoders, feature_cols

@st.cache_resource
def load_keras_model(model_filename):
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_evaluation_reports():
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    norm_path = os.path.join(REPORTS_DIR, "normalized_scores.csv")
    
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    norm_df = pd.read_csv(norm_path) if os.path.exists(norm_path) else None
    
    preds_dict = {}
    for k, fname in [('LSTM', 'lstm_predictions.npy'), ('GRU', 'gru_predictions.npy'), ('Hybrid', 'hybrid_predictions.npy')]:
        p = os.path.join(BASE_DIR, "data", "processed", fname)
        if os.path.exists(p):
            preds_dict[k] = np.load(p)
            
    y_true = np.load(TEST_TRUE_PATH) if os.path.exists(TEST_TRUE_PATH) else None
    return comp_df, norm_df, preds_dict, y_true

# Load all core assets
engineered_df, raw_df = load_datasets()
scaler_X, scaler_y, encoders, feature_cols = load_artifacts()
comp_df, norm_df, test_predictions, y_true_test = load_evaluation_reports()

if engineered_df is None:
    st.error("Engineered dataset not found. Please ensure data/processed/full_engineered.csv exists.")
    st.stop()

# --- Raw CSV Scaling & Feature Engineering Pipeline ---
def preprocess_and_scale_uploaded_csv(uploaded_df, scaler_X, scaler_y, encoders, feature_cols):
    """
    Transforms raw unscaled user CSV into model-ready scaled features.
    1. Validates columns & dates.
    2. Scales raw numeric features with scaler_X.
    3. Computes cyclical date features.
    4. Computes rolling statistics and lag features.
    5. Returns engineered DataFrame aligned with the model's 30 top features.
    """
    df = uploaded_df.copy()
    if 'Date' not in df.columns:
        raise ValueError("Uploaded CSV must have a 'Date' column.")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    numeric_features = [
        'Inventory_Level', 'Supplier_Lead_Time_Days', 'Reorder_Point',
        'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Demand_Forecast'
    ]
    missing_num = [c for c in numeric_features if c not in df.columns]
    if missing_num:
        raise ValueError(f"Missing required numeric columns: {missing_num}")
        
    if 'Units_Sold' not in df.columns:
        raise ValueError("Missing 'Units_Sold' column required for sequence history.")

    # 1. Scale numeric features with saved scaler_X
    df[numeric_features] = scaler_X.transform(df[numeric_features])
    
    # Scale Units_Sold target using saved scaler_y
    df['Units_Sold_scaled'] = scaler_y.transform(df[['Units_Sold']])
    
    # 2. Date cyclical features
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # 3. Rolling & Lag Feature Engineering
    target = 'Units_Sold'
    for win in [7, 14, 30]:
        df[f'rolling_mean_{win}'] = df[target].shift(1).rolling(win, min_periods=1).mean().fillna(0)
        df[f'rolling_std_{win}']  = df[target].shift(1).rolling(win, min_periods=1).std().fillna(0)
        df[f'rolling_min_{win}']  = df[target].shift(1).rolling(win, min_periods=1).min().fillna(0)
        df[f'rolling_max_{win}']  = df[target].shift(1).rolling(win, min_periods=1).max().fillna(0)
        df[f'ewm_{win}']          = df[target].shift(1).ewm(span=win, adjust=False).mean().fillna(0)

    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df[target].shift(lag).fillna(0)

    if 'Promotion_Flag' not in df.columns:
        df['Promotion_Flag'] = 0

    # Ensure all 30 top_features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df

# --- Rolling Auto-Regressive Forecast Engine ---
def run_forecast(df, sku_id, feature_cols, model, scaler_y, n_steps=14, seq_length=30, promo_flag=False, is_custom=False):
    if is_custom:
        sku_data = df.sort_values('Date').fillna(0)
    else:
        sku_data = df[df['SKU_ID'] == sku_id].sort_values('Date').fillna(0)
        
    if len(sku_data) < seq_length:
        return None, None
    
    history = sku_data[feature_cols].values[-seq_length:].astype(np.float32)
    target_idx = feature_cols.index('Units_Sold_scaled') if 'Units_Sold_scaled' in feature_cols else 0
    promo_idx = feature_cols.index('Promotion_Flag') if 'Promotion_Flag' in feature_cols else None

    predictions, current_window = [], history.copy()
    for _ in range(n_steps):
        inp = current_window[-seq_length:].reshape(1, seq_length, len(feature_cols))
        pred_scaled = model.predict(inp, verbose=0)[0][0]
        pred_actual = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        pred_actual = max(0.0, float(pred_actual))
        predictions.append(pred_actual)
        
        new_row = current_window[-1].copy()
        new_row[target_idx] = pred_scaled
        if promo_flag and promo_idx is not None:
            new_row[promo_idx] = 1.0
            
        current_window = np.vstack([current_window[1:], new_row])

    last_date = sku_data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'SKU_ID': sku_id if not is_custom else "UPLOADED_DATA",
        'Forecasted_Demand': [round(p, 1) for p in predictions]
    })
    return sku_data, forecast_df

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/delivery.png", width=70)
st.sidebar.title("Demand AI Control")

data_mode = st.sidebar.radio(
    "📂 Data Input Mode",
    ["Select from Dataset", "Upload Raw CSV"],
    index=0
)

uploaded_file = None
uploaded_df = None
custom_scaled_df = None
custom_error = None

if data_mode == "Upload Raw CSV":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload Raw CSV (min 30 rows)", type=['csv'])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            custom_scaled_df = preprocess_and_scale_uploaded_csv(
                uploaded_df, scaler_X, scaler_y, encoders, feature_cols
            )
            st.sidebar.success(f"✅ Loaded & Scaled {len(custom_scaled_df)} rows")
        except Exception as e:
            custom_error = str(e)
            st.sidebar.error(f"Error: {e}")
    selected_sku = "Uploaded_Custom_Data"
else:
    available_skus = sorted(engineered_df['SKU_ID'].unique().tolist())
    selected_sku = st.sidebar.selectbox("🎯 Select SKU", available_skus, index=0)

selected_model_name = st.sidebar.selectbox("🧠 Select Neural Architecture", list(MODEL_OPTIONS.keys()), index=0)
model_filename = MODEL_OPTIONS[selected_model_name]
model = load_keras_model(model_filename)

forecast_horizon = st.sidebar.slider("📅 Forecast Horizon (Days)", min_value=7, max_value=30, value=14, step=7)
safety_multiplier = st.sidebar.slider("🛡️ Safety Stock Multiplier", min_value=1.0, max_value=2.5, value=1.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Scenario Simulation")
promo_override = st.sidebar.checkbox("Simulate Promotional Campaign (Promotion_Flag = 1)", value=False)

# Main Page Header
st.title("📦 Supply Chain Demand & Inventory Forecaster")
st.markdown(f"**Target:** `{selected_sku}` | **Active Model:** `{selected_model_name}` | **Horizon:** `{forecast_horizon} Days`")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Demand Forecasting", "📊 Model Benchmark & Evaluation", "🔍 Supply Chain Explorer"])

# ----------------- TAB 1: DEMAND FORECASTING -----------------
with tab1:
    is_custom = (data_mode == "Upload Raw CSV")
    active_df = custom_scaled_df if is_custom else engineered_df
    
    if is_custom:
        st.subheader("📤 Custom CSV Upload & Scaling Pipeline")
        st.markdown("""
        Upload raw, unscaled supply chain records. The app automatically executes:
        1. **`scaler_X.transform()`**: Normalizes numeric inputs (`Inventory_Level`, `Reorder_Point`, `Lead_Time`, `Price`, etc.).
        2. **`scaler_y.transform()`**: Normalizes historical sales (`Units_Sold`).
        3. **Feature Generation**: Calculates rolling window statistics (7, 14, 30 days) and date cyclical components.
        4. **Inversion**: Uses `scaler_y.inverse_transform()` to convert neural network predictions back into exact units sold.
        """)
        
        # Download Sample Template Button
        if raw_df is not None:
            sample_template = raw_df[raw_df['SKU_ID'] == 'SKU_1'].tail(35).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Sample CSV Template (35 rows)",
                data=sample_template,
                file_name="sample_raw_supply_chain_data.csv",
                mime="text/csv"
            )
            
        if uploaded_file is None:
            st.info("👆 Please upload a CSV in the sidebar to start inference, or download the template above.")
        elif custom_error:
            st.error(f"Preprocessing error: {custom_error}")

    if active_df is not None and model is not None and (not is_custom or uploaded_file is not None):
        with st.spinner("Executing neural network rolling auto-regressive forecast..."):
            sku_data, forecast_df = run_forecast(
                active_df, selected_sku, feature_cols, model, scaler_y, 
                n_steps=forecast_horizon, promo_flag=promo_override, is_custom=is_custom
            )

        if sku_data is not None and forecast_df is not None:
            # Inventory & reorder metrics
            if is_custom and uploaded_df is not None:
                curr_inventory = float(uploaded_df['Inventory_Level'].iloc[-1])
                reorder_point = float(uploaded_df['Reorder_Point'].iloc[-1])
                lead_time = float(uploaded_df['Supplier_Lead_Time_Days'].iloc[-1])
            elif raw_df is not None:
                raw_sku = raw_df[raw_df['SKU_ID'] == selected_sku].sort_values('Date')
                curr_inventory = float(raw_sku['Inventory_Level'].iloc[-1])
                reorder_point = float(raw_sku['Reorder_Point'].iloc[-1])
                lead_time = float(raw_sku['Supplier_Lead_Time_Days'].iloc[-1])
            else:
                curr_inventory = float(sku_data['Inventory_Level'].iloc[-1])
                reorder_point = float(sku_data['Reorder_Point'].iloc[-1])
                lead_time = float(sku_data['Supplier_Lead_Time_Days'].iloc[-1])
            
            total_projected_demand = float(forecast_df['Forecasted_Demand'].sum())
            avg_daily_demand = float(forecast_df['Forecasted_Demand'].mean())
            safety_stock = round(avg_daily_demand * lead_time * 0.1 * safety_multiplier, 1)
            
            if curr_inventory < reorder_point:
                status = "CRITICAL: Under Reorder Point"
                status_color = "status-critical"
            elif curr_inventory < (reorder_point * 1.5):
                status = "WARNING: Approaching Reorder Point"
                status_color = "status-warning"
            else:
                status = "HEALTHY: Stock Adequate"
                status_color = "status-ok"

            # Top KPI metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Projected Demand", f"{int(total_projected_demand):,} units")
            with col2:
                st.metric("Current Warehouse Stock", f"{int(curr_inventory):,} units")
            with col3:
                st.metric("Reorder Threshold", f"{int(reorder_point):,} units")
            with col4:
                st.metric("Recommended Safety Stock", f"{int(safety_stock):,} units")

            st.markdown(f"**Inventory Alert Status:** <span class='{status_color}'>{status}</span>", unsafe_allow_html=True)
            st.markdown("---")

            # Plotly Chart
            hist_view = sku_data.tail(60)
            fig = go.Figure()

            # Real Units_Sold
            actual_y = uploaded_df['Units_Sold'].tail(60) if is_custom and uploaded_df is not None else hist_view['Units_Sold']
            fig.add_trace(go.Scatter(
                x=hist_view['Date'], 
                y=actual_y,
                mode='lines+markers',
                name='Actual Historical Sales',
                line=dict(color='#1E88E5', width=2),
                marker=dict(size=4)
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], 
                y=forecast_df['Forecasted_Demand'],
                mode='lines+markers',
                name=f'Predicted Demand ({selected_model_name})',
                line=dict(color='#FF7043', width=3, dash='dash'),
                marker=dict(size=6, symbol='circle')
            ))

            fig.update_layout(
                title=f"Historical Demand vs Next {forecast_horizon}-Day Rolling Forecast ({selected_sku})",
                xaxis_title="Date",
                yaxis_title="Units Sold / Demand",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Daily Schedule & Inventory Depletion
            st.subheader("📋 Daily Inventory Depletion Schedule")
            schedule_df = forecast_df.copy()
            sim_stock = curr_inventory
            stock_timeline, order_recs = [], []
            
            for d in schedule_df['Forecasted_Demand']:
                sim_stock -= d
                stock_timeline.append(int(sim_stock))
                if sim_stock <= reorder_point:
                    order_recs.append("🚨 Place Purchase Order")
                else:
                    order_recs.append("✅ Stock Level OK")

            schedule_df['Projected_Ending_Stock'] = stock_timeline
            schedule_df['Inventory_Advisory'] = order_recs
            
            st.dataframe(
                schedule_df.style.map(
                    lambda v: 'color: red; font-weight: bold;' if '🚨' in str(v) else 'color: green;',
                    subset=['Inventory_Advisory']
                ),
                use_container_width=True
            )
            
            csv = schedule_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"demand_forecast_{selected_sku}_{forecast_horizon}days.csv",
                mime="text/csv"
            )
        elif is_custom and uploaded_file is not None:
            st.warning("Insufficient rows in uploaded CSV. The neural network requires at least 30 consecutive days of history.")

# ----------------- TAB 2: MODEL BENCHMARK -----------------
with tab2:
    st.subheader("🏆 Model Comparison & Evaluation Leaderboard")
    st.markdown("Loaded directly from test evaluation reports (`outputs/reports/model_comparison.csv`).")
    
    if comp_df is not None:
        st.dataframe(
            comp_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE_%'], color='#c8e6c9')
                         .highlight_max(subset=['R2'], color='#c8e6c9'),
            use_container_width=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_rmse = px.bar(
                comp_df, 
                x='Model', 
                y='RMSE', 
                title="Root Mean Squared Error (Lower = Better)",
                color='RMSE',
                color_continuous_scale='Blues_r',
                text_auto='.3f'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_b:
            fig_mae = px.bar(
                comp_df, 
                x='Model', 
                y='MAE', 
                title="Mean Absolute Error (Lower = Better)",
                color='MAE',
                color_continuous_scale='Oranges_r',
                text_auto='.3f'
            )
            st.plotly_chart(fig_mae, use_container_width=True)

    if y_true_test is not None and test_predictions:
        st.subheader("📈 Test Set Predictions vs True Sales (First 150 Steps)")
        n_plot = min(150, len(y_true_test))
        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(y=y_true_test[:n_plot], mode='lines', name='Actual True Sales', line=dict(color='black', width=2)))
        
        colors = {'LSTM': '#1E88E5', 'GRU': '#43A047', 'Hybrid': '#E53935'}
        for name, pred in test_predictions.items():
            test_fig.add_trace(go.Scatter(y=pred[:n_plot], mode='lines', name=f'{name} Prediction', line=dict(color=colors.get(name, '#FB8C00'), width=1.5, dash='dash')))
            
        test_fig.update_layout(
            title="Comparison of Model Predictions on Out-of-Sample Test Data",
            xaxis_title="Time Step (Days)",
            yaxis_title="Units Sold",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(test_fig, use_container_width=True)

# ----------------- TAB 3: DATA EXPLORER -----------------
with tab3:
    display_df = raw_df if raw_df is not None else engineered_df
    sku_to_explore = selected_sku if selected_sku in display_df['SKU_ID'].values else display_df['SKU_ID'].iloc[0]
    st.subheader(f"📊 Supply Chain & SKU Analytics ({sku_to_explore})")
    
    sku_full = display_df[display_df['SKU_ID'] == sku_to_explore]
    
    col_x, col_y = st.columns(2)
    with col_x:
        fig_promo = px.box(
            sku_full, 
            x='Promotion_Flag', 
            y='Units_Sold', 
            color='Promotion_Flag',
            title=f"Promotion Impact on Units Sold ({sku_to_explore})",
            labels={'Promotion_Flag': 'Promotion Active (1=Yes, 0=No)', 'Units_Sold': 'Units Sold'}
        )
        st.plotly_chart(fig_promo, use_container_width=True)

    with col_y:
        fig_scatter = px.scatter(
            sku_full.tail(150),
            x='Inventory_Level',
            y='Units_Sold',
            color='Stockout_Flag',
            title=f"Inventory Level vs Units Sold (Last 150 records for {sku_to_explore})",
            labels={'Stockout_Flag': 'Stockout Flag', 'Inventory_Level': 'Inventory Level'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 🏬 Warehouse & Regional Volume Distribution")
    summary_group = display_df.groupby(['Warehouse_ID', 'Region'])['Units_Sold'].agg(['count', 'sum', 'mean']).reset_index()
    summary_group.columns = ['Warehouse ID', 'Region', 'Transactions', 'Total Units Sold', 'Average Daily Sales']
    summary_group['Average Daily Sales'] = summary_group['Average Daily Sales'].round(2)
    st.dataframe(summary_group, use_container_width=True)
