# 📦 Supply Chain Demand Forecaster & Inventory Intelligence

An interactive, deep learning-powered supply chain management dashboard built with **Streamlit** and **TensorFlow/Keras**. 

This system delivers multi-step auto-regressive demand forecasts, monitors inventory levels across multiple warehouses, detects impending stockout risks, and calculates optimal purchase order timelines.

---

## 🚀 Key Highlights & Architecture

- **Deep Learning Architectures**:
  - **Stacked LSTM**: 3-layer recurrent network with Dropout and Batch Normalization (Best performer: RMSE **6.46**, MAE **5.13**).
  - **Hybrid CNN-BiLSTM-GRU**: Conv1D extracts localized temporal motifs, BiLSTM captures bidirectional dependencies, and GRU summarizes sequences (RMSE **6.50**, MAE **5.17**).
  - **Stacked GRU**: Lightweight recurrent network for fast inference (RMSE **7.30**, MAE **5.88**).
- **Automated Dual-Mode Preprocessing**:
  - **Historical SKU Slicing**: Direct forecasting on pre-engineered, calibrated historical series across 50 SKUs.
  - **Dynamic Raw CSV Upload**: Automatically applies fitted `scaler_X.pkl` (MinMaxScaler), cyclical calendar transforms (sin/cos), lag generators, and rolling statistics (7, 14, 30 days) to new data before feeding to the neural networks.
- **Auto-Regressive Forecasting Loop**: Predicts multi-horizon windows (7, 14, 21, 30 days) and inverts outputs back to natural unit sales via `scaler_y.pkl`.
- **Automated Inventory Advisory**: Real-time depletion tracking with automated alert badges (`CRITICAL`, `WARNING`, `HEALTHY`) and Purchase Order recommendations based on lead times and safety stock multipliers.

---

## 📂 Project Directory Structure

```text
D:\NNDL\
│
├── app.py                             # Main interactive Streamlit application
├── requirements.txt                   # Production Python dependencies
├── run_app.bat                        # Windows 1-click batch launcher
├── README.md                          # Project documentation
│
├── data/
│   ├── raw/
│   │   └── supply_chain_dataset1.csv  # 91,250 transaction rows across 15 supply chain features
│   └── processed/
│       ├── full_engineered.csv        # Preprocessed dataset with 56 engineered features
│       ├── scaler_X.pkl               # Fitted MinMaxScaler for numerical input features
│       ├── scaler_y.pkl               # Fitted MinMaxScaler for target (Units_Sold)
│       ├── encoders.pkl               # LabelEncoders for SKUs, warehouses, suppliers, regions
│       ├── top_features.pkl           # 30 high-impact feature names selected by F-regression
│       ├── train.csv / val.csv / test.csv
│       └── y_true_test.npy            # Test set ground truth values
│
├── models/
│   ├── lstm_model.keras               # Trained Stacked LSTM model checkpoint
│   ├── hybrid_cnn_lstm_gru.keras      # Trained Hybrid CNN-BiLSTM-GRU model checkpoint
│   └── gru_model.keras                # Trained Stacked GRU model checkpoint
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Exploratory Data Analysis & visual inspections
│   ├── 02_data_preprocessing.ipynb    # Missing value handling, encoding & MinMax scaling
│   ├── 03_feature_engineering.ipynb   # Rolling statistics, cyclical date encoding & feature selection
│   ├── 04_lstm_model.ipynb            # Architecture build, callbacks & training for LSTM
│   ├── 05_gru_model.ipynb             # Architecture build, callbacks & training for GRU
│   ├── 06_hybrid_model.ipynb          # Multi-branch CNN-BiLSTM-GRU architecture build & training
│   ├── 07_model_evaluation.ipynb      # Comparative evaluation, residual analysis & test benchmarks
│   └── 08_deployment_inference.ipynb  # Rolling auto-regressive forecaster & reorder alert engine
│
└── outputs/
    ├── plots/                         # Visualizations, residual plots & training curves
    └── reports/
        ├── model_comparison.csv       # Test metrics (RMSE, MAE, R2, MAPE)
        ├── normalized_scores.csv      # Unified ranking scores
        └── *.json                     # JSON metrics for individual models
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python**: 3.9, 3.10, or 3.11
- **Hardware**: CPU or CUDA-compatible GPU

### Step 1: Clone or Navigate to the Directory
```bash
cd D:\NNDL
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Using Streamlit CLI
```bash
streamlit run app.py
```

### Option B: Windows 1-Click Launcher
Double-click `run_app.bat` inside `D:\NNDL\`.

The app will automatically launch in your browser at:
```text
http://localhost:8501
```

---

## 📊 Features & Walkthrough

### 1. 🔮 Demand Forecasting (Tab 1)
- **Interactive Controls (Sidebar)**:
  - Toggle between **Pre-engineered Dataset** or **Raw CSV Upload**.
  - Select target SKU (`SKU_1` to `SKU_50`).
  - Choose active architecture (`LSTM`, `Hybrid`, `GRU`).
  - Adjust Forecast Horizon ($7$ to $30$ days) and Safety Stock Multiplier ($1.0\times$ to $2.5\times$).
  - Toggle What-If Promotional Campaigns (`Promotion_Flag = 1`).
- **Interactive Plotly Curves**: Historical sales curve followed by the model's auto-regressive multi-step forecast trajectory.
- **Inventory Depletion Schedule**: Daily tabular simulation tracking ending stock and Purchase Order alerts (`🚨 Place Purchase Order` vs `✅ Stock Level OK`).
- **CSV Export**: Single-click button to download the forecast report.

### 2. 🏆 Model Benchmark & Evaluation (Tab 2)
- Comparative leaderboard loading directly from `outputs/reports/model_comparison.csv`:
  | Model | Test RMSE | Test MAE | Test MAPE (%) |
  | :--- | :---: | :---: | :---: |
  | **LSTM** | **6.46** | **5.13** | **36.18%** |
  | **Hybrid CNN-BiLSTM-GRU** | 6.50 | 5.17 | 36.61% |
  | **GRU** | 7.30 | 5.88 | 38.79% |
- Real out-of-sample visual comparison plotting ground truth (`y_true_test.npy`) against test predictions from all three models.

### 3. 🔍 Supply Chain & Inventory Explorer (Tab 3)
- Promotion vs. Baseline sales volume boxplots.
- Stockout event detection scatter plots (Inventory Level vs. Units Sold).
- Warehouse node (`WH_1` to `WH_5`) and Regional volume summaries.

---

## 📥 Custom CSV Upload Specifications

When choosing **"Upload Raw CSV"** in the sidebar, the file must contain at least **30 consecutive daily records** with the following columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Date` | YYYY-MM-DD | Daily date identifier |
| `Units_Sold` | Numeric | Historical units sold (target) |
| `Inventory_Level` | Numeric | Unscaled warehouse on-hand stock |
| `Supplier_Lead_Time_Days`| Numeric | Lead time for replenishment |
| `Reorder_Point` | Numeric | Minimum inventory safety threshold |
| `Order_Quantity` | Numeric | Standard batch reorder size |
| `Unit_Cost` | Numeric | Cost per unit from supplier |
| `Unit_Price` | Numeric | Selling price per unit |
| `Demand_Forecast` | Numeric | Baseline external demand projection |
| `Promotion_Flag` | 0 or 1 | (Optional) Active marketing flag |

*Note: You can download a pre-formatted 35-row sample template directly from the app interface.*

---

## ⚠️ Known Limitations & Production Roadmap

1. **Auto-Regressive Error Accumulation**: Rolling step-by-step prediction can accumulate errors over extended horizons ($> 21$ days). Direct multi-horizon prediction models (e.g., Temporal Fusion Transformer or N-BEATS) are recommended for enterprise scale.
2. **Tabular Gradient Saturation**: Recurrent models on noisy supply chain data can over-smooth predictions. Future releases can incorporate tree-based baselines (**LightGBM / CatBoost**).
3. **Serving Decoupling**: For multi-tenant ERP integrations, decouple model inference into a dedicated **FastAPI** service with asynchronous background workers.
