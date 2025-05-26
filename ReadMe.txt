retail_sales_forecasting/         # Project root
├── venv/                         # Python virtual environment (created via python -m venv venv)
│   ├── bin/ or Scripts/          # Executables for Linux/macOS or Windows
│   ├── lib/                      # Installed packages
│   └── ...
├── data/                         # Raw and processed data
│   ├── raw/                      # Original downloaded files
│   │   └── store_sales_2024.csv
│   └── processed/                # Cleaned/engineered data
│       └── sales_features.parquet
├── notebooks/                    # Exploratory analysis & prototyping
│   └── 01-sales-eda.ipynb
├── src/                          # Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # ETL and feature engineering
│   ├── eda.py                    # Functions to generate plots & stats
│   ├── model_training.py         # Train, evaluate, and save model
│   └──                 
├── dashboard/                    # Streamlit app for visualization
│   └── app.py
├── models/                       # Saved model artifacts
│   └── xgb_regressor.joblib
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and setup instructions


# Retail Sales Forecasting with AI
This project predicts monthly sales for each store using XGBoost and provides an interactive dashboard.
## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data:
   ```bash
   python -m src.data_preprocessing \
  --input data/raw/store_sales_2024.csv \           
  --output data/processed/sales_features.parquet
   ```

4. Train model:
   ```bash
   python -m src.model_training \    
  --features data/processed/sales_features.parquet \
  --output models/xgb_regressor.joblib
   ```

5. Run dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
5. create a average sales chart from eda.py:
   ```bash
   python -m src.eda --data data/processed/sales_features.parquet

   ```