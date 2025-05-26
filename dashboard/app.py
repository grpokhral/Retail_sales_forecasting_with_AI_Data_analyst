import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_data_and_model():
    df = pd.read_parquet('data/processed/sales_features.parquet')
    model = joblib.load('models/xgb_regressor.joblib')
    return df, model

df, model = load_data_and_model()

st.title("📈 Retail Sales Forecast Dashboard by rahul")

# Store selector
stores = sorted(df['Store'].unique())
store_id = st.selectbox("🏪 Select Store", stores)

# Year selector
years = sorted(df['Year'].unique())
year_selected = st.selectbox("📅 Select Year", years)

# Filter data by store and year
df_store = df[(df['Store'] == store_id) & (df['Year'] == year_selected)].copy()

# Predict sales
X = df_store.drop(columns=['Sales', 'Date'])
df_store['Predicted_Sales'] = model.predict(X)

# Line chart of actual vs predicted
df_store.set_index('Date', inplace=True)
st.line_chart(df_store[['Sales', 'Predicted_Sales']])

# RMSE
rmse = ((df_store['Sales'] - df_store['Predicted_Sales']) ** 2).mean() ** 0.5
st.write(f"📉 RMSE for Store {store_id} in {year_selected}: {rmse:.2f}")
