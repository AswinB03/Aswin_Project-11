import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Air Quality Analysis and Prediction")

file_path = "air_quality_data.csv"
if not os.path.exists(file_path):
    st.error("File 'air_quality_data.csv' not found. Please place it in the same folder as app.py.")
    st.stop()

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df = df.drop_duplicates()
df = df.dropna(subset=['City'])

non_gas_cols = ['City', 'AQI', 'Date', 'Time']
gas_cols = [col for col in df.columns if col not in non_gas_cols]

for col in gas_cols + ['AQI']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['AQI'] + gas_cols, inplace=True)

st.subheader("Preview of Cleaned Data")
st.dataframe(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heat
