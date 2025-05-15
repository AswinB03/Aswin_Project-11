import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Air Quality Analysis and Prediction App")

# === File path check ===
file_path = "air_quality_data.csv"

if not os.path.exists(file_path):
    st.error(" 'air_quality_data.csv' not found. Please make sure it's in the same folder as app.py.")
    st.stop()

# === Load and clean data ===
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # Clean column names
df = df.drop_duplicates()
df = df.dropna(subset=['City'])  # Drop rows without City

# Convert gas columns to numeric
non_gas_columns = ['City', 'AQI', 'Date', 'Time']
gas_columns = [col for col in df.columns if col not in non_gas_columns]

for gas in gas_columns:
    df[gas] = pd.to_numeric(df[gas], errors='coerce')

# Show data
st.subheader("Preview of Cleaned Data")
st.dataframe(df.head())

# === Missing values ===
st.subheader(" Missing Values")
st.write(df.isnull().sum())

# === Correlation Heatmap ===
st.subheader("Correlation Heatmap (Numeric Features Only)")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="hot")
st.pyplot()

# === AQI Distribution ===
st.subheader("📉 Distribution of AQI")
plt.figure(figsize=(6, 4))
sns.histplot(df['AQI'], kde=True, color='orange')
plt.xlabel("AQI")
st.pyplot()

# === AQI by City ===
st.subheader("AQI by City")
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
df_sorted = df.sort_values(by='AQI', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_sorted['City'], df_sorted['AQI'], color='skyblue')
plt.xlabel("AQI Value")
plt.ylabel("City")
plt.title("Air Quality Index (AQI) by City")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
st.pyplot()

# === Pollution Level (Gas) by City ===
st.subheader("Average Pollution Gas Levels by City")
city_avg = df.groupby("City")[gas_columns].mean().reset_index()
melted_df = city_avg.melt(id_vars='City', var_name='Gas', value_name='Level')

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=melted_df, x='Level', y='City', hue='Gas')
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3)
plt.tight_layout()
st.pyplot()

# === Modeling ===
st.subheader("AQI Prediction Models")

# Features and target
X = df.drop('AQI', axis=1)
X_encoded = pd.get_dummies(X)
y = df['AQI']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

# === Evaluation Metrics ===
st.markdown("###Linear Regression")
st.write("MAE:", round(mean_absolute_error(y_test, lr_preds), 2))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2))
st.write("R² Score:", round(r2_score(y_test, lr_preds), 2))

st.markdown("###Decision Tree Regressor")
st.write("MAE:", round(mean_absolute_error(y_test, dt_preds), 2))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, dt_preds)), 2))
st.write("R² Score:", round(r2_score(y_test, dt_preds), 2))

st.success("Model training and evaluation completed successfully!")
