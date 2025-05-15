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

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# AQI Distribution
st.subheader("AQI Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['AQI'], kde=True, color='green', ax=ax)
ax.set_xlabel("AQI")
st.pyplot(fig)

# AQI by City
st.subheader("AQI by City")
city_mean = df.groupby("City")['AQI'].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(city_mean.index, city_mean.values, color='skyblue')
ax.set_xlabel("Average AQI")
ax.set_title("Average AQI by City")
ax.grid(axis='x', linestyle='--', alpha=0.6)
st.pyplot(fig)

# Average Gas Levels by City
st.subheader("Average Pollution Gas Levels by City")
city_avg = df.groupby("City")[gas_cols].mean().reset_index()
melted = city_avg.melt(id_vars='City', var_name='Gas', value_name='Level')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=melted, x='Level', y='City', hue='Gas', ax=ax)
fig.tight_layout()
st.pyplot(fig)

# Machine Learning Model
st.subheader("AQI Prediction using Machine Learning")

df_model = df.dropna(subset=['AQI'])
X = df_model.drop(['AQI', 'Date', 'Time'], axis=1)
X = pd.get_dummies(X)
y = df_model['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

st.markdown("Linear Regression Performance")
st.write("MAE:", round(mean_absolute_error(y_test, lr_preds), 2))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2))
st.write("R² Score:", round(r2_score(y_test, lr_preds), 2))

st.markdown("Decision Tree Regressor Performance")
st.write("MAE:", round(mean_absolute_error(y_test, dt_preds), 2))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, dt_preds)), 2))
st.write("R² Score:", round(r2_score(y_test, dt_preds), 2))

st.success("Analysis and prediction completed successfully.")
