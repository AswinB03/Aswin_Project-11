# app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("Air Quality Analysis and Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload Air Quality CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    df = df.drop_duplicates()

    # AQI Distribution
    st.subheader("AQI Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['AQI'], kde=True, color='orange', ax=ax1)
    ax1.set_title("Distribution of AQI")
    st.pyplot(fig1)

    # Correlation Matrix
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig2, ax2 = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="hot", ax=ax2)
    st.pyplot(fig2)

    # Encode categorical columns
    X = pd.get_dummies(df.drop('AQI', axis=1))
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

    # Predictions
    lr_preds = lr.predict(X_test)
    dt_preds = dt.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")
    st.markdown("### Linear Regression")
    st.write("MAE:", mean_absolute_error(y_test, lr_preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
    st.write("R² Score:", r2_score(y_test, lr_preds))

    st.markdown("#Decision Tree Regressor")
    st.write("MAE:", mean_absolute_error(y_test, dt_preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, dt_preds)))
    st.write("R² Score:", r2_score(y_test, dt_preds))

    # AQI by City
    st.subheader("AQI by City")
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df_sorted = df.sort_values(by='AQI', ascending=True)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.barh(df_sorted['City'], df_sorted['AQI'], color='skyblue')
    ax3.set_xlabel("AQI Value")
    ax3.set_title("Air Quality Index by City")
    st.pyplot(fig3)

    # Gas levels by City
    non_gas_columns = ['City', 'AQI', 'Date', 'Time'] if 'Date' in df.columns else ['City', 'AQI']
    gas_columns = [col for col in df.columns if col not in non_gas_columns]

    city_avg = df.groupby("City")[gas_columns].mean().reset_index()
    melted_df = city_avg.melt(id_vars='City', var_name='Gas', value_name='Level')

    st.subheader("Average Pollution Gas Levels by City")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted_df, x='Level', y='City', hue='Gas', ax=ax4)
    ax4.set_title("Gas Levels by City")
    st.pyplot(fig4)

else:
    st.warning("📌 Please upload a valid air_quality_data.csv file to continue.")
  
