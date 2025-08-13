import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib

st.set_page_config(page_title="Forest Diversity & Carbon", layout="wide")
st.title("🌳 Forest Diversity & Carbon Sequestration Prediction")

uploaded = st.file_uploader("Upload your raw Dataset.csv", type="csv")
if not uploaded:
    st.info("Please upload a CSV file with your tree inventory data.")
    st.stop()

# Load & clean data
df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip().str.replace(" ", "")
df = df.dropna(subset=["Dbh(cm)", "Ht(m)", "TreeSpecies"])

# Convert columns to numeric that might be needed for EDA and predictions
for col in ["Db(m)", "Dbh(cm)", "Dt(m)", "Dm(m)", "Ht(m)", "Density"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional: Remove outliers based on IQR for Dbh and Ht
for col in ["Dbh(cm)", "Ht(m)"]:
    if col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

# Feature engineering for EDA and calculations
if all(x in df.columns for x in ["Dbh(cm)", "Ht(m)"]):
    df["Basal_Area(m2)"] = math.pi * (df["Dbh(cm)"] / 200) ** 2
else:
    df["Basal_Area(m2)"] = np.nan

if all(x in df.columns for x in ["Db(m)", "Dm(m)", "Dt(m)", "Ht(m)"]):
    df["Volume(m3)"] = math.pi * df["Ht(m)"] / 24 * (
        df["Db(m)"] ** 2 + 4 * df["Dm(m)"] ** 2 + df["Dt(m)"] ** 2
    )
else:
    df["Volume(m3)"] = np.nan

if "Density" in df.columns:
    df["Density_kg_m3"] = df["Density"] * 1000
else:
    df["Density_kg_m3"] = np.nan

df["Carbon(kg)"] = df["Volume(m3)"] * df["Density_kg_m3"] * 0.5

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Height distribution bar plot (binned)
if "Ht(m)" in df.columns:
    height_bins = [0, 5, 10, 15, 20, 25, 30, 40]
    height_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40"]
    df["Height_Class"] = pd.cut(df["Ht(m)"], bins=height_bins, labels=height_labels, right=False)
    height_counts = df["Height_Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    height_counts.plot(kind="bar", ax=ax, color="forestgreen")
    ax.set_title("Tree Height Distribution (Binned)")
    ax.set_xlabel("Height Class (m)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# DBH line graph using histogram bins
if "Dbh(cm)" in df.columns:
    bin_edges = np.histogram_bin_edges(df["Dbh(cm)"], bins=30)
    hist_counts, _ = np.histogram(df["Dbh(cm)"], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bin_centers, hist_counts, marker="o", linestyle="-")
    ax.set_title("Diameter Distribution (cm)")
    ax.set_xlabel("Diameter (cm)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    st.pyplot(fig)

# Correlation matrix of relevant numeric variables
num_cols = [col for col in ["Dbh(cm)", "Ht(m)", "Basal_Area(m2)", "Volume(m3)", "Density", "Carbon(kg)"] if col in df.columns]
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Shannon diversity index calculation
def shannon(series):
    cnts = Counter(series)
    vals = np.array(list(cnts.values()))
    props = vals / vals.sum()
    return -np.sum(props * np.log(props))

if "ForestId" in df.columns and df["ForestId"].notnull().any():
    div_forest = df.groupby("ForestId")["TreeSpecies"].apply(shannon).reset_index(name="Shannon_Index")
else:
    total_shannon = shannon(df["TreeSpecies"])
    div_forest = pd.DataFrame([{"ForestId": "All", "Shannon_Index": total_shannon}])

# Load saved carbon prediction model and predict per individual tree
try:
    loaded_obj = joblib.load("best_carbon_model.joblib")
    best_model = loaded_obj['model']  # actual trained sklearn model
    
    # Use features from the saved dict if available, else default to specified list
    feature_columns = loaded_obj.get('features', ["Db(m)", "Dbh(cm)", "Dt(m)", "Dm(m)", "Ht(m)"])
    
    # Convert features to numeric and drop rows with missing values
    for col in feature_columns:
        if col not in df.columns:
            st.error(f"Expected feature column missing for prediction: {col}")
            st.stop()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    X_pred = df[feature_columns].dropna()
    
    carbon_pred = best_model.predict(X_pred)
    df.loc[X_pred.index, "Carbon_Predicted(kg)"] = carbon_pred
    
    st.success("Best carbon prediction model loaded and applied successfully.")
    
except Exception as e:
    st.warning(f"Could not load or use the best carbon prediction model: {e}")
    df["Carbon_Predicted(kg)"] = np.nan

# Aggregate predicted carbon at forest level or overall
if "ForestId" in df.columns and df["ForestId"].notnull().any():
    forest_carbon_pred = df.groupby("ForestId")["Carbon_Predicted(kg)"].sum().reset_index()
else:
    total_carbon_pred = df["Carbon_Predicted(kg)"].sum()
    forest_carbon_pred = pd.DataFrame([{"ForestId": "All", "Carbon_Predicted(kg)": total_carbon_pred}])

# Merge Shannon diversity index with predicted carbon
forest_results = pd.merge(div_forest, forest_carbon_pred, on="ForestId", how="left")

# Display results
st.header("Forest-Level Shannon Diversity and Predicted Carbon")
st.dataframe(forest_results)

# Download button for results CSV
csv = forest_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Forest-Level Predictions as CSV",
    data=csv,
    file_name="forest_level_predictions.csv",
    mime="text/csv"
)
