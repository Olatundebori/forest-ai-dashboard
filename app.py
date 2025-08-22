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

# Convert to numeric where needed
for col in ["Db(m)", "Dbh(cm)", "Dt(m)", "Dm(m)", "Ht(m)", "Density"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove outliers based on IQR 
for col in ["Dbh(cm)", "Ht(m)", "Db(m)", "Dt(m)", "Dm(m)"]:
    if col in df.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

# Feature engineering
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

# Height distribution bar plot
if "Ht(m)" in df.columns:
    height_bins = [0, 5, 10, 15, 20, 25, 30, 40]
    height_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40"]
    df["Height_Class"] = pd.cut(df["Ht(m)"], bins=height_bins, labels=height_labels, right=False)
    height_counts = df["Height_Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    height_counts.plot(kind="bar", ax=ax, color="forestgreen")
    ax.set_title("Tree Height Distribution")
    ax.set_xlabel("Height Class (m)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# DBH diameter class bar plot (binned)
if "Dbh(cm)" in df.columns:
    size_bins = [0, 10, 20, 30, 40, 50, 100, 150]
    size_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-100", "100-150"]
    df['DBH_Class'] = pd.cut(df['Dbh(cm)'], bins=size_bins, labels=size_labels, right=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='DBH_Class', data=df, palette='Greens', ax=ax)
    ax.set_title('Tree Size Class Distribution by DBH')
    ax.set_xlabel('DBH Class (cm)')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Correlation matrix heatmap
num_cols = [col for col in ["Dbh(cm)", "Ht(m)", "Basal_Area(m2)", "Volume(m3)", "Density", "Carbon(kg)"] if col in df.columns]
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

# Tree Species Distribution Bar Plot with top 20 only if more species
if "TreeSpecies" in df.columns:
    species_counts = df["TreeSpecies"].value_counts()
    st.subheader("Tree Species Frequency Bar Plot (Top 20 if more than 20 species)")
    if species_counts.shape[0] > 20:
        fig, ax = plt.subplots(figsize=(10, 5))
        species_counts.head(20).plot(kind="bar", ax=ax, color="cornflowerblue")
        ax.set_title("Top 20 Tree Species by Count")
        ax.set_xlabel("Tree Species")
        ax.set_ylabel("Number of Trees")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        species_counts.plot(kind="bar", ax=ax, color="cornflowerblue")
        ax.set_title("Tree Species by Count")
        ax.set_xlabel("Tree Species")
        ax.set_ylabel("Number of Trees")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    # Full species frequency table
    st.subheader("Full Tree Species Frequency Table")
    freq_df = pd.DataFrame({"TreeSpecies": species_counts.index, "Count": species_counts.values})
    st.dataframe(freq_df)

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
    feature_columns = loaded_obj.get('features', ["Db(m)", "Dbh(cm)", "Dt(m)", "Dm(m)", "Ht(m)"])
    for col in feature_columns:
        if col not in df.columns:
            st.error(f"Expected feature column missing for prediction: {col}")
            st.stop()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    X_pred = df[feature_columns].dropna()
    carbon_pred = best_model.predict(X_pred)
    df.loc[X_pred.index, "Carbon_Predicted(kg)"] = carbon_pred
    st.success("Best carbon prediction model loaded and applied successfully.")
    
    # Tree Species Contribution to Total Predicted Carbon Table and Plot (Top 20)
    if "TreeSpecies" in df.columns:
        species_carbon = df.groupby("TreeSpecies")["Carbon_Predicted(kg)"].sum().sort_values(ascending=False)
        st.subheader("Tree Species Contribution to Total Predicted Carbon - Table")
        carbon_table = pd.DataFrame({"TreeSpecies": species_carbon.index, "Total Predicted Carbon (kg)": species_carbon.values})
        st.dataframe(carbon_table)
        
        st.subheader("Tree Species Contribution to Total Predicted Carbon - Top 20 Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        species_carbon.head(20).plot(kind="bar", ax=ax, color="darkgreen")
        ax.set_title("Top 20 Tree Species Contribution to Total Predicted Carbon")
        ax.set_xlabel("Tree Species")
        ax.set_ylabel("Total Predicted Carbon (kg)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
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
