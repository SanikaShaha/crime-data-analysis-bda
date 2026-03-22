import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Crime Data Analysis", layout="wide")
st.title("🚔 Crime Data Analysis using K-Means Clustering")

uploaded_file = st.file_uploader("📥 Upload crime_output.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    # Force numeric columns
    numeric_cols = df.columns

    col1 = st.selectbox("Select Feature 1", numeric_cols)
    col2 = st.selectbox("Select Feature 2", numeric_cols)

    if col1 == col2:
        st.error("❌ Please select two DIFFERENT features")
        st.stop()

    # Convert to numeric safely
    x = pd.to_numeric(df[col1], errors="coerce")
    y = pd.to_numeric(df[col2], errors="coerce")

    clean_df = pd.DataFrame({
        col1: x,
        col2: y
    }).dropna()

    if clean_df.empty or len(clean_df) < 2:
        st.error("❌ Not enough valid numeric data for clustering")
        st.stop()

    k = st.slider("Number of clusters (K)", 2, 6, 3)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(clean_df)

    st.subheader("📊 K-Means (Mean-based) Clustering")

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        clean_df[col1].values.ravel(),
        clean_df[col2].values.ravel(),
        c=clusters,
        cmap="viridis"
    )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c="red",
        marker="X",
        label="Centroids"
    )

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("K-Means Clustering")
    ax.legend()
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Output
    output_df = clean_df.copy()
    output_df["Cluster"] = clusters

    st.subheader("📤 Download Output")
    st.download_button(
        "⬇️ Download Clustered CSV",
        output_df.to_csv(index=False),
        file_name="crime_clustered_output.csv",
        mime="text/csv"
    )

else:
    st.info("Upload CSV to start analysis")
