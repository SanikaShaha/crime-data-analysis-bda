import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Crime Data Analysis", layout="wide")
st.title("🚔 Crime Data Analysis using K-Means Clustering")

uploaded_file = st.file_uploader("📥 Upload crime_output.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) >= 2:
        col1 = st.selectbox("Select Feature 1", numeric_cols)
        col2 = st.selectbox("Select Feature 2", numeric_cols)
        k = st.slider("Number of clusters (K)", 2, 6, 3)

        clean_df = df[[col1, col2]].dropna()

        kmeans = KMeans(n_clusters=k, random_state=42)
        clean_df["Cluster"] = kmeans.fit_predict(clean_df)

        st.subheader("📊 K-Means (Mean-based) Clustering")

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            clean_df[col1],
            clean_df[col2],
            c=clean_df["Cluster"]
        )

        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker="X"
        )

        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title("K-Means Clustering")
        plt.colorbar(scatter)
        st.pyplot(fig)

        st.subheader("📤 Download Output")
        st.download_button(
            "⬇️ Download Clustered CSV",
            clean_df.to_csv(index=False),
            file_name="crime_clustered_output.csv",
            mime="text/csv"
        )
    else:
        st.warning("Need at least 2 numeric columns")
else:
    st.info("Upload CSV to begin analysis")
