import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Crime Data Analysis", layout="wide")

st.title("🚔 Crime Data Analysis using K-Means Clustering")

# Upload CSV (Spark output OR any crime CSV)
uploaded_file = st.file_uploader("📥 Upload crime_output.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Crime Dataset")
    st.dataframe(df)

    # Numeric columns only
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) >= 2:
        col1 = st.selectbox("Select Feature 1", numeric_cols)
        col2 = st.selectbox("Select Feature 2", numeric_cols)

        X = df[[col1, col2]]

        k = st.slider("Select number of clusters (K)", 2, 6, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X)

        st.subheader("📊 K-Means Clustering Output")

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            X[col1],
            X[col2],
            c=df["Cluster"]
        )
        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker="X"
        )
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title("Mean-based (K-Means) Clustering")
        plt.colorbar(scatter)
        st.pyplot(fig)

        # Download output
        st.subheader("📤 Download Clustered Output")
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            file_name="crime_clustered_output.csv",
            mime="text/csv"
        )
    else:
        st.warning("Dataset must contain at least 2 numeric columns")
else:
    st.info("⬆️ Upload crime_output.csv to start analysis")
