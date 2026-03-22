import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Crime Data Analysis", layout="wide")

st.title("🚔 Crime Data Analysis Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload crime_output.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Crime Dataset")
    st.dataframe(df)

    # ---- CLUSTERING ----
    if {'Latitude', 'Longitude'}.issubset(df.columns):
        st.subheader("📊 Crime Hotspot Clustering")

        X = df[['Latitude', 'Longitude']]

        k = st.slider("Number of clusters", 2, 6, 3)
        model = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = model.fit_predict(X)

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df['Longitude'],
            df['Latitude'],
            c=df['Cluster'],
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Crime Clusters")
        plt.colorbar(scatter)
        st.pyplot(fig)

    else:
        st.warning("Latitude & Longitude columns not found!")

else:
    st.info("⬆️ Upload crime_output.csv to view table & clustering")
