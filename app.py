import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import io

# Page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #F8F9FA;}
    h1, h2, h3 {color: #333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛍️ Mall Customer Segmentation App")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Upload Dataset", "Data Overview", "Clustering", "Visualization"])

# Upload dataset
if options == "Upload Dataset":
    st.header("Upload Your Dataset (CSV)")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        customer_data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.session_state['data'] = customer_data
        st.dataframe(customer_data.head())

elif options == "Data Overview":
    st.header("Dataset Overview")

    if 'data' in st.session_state:
        customer_data = st.session_state['data']
        st.subheader("Dataset Preview")
        st.dataframe(customer_data.head())

        st.subheader("Basic Information")
        buffer = io.StringIO()
        customer_data.info(buf=buffer)
        s = buffer.getvalue()
        st.code(s)

        st.subheader("Missing Values")
        st.dataframe(customer_data.isnull().sum())
    else:
        st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")

elif options == "Clustering":
    st.header("Apply K-Means Clustering")

    if 'data' in st.session_state:
        customer_data = st.session_state['data']

        X = customer_data.iloc[:, [3, 4]].values

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        st.subheader("Elbow Method Graph")
        fig, ax = plt.subplots()
        sns.set()
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax.set_title('The Elbow Point Graph')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
        Y = kmeans.fit_predict(X)

        st.session_state['X'] = X
        st.session_state['Y'] = Y
        st.session_state['kmeans'] = kmeans

    else:
        st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")

elif options == "Visualization":
    st.header("Customer Clusters Visualization")

    if 'X' in st.session_state and 'Y' in st.session_state and 'kmeans' in st.session_state:
        X = st.session_state['X']
        Y = st.session_state['Y']
        kmeans = st.session_state['kmeans']

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = ['green', 'red', 'yellow', 'violet', 'blue', 'orange', 'pink', 'purple', 'brown', 'gray']

        for cluster in np.unique(Y):
            ax.scatter(X[Y == cluster, 0], X[Y == cluster, 1], 
                       s=70, c=colors[cluster % len(colors)], label=f'Cluster {cluster + 1}')

        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=200, c='black', marker='X', label='Centroids')
        ax.set_title("Customer Segmentation Clusters")
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Spending Score")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please perform clustering first in the 'Clustering' section.")

# Footer
st.markdown("---")
st.caption("Developed by APPE Company • Powered by Streamlit")

