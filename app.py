import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("Mall Customer Segmentation using KMeans")

uploaded_file = st.file_uploader("Upload Mall_Customers.csv file", type=["csv"])

if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(customer_data.head())




    st.write("### Dataset Info")
    buffer = io.StringIO()
    customer_data.info(buf=buffer)
    s = buffer.getvalue()
    st.code(s)


    st.write("### Checking for Missing Values")
    st.write(customer_data.isnull().sum())

    # Selecting features for clustering
    X = customer_data.iloc[:, [3, 4]].values

    # Elbow Method to find optimal k
    st.subheader("Elbow Method to Find Optimal Number of Clusters (k)")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    sns.set()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

    # Select k using slider (default 5 like your code)
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    Y = kmeans.fit_predict(X)

    st.subheader("Cluster Labels")
    st.write(Y)

    # Plotting Clusters
    st.subheader("Visualization of Clusters and Centroids")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ['green', 'red', 'yellow', 'violet', 'blue', 'orange', 'brown', 'pink', 'gray', 'purple']

    for i in range(k):
        ax2.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    
    # Centroids
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='cyan', marker='X', label='Centroids')

    ax2.set_title("Customer Groups")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.legend()
    st.pyplot(fig2)

    # Download clustered data
    result_df = customer_data.copy()
    result_df['Cluster'] = Y
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", data=csv, file_name='clustered_customers.csv', mime='text/csv')

else:
    st.info("Upload the dataset to start clustering.")

