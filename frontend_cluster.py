import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
import pickle

# Title of the web app
st.title("KMeans Clustering for Mall Customers")

# Upload file functionality
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(dataset.head())

    # Extract the relevant columns for clustering
    x = dataset.iloc[:, [3, 4]].values

    # The Elbow Method for optimal number of clusters
    st.subheader("Elbow Method for Optimal Clusters")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plot Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method', fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('WCSS', fontsize=14)
    st.pyplot(plt)

    # KMeans clustering with 5 clusters (based on Elbow Method)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    # Visualize the clusters
    st.subheader("Cluster Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of Customers', fontsize=16)
    plt.xlabel('Annual Income (k$)', fontsize=14)
    plt.ylabel('Spending Score (1-100)', fontsize=14)
    plt.legend()
    st.pyplot(plt)

    # Button to save the model as a pickle file
    if st.sidebar.button("Save Model"):
        filename = 'k_means.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(kmeans, file)
        st.sidebar.success(f"Model has been pickled and saved as {filename}")
else:
    st.write("Please upload a CSV file to proceed.")