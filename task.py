import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Drop the CustomerID column
data.drop('CustomerID', axis=1, inplace=True)

# Extract relevant features
final = data.iloc[:, [2, 3]].values

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Visualize initial data
axes[0].scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c='blue')
axes[0].set_title('Initial Data')
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score (1-100)')

# 2. Elbow method to determine the optimal number of clusters
ilist = []
n = 11
for i in range(1, n):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(final)
    ilist.append(km.inertia_)

axes[1].plot(range(1, n), ilist, marker='o')
axes[1].set_title('Elbow Method')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Inertia')

# 3. Apply KMeans with optimal clusters (e.g., 5)
km = KMeans(n_clusters=5, random_state=42)
y_pred = km.fit_predict(final)

# Visualize clusters with centroids
axes[2].scatter(final[y_pred == 0, 0], final[y_pred == 0, 1], label='Cluster 1', c='r', s=100)
axes[2].scatter(final[y_pred == 1, 0], final[y_pred == 1, 1], label='Cluster 2', c='g', s=100)
axes[2].scatter(final[y_pred == 2, 0], final[y_pred == 2, 1], label='Cluster 3', c='c', s=100)
axes[2].scatter(final[y_pred == 3, 0], final[y_pred == 3, 1], label='Cluster 4', c='b', s=100)
axes[2].scatter(final[y_pred == 4, 0], final[y_pred == 4, 1], label='Cluster 5', c='m', s=100)
axes[2].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], label='Centroids', c='y', s=300)
axes[2].legend(bbox_to_anchor=(1.05, 1))
axes[2].set_title('Customer Segments')
axes[2].set_xlabel('Annual Income (k$)')
axes[2].set_ylabel('Spending Score (1-100)')

# Adjust layout
plt.tight_layout()
plt.show()
