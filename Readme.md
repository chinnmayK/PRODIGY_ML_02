
---

# K-Means Clustering for Retail Customer Segmentation

This project uses the K-means clustering algorithm to group retail customers based on their `Annual Income (k$)` and `Spending Score (1-100)` to identify distinct customer segments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kmeans-customer-clustering.git
   ```
2. Install required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```

## Usage

1. Load the dataset (`Mall_Customers.csv`).
2. Run the script to:
   - Visualize customer data.
   - Determine the optimal number of clusters using the Elbow method.
   - Apply K-means and visualize the customer segments.
   
   ```bash
   python kmeans_clustering.py
   ```

## Methodology

- **Elbow Method**: Determines the optimal number of clusters.
- **K-Means**: Clusters customers into distinct groups and visualizes the results.

## Visualization

Three plots are generated:
- Initial customer data.
- Elbow curve for optimal clusters.
- Customer segments with centroids.

---