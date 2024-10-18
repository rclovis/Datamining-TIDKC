from sklearn_extra.cluster import KMedoids
import tslearn.metrics as metrics
from tqdm import tqdm

from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory
import numpy as np

# Load the dataset
print("Loading and preprocessing the dataset...")
data, labels = load_and_preprocess_data('geolife')
print("Data loaded. Number of time series:", len(data))

# Prepare an empty matrix to store DTW distances
n_samples = len(data)
dists = np.zeros((n_samples, n_samples))

# Calculate DTW distance matrix with a progress bar
print("Calculating DTW distance matrix...")
for i in tqdm(range(n_samples), desc="Computing distances"):
    for j in range(i + 1, n_samples):  # Matrix is symmetric
        dist = metrics.cdist_dtw(data[i:i+1], data[j:j+1])[0][0]
        dists[i, j] = dist
        dists[j, i] = dist
print("Distance matrix computed.")

# Initialize KMedoids with precomputed distance matrix
print("Clustering with KMedoids...")
kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict the KMedoids model with the distance matrix
cluster_labels = kmedoids.fit_predict(dists)
print("Clustering completed.")

# Print the labels
print("Cluster labels:")
print(cluster_labels)

# Visualize the trajectory with the computed labels
print("Visualizing the clustered trajectories...")
visualize_trajectory(data, cluster_labels)
print("Visualization complete.")

