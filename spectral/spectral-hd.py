from sklearn.cluster import SpectralClustering
import numpy as np
from distance_measure import hausdorff_distance

from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory

# Load the dataset
data, labels = load_and_preprocess_data('TRAFFIC')

# Calculate hausdorff distance matrix
dists = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        dist = hausdorff_distance(data[i], data[j])
        dists[i,j] = dist
        dists[j,i] = dist

# Calculate similarity matrix
similarity = np.exp(-dists)

spectral_clustering = SpectralClustering(n_clusters=5, random_state=0, affinity='precomputed')

# Fit and predict the SpectralClustering model with the distance matrix
spectral_clustering.fit_predict(similarity)

# # Print the labels
print(spectral_clustering.labels_)

visualize_trajectory(data, spectral_clustering.labels_)