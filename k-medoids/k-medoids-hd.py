from sklearn_extra.cluster import KMedoids
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

kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict the KMedoids model with the distance matrix
kmedoids.fit_predict(dists)

# # Print the labels
print(kmedoids.labels_)

visualize_trajectory(data, kmedoids.labels_)