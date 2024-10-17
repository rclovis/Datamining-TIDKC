from sklearn_extra.cluster import KMedoids
import tslearn.metrics as metrics

from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory

# Load the dataset
data, labels = load_and_preprocess_data('TRAFFIC')

# Calculate DTW distance matrix
dists = metrics.cdist_dtw(data)

kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict the KMedoids model with the distance matrix
kmedoids.fit_predict(dists)

# Print the labels
print(kmedoids.labels_)

visualize_trajectory(data, kmedoids.labels_)