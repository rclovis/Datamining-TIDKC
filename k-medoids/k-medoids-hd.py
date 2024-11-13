from sklearn_extra.cluster import KMedoids
import numpy as np
from utils.distance_measure import hausdorff_distance

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

"""
"""Compare this with above """
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from scipy.spatial.distance import directed_hausdorff
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory

# Define Hausdorff Distance function
def hausdorff_distance(p, q):
    return max(directed_hausdorff(p, q)[0], directed_hausdorff(q, p)[0])

# Calculate the Hausdorff distance matrix with multi-threading
def hd_distance_matrix(data, n_jobs=-1):
    n = len(data)
    hd_dists = np.zeros((n, n))

    # Compute pairwise Hausdorff distances in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(hausdorff_distance)(data[i], data[j]) for i in range(n) for j in range(i, n))

    # Fill the symmetric distance matrix
    k = 0
    for i in range(n):
        for j in range(i, n):
            hd_dists[i, j] = results[k]
            hd_dists[j, i] = results[k]  # Ensure symmetry
            k += 1

    return hd_dists

# Load the dataset
data, labels = load_and_preprocess_data('CASIA')

# Compute the Hausdorff distance matrix
hd_dists = hd_distance_matrix(data)

# Initialize and fit the KMedoids model with the Hausdorff distance matrix
kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')
kmedoids.fit_predict(hd_dists)

# Print the resulting labels
print(kmedoids.labels_)
visualize_trajectory(data, kmedoids.labels_)
"""
