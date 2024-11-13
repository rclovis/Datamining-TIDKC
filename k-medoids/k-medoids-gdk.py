"""from sklearn_extra.cluster import KMedoids
import numpy as np
import os, sys

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from utils.eval_clusters import nmi_score, ari_score
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory


# Gaussian Dynamic Kernel (GDK) function - requires custom kernel implementation or a library that supports it.
def gaussian_dynamic_kernel(X, sigma=1.0):
    ""Compute the Gaussian Dynamic Kernel (GDK) similarity matrix for trajectories.
    N = len(X)
    kernel_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist = np.linalg.norm(X[i] - X[j])  # or a more complex similarity based on dynamic features
            kernel_matrix[i, j] = np.exp(-dist**2 / (2 * sigma**2))
    return kernel_matrix

# Load the dataset (assuming data is preprocessed appropriately)
data, labels = load_and_preprocess_data('TRAFFIC')

# Calculate the GDK similarity matrix
gdk_similarity_matrix = gaussian_dynamic_kernel(data)

kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict with the similarity matrix
kmedoids.fit_predict(1 - gdk_similarity_matrix)  # converting similarity to a pseudo-distance

print(kmedoids.labels_)
print("ARI: ", ari_score(labels, kmedoids.labels_))
print("NMI: ", nmi_score(labels, kmedoids.labels_))

visualize_trajectory(data, kmedoids.labels_)
"""
"""Commented above code to try the following"""
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import rbf_kernel
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory

# Define the GDK function
def gdk(p, q):
    gamma = 1.0 / p.shape[1]  # Adjust gamma based on feature dimension
    similarity_matrix = rbf_kernel(p, q, gamma=gamma)
    k_g = np.mean(similarity_matrix)
    return k_g

# Calculate the GDK distance matrix for clustering
def gdk_distance_matrix(data):
    n = len(data)
    gdk_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = gdk(data[i], data[j])
            gdk_dists[i, j] = dist
            gdk_dists[j, i] = dist  # Ensure symmetry
    return 1 - gdk_dists  # Convert similarities to "pseudo-distances" for KMedoids

# Load the dataset
data, labels = load_and_preprocess_data('TRAFFIC')

# Compute the GDK distance matrix
gdk_dists = gdk_distance_matrix(data)

# Initialize and fit the KMedoids model with the GDK distance matrix
kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')
kmedoids.fit_predict(gdk_dists)

# Print the resulting labels
print(kmedoids.labels_)
visualize_trajectory(data, kmedoids.labels_)
