from sklearn_extra.cluster import KMedoids
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory
from IDK2_REUSE import idk_kernel_map

"""  
This program is by far runnable, but didn't check the correctness
"""

# Load the dataset
data, labels = load_and_preprocess_data('TRAFFIC')

# Calculate IDK distance matrix
psi = 4  # You can choose an appropriate value for psi
idk_dists = idk_kernel_map(data, psi)

# Only use the first n_samples columns of the distance matrix
idk_dists = idk_dists[:, :len(data)]

kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict the KMedoids model with the IDK distance matrix
kmedoids.fit_predict(idk_dists)


# Print the labels
print(kmedoids.labels_)

visualize_trajectory(data, kmedoids.labels_)
