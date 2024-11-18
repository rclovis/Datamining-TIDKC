import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state

class IDK:
    def __init__(self, psi=16, estimators=500, random_seed=5210):
        self.psi = psi              # Subsample size for IK
        self.estimators = estimators  # Number of subsamples
        self.random_seed = random_seed  # Random seed for reproducibility

    def inne(self, data):
        n_samples, n_features = data.shape
        centroids = np.empty([self.estimators * self.psi, n_features])
        centroids_radius = np.empty([self.estimators * self.psi])

        # Generate random seeds for reproducibility
        seeds = check_random_state(self.random_seed).randint(0, 2**32 - 1, size=self.estimators)

        isolation_scores = np.zeros([n_samples, self.estimators * self.psi])
        for i in range(self.estimators):
            # Randomly select psi samples as centroids
            sample_indices = check_random_state(seeds[i]).choice(n_samples, self.psi, replace=False)
            centroids[i * self.psi : (i + 1) * self.psi, :] = data[sample_indices, :]

            # Compute pairwise distances among centroids
            centroid_distances = euclidean_distances(centroids[i * self.psi : (i + 1) * self.psi, :])
            np.fill_diagonal(centroid_distances, np.inf)

            # Radius is the minimum distance to nearest centroid
            centroids_radius[i * self.psi : (i + 1) * self.psi] = np.min(centroid_distances, axis=1)

            # Compute distances from all data points to centroids
            distances = euclidean_distances(data, centroids[i * self.psi : (i + 1) * self.psi, :])

            # Points within centroid radius get a score of 1
            cover_radius = distances <= centroids_radius[i * self.psi : (i + 1) * self.psi]
            isolation_scores[:, i * self.psi : (i + 1) * self.psi] = cover_radius.astype(float)

        return isolation_scores

    def create_idk(self, data):
        # Compute the Isolation Distributional Kernel (IDK)
        isolation_scores = self.inne(data)
        # Normalize the isolation scores to get the feature mapping
        idk_features = isolation_scores / isolation_scores.shape[1]
        return idk_features
