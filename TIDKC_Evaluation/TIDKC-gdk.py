
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
import time


class TIDKC:
    def __init__(self, n_clusters, max_iter=100, random_state=0, n_jobs=1, chunk_size=100, target_length=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs  # Reduced to 1 for debugging
        self.chunk_size = chunk_size
        self.target_length = target_length
        self.labels_ = None
        self.centers_ = None
        self.dist_matrix_cache = None

    def normalize_trajectory_length(self, trajectory):
        """Normalize trajectory to target length using linear interpolation"""
        if trajectory.shape[0] == self.target_length:
            return trajectory

        original_length = trajectory.shape[0]
        dims = trajectory.shape[1]

        # Create time points
        original_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, self.target_length)

        # Interpolate each dimension
        normalized_trajectory = np.zeros((self.target_length, dims))
        for d in range(dims):
            normalized_trajectory[:, d] = np.interp(
                new_indices,
                original_indices,
                trajectory[:, d]
            )

        return normalized_trajectory

    def gdk_distance(self, p, q):
        """
        Calculate Gaussian Dynamic Kernel (GDK) distance
        More robust and efficient implementation
        """
        # Normalize trajectories to target length
        p_norm = self.normalize_trajectory_length(p)
        q_norm = self.normalize_trajectory_length(q)

        # Flatten normalized trajectories
        p_flat = p_norm.flatten()
        q_flat = q_norm.flatten()

        # Calculate gamma based on the flattened dimension
        gamma = 1.0 / len(p_flat)

        # Compute RBF kernel similarity
        similarity = np.exp(-gamma * np.sum((p_flat - q_flat) ** 2))

        # Convert to distance
        return 1 - similarity

    def calculate_distance_matrix(self, data):
        """
        Calculate distance matrix with improved performance
        """
        start_time = time.time()
        print("Starting distance matrix calculation...")

        n = len(data)
        dist_matrix = np.zeros((n, n))

        # Use nested loops with progress bar for visibility
        for i in tqdm(range(n), desc="Calculating distances"):
            for j in range(i, n):
                dist = self.gdk_distance(data[i], data[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        end_time = time.time()
        print(f"Distance matrix calculation completed in {end_time - start_time:.2f} seconds")

        return dist_matrix

    def initialize_centers(self, n_samples, dist_matrix):
        """Initialize centers using k-means++ like strategy"""
        np.random.seed(self.random_state)
        centers = [np.random.randint(n_samples)]

        for _ in range(self.n_clusters - 1):
            distances = dist_matrix[:, centers].min(axis=1)
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_center = np.random.choice(n_samples, p=probabilities)
            centers.append(next_center)

        return np.array(centers)

    def assign_labels_vectorized(self, dist_matrix, centers):
        """Vectorized label assignment"""
        return np.argmin(dist_matrix[:, centers], axis=1)

    def update_centers_vectorized(self, dist_matrix, labels):
        """Vectorized center updates"""
        new_centers = []
        for k in range(self.n_clusters):
            cluster_points = np.where(labels == k)[0]
            if len(cluster_points) > 0:
                cluster_distances = dist_matrix[cluster_points][:, cluster_points]
                max_distances = np.max(cluster_distances, axis=1)
                new_center = cluster_points[np.argmin(max_distances)]
                new_centers.append(new_center)
            else:
                unused_points = np.setdiff1d(np.arange(len(dist_matrix)), new_centers)
                new_centers.append(np.random.choice(unused_points) if len(unused_points) > 0
                                   else np.random.randint(len(dist_matrix)))
        return np.array(new_centers)

    def fit(self, data):
        n_samples = len(data)

        # Calculate distance matrix with progress bar
        print("Calculating distance matrix...")
        dist_matrix = self.calculate_distance_matrix(data)

        # Initialize centers using k-means++ like strategy
        print("Initializing centers...")
        centers = self.initialize_centers(n_samples, dist_matrix)

        # Main clustering loop
        print("Clustering...")
        for iteration in range(self.max_iter):
            old_centers = centers.copy()

            # Assign labels using vectorized operations
            labels = self.assign_labels_vectorized(dist_matrix, centers)

            # Update centers using vectorized operations
            centers = self.update_centers_vectorized(dist_matrix, labels)

            # Check convergence
            if np.array_equal(old_centers, centers):
                print(f"Converged after {iteration + 1} iterations")
                break

        self.labels_ = labels
        self.centers_ = centers
        return self

    def fit_predict(self, data):
        self.fit(data)
        return self.labels_


# Usage example:
if __name__ == "__main__":
    # Load the dataset
    data, labels = load_and_preprocess_data('TRAFFIC')  # or any other dataset

    # Initialize and fit TIDKC
    tidkc = TIDKC(
        n_clusters=5,
        random_state=0,
        n_jobs=1,  # Set to 1 for debugging
        chunk_size=100,
        target_length=100  # Set the target length for trajectory normalization
    )

    # Measure total execution time as for larger datasets it takes some time to run
    start_total = time.time()
    predicted_labels = tidkc.fit_predict(data)
    end_total = time.time()

    # Print results
    print("Cluster Labels:", predicted_labels)
    print(f"Total Execution Time: {end_total - start_total:.2f} seconds")

    # Visualize results
    visualize_trajectory(data, predicted_labels)
    #Somewhat working