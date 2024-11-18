import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from TIDKCC import tidkc
from utils.dataloader import load_and_preprocess_data

plt.style.use("ggplot")

class TrajClustering:
    def __init__(self):
        self.data = []
        self.ground_truth_labels = []
        self.labels = []

    def tidkc_clustering(self, number_of_clusters):
        # Option A: Flatten trajectories
        flattened_data = [traj.flatten() for traj in self.data]
        # Convert list to 2D NumPy array
        data_array = np.array(flattened_data)
        
        # Check data shape
        print("Data shape:", data_array.shape)
        
        # Proceed with clustering
        self.labels = tidkc(data_array, k=number_of_clusters)


    def plot_clusters(self):
        print("Plotting clusters")
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            for idx in label_indices:
                plt.plot(
                    self.data[idx][:, 0],
                    self.data[idx][:, 1],
                    color=plt.cm.jet(label / max(unique_labels)),
                )
        plt.title("Clustering Results")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def load_dataset(self, dataset_name):
        try:
            self.data, self.ground_truth_labels = load_and_preprocess_data(dataset_name)
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found")
            sys.exit(1)

    def run_clustering(self, clustering_method, number_of_clusters=3):
        print(f"Running clustering method {clustering_method}")
        start_time = time.time()
        if clustering_method == "TIDKC":
            self.tidkc_clustering(number_of_clusters)
        else:
            print(f"Clustering method {clustering_method} not implemented")
            return
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        if hasattr(self, 'ground_truth_labels') and len(self.ground_truth_labels) == len(self.labels):
            print("Computing NMI")
            nmi = normalized_mutual_info_score(self.ground_truth_labels, self.labels)
            print(f"NMI: {nmi:.6f}")

            print("Computing ARI")
            ari = adjusted_rand_score(self.ground_truth_labels, self.labels)
            print(f"ARI: {ari:.6f}")
        else:
            print("Ground truth labels not available or do not match the number of predicted labels.")
