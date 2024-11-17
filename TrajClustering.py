import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.metrics import dtw

from IDK import IDK
from tidkc import tidkc
from utils.dataloader import load_and_preprocess_data
from utils.distance_measure import gdk

plt.style.use("ggplot")


class TrajClustering:
    def __init__(self):
        self.data = []
        self.ground_truth_labels = []
        self.score = []
        self.labels = []
        pass

    def idk2_metric(self):
        idk = IDK(random_seed=42)
        self.score = idk.idk_square(self.data, 16, 16, 400, 400)

    def idk_metric(self):
        idk = IDK(random_seed=42)
        self.score = idk.idk(self.data, 16, 400)

    def hausdorff_metric(self):
        self.score = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                self.score[i][j] = distance.directed_hausdorff(
                    self.data[i], self.data[j]
                )[0]
                self.score[j][i] = self.score[i][j]

    def dtw_metric(self):
        self.score = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                self.score[i][j] = dtw(self.data[i], self.data[j])
                self.score[j][i] = self.score[i][j]

    def emd_metric(self):
        self.score = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                data1 = np.hstack(self.data[i])
                data2 = np.hstack(self.data[j])
                self.score[i][j] = wasserstein_distance(data1, data2)
                self.score[j][i] = self.score[i][j]

    def gdk_metric(self):
        self.score = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                self.score[i][j] = gdk(self.data[i], self.data[j])
                self.score[j][i] = self.score[i][j]

    def spectral_clustering(self, number_of_clusters):
        similarity = cosine_similarity(self.score)
        print(similarity.shape)
        print(similarity)
        spectral_clustering = SpectralClustering(
            n_clusters=number_of_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
        )
        self.labels = spectral_clustering.fit_predict(similarity)

    def kmeans_clustering(self, number_of_clusters):
        similarity = cosine_similarity(self.score)
        print(similarity.shape)
        kmeans = KMeans(n_clusters=number_of_clusters)
        self.labels = kmeans.fit_predict(self.score)

    def tidkc_clustering(self, number_of_clusters):
        self.labels = tidkc(self.data, number_of_clusters, rho=0.9)

    def plot_mds(self):
        print("plotting MDS")
        mds = MDS(n_components=2, random_state=42)
        mds_transformed = mds.fit_transform(self.score)
        plt.scatter(
            mds_transformed[:, 0],
            mds_transformed[:, 1],
            c=self.ground_truth_labels,
            cmap="jet",
        )
        plt.show()

    def plot_ground_truth(self):
        print("plotting ground truth")
        largest_label = max(self.ground_truth_labels)
        for i in range(len(self.data)):
            plt.plot(
                self.data[i][:, 0],
                self.data[i][:, 1],
                color=plt.cm.jet(self.ground_truth_labels[i] / largest_label),
            )
        plt.show()

    def plot_clusters(self):
        print("plotting clusters")
        largest_label = max(self.labels)
        for i in range(len(self.data)):
            plt.plot(
                self.data[i][:, 0],
                self.data[i][:, 1],
                color=plt.cm.jet(self.labels[i] / largest_label),
            )
        plt.show()

    def load_dataset(self, dataset_name):
        try:
            self.data, self.ground_truth_labels = load_and_preprocess_data(dataset_name)
            self.data = self.data[:500]
            self.ground_truth_labels = self.ground_truth_labels[:500]
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found")
            sys.exit(1)

    def run_distance(self, distance_metric):
        print(f"Running distance metric {distance_metric}")
        start_time = time.time()
        match distance_metric:
            case "IDK2":
                self.idk2_metric()
            case "IDK":
                self.idk_metric()
            case "Hausdorff":
                self.hausdorff_metric()
            case "DTW":
                self.dtw_metric()
            case "EMD":
                self.emd_metric()
            case "GDK":
                self.gdk_metric()
            case _:
                print(f"Distance metric {distance_metric} not found")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        return

    def run_clustering(self, clustering_method, number_of_clusters=3):
        print(f"Running clustering method {clustering_method}")
        start_time = time.time()
        match clustering_method:
            case "KMeans":
                self.kmeans_clustering(number_of_clusters)
            case "Spectral":
                self.spectral_clustering(number_of_clusters)
            case "TIDKC":
                self.tidkc_clustering(number_of_clusters)
            case _:
                print(f"Clustering method {clustering_method} not found")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        print("Computing NMI")
        nmi = normalized_mutual_info_score(self.ground_truth_labels, self.labels)
        print(nmi)
        print("Computing ARI")
        ari = adjusted_rand_score(self.ground_truth_labels, self.labels)
        print(ari)
        return nmi, ari
