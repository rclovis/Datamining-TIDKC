import time
import sys

import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


from sklearn.manifold import MDS
from tslearn.metrics import dtw

from utils.dataloader import load_and_preprocess_data
from utils.distance_measure import gdk
from IDK import *

plt.style.use('ggplot')

class TrajClustering:
    def __init__(self):
        self.data = []
        self.labels = []
        self.score = []
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
                self.score[i][j] = distance.directed_hausdorff(self.data[i], self.data[j])[0]
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

    def plot_mds(self):
        print("plotting MDS")
        mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
        mds_transformed = mds.fit_transform(self.score)
        plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], c=self.labels, cmap='jet')
        plt.show()

    def plot_ground_truth(self):
        print("plotting ground truth")
        largest_label = max(self.labels)
        for i in range(len(self.data)):
            plt.plot(self.data[i][:, 0], self.data[i][:, 1], color=plt.cm.jet(self.labels[i] / largest_label))
        plt.show()

    def load_dataset(self, dataset_name):
        try:
            self.data, self.labels = load_and_preprocess_data(dataset_name)
            self.data = self.data[:500]
            self.labels = self.labels[:500]
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found")
            sys.exit(1)

    def run_distance (self, distance_metric):
        start_time = time.time()
        match distance_metric:
            case 'IDK2':
                self.idk2_metric()
            case 'IDK':
                self.idk_metric()
            case 'Hausdorff':
                self.hausdorff_metric()
            case 'DTW':
                self.dtw_metric()
            case 'EMD':
                self.emd_metric()
            case 'GDK':
                self.gdk_metric()
            case _:
                print(f"Distance metric {distance_metric} not found")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        return self.score

    def run_clustering (self, clustering_method):
        start_time = time.time()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        return
