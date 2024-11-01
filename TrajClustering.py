import time
import sys

import matplotlib
import matplotlib.pyplot as plt
from numba.cuda.simulator import kernel
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from tslearn.metrics import dtw

from utils.dataloader import load_and_preprocess_data
from IDK import *

plt.style.use('ggplot')

class TrajClustering:
    def __init__(self):
        self.data = []
        self.labels = []
        self.score = []
        pass

    def idk(self):
        idk = IDK(random_seed=42)
        self.score = idk.idk_square(self.data, 16, 16, 400, 400)

        # colors = np.array(["#000000", "#43cc5c"])
        # for i in range(len(self.data)):
        #     plt.plot(self.data[i][:, 0], self.data[i][:, 1], color=plt.cm.jet(self.score[i]))
        # plt.show()
        #
    def hausdorff_metric(self):
        pass

    def dtw_metric(self):
        self.score = np.zeros((len(self.data), len(self.data)))
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                self.score[i][j] = dtw(self.data[i], self.data[j])


    def emd_metric(self):
        pass

    def gdk_metric(self):

        alldata = []
        index_lines = np.array([0])
        for i in range(len(self.data)):
            for data_point in self.data[i]:
                alldata.append(data_point)
            index_lines = np.append(index_lines, len(alldata))
        alldata = np.array(alldata)
        kernel = gaussian_kde(alldata.T)
        self.score = np.zeros((len(self.data), len(alldata)))

        for i in range(len(self.data)):
            self.score[i] = kernel(self.data[i].T).shape
        print(self.score.shape)


    def plot_mds(self):
        print("plot")
        mds = MDS(n_components=2, random_state=42)
        mds_transformed = mds.fit_transform(self.score)
        plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], c=self.labels, cmap='jet')
        plt.show()

    def load_data(self, dataset_name):
        try:
            self.data, self.labels = load_and_preprocess_data(dataset_name)
            self.data = self.data[:500]
            self.labels = self.labels[:500]
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found")
            sys.exit(1)

    def run_distance (self, dataset_name, distance_metric):
        self.load_data(dataset_name)
        start_time = time.time()
        match distance_metric:
            case 'IDK':
                self.idk()
            case 'Hausdorff':
                self.hausdorff_metric()
            case 'DTW':
                self.dtw_metric()
            case 'EMD':
                self.emd_metric()
            case 'GDK':
                self.gdk_metric()

        return

    def run_clustering (self, dataset_name, clustering_method):
        self.load_data(dataset_name)
        start_time = time.time()
        return
