import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state


class IDK:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed  # random seed

    def inne(self, data, psi, estimators):
        # psi = ψ number of samples to be selected for each centroid
        # estimators = t number of sub-samples to be selected (number of centroids)
        x_data, y_data = data.shape
        centroids = np.empty([estimators * psi])
        centroids_radius = np.empty([estimators * psi])

        ## Generate a random seed for each estimator
        seeds = check_random_state(self.random_seed).randint(
            0, 2**32 - 1, size=estimators
        )

        isolation_scores = np.ones([data.shape[0], estimators * psi])
        for i in range(estimators):

            # choose ψ samples from dataset to use as centroid
            samples = check_random_state(seeds[i]).choice(x_data, psi, replace=False)
            centroids[i * psi : (i + 1) * psi] = samples
            #

            tdata = data[samples, :]

            # calculate the distance between each centroid and all other centroids to create a pairwise distance matrix
            pairwise_distance = euclidean_distances(tdata, tdata, squared=True)
            np.fill_diagonal(pairwise_distance, np.inf)
            #

            # the radius of each centroids is the distance to its nearest centroid neighbor
            centroids_radius[i * psi : (i + 1) * psi] = np.amin(
                pairwise_distance, axis=1
            )
            #

            # cnn_index is the number of the nearest centroid for each centroid
            # cnn_radius is the radius of each cnn_index centroid
            index_nearest_centroid = np.argmin(pairwise_distance, axis=1)
            radius_nearest_centroid = centroids_radius[i * psi : (i + 1) * psi][
                index_nearest_centroid
            ]
            #

            x_dists = euclidean_distances(data, tdata, squared=True)
            # find instances that are covered by at least one hypersphere.

            # check if the point of the dataset is inside the hypersphere of the centroids
            cover_radius = np.where(
                x_dists <= centroids_radius[i * psi : (i + 1) * psi], 1.0, 0.0
            )
            for u in range(data.shape[0]):
                isolation_scores[u][i * psi : (i + 1) * psi] = cover_radius[u]
        return isolation_scores

    def idk(self, data, psi=4, estimators=100):
        alldata = []
        index_lines = np.array([0])
        for i in range(len(data)):
            for data_point in data[i]:
                alldata.append(data_point)
            index_lines = np.append(index_lines, len(alldata))
        alldata = np.array(alldata)

        score = self.inne(alldata, psi, estimators)
        idkmap = []
        for i in range(len(data)):
            idkmap.append(
                np.sum(score[index_lines[i] : index_lines[i + 1]], axis=0)
                / (index_lines[i + 1] - index_lines[i])
            )
        idkmap = np.array(idkmap)

        # idkm2_mean = np.average(idkmap, axis=0) / estimators
        # idk_score = np.dot(idkmap, idkm2_mean.T)
        return idkmap

    def idk_square(self, data, psi1=4, psi2=4, estimators1=100, estimators2=100):
        alldata = []
        index_lines = np.array([0])
        for i in range(len(data)):
            for data_point in data[i]:
                alldata.append(data_point)
            index_lines = np.append(index_lines, len(alldata))
        alldata = np.array(alldata)

        score1 = self.inne(alldata, psi1, estimators1)
        idkmap1 = []
        for i in range(len(data)):
            idkmap1.append(
                np.sum(score1[index_lines[i] : index_lines[i + 1]], axis=0)
                / (index_lines[i + 1] - index_lines[i])
            )
        idkmap1 = np.array(idkmap1)

        score2 = self.inne(idkmap1, psi2, estimators2)
        # idkm2_mean = np.average(score2, axis=0) / estimators2
        # score2 = np.dot(score2, idkm2_mean.T)
        return score2

    def k2(self, data, psi=4, estimators=100):
        score = self.inne(data, psi, estimators)
        # TODO: investigate weird
        idkm2_mean = np.average(score, axis=0) / estimators
        score = np.dot(score, idkm2_mean.T)
        return score

    def k2_draft(self, data, psi=4, estimators=100):
        """
        alldata = []"""
        index_lines = np.array([0])

        for i in range(len(data)):
            # for data_point in data[i]:
            # alldata.append(data_point)
            index_lines = np.append(index_lines, len(alldata))
        alldata = np.array(alldata)

        score = self.inne(data, psi, estimators)
        idkmap = []
        for i in range(len(data)):
            idkmap.append(
                np.sum(score[index_lines[i] : index_lines[i + 1]], axis=0)
                / (index_lines[i + 1] - index_lines[i])
            )
        idkmap = np.array(idkmap)
        return idkmap.max()
