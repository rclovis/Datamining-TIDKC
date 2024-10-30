import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances

import random
import sys

class IsolationNNE:
    def __init__(self, psi=16, estimators=200, contamination=0.15, random_seed=None):
        self.psi = psi # ψ number of samples to be selected for each centroid
        self.estimators = estimators # t number of sub-samples to be selected (number of centroids)
        self.contamination = contamination # contamination rate. Threshold for the decision function
        self.random_seed = random_seed # random seed

        self._centroids = []
        self._centroids_radius = []
        self._ratio = []

    def generate_centroid(self, data):
        x_data, y_data = data.shape

        self._centroids = np.empty([self.estimators, self.psi, y_data])
        self._ratio = np.empty([self.estimators, self.psi])

        self._centroids_radius = np.empty([self.estimators, self.psi])

        random_state = check_random_state(self.random_seed)
        self._seeds = random_state.randint(sys.maxsize, size=self.estimators)

        for i in range(self.estimators):
            rnd = check_random_state(self._seeds[i])

            # choose ψ samples from dataset to use as centroid
            self._centroids[i] = data[rnd.choice(x_data, self.psi, replace=False)]
            #

            # calculate the distance between each centroid and all other centroids to create a pairwise distance matrix
            pairwise_distance = euclidean_distances(self._centroids[i], self._centroids[i], squared=True)
            np.fill_diagonal(pairwise_distance, np.inf)
            #

            # the radius of each centroids is the distance to its nearest centroid neighbor
            self._centroids_radius[i] = np.amin(pairwise_distance, axis=1)
            #

            # cnn_index is the number of the nearest centroid for each centroid
            # cnn_radius is the radius of each cnn_index centroid
            index_nearest_centroid = np.argmin(pairwise_distance, axis=1)
            radius_nearest_centroid = self._centroids_radius[i][index_nearest_centroid]
            #

            # the ratio is the proportion of the radius of the nearest centroid to the radius of the centroid
            # the ratio is used to calculate the isolation score
            # the higher the ratio, the more isolated the centroid is
            self._ratio[i] = 1 - (radius_nearest_centroid + np.finfo(float).eps) / (self._centroids_radius[i] + np.finfo(float).eps)
            #

        return self

    def predict(self, data):
        decision_func = self.score_samples(data)
        is_inlier = np.ones_like(decision_func, dtype=int)

        # 100.0 * self.contamination percent of the points are above the threshold
        threshold = np.percentile(decision_func, 100.0 * self.contamination)
        #

        is_inlier = np.where(decision_func < threshold, 0, 1)
        return is_inlier

    def score_samples(self, data):
        isolation_scores = np.ones([self.estimators, data.shape[0]])
        # each test instance is evaluated against estimators sets of hyperspheres
        for i in range(self.estimators):
            x_dists = euclidean_distances(data, self._centroids[i],  squared=True)
            # find instances that are covered by at least one hypersphere.

            # check if the point of the dataset is inside the hypersphere of the centroids
            cover_radius = np.where(x_dists <= self._centroids_radius[i], self._centroids_radius[i], np.nan)

            # get the indices of the points of the dataset that are inside at leat one hypersphere of a centroids
            x_covered = np.where(~np.isnan(cover_radius).all(axis=1))

            # the centroid of the hypersphere covering x and having the smallest radius
            cnn_x = np.nanargmin(cover_radius[x_covered], axis=1)
            isolation_scores[i][x_covered] = self._ratio[i][cnn_x]
        # the isolation scores are averaged to produce the anomaly score
        scores = 1 - np.mean(isolation_scores, axis=0)
        return scores
