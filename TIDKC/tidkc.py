import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

# STEPS:

# initialise/inputs
# k = num clusters, determined by unique ground truth labels in the dataset
knn = 70
# % v is the learning rate (???)
# s is the sample size for mode selection (???)


# 1: trajectory -> point mapping (RKHS, ruckus package?)

# 2: identify cluster centres / findmode

    # count # features in dataset
    # initialise cluster sums arr (csum) and cluster sizes array (csize)
    # loop for assigning cluster seeds -- findmode must already be implemented here

def find_mode(dataset, k, knn) -> np.ndarray:
    # inputs: dataset, k, knn
    # outputs: ID (indices of cluster centres in the dataset)
    mean_array = np.mean(dataset, axis=0)
    density = np.dot(mean_array, dataset)
    dist = squareform(pdist(dataset, metric='euclidean'))

    lc_density = get_lc(dist, density, knn)

    # variable initialisation
    max_dist = np.max(dist) # max distance found in the dist ndarray
    num_datapoints = dist.shape[1]
    min_dists = np.full(num_datapoints, max_dist) # initialise ndarray storing min distance from higher-density point, default=max_dist
    sorted_density = np.argsort(lc_density)[::-1] # sorts from highest to lowest density
    nearest_neighbour = np.zeros(num_datapoints, dtype=int)

    # finds min_dists value for all points except highest-density pt
    for i in range(1, num_datapoints):
        for j in range(i): # if dist between current pt and previous pts < current min_dist, update min_dist + nearest_neighbour
            if dist[sorted_density[i], sorted_density[j]] < min_dists[sorted_density[i]]:
                min_dists[sorted_density[i]] = dist[sorted_density[i], sorted_density[j]]
                nearest_neighbour[sorted_density[i]] = sorted_density[j]
    min_dists[sorted_density[0]] = np.max(min_dists) # highest density pt's min dist = max min_dists value

    # rank elements based on how often they appear in density + min dists, return the k elements with highest rank
    density_ranks = rankdata(density, method='ordinal')
    min_dist_ranks = rankdata(min_dists, method='ordinal')
    mult = density_ranks * min_dist_ranks
    sorted_mult = np.argsort(mult)[::-1]

    return sorted_mult[:k]



# local contrast estimation to find cluster seeds:
def get_lc(dist: np.ndarray, density: np.ndarray, knn: int) -> np.ndarray:
    # dist: nxn distance matrix of dataset
    # density: density vector of dataset
    n = density.shape[0] # get num dimensions of the density vector
    lc = np.zeros(n) # instantiate matrix filled w zeros, shape = nx1

    for i in range(n):
        inx = np.argsort(dist[i, :])  # find dist of all points from i, sort from lowest to highest
        knn_array = inx[1:knn+1]  # identify k-nearest neighbours
        lc[i] = np.sum(density[i] > density[knn_array]) # count num neighbours where i has higher density, assign

    return lc
