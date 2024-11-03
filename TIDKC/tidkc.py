import numpy as np

# STEPS:

# initialise/inputs
# k = num clusters, determined by unique ground truth labels in the dataset
# k = size of of k-nearest neighbours, hardcoded to 70 as per the IDKC_traj code
# % v is the learning rate (???)
# s is the sample size for mode selection (???)


# 1: trajectory -> point mapping (RKHS, ruckus package?)

# 2: identify cluster centres / findmode

    # count # features in dataset
    # initialise cluster sums arr (csum) and cluster sizes array (csize)
    # loop for assigning cluster seeds -- findmode must already be implemented here
    # also the number of clusters is determined by the ground truth label

def find_mode(dataset, k, knn) -> np.ndarray:
    mean_array = np.mean(dataset, axis=0)
    density = np.dot(mean_array, dataset)

    # inputs: dataset, k, knn
    # outputs: ID (indices of cluster centres in the dataset)
    # 1. create numeric array 'density', containing density of each point in dataset.
    # first create array (vector) containing mean of all points for each feature
    # then calculate density by doing dot product: np.dot(dataset, vector)



# 2.5: local contrast estimation for finding cluster seeds (finding points with a much higher density than their k-nearest neighbours):
def get_lc(dist: np.ndarray, density: np.ndarray, knn: int) -> np.ndarray:
    # dist: nxn distance matrix of dataset
    # density: density vector of dataset

    n = density.shape[0] # get num dimensions of the density vector
    lc = np.zeros(n) # instantiate matrix filled w zeros, shape = nx1

    for i in range(n):
        inx = np.argsort(dist[i, :])  # find the distances of all points from i, and sort them from lowest to highest distance
        knn_array = inx[1:knn+1]  # identify the k-nearest neighbours of i
        lc[i] = np.sum(density[i] > density[knn_array]) # count the number of neighbours where i has a higher density than them, assign this count to lc[i]

    return lc
