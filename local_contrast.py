import numpy as np


def local_contrast(
    distance_matrix: np.ndarray, density_vector: np.ndarray, kn: int
) -> np.ndarray:
    """
    Local Contrast of an instance x is defined as the number of times that x has
    higher density than its K nearest neighbours

    Calculate Local Contrast (LC) for each point in a dataset.

    Parameters:
    - distance_matrix: 2D numpy array (N x N), distance matrix for the dataset
    - density_vector: 1D numpy array (N,), density values for each point in the dataset
    - kn: int, number of nearest neighbors to consider

    Returns:
    - LC: 1D numpy array (N,), Local Contrast values for each point
    """
    N = len(density_vector)
    LC = np.zeros(N)

    for i in range(N):
        # Get the indices of the nearest neighbors, excluding the point itself
        knn_indices = np.argsort(distance_matrix[i])[1 : kn + 1]
        # Calculate local contrast for point i
        LC[i] = np.sum(density_vector[i] > density_vector[knn_indices])

    return LC


# local contrast estimation to find cluster seeds:
def get_lc(dist: np.ndarray, density: np.ndarray, knn: int) -> np.ndarray:
    # dist: nxn distance matrix of dataset
    # density: density vector of dataset
    n = density.shape[0]  # get num dimensions of the density vector
    lc = np.zeros(n)  # instantiate matrix filled w zeros, shape = nx1

    for i in range(n):
        inx = np.argsort(
            dist[i, :]
        )  # find dist of all points from i, sort from lowest to highest
        knn_array = inx[1 : knn + 1]  # identify k-nearest neighbours
        lc[i] = np.sum(
            density[i] > density[knn_array]
        )  # count num neighbours where i has higher density, assign

    return lc
