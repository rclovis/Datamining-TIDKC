import numpy as np
from local_contrast import get_lc
from scipy.spatial.distance import pdist, squareform


def find_mode(dataset, k, knn) -> np.ndarray:
    # inputs: dataset, k, knn
    # outputs: ID (indices of cluster centres in the dataset)
    mean_array = np.mean(dataset, axis=0)
    density = np.dot(mean_array, dataset)
    dist = squareform(pdist(dataset, metric="euclidean"))

    lc_density = get_lc(dist, density, knn)

    # variable initialisation
    max_dist = np.max(dist)  # max distance found in the dist ndarray
    num_datapoints = dist.shape[1]
    min_dists = np.full(
        num_datapoints, max_dist
    )  # initialise ndarray storing min distance from higher-density point, default=max_dist
    sorted_density = np.argsort(lc_density)[
        ::-1
    ]  # sorts from highest to lowest density
    nearest_neighbour = np.zeros(num_datapoints, dtype=int)

    # finds min_dists value for all points except highest-density pt
    for i in range(1, num_datapoints):
        for j in range(
            i
        ):  # if dist between current pt and previous pts < current min_dist, update min_dist + nearest_neighbour
            if (
                dist[sorted_density[i], sorted_density[j]]
                < min_dists[sorted_density[i]]
            ):
                min_dists[sorted_density[i]] = dist[
                    sorted_density[i], sorted_density[j]
                ]
                nearest_neighbour[sorted_density[i]] = sorted_density[j]
    min_dists[sorted_density[0]] = np.max(
        min_dists
    )  # highest density pt's min dist = max min_dists value

    # rank elements based on how often they appear in density + min dists, return the k elements with highest rank
    density_ranks = rankdata(density, method="ordinal")
    min_dist_ranks = rankdata(min_dists, method="ordinal")
    mult = density_ranks * min_dist_ranks
    sorted_mult = np.argsort(mult)[::-1]

    return sorted_mult[:k]
