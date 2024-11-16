import numpy as np
from scipy.spatial import distance_matrix
from utils.dataloader import load_and_preprocess_data
import sys, os
from IDK import *
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform

"""TIDKC Algorithm -- from paper
1:  Map each trajectory Ti ∈ D to a point gi = Φ1(PTi ) in RKHS H1
    using feature map Φ1 of K1 to yield G = {g1,...,gn}
2:  Select k cluster seeds cj ∈ G based on local-contrast estimation;
    and initialize clusters Cj = {cj }, j = 1, ..., k
3:  Initialize N ← G \\ ∪j Cj
4:  Initialize τ ← max g∈N,∈[1,k] K2(δ(g), PC )
5:  repeat
6:  τ ← rho × τ
7:  Expand cluster Cj to include unassigned point
    g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ
8:  N ← G \\ ∪j Cj
9:  until |N| = 0 or τ < 0.00001
10: Assign each unassigned point g to nearest cluster C via K2(δ(g), PC )
11: Cluster Ej ⊂ D corresponds to Cj ⊂ G, j = 1, . . . , k
"""

def tidkc(D: np.ndarray, k: int, idk:IDK):
    """Clustering using IDK
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    """
    kn = 70  # K neighbor threshhold
    t = 100  # number of estimator (used in IDK)
    psi = 8  # number of seeds (used in IDK)
    rho = 0.9  # growth rate

    ## Step 1 - Map each trajectory in RKHS using K1
    G = idk.idk(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    ## and initialise cluster array Cj of length k
    c_seeds = find_mode(G, k, kn)
    Cj = np.empty(k, dtype=np.ndarray)
    for i in range(k):
        Cj[i] = []
        Cj[i].append(G[c_seeds[i]])
    
    ## Step 3 - Initialize N, the difference between G and the set of all
    ## cluster seeds
    N = np.delete(G, c_seeds, axis=0)

    ## Step 4 - Initialize T, the similarity threshold
    # τ ← max g∈N, L∈[1,k] K2(δ(g), PCL)
    cluster_seeds = G[c_seeds]
    max_similarity = -np.inf
    
    for g in N:
        for l in range(k):
            similarity_score = np.dot(idk.k2(N), cluster_seeds[l])

    tau = max_similarity

    ## Steps 5 & 9 - begin loop, set conditions for ending loop:
    while(abs(N) != 0 and tau >= 0.00001):
        ## Step 6 - update value of tau (τ)
        tau *= rho

        ## Step 7 - Expand cluster Cj to include unassigned point
            ## g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ
        newly_assigned_points = []
        for g in N:
            max_similarity = -np.inf
            best_cluster = -1
            for j in range(k):
                similarity_score = np.dot(idk.k2(g.reshape(1, -1)), cluster_seeds[j])
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    best_cluster = j
            if max_similarity > tau:
                Cj[best_cluster].append(g)
                newly_assigned_points.append(g)
            
        ## Step 8 - update value of N:  N ← G \\ ∪j Cj
        N = np.array([g for g in N if g not in newly_assigned_points])
    
    ## Step 10 - Assign each unassigned point g to nearest cluster C
    ## via K2(δ(g), PC )
    while(len(G) > 0):
        x = 0 # placeholder, this means nothing


    ## Step 11 - Cluster Ej ⊂ D corresponds to Cj ⊂ G,j = 1,...,k

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


def find_mode(dataset, k, knn) -> np.ndarray:
    # inputs: dataset, k, knn
    # outputs: ID (indices of cluster centres in the dataset)
    mean_array = np.mean(dataset, axis=0)
    density = np.dot(dataset, mean_array)
    dist = squareform(pdist(dataset, metric="euclidean"))
    
    lc_density = local_contrast(dist, density, knn)
    
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


print("started main")
idk = IDK(random_seed=42)
print("created idk")
data, labels = load_and_preprocess_data('TRAFFIC')
print("created dataset")
k = 11
N = tidkc(data, 11, idk)
print(N)
