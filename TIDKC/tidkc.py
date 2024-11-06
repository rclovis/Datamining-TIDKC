import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import rankdata
from ref.idk import idk_kernel_map as K1, idk_square as K2
from .local_contrast import local_contrast
from matplotlib import pyplot as plt

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


"""TIDKC Algorithm -- from paper
2:  Select k cluster seeds cj ∈ G based on local-contrast estimation;
    and initialize clusters Cj = {cj }, j = 1, ..., k
3:  Initialize N ← G \\ ∪j Cj
4:  Initialize τ ← max g∈N,∈[1,k] K2(δ(g), PC )
5:  repeat
6:  τ ←  × τ
7:  Expand cluster Cj to include unassigned point
    g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ
8:  N ← G \\ ∪j Cj
9:  until |N | = 0 or τ < 0.00001
10: Assign each unassigned point g to nearest cluster C via K2(δ(g), PC )
11: Cluster Ej ⊂ D corresponds to Cj ⊂ G, j = 1, . . . , k
"""


def tidkc(D: np.ndarray, k: int, rho: float):
    """Clustering using IDK
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    rho : growth rate in [0..1]
    """
    # Constants ?
    kn = 70  # K neighbor threshhold
    t = 100  # number of estimator (used in IDK)
    psi = 8  # number of seeds ? (used in IDK)
    v = 0.9  # learning rate ?

    ## Step 1 - Map each trajectory in RKHS using K1
    G = K1(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    G_dist = distance_matrix(G, G)
    # Multiply G by (the mean matrix of its columns shifted from line to column matrix)
    G_density = G @ np.mean(G, axis=0)[:, np.newaxis]  # bruh

    LC = local_contrast(G_dist, G_density, kn)
    # Select the k most dense points in LC (index)
    # This solves only part of the problem. Here we get the most dense points
    # in the space, but in order to have good clustering we need to have not
    # only dense points but point that need to be **both** *dense* and *have
    # a large distance to other dense points*. (Seems like that's what FindMode
    # does)
    c_seeds = np.argsort(LC)[::-1][:k]

    ## Step 3 - Initialize N, the difference between G and the set of all
    ## cluster seeds (points without clusters)

    N = np.delete(G, c_seeds, axis=0)

    ## Step 4 - Initialize T, the similarity threshold
    t =
