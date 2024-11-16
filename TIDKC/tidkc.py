import numpy as np
from scipy.spatial import distance_matrix
<<<<<<< HEAD
<<<<<<< HEAD
=======
import sys, os
>>>>>>> b4ffe4c (resolve merge conflicts)
from ..IDK import *
from .local_contrast import local_contrast
from matplotlib import pyplot as plt
<<<<<<< HEAD
# what is going on :(
=======
from scipy.stats import rankdata
from ref.idk import idk_kernel_map as K1, idk_square as K2
from .local_contrast import local_contrast
>>>>>>> 544b9c5 (Reorganize TIDKC)

# STEPS:

# initialise/inputs
# k = num clusters, determined by unique ground truth labels in the dataset
# s is the sample size for mode selection (???)


# 1: trajectory -> point mapping (RKHS, ruckus package?)
<<<<<<< HEAD
=======
from local_contrast import local_contrast
from find_mode import *

""" sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from IDK import * """
>>>>>>> b4ffe4c (resolve merge conflicts)


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
9:  until |N | = 0 or τ < 0.00001
10: Assign each unassigned point g to nearest cluster C via K2(δ(g), PC )
11: Cluster Ej ⊂ D corresponds to Cj ⊂ G, j = 1, . . . , k
"""


def tidkc(D: np.ndarray, k: int, idk:IDK):
    """Clustering using IDK
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    """
    # Constants ?
    kn = 70  # K neighbor threshhold
    t = 100  # number of estimator (used in IDK)
    psi = 8  # number of seeds (used in IDK)
    rho = 0.9  # growth rate

    ## Step 1 - Map each trajectory in RKHS using K1
    G = idk.idk(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    c_seeds = find_mode(G, k, kn)

    ## Step 3 - Initialize N, the difference between G and the set of all
    ## cluster seeds
    N = np.delete(G, c_seeds, axis=0)

    ## Step 4 - Initialize T, the similarity threshold
    # τ ← max g∈N, L∈[1,k] K2(δ(g), PCL)
    cluster_means = G[c_seeds]
    max_similarity = -np.inf
    
    for g in N:
        for l in range(k):
            similarity_score = np.dot(idk.k2(g.reshape(1, -1)), cluster_means[l])
            if similarity_score > max_similarity:
                max_similarity = similarity_score

    tau = max_similarity

<<<<<<< HEAD
    # is this part even necessary???????????
    # populate datapoints_dict + clusters_sum with seeds
    for i in range(k):
        # datapoints_dicts contains key:value
        # [cluster's number (index in c_seeds)] : [seed's value, aka coordinates]
        datapoints_dict[i] = G[c_seeds[i], :]
        clusters_sum[i, :] = np.sum(datapoints_dict[i][1:], axis=0)

    # "It is interesting to note that, independent of the algorithm
    # used to maximize the objective function, the similarity distribution 
    # derived from the kernel K can be used to describe the data
    # distribution of a dataset as follows:
    # Definition 2. Given cluster Cj, j = 1 . . . , k in a dataset D
    # and the distributional kernel K derived from D, the K˜ similarity
    # distribution is defined as:
    # K˜(x|D) = maxj K(δ(x), PCj), ∀x ∈ Rd"

    # we have to calculate K2(δ(g), PCL) for all points g and clusters CL?
    # "K2 is just the last four lines of IDK squared"?
    # where δ(g) = dirac measure of g (1 if g in CL, 0 if not)
=======
# 2: identify cluster centres / findmode

# count # features in dataset
# initialise cluster sums arr (csum) and cluster sizes array (csize)
# loop for assigning cluster seeds -- findmode must already be implemented here
>>>>>>> 544b9c5 (Reorganize TIDKC)


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

<<<<<<< HEAD
    T = np.argmax()
    tau = 0
=======
    return tau
>>>>>>> b4ffe4c (resolve merge conflicts)

    ## Steps 5 & 9 - begin loop, set conditions for ending loop:
    while(abs(N) != 0 and T >= 0.00001):
        ## Step 6 - update value of tau (τ)
        tau *= rho

        ## Step 7 - Expand cluster Cj to include unassigned point
            ## g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ

        
        ## Step 8 - update value of N:  N ← G \\ ∪j Cj
        N = np.delete(G, c_seeds, axis=0)
    
    ## Step 10 - Assign each unassigned point g to nearest cluster C
    ## via K2(δ(g), PC )
    while(len(G) > 0):
        x = 0 # placeholder, this means nothing


    ## Step 11 - Cluster Ej ⊂ D corresponds to Cj ⊂ G,j = 1,...,k
=======

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
    v = 0.9  # what is v ? :(

    ## Step 1 - Map each trajectory in RKHS using K1
    G = K1(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    G_dist = distance_matrix(G, G)
    # Multiply G by (the mean matrix of its columns shifted from line to column matrix)
    G_density = G @ np.mean(G, axis=0)[:, np.newaxis]  # bruh

    LC = local_contrast(G_dist, G_density, kn)
    # Select the k most dense points in LC (index)
    c_seeds = np.argsort(LC)[::-1][:k]
>>>>>>> 544b9c5 (Reorganize TIDKC)
