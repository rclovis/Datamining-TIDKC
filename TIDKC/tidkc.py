import numpy as np
from scipy.spatial import distance_matrix
from ..IDK import *
from .local_contrast import local_contrast
from .find_mode import find_mode
from matplotlib import pyplot as plt
# what is going on :(

# STEPS:

# initialise/inputs
# k = num clusters, determined by unique ground truth labels in the dataset
# s is the sample size for mode selection (???)


# 1: trajectory -> point mapping (RKHS, ruckus package?)


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


def tidkc(D: np.ndarray, k: int):
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
    idk = IDK(random_seed=42)
    G = idk.idk(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    c_seeds = find_mode(G, k, kn)

    ## Step 3 - Initialize N, the difference between G and the set of all
    ## cluster seeds
    N = np.delete(G, c_seeds, axis=0)

    ## Step 4 - Initialize T, the similarity threshold
    # τ ← max g∈N, L∈[1,k] K2(δ(g), PCL)
    num_features = G.shape[1]

    clusters_sum = np.zeros((k, num_features)) # sum of the datapoints per cluster
    datapoints_dict = {} # actual datapoints stored per cluster

    # populate datapoints_dict + clusters_sum with seeds
    for i in range(k):
        datapoints_dict[i] = G[c_seeds[i], :]
        clusters_sum[i, :] = np.sum(datapoints_dict[i][1:], axis=0)

    # we have to calculate K2(δ(g), PCL) for all points g and clusters CL?
    # where δ(g) = dirac measure of g (1 if g in CL, 0 if not)



    T = np.argmax()
    tau = 0

    ## Steps 5 & 9 - begin loop, set conditions for ending loop:
    while(abs(N) != 0 and T >= 0.00001):
        ## Step 6 - update value of tau (τ)
        tau = rho * tau

        ## Step 7 - Expand cluster Cj to include unassigned point
            ## g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ

        
        ## Step 8 - update value of N:  N ← G \\ ∪j Cj
        N = np.delete(G, c_seeds, axis=0)
    
    ## Step 10 - Assign each unassigned point g to nearest cluster C
    ## via K2(δ(g), PC )
    while(len(G) > 0):
        x = 0 # placeholder, this means nothing


    ## Step 11 - Cluster Ej ⊂ D corresponds to Cj ⊂ G,j = 1,...,k
