import numpy as np

from find_mode import find_mode
from IDK import IDK
from utils.dataloader import load_and_preprocess_data


def tidkc(D: np.ndarray, k: int):
    """Clustering using IDK
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    """
    kn = 70  # K neighbor threshhold
    t = 100  # number of estimator (used in IDK)
    psi = 8  # number of seeds (used in IDK)
    rho = 0.9  # growth rate

    idk = IDK(random_seed=42)

    ## Step 1 - Map each trajectory in RKHS using K1
    G = idk.idk(D, psi, t)

    ## Step 2 - Select k cluster seeds using local-constrast
    ## and initialise cluster array Cj of length k
    c_seeds = find_mode(G, k, kn)
    Cj = np.empty(k, dtype=np.ndarray)
    for i in range(k):
        Cj[i] = []
        Cj[i].append(c_seeds[i])

    ## Step 3 - Initialize N, the difference between G and the set of all
    ## cluster seeds
    N = np.delete(G, c_seeds, axis=0)

    ## TODO: Step 4 - Initialize tau, the similarity threshold
    # τ ← max g∈N, L∈[1,k] K2(δ(g), PCL)
    tau = 0

    while abs(N) != 0 and tau >= 0.00001:
        ## Step 6 - update value of tau (τ)
        tau *= rho

        ## TODO: Step 7 - Expand cluster Cj to include unassigned point
        ## g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ
        newly_assigned_points = []

        ## Step 8 - update value of N:  N ← G \\ ∪j Cj
        N = np.array([g for g in N if g not in newly_assigned_points])

    ## TODO: Step 10 - Assign each unassigned point g to nearest cluster C
    ## via K2(δ(g), PC )
    while len(G) > 0:
        pass

    ## Step 11 - Cluster Ej ⊂ D corresponds to Cj ⊂ G,j = 1,...,k
    r = []
    for c in Cj:
        r.append(list(map(lambda x: D[x], c)))
    return r


print("started main")
idk = IDK(random_seed=42)
print("created idk")
data, labels = load_and_preprocess_data("TRAFFIC")
print("created dataset")
k = 11
N = tidkc(data, 11, idk)
print(N)
