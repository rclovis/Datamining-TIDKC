import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from find_mode import find_mode
from IDK import IDK
from utils.dataloader import load_and_preprocess_data


def to_label(C, D):
    """Convert cluster to label
    C : cluster of trajectories
    """
    labels = np.zeros(len(D))
    for i in range(len(C)):
        for j in C[i]:
            labels[j] = i
    return labels


def cluster_similarity(traj: np.ndarray, cluster: np.ndarray):
    """Compute similarity between a trajectory and a cluster
    traj : trajectory
    cluster : cluster of trajectories
    """
    cluster_centroid = np.average(cluster, axis=0)
    traj = traj.reshape(1, -1)
    cluster_centroid = cluster_centroid.reshape(1, -1)
    return cosine_similarity(traj, cluster_centroid)[0][0]


def extract_cluster(D, C) -> np.ndarray:
    """Extract cluster from dataset
    D : dataset of trajectories {T1..Ti}
    C : cluster of trajectories
    """
    return np.array([D[i] for i in C])


def seed_selection(D, k, kn):
    """Select k cluster seeds using local-constrast
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    kn : K neighbor threshhold
    """
    c_seeds = find_mode(D, k, kn)
    Cj = {}
    for i in range(k):
        Cj[i] = np.array([c_seeds[i]])
    return Cj, c_seeds


def tidkc(D: np.ndarray, k: int, rho: float = 0.8):
    """Clustering using IDK
    D : dataset of trajectories {T1..Ti}
    k : number of clusters to identify
    rho : user defined growth rate of tau
    """

    ## Constants
    kn = 70  # K neighbor threshhold
    t = 100  # number of estimator (used in IDK)
    psi = 8  # number of seeds (used in IDK)

    idk = IDK(random_seed=42)
    K2 = idk.idk_square(D, psi, psi, t, t)

    G = idk.idk(D, psi, t)  ## Step 1 - Generate G using IDK
    Cj, c_seeds = seed_selection(G, k, kn)  ## Step 2 - Select k cluster seeds

    ## Step 3 - Initialize N: N ← G \\ C
    N = np.array([i for i in range(len(G)) if i not in c_seeds])

    ## TODO: Step 4 - Initialize tau, the similarity threshold
    # τ ← max g∈N, L∈[1,k] K2(δ(g), PCL)
    tau = max(
        [
            max(
                [
                    cluster_similarity(K2[i], extract_cluster(K2, Cj[l]))
                    for l in range(k)
                ]
            )
            for i in N
        ]
    )

    print(tau)

    while len(N) != 0 and tau >= 0.00001:
        ## Step 6 - update value of tau (τ)
        tau *= rho

        ## TODO: Step 7 - Expand cluster Cj to include unassigned point
        ## g ∈ N for j = arg max∈[1,k] K2(δ(g), PC ) and K2(δ(g), PCj ) > τ
        newly_assigned_points = []

        for i in N:
            max_cluster: int = -1
            max_similarity: float = -1.0
            for j in range(k):
                similarity = cluster_similarity(K2[i], extract_cluster(K2, Cj[j]))
                if similarity > max_similarity:
                    max_cluster = j
                    max_similarity = similarity
            if max_similarity > tau:
                Cj[max_cluster] = np.append(Cj[max_cluster], i)
                newly_assigned_points = np.append(newly_assigned_points, i)

        ## Step 8 - update value of N:  N ← G \\ ∪j Cj
        N = np.setdiff1d(N, newly_assigned_points)

    ## TODO: Step 10 - Assign each unassigned point g to nearest cluster C
    ## via K2(δ(g), PC )
    for i in N:
        min_cluster: int = -1
        min_similarity: float = 1.0
        for j in range(k):
            similarity = cluster_similarity(K2[i], extract_cluster(K2, Cj[j]))
            if similarity < min_similarity:
                min_cluster = j
                min_similarity = similarity
        Cj[min_cluster] = np.append(Cj[min_cluster], i)

    return to_label(Cj, D)
