import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from IDK import IDK
import random

def seeds_selection(K, D_indices, k, q):
    # K: Kernel matrix of the subset D'
    # D_indices: Indices of the data points in D'
    # k: Number of clusters
    # q: Number of nearest neighbors

    n_samples = K.shape[0]
    local_contrast = np.zeros(n_samples)
    for i in range(n_samples):
        # Similarities to all other points
        similarities = K[i]
        # Exclude self and find q nearest neighbors
        nearest_indices = np.argsort(-similarities)[1:q+1]
        # Compute local contrast
        local_contrast[i] = np.sum(similarities[i] - similarities[nearest_indices])

    # Select seeds with highest local contrast
    seed_indices_local = np.argsort(-local_contrast)[:k]
    seed_indices_global = [D_indices[idx] for idx in seed_indices_local]
    return seed_indices_global

def tidkc(D, psi=16, t=500, k=3, q=5, s=1000, rho=0.9):
    # Step 1: Create IDK
    idk = IDK(psi=psi, estimators=t)
    # Map trajectories to feature space using IDK
    D_mapped = idk.create_idk(D)
    # Normalize the mapped data
    D_mapped = normalize(D_mapped)

    n_samples = D_mapped.shape[0]
    all_indices = np.arange(n_samples)

    # Step 2: Randomly select s data points from D for seed selection
    s = min(s, n_samples)  # Ensure s does not exceed dataset size
    #set random seed
    random.seed(42)
    sample_indices = random.sample(list(all_indices), s)
    D_subset = D_mapped[sample_indices]

    # Step 3: Select initial seeds
    K_subset = cosine_similarity(D_subset)
    seed_indices = seeds_selection(K_subset, sample_indices, k, q)

    # Initialize clusters with seeds
    clusters = {i: [seed_indices[i]] for i in range(k)}
    N = set(all_indices) - set(seed_indices)

    # Step 4: Initialize similarity threshold τ
    K_full = cosine_similarity(D_mapped)
    tau = max([K_full[i, seed_indices].max() for i in N])

    # Step 5: Cluster growing
    while tau > 1e-5 and N:
        tau *= rho  # Update similarity threshold
        N_new = set()
        for i in N:
            # Compute similarities to cluster seeds
            similarities = [K_full[i, seed] for seed in seed_indices]
            max_similarity = max(similarities)
            cluster_assignment = np.argmax(similarities)
            if max_similarity > tau:
                clusters[cluster_assignment].append(i)
            else:
                N_new.add(i)
        if N == N_new:
            break  # No changes in N, terminate
        N = N_new

    # Step 6: Assign remaining points
    for i in N:
        similarities = [K_full[i, seed] for seed in seed_indices]
        cluster_assignment = np.argmax(similarities)
        clusters[cluster_assignment].append(i)

    # Step 7: (Optional) Refinement step - Not specified in detail
    # Skipping refinement due to lack of specifics

    # Prepare labels
    labels = np.zeros(n_samples, dtype=int)
    for cluster_idx, indices in clusters.items():
        labels[indices] = cluster_idx

    return labels
