"""
It computes trajectory distance, including hausdoff distance, dtw distance, emd distance, gdk similarity.
p, q are two numpy.ndarray with shape (d, 2).
"""

import scipy.io as io
import numpy as np
from scipy import spatial
from dtaidistance import dtw, dtw_ndim
from scipy.stats import wasserstein_distance_nd
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler


def hausdorff_distance(p, q):
    haus_dist1, _, _ = spatial.distance.directed_hausdorff(p, q)
    haus_dist2, _, _ = spatial.distance.directed_hausdorff(q, p)
    return max(haus_dist1, haus_dist2)

def dtw_distance(p, q):
    dtw_dist = dtw_ndim.distance(p, q)
    return dtw_dist

def emd(p, q):
    emd_dist = wasserstein_distance_nd(p, q)
    return emd_dist

def gdk(p, q):
    '''
    # Basic empirical estimation
    k_g = np.sum(rbf_kernel(p, q)) / (p.shape[0] * q.shape[0])

    pq = np.vstack((p, q))

    # Nystroem approximation
    fmap_nystroem = Nystroem(gamma=1, n_components=10)
    fmap_nystroem.fit(pq)
    transformed_data_nystroem_p = fmap_nystroem.transform(p)
    transformed_data_nystroem_q = fmap_nystroem.transform(q)
    nystroem = np.dot(np.mean(transformed_data_nystroem_p, axis=0), np.mean(transformed_data_nystroem_q, axis=0))
    # print(nystroem)

    # Random Fourier Features approximation
    fmap_rff = RBFSampler(gamma=1, n_components=10)
    fmap_rff.fit(pq)
    transformed_data_rff_p = fmap_rff.transform(p)
    transformed_data_rff_q = fmap_rff.transform(q)
    rff = np.dot(np.mean(transformed_data_rff_p, axis=0), np.mean(transformed_data_rff_q, axis=0))
    # print(rff)
    '''

    # faster implementation, honestly I don't know why but it's faster
    gamma = 1.0 / p.shape[1]
    similarity_matrix = rbf_kernel(p, q, gamma=gamma)
    k_g = np.mean(similarity_matrix)

    return k_g


if __name__ == '__main__':
    raw_mat = io.loadmat("./TIDKC-main/datasets/geolife.mat")
    data = raw_mat['data'][0]
    p = data[0]
    q = data[1]
    dist = gdk(p, q)
    print(dist)
