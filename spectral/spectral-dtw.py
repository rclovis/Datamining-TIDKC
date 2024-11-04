# BUGGY!!!!!!!
from sklearn.cluster import SpectralClustering
import os, sys
import tslearn.metrics as metrics

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from utils.eval_clusters import nmi_score, ari_score
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory


# Load the dataset -- can be CASIA, cross, geolife, pedes3, pedes4, TRAFFIC, or cyclists
args = sys.argv
if len(args) < 2:
    raise IndexError("You forgot to specify a dataset!")

dataset = sys.argv[1]
data, labels = load_and_preprocess_data(dataset)

num_clusters = len(set(labels))

# Calculate DTW distance matrix
dists = metrics.cdist_dtw(data)

# Fit and predict the spectral model with the distance matrix

spectrals = SpectralClustering(n_clusters=num_clusters, random_state=0, affinity='nearest_neighbors')

spectrals.fit_predict(dists)

# Print the labels
print(spectrals.labels_)
print("ARI: ", ari_score(labels, spectrals.labels_))
print("NMI: ", nmi_score(labels, spectrals.labels_))

visualize_trajectory(data, spectrals.labels_)

