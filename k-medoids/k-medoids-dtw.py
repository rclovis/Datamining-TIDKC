from sklearn_extra.cluster import KMedoids
import sys
import tslearn.metrics as metrics
from eval_clusters import nmi_score, ari_score
from utils.dataloader import load_and_preprocess_data
from utils.visualizer import visualize_trajectory

# Load the dataset -- can be CASIA, cross, geolife, pedes3, pedes4, TRAFFIC, or cyclists

args = sys.argv
if len(args) < 2:
    raise IndexError("You forgot to specify a dataset!")

dataset = sys.argv[1]
data, labels = load_and_preprocess_data(dataset)


# Calculate DTW distance matrix
dists = metrics.cdist_dtw(data)

kmedoids = KMedoids(n_clusters=5, random_state=0, metric='precomputed')

# Fit and predict the KMedoids model with the distance matrix
kmedoids.fit_predict(dists)

# Print the labels
print(kmedoids.labels_)
print("ARI: ", ari_score(labels, kmedoids.labels_))
print("NMI: ", nmi_score(labels, kmedoids.labels_))

visualize_trajectory(data, kmedoids.labels_)

