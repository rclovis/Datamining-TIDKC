# functionality for evaluating adjusted rand index
# rand index: "computes similarity measure between two clusterings by considering all pairs of samples
# and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings"

# ARI = (RI - expected RI) / (max(RI) - expected RI)

# should normally be between 0 and 1, but can be as low as -0.5
# note: ari(a, b) == ari(b, a)

from sklearn.metrics import adjusted_rand_score

# params: true_labels (representing the actual cluster labels) and pred_labels (the labels predicted by tidkc)
# datatypes of both params must be 'array-like', i.e. numpy arrays, lists, pandas dataframes, etc.
# should return score between -0.5 (literally worse than random) and 1.0 (perfect)

def ari_score(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ari
