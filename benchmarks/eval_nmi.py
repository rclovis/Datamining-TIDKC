# functionality for evaluating normalised mutual information
# NMI(Y,C) = (2*I(Y;C))/(H(Y) + H(C))
# where Y = class labels, C = cluster labels, H(x) = entropy, I(Y;C) = mutual info between Y and C
# essentially, evaluates how consistently predicted clusters match what the actual clusters should be (ground truth)

from sklearn.metrics import normalized_mutual_info_score

# params: true_labels (representing the actual cluster labels) and pred_labels (the labels predicted by tidkc)
# datatypes of both params must be 'array-like', i.e. numpy arrays, lists, pandas dataframes, etc.
# should return score between 0.0 (bad) and 1.0 (good)

def nmi_score(true_labels, pred_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return nmi

