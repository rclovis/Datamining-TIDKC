# evaluating normalised mutual information

# NMI(Y,C) = (2*I(Y;C))/(H(Y) + H(C))
# where Y = class labels, C = cluster labels, H(x) = entropy, I(Y;C) = mutual info between Y and C
# essentially, evaluates how consistently predicted clusters match what the actual clusters should be (ground truth)

from sklearn.metrics import normalized_mutual_info_score
# option 1: simply use scikit-learn
# example:
# given true_labels (representing the actual cluster labels) and pred_labels (the labels predicted by tidkc)
# we can calculate NMI by doing:
# nmi = normalized_mutual_info_score(true_labels, pred_labels)


# option 2: develop own nmi function
# we'd probably have to do it in python anyways
# would rather focus on implementing other areas in a different language

# on how to present the info:
# let's just do a simple table like in the paper
# we can build that in latex/google docs in 2 seconds
