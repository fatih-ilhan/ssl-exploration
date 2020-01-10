import os
import random
import numpy as np
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import config


def simulate_missing_labels(x, y, params):
    """
    Simulates missing label patterns over given data and label
    """
    simulate_func_dispatcher = {"random": simulate_missing_labels_random,
                                "class": simulate_missing_labels_class,
                                "feature": simulate_missing_labels_feature,
                                "none": simulate_missing_labels_pass}

    method_type = params["method"]
    method_kwargs = {key: params[key] for key in params.keys() if method_type in key}

    return simulate_func_dispatcher[method_type](x, y, **method_kwargs)


def simulate_missing_labels_random(x, y, random_p):
    num_data = x.shape[0]
    idxs_to_miss = random.sample(list(range(num_data)), int(num_data * (1 - random_p)))
    y[idxs_to_miss] = -1
    return x, y


def simulate_missing_labels_class(x, y, class_p):
    num_classes = y.max() + 1
    for i in range(num_classes):
        p = np.random.uniform(class_p[0], class_p[1])
        num_data = (y == i).sum()
        idxs_to_miss = random.sample(np.where(y == i)[0].tolist(), int(num_data * (1 - p)))
        y[idxs_to_miss] = -1
    return x, y


def simulate_missing_labels_feature(x, y, feature_p):
    x_ = x.copy()
    y_ = y.copy()

    PCA = decomposition.PCA(whiten=True)
    PCA.n_components = x.shape[1]
    PCA.fit(x)

    n_components = np.where(PCA.explained_variance_ratio_.cumsum() > 0.95)[0][0]
    PCA = decomposition.PCA(n_components=n_components, whiten=True)

    PCA.fit(x)
    x = PCA.transform(x)

    num_classes = y.max() + 1
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(x)
    centers = kmeans.cluster_centers_

    diff = pairwise_distances(x, centers)
    a = np.min(diff, axis=1)
    b = np.sort(diff, axis=1)[:, 1]
    diff = (b - a) / b

    idx_sorted = np.argsort(diff)
    num_data_to_miss = int(x.shape[0] * (1 - feature_p))
    idx_to_miss = idx_sorted[:num_data_to_miss]

    y[idx_to_miss] = -1

    return x_, y


def simulate_missing_labels_pass(x, y):
    return x, y



