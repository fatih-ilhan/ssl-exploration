import os
import random
import numpy as np


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


def simulate_missing_labels_feature(x, y):
    return x, y


def simulate_missing_labels_pass(x, y):
    return x, y



