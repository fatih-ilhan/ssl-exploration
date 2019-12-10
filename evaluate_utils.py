import sklearn.metrics as skmetrics


def evaluate(x, y, metric_list):
    result_dict = {}

    for metric in metric_list:
        if metric == "confusion_matrix":
            result = skmetrics.confusion_matrix(y[y != -1], x[y != -1])
            result_dict[metric] = result
        elif metric == "accuracy":
            result = skmetrics.accuracy_score(y[y != -1], x[y != -1])
            result_dict[metric] = result
        elif metric == "balanced_accuracy":
            result = skmetrics.balanced_accuracy_score(y[y != -1], x[y != -1])
            result_dict[metric] = result
        elif metric == "f1":
            result = skmetrics.f1_score(y[y != -1], x[y != -1])
            result_dict[metric] = result

    return result_dict

