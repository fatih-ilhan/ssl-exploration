import os
import sys
import warnings
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import decomposition

import config

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class CoTrainer(BaseEstimator, ClassifierMixin):
    model_dispatcher = {"mlp": MLPClassifier, "svm": LinearSVC, "tree": DecisionTreeClassifier,
                        "forest": RandomForestClassifier}

    score_func_dispatcher = {"mlp": "predict_proba", "svm": "decision_function", "tree": "predict_proba",
                             "forest": "predict_proba"}

    def __init__(self, params):
        self.model_name = params["model_name"]
        self.model_params = params["model_params"][self.model_name]
        self.num_feature_splits = params["num_feature_splits"]
        self.num_splits = params["num_splits"]
        self.metric_key = params["metric_key"]
        self.confidence_threshold = params["confidence_threshold"]
        self.max_iter = params["max_iter"]
        self.standardize_flag = params['standardize_flag']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(whiten=True)

        self.feature_idx_list = [[]] * self.num_feature_splits
        self.estimator_list = [None] * self.num_feature_splits
        self.best_val_score_list = [0] * self.num_feature_splits
        self.best_val_score = 0

    def fit(self, x_input, y_input):

        x = x_input.copy()
        y = y_input.copy()

        # standardize input
        if self.standardize_flag:
            self.scaler.fit(x)
            x = self.scaler.transform(x)

        # apply PCA
        if config.PCA_VAR_THR < 1:
            if self.PCA.n_components is None:
                self.PCA.n_components = x.shape[1]
                self.PCA.fit(x)
                n_components = np.where(self.PCA.explained_variance_ratio_.cumsum() > config.PCA_VAR_THR)[0][0]
                self.PCA = decomposition.PCA(n_components=n_components, whiten=True)
            self.PCA.fit(x)
            x = self.PCA.transform(x)

        x_list = []
        y_list = []
        is_unlabeled_list = []
        feature_idx_list = []

        num_feature_per_split = int(np.ceil(x.shape[-1] * (1 / self.num_feature_splits + 0.5)))
        for i in range(self.num_feature_splits):
            feature_list = np.random.choice(np.arange(x.shape[-1]), num_feature_per_split)
            feature_idx_list.append(feature_list)
            x_list.append(x[:, feature_list])
            y_list.append(y)
            is_unlabeled_list.append(y == -1)

        best_val_score_list = [0] * self.num_feature_splits
        best_estimator_list = [None] * self.num_feature_splits
        for i in range(self.max_iter):
            print("--------------------")
            print(f"Iteration {i+1}/{self.max_iter}")
            print(f"Average unlabeled data count: {sum([is_unlabeled.sum() for is_unlabeled in is_unlabeled_list]) / self.num_feature_splits}")

            for j in range(self.num_feature_splits):
                x = x_list[j]
                y = y_list[j]
                is_unlabeled = is_unlabeled_list[j]
                cv = GridSearchCV(self.model_dispatcher[self.model_name](), self.model_params,
                                  scoring=self.metric_key,
                                  cv=self.num_splits,
                                  n_jobs=config.N_JOBS)
                cv.fit(x[~is_unlabeled], y[~is_unlabeled])

                if is_unlabeled.sum() > 0:
                    preds = cv.predict(x[is_unlabeled])
                    preds_scores = getattr(cv, self.score_func_dispatcher[self.model_name])(x[is_unlabeled])

                    if preds_scores.ndim == 1:
                        preds_scores = preds_scores[:, None]

                    is_labeled_from_unlabeled = np.max(preds_scores, axis=1) >= self.confidence_threshold
                    new_labeled_indices = np.where(is_unlabeled)[0][is_labeled_from_unlabeled]

                    for k in range(self.num_feature_splits):
                        if k == j:
                            continue
                        y_list[k][new_labeled_indices] = preds[is_labeled_from_unlabeled]
                        is_unlabeled_list[k][new_labeled_indices] = 0

                val_score = cv.cv_results_["mean_test_score"].max()
                if val_score > best_val_score_list[j]:
                    best_val_score_list[j] = val_score
                    best_estimator_list[j] = cv.best_estimator_

            print(f"Best average validation score: {np.mean(best_val_score_list)}")

        self.estimator_list = best_estimator_list
        self.best_val_score_list = best_val_score_list
        self.feature_idx_list = feature_idx_list
        self.best_val_score = np.mean(self.best_val_score_list)

    def predict(self, x_input):

        x = x_input.copy()

        if self.standardize_flag:
            x = self.scaler.transform(x)

        if config.PCA_VAR_THR < 1:
            x = self.PCA.transform(x)

        pred_list = []

        for i in range(self.num_feature_splits):
            pred = self.estimator_list[i].predict(x[:, self.feature_idx_list[i]])
            pred_list.append(pred)

        pred = np.array([Counter(col).most_common(1)[0][0] for col in zip(*np.array(pred_list))])

        return pred

