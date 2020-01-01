import os
import sys
import warnings
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


class SelfTrainer(BaseEstimator, ClassifierMixin):
    model_dispatcher = {"mlp": MLPClassifier, "svm": LinearSVC, "tree": DecisionTreeClassifier,
                        "forest": RandomForestClassifier}

    score_func_dispatcher = {"mlp": "predict_proba", "svm": "decision_function", "tree": "predict_proba",
                             "forest": "predict_proba"}

    def __init__(self, params):
        self.model_name = params["model_name"]
        self.model_params = params["model_params"][self.model_name]
        self.num_splits = params["num_splits"]
        self.metric_key = params["metric_key"]
        self.confidence_threshold = params["confidence_threshold"]
        self.max_iter = params["max_iter"]
        self.standardize_flag = params['standardize_flag']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(whiten=True)

        self.estimator = None
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

        is_unlabeled = y == -1

        best_val_score = 0
        best_estimator = None
        for i in range(self.max_iter):
            print("--------------------")
            print(f"Iteration {i+1}/{self.max_iter}")
            print(f"Unlabeled data count: {is_unlabeled.sum()}")

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
                y[new_labeled_indices] = preds[is_labeled_from_unlabeled]
                is_unlabeled[new_labeled_indices] = 0

            val_score = cv.cv_results_["mean_test_score"].max()
            if val_score > best_val_score:
                best_val_score = val_score
                best_estimator = cv.best_estimator_

            print(f"Validation score: {val_score}")
            print(f"Best validation score: {best_val_score}")

            if is_unlabeled.sum() == 0:
                break

        self.estimator = best_estimator
        self.best_val_score = best_val_score

    def predict(self, x_input):

        x = x_input.copy()

        if self.standardize_flag:
            x = self.scaler.transform(x)

        if config.PCA_VAR_THR < 1:
            x = self.PCA.transform(x)

        return self.estimator.predict(x)

