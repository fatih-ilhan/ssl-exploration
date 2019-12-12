import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


class SelfTrainer(BaseEstimator, ClassifierMixin):
    model_dispatcher = {"mlp": MLPClassifier, "svm": SVC}

    def __init__(self, params):
        self.model_name = params["model_name"]
        self.model_params = params["model_params"][self.model_name]
        self.num_splits = params["num_splits"]
        self.metric_key = params["metric_key"]
        self.confidence_threshold = params["confidence_threshold"]
        self.max_iter = params["max_iter"]
        self.PCA_dim = params['PCA_dim']
        self.standardize_flag = params['standardize_flag']

        self.classifier = GridSearchCV(self.model_dispatcher[self.model_name](), self.model_params,
                                       scoring=self.metric_key, cv=self.num_splits)
        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(n_components=self.PCA_dim, whiten=True)

    def fit(self, x_input, y_input):

        x = x_input.copy()
        y = y_input.copy()

        # standardize input
        if self.standardize_flag:
            self.scaler.fit(x)
            x = self.scaler.transform(x)

        # apply PCA
        if self.PCA_dim > 0:
            self.PCA.fit(x)
            x = self.PCA.transform(x)

            print("PCA explained variance_ratio:", self.PCA.explained_variance_ratio_.cumsum())

        is_unlabeled = y == -1

        best_val_score = 0
        best_model = self.classifier
        for iter in range(self.max_iter):
            print("--------------------")
            print(f"Iteration {iter+1}/{self.max_iter}")
            print(f"Unlabeled data count: {is_unlabeled.sum()}")

            self.classifier.fit(x[~is_unlabeled], y[~is_unlabeled])

            if is_unlabeled.sum() == 0:
                break

            preds = self.classifier.predict(x[is_unlabeled])
            preds_scores = self.classifier.predict_proba(x[is_unlabeled])

            is_labeled_from_unlabeled = np.max(preds_scores, axis=1) >= self.confidence_threshold
            new_labeled_indices = np.where(is_unlabeled)[0][is_labeled_from_unlabeled]
            y[new_labeled_indices] = preds[is_labeled_from_unlabeled]
            is_unlabeled[new_labeled_indices] = 0

            if is_unlabeled.sum() == 0:
                break

            val_score = self.classifier.cv_results_["mean_test_score"]
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = self.classifier

            print(f"Validation score: {val_score}")
            print(f"Best validation score: {best_val_score}")

        self.classifier = best_model

    def predict(self, x_input):

        x = x_input.copy()

        if self.standardize_flag:
            x = self.scaler.transform(x)

        if self.PCA_dim:
            x = self.PCA.transform(x)

        return self.classifier.predict(x)

