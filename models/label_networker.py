import os
import sys
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

import config

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
#     os.environ["PYTHONWARNINGS"] = "ignore"


class LabelNetworker(BaseEstimator, ClassifierMixin):

    model_dispatcher = {"prop": LabelPropagation, "spread": LabelSpreading}

    def __init__(self, params):
        self.model_name = params["model_name"]
        self.model_params = params["model_params"][self.model_name]
        self.num_splits = params["num_splits"]
        self.metric_key = params["metric_key"]
        self.PCA_dim = params['PCA_dim']
        self.standardize_flag = params['standardize_flag']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(n_components=self.PCA_dim, whiten=True)

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
        if self.PCA_dim > 0:
            self.PCA.fit(x)
            x = self.PCA.transform(x)

            print("PCA explained variance_ratio:", self.PCA.explained_variance_ratio_.cumsum())

        cv = GridSearchCV(LabelPropagation(),
                          self.model_params,
                          scoring=self.metric_key,
                          cv=self.num_splits,
                          n_jobs=config.N_JOBS)

        cv.fit(x, y)

        self.estimator = cv.best_estimator_
        self.best_val_score = cv.cv_results_["mean_test_score"].max()

    def predict(self, x_input):

        x = x_input.copy()

        if self.standardize_flag:
            x = self.scaler.transform(x)

        if self.PCA_dim:
            x = self.PCA.transform(x)

        return self.estimator.predict(x)
