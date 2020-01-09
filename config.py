TEST_RATIO = 0.3
N_JOBS = 4
PCA_VAR_THR = 1  # 1 to disable


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self):
        self.experiment_params = {"evaluation_metric": ["balanced_accuracy"]}
        self.simulation_params = {"method": "none",
                                  "random_p": 0.5,  # labeled probability
                                  "class_p": [0.2, 0.8],  # min max labeled probabilities
                                  "feature_p": []}
        self.self_trainer_params = {"model_name": "svm",
                                    "model_params": {"mlp": {"hidden_layer_sizes": [(20,), (10, 4,)],
                                                             "activation": ["relu", "logistic", ],
                                                             "alpha": [1e-4, 1e-2, ],
                                                             "max_iter": [25, 100, 250, 500, ],
                                                             },
                                                     "svm": {"C": [10, 1, 1e-1, ],
                                                             "max_iter": [10, 100, 300, 1000, ]},
                                                     "tree": {"max_depth": [None, 5, 20, ],
                                                              "min_samples_split": [2, 4, 8, ],
                                                              "min_samples_leaf": [1, 3, 10], },
                                                     "forest": {"max_depth": [None, 5, 20, ],
                                                                "min_samples_split": [2, 4, 8, ],
                                                                "n_jobs": [4, ],
                                                                "n_estimators": [30, 100, ]},
                                                     },
                                    "num_splits": 4,
                                    "metric_key": "balanced_accuracy",
                                    "confidence_threshold": 0.5,
                                    "max_iter": 10,
                                    "standardize_flag": True
                                    }
        self.co_trainer_params = {"model_name": "svm",
                                  "model_params": {"mlp": {"hidden_layer_sizes": [(20,), (10, 4,)],
                                                             "activation": ["relu", "logistic", ],
                                                             "alpha": [1e-4, 1e-2, ],
                                                             "max_iter": [25, 100, 250, 500, ],
                                                             },
                                                     "svm": {"C": [10, 1, 1e-1, ],
                                                             "max_iter": [10, 100, 1000, 3000, ]},
                                                     "tree": {"max_depth": [None, 5, 20, ],
                                                              "min_samples_split": [2, 4, 8, ],
                                                              "min_samples_leaf": [1, 3, 10], },
                                                     "forest": {"max_depth": [None, 5, 20, ],
                                                                "min_samples_split": [2, 4, 8, ],
                                                                "n_jobs": [4, ],
                                                                "n_estimators": [30, 100, ]},
                                                     },
                                  "num_feature_splits": 3,
                                  "num_splits": 2,
                                  "metric_key": "balanced_accuracy",
                                  "confidence_threshold": 0.95,
                                  "max_iter": 10,
                                  "standardize_flag": True
                                  }
        self.s3vm_params = {}
        self.gmm_params = {"max_steps": 1000,
                           "stopping_epsilon": 1e-8,
                           "standardize_flag": True}
        self.label_networker_params = {"model_name": "spread",
                                       "model_params": {"prop": {"max_iter": [10, 100, 1000, ],
                                                                 "n_jobs": [4, ],
                                                                 "gamma": [0, 0.5, 1, 5, 10, 20, 40, ],
                                                                 "n_neighbors": [2, 4, 8, 16, ]},
                                                        "spread": {"max_iter": [10, 100, 1000, ],
                                                                   "n_jobs": [4, ],
                                                                   "gamma": [0, 0.5, 1, 5, 10, 20, 40, ],
                                                                   "n_neighbors": [2, 4, 8, 16, ]}},
                                       "num_splits": 4,
                                       "metric_key": "balanced_accuracy",
                                       "standardize_flag": True}
