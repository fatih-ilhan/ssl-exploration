TEST_RATIO = 0.3
N_JOBS = 4


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self):
        self.experiment_params = {"evaluation_metric": ["balanced_accuracy"]}
        self.pca_params = {}
        self.self_trainer_params = {"model_name": "tree",
                                    "model_params": {"mlp": {"hidden_layer_sizes": [(20,), (10, 4,)],
                                                             "activation": ["relu", "logistic",],
                                                             "alpha": [1e-4, 1e-2,],
                                                             "max_iter": [25, 100, 250, 500,],
                                                             },
                                                     "svm": {"C": [10, 1, 1e-1],
                                                             "max_iter": [100, 1000]},
                                                     "tree": {"max_depth": [None, 5, 20],
                                                              "min_samples_split": [2, 4, 8],
                                                              "min_samples_leaf": [1, 3, 10]},
                                                     "forest": {"max_depth": [None, 5, 20],
                                                                "min_samples_split": [2, 4, 8],
                                                                "n_jobs": [4],
                                                                "n_estimators": [30, 100]},
                                                     },
                                    "num_splits": 4,
                                    "metric_key": "balanced_accuracy",
                                    "confidence_threshold": 0.9,
                                    "PCA_dim": 30,
                                    "max_iter": 10,
                                    "standardize_flag": True
                                    }
        self.co_trainer_params = {"model_name": "mlp",
                                  "model_params": {},
                                  "num_splits": 5}
        self.s3vm_params = {}
        self.gmm_params = {"max_steps": 1000,
                           "PCA_dim": 5,  # 0 to disable
                           "stopping_epsilon": 1e-8,
                           "standardize_flag": True}
