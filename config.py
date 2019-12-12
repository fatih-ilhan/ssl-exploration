TEST_RATIO = 0.3


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self):
        self.experiment_params = {"evaluation_metric": ["accuracy", "balanced_accuracy", "balanced_f1", "confusion_matrix"]}
        self.pca_params = {}
        self.self_trainer_params = {"model_name": "mlp",
                                    "model_params": {"mlp": {"hidden_layer_sizes": [(100,)],
                                                             "activation": ["relu"]},
                                                     "svm": {"C": [1],
                                                             "kernel": ["rbf"],
                                                             "probability": [True]}
                                                     },
                                    "num_splits": 5,
                                    "metric_key": "balanced_accuracy",
                                    "confidence_threshold": 0.99,
                                    "PCA_dim": 0,
                                    "max_iter": 2,
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
