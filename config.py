DATA_DISPATCHER = {1: "arrhythmia",
                   2: "cardiotocography",
                   3: "smartphone_activity",
                   4: "gastrointestinal"}
TEST_RATIO = 0.3


class Config:
    """
    This object contains manually given parameters
    """
    def __init__(self):
        self.experiment_params = {"evaluation_metric": "accuracy"}
        self.pca_params = {}
        self.mlp_params = {}
        self.s3vm_params = {}
        self.gmm_params = {}