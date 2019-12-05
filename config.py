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