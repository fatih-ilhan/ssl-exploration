import argparse
import pickle

from config import Config
from models.gmm import GMM
from models.mlp import MLP
from models.s3vm import S3VM
from utils import prepare_data, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, nargs='+')  # "train", "test"
    parser.add_argument('model_name', type=str)  # "mlp", "s3vm", "gmm"
    parser.add_argument('model_path', type=str, default=None)  # path/to/model.pkl
    parser.add_argument('data_id', type=int)  # 1, 2 ...

    args = parser.parse_args()

    config_obj = Config()
    model_dispatcher = {"mlp": MLP,
                        "s3vm": S3VM,
                        "gmm": GMM}
    params_dispatcher = {"mlp": config_obj.mlp_params,
                       "s3vm": config_obj.s3vm_params,
                       "gmm": config_obj.gmm_params}

    data_dict = prepare_data(args.data_id)
    x_train, y_train, x_test, y_test = data_dict.values()

    if args.model_path is not None:
        model_params = params_dispatcher[args.model_name]
        model = model_dispatcher[args.model_name](model_params)
    else:
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)

    if "train" in args.mode:
        model.fit(x_train, y_train)
        train_preds = model.predict(x_train)
        train_results = evaluate(train_preds, y_train, config_obj.experiment_params["evaluation_metric"])
        print(train_results)

    if "test" in args.mode:
        test_preds = model.predict(x_test)
        test_results = evaluate(test_preds, y_test, config_obj.experiment_params["evaluation_metric"])
        print(test_results)
