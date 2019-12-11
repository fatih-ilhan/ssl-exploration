import argparse
import pickle
import json

from config import Config
from models.gmm import GMM
from models.self_trainer import SelfTrainer
from models.s3vm import S3VM
from utils.data_utils import prepare_data
from utils.evaluate_utils import evaluate, merge_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, nargs='+')  # "train", "test"
    parser.add_argument('--model_name', type=str)  # "self_trainer", "s3vm", "gmm"
    parser.add_argument('--model_path', type=str, default=None)  # path/to/model.pkl
    parser.add_argument('--dataset_name', type=int)  # string
    parser.add_argument('--num_repeat', type=int)  # repeat train + test

    args = parser.parse_args()

    config_obj = Config()
    model_dispatcher = {"self_trainer": SelfTrainer,
                        "s3vm": S3VM,
                        "gmm": GMM}
    params_dispatcher = {"self_trainer": config_obj.self_trainer_params,
                         "s3vm": config_obj.s3vm_params,
                         "gmm": config_obj.gmm_params}

    data_dict = prepare_data(args.dataset_name)
    x_train, y_train, x_test, y_test = data_dict.values()

    if args.model_path is None:
        model_params = params_dispatcher[args.model_name]
        model = model_dispatcher[args.model_name](model_params)
    else:
        with open(args.model_path, "rb") as f:
            model = pickle.load(f)

    train_results_list = []
    test_results_list = []

    for rep in range(args.num_repeat):
        if "train" in args.mode:
            model.fit(x_train, y_train)
            train_preds = model.predict(x_train)
            train_results = evaluate(train_preds, y_train, config_obj.experiment_params["evaluation_metric"])
            train_results_list.append(train_results)

        if "test" in args.mode:
            test_preds = model.predict(x_test)
            test_results = evaluate(test_preds, y_test, config_obj.experiment_params["evaluation_metric"])
            test_results_list.append(test_results)

    average_train_results_mean = merge_results(train_results_list, "mean")
    average_train_results_std = merge_results(train_results_list, "std")

    average_test_results_mean = merge_results(test_results_list, "mean")
    average_test_results_std = merge_results(test_results_list, "std")

    print("Train results (mean):", json.dumps(average_train_results_mean, indent=4))
    print("Train results (std):", json.dumps(average_train_results_std, indent=4))
    print("Test results (mean):", json.dumps(average_test_results_mean, indent=4))
    print("Test results (std):", json.dumps(average_test_results_std, indent=4))
