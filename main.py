import argparse
import pickle
import json
import os

from config import Config
from models.gmm import GMM
from models.self_trainer import SelfTrainer
from models.s3vm import S3VM
from models.label_networker import LabelNetworker
from utils.data_utils import prepare_data
from utils.evaluation_utils import evaluate, merge_results
from utils.simulation_utils import simulate_missing_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, nargs='+')  # "train", "test"
    parser.add_argument('--model_name', type=str)  # "self_trainer", "s3vm", "gmm"
    parser.add_argument('--dataset_list', type=str, nargs='+')  # string list
    parser.add_argument('--num_repeat', type=int, default=1)  # repeat train + test
    parser.add_argument('--save', type=bool, default=1)  # save results flag
    parser.add_argument('--load', type=bool, default=0)  # load previous pkl

    args = parser.parse_args()

    config_obj = Config()
    model_dispatcher = {"self_trainer": SelfTrainer,
                        "s3vm": S3VM,
                        "gmm": GMM,
                        "label_networker": LabelNetworker}
    params_dispatcher = {"self_trainer": config_obj.self_trainer_params,
                         "s3vm": config_obj.s3vm_params,
                         "gmm": config_obj.gmm_params,
                         "label_networker": config_obj.label_networker_params}

    for dataset_name in args.dataset_list:

        model_path = "output", "models", args.model_name + "_" + dataset_name + ".pkl"

        train_results_list = []
        val_results_list = []
        test_results_list = []

        data_dict = prepare_data(dataset_name)

        for rep in range(args.num_repeat):
            print("********************")
            print(f"Dataset: {dataset_name}, Repeat index: {rep+1}")

            data_split = data_dict.values()
            x_train, y_train, x_test, y_test = data_split
            x_train_simulated, y_train_simulated = \
                simulate_missing_labels(x_train, y_train, config_obj.simulation_params)

            if ~args.load:
                model_params = params_dispatcher[args.model_name]
                model = model_dispatcher[args.model_name](model_params)
            else:
                with open(args.model_path, "rb") as f:
                    model = pickle.load(f)

            if "train" in args.mode:
                model.fit(x_train_simulated, y_train_simulated)
                train_preds = model.predict(x_train[y_train != -1])
                train_results = evaluate(train_preds, y_train[y_train != -1], config_obj.experiment_params["evaluation_metric"])
                train_results_list.append(train_results)
                val_results_list.append({"balanced_accuracy": model.best_val_score})

            if "test" in args.mode:
                test_preds = model.predict(x_test[y_test != -1])
                test_results = evaluate(test_preds, y_test[y_test != -1], config_obj.experiment_params["evaluation_metric"])
                test_results_list.append(test_results)

        average_train_results_mean = merge_results(train_results_list, "mean")
        average_train_results_std = merge_results(train_results_list, "std")
        
        average_val_results_mean = merge_results(val_results_list, "mean")
        average_val_results_std = merge_results(val_results_list, "std")

        average_test_results_mean = merge_results(test_results_list, "mean")
        average_test_results_std = merge_results(test_results_list, "std")

        print("Train results (mean):", json.dumps(average_train_results_mean, indent=4))
        print("Train results (std):", json.dumps(average_train_results_std, indent=4))
        print("Validation results (mean):", json.dumps(average_val_results_mean, indent=4))
        print("Validation results (std):", json.dumps(average_val_results_std, indent=4))
        print("Test results (mean):", json.dumps(average_test_results_mean, indent=4))
        print("Test results (std):", json.dumps(average_test_results_std, indent=4))

        all_results = {}
        all_results["average_train_results_mean"] = average_train_results_mean
        all_results["average_train_results_std"] = average_train_results_std
        all_results["average_val_results_mean"] = average_val_results_mean
        all_results["average_val_results_std"] = average_val_results_std
        all_results["average_test_results_mean"] = average_test_results_mean
        all_results["average_test_results_std"] = average_test_results_std

        if args.save:
            pickle.dump(all_results, open(os.path.join("output", "results",
                                                       args.model_name + "_" + dataset_name + ".pkl"), "wb"))
            pickle.dump(model, open(os.path.join("output", "models",
                                                 args.model_name + "_" + dataset_name + ".pkl"), "wb"))
