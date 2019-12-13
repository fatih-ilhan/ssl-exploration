import os
import random
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.io import loadmat
import pickle

import config

"""
Includes utilization functions
"""


def prepare_data(dataset_name, overwrite_flag=False):
    """
    :param dataset_name: str
    :param overwrite_flag: bool
    :return: dict of x_train (n_train x d), y_train (n_train), x_test (n_test x d), y_test (n_test)
    """
    data_path = os.path.join("data", dataset_name + ".pkl")

    if os.path.isfile(data_path) and not overwrite_flag:
        data_dict = pickle.load(open(data_path, "rb"))
        return data_dict

    helper_dispatcher = {"arrhythmia": prepare_data_arrhythmia,  #
                         "cardio": prepare_data_cardio,
                         "smartphone": prepare_data_smartphone,
                         "gastro": prepare_data_gastro,
                         "breast": prepare_data_breast,
                         "credit": prepare_data_credit,  #
                         "letter": prepare_data_letter,
                         "mushroom": prepare_data_mushroom,  #
                         "usps": prepare_data_ssl_book,
                         "coil": prepare_data_ssl_book,
                         "bci": prepare_data_ssl_book,
                         "sec_str": prepare_data_ssl_book}

    args = []
    if dataset_name in ["usps", "coil", "bci", "sec_str"]:
        args.append(dataset_name)

    x_train, y_train, x_test, y_test = helper_dispatcher[dataset_name](*args)
    data_dict = {"x_train": x_train,
                 "y_train": y_train,
                 "x_test": x_test,
                 "y_test": y_test}

    pickle.dump(data_dict, open(data_path, "wb"))

    return data_dict


def prepare_data_arrhythmia():
    raise NotImplementedError

    # data_path = os.path.join("data_raw", "fully_labeled", "arrhythmia", "arrhythmia.data")
    # df = pd.read_csv(data_path, header=None)
    # df = df.replace("?", np.nan)
    # df = df.astype(float)
    #
    # test_mask = np.random.rand(len(df)) < config.TEST_RATIO
    # train_df = df[~test_mask]
    # test_df = df[test_mask]
    #
    # # Impute missing values
    # imp = IterativeImputer(max_iter=10, random_state=0)
    # imp.fit(train_df)
    #
    # train_np = imp.transform(train_df)
    # test_np = imp.transform(test_df)
    #
    # x_train = train_np[:, :-1]
    # y_train = (train_np[:, -1] - 1).astype(int)
    # x_test = test_np[:, :-1]
    # y_test = (test_np[:, -1] - 1).astype(int)
    #
    # return x_train, y_train, x_test, y_test


def prepare_data_cardio():
    data_path = os.path.join("data_raw", "partially_labeled", "cardiotocography", "CTG.xls")
    df = pd.read_excel(data_path, header=None, sheet_name="Data")
    x_df = df.iloc[2:-3, 10: 31].astype(float)
    y_df = df.iloc[2:-3:, -1].astype(int)
    y_df[y_df == 1] = 0
    y_df[y_df == 2] = -1
    y_df[y_df == 3] = 1

    test_mask = np.zeros(y_df.shape[0], dtype=bool)
    labeled_idx = np.argwhere(y_df >= 0)[:, 0]
    test_idx = random.sample(list(labeled_idx), int(len(labeled_idx) * config.TEST_RATIO))
    test_mask[test_idx] = True

    x_train = np.array(x_df[~test_mask])
    y_train = np.array(y_df[~test_mask])
    x_test = np.array(x_df[test_mask])
    y_test = np.array(y_df[test_mask])

    return x_train, y_train, x_test, y_test


def prepare_data_smartphone():
    x_train_data_path = os.path.join("data_raw", "fully_labeled", "smartphone_activity", "Train", "X_train.txt")
    x_train_df = pd.read_csv(x_train_data_path, header=None, sep=" ")
    x_train = np.array(x_train_df)

    y_train_data_path = os.path.join("data_raw", "fully_labeled", "smartphone_activity", "Train", "y_train.txt")
    y_train_df = pd.read_csv(y_train_data_path, header=None, sep=" ")
    y_train = np.array(y_train_df)[:, 0] - 1

    x_test_data_path = os.path.join("data_raw", "fully_labeled", "smartphone_activity", "Test", "X_test.txt")
    x_test_df = pd.read_csv(x_test_data_path, header=None, sep=" ")
    x_test = np.array(x_test_df)

    y_test_data_path = os.path.join("data_raw", "fully_labeled", "smartphone_activity", "Test", "y_test.txt")
    y_test_df = pd.read_csv(y_test_data_path, header=None, sep=" ")
    y_test = np.array(y_test_df)[:, 0] - 1

    return x_train, y_train, x_test, y_test


def prepare_data_gastro():
    x_data_path = os.path.join("data_raw", "partially_labeled", "gastrointestinal", "data.txt")
    x_df = pd.read_csv(x_data_path)
    x_df = x_df.iloc[2:]
    x = np.array(x_df).T

    y_data_path = os.path.join("data_raw", "partially_labeled", "gastrointestinal", "HumanEvaluation.xlsx")
    y_df = pd.read_excel(y_data_path, header=2)

    y_df = y_df[["GROUND TRUTH", "EXPERT 1", "EXPERT 2", "EXPERT 3", "EXPERT 4", "BEGINNER 1", "BEGINNER 2", "BEGINNER 3"]]
    y = np.array(y_df.iloc[:76])
    y[y == "serrated"] = 0  # malignant
    y[y == "adenoma"] = 1  # malignant
    y[y == "hyperplasic"] = 2  # benign
    y = y.astype(int)

    is_labeled_array = np.zeros(y.shape[0], dtype=bool)
    for i in range(y.shape[0]):
        if max(np.unique(y[i, 1:], return_counts=True)[1]) >= 5:
            is_labeled_array[i] = True

    y = y[:, 0]
    y[~is_labeled_array] = -1  # low confidence labels
    y = np.repeat(y, 2)  # there are two realizations of each sample

    test_mask = np.zeros(y.shape[0], dtype=bool)
    labeled_idx = np.argwhere(y >= 0)[:, 0]
    test_idx = random.sample(list(labeled_idx), int(len(labeled_idx) * config.TEST_RATIO))
    test_mask[test_idx] = True

    x_train = x[~test_mask]
    y_train = y[~test_mask]

    x_test = x[test_mask]
    y_test = y[test_mask]

    return x_train, y_train, x_test, y_test


def prepare_data_breast():
    data_path = os.path.join("data_raw", "fully_labeled", "breast_w", "breast_w.csv")
    df = pd.read_csv(data_path)
    df = df.iloc[:, 1:]
    df = df.replace("?", np.nan)
    df = df.replace("benign", 0)
    df = df.replace("malignant", 1)
    df = df.astype(float)

    test_mask = np.random.rand(len(df)) < config.TEST_RATIO
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # Impute missing values
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(train_df)

    train_np = imp.transform(train_df)
    test_np = imp.transform(test_df)

    x_train = train_np[:, :-1].astype(float)
    y_train = train_np[:, -1].astype(int)

    x_test = test_np[:, :-1].astype(float)
    y_test = test_np[:, -1].astype(int)

    return x_train, y_train, x_test, y_test


def prepare_data_credit():
    raise NotImplementedError


def prepare_data_letter():
    data_path = os.path.join("data_raw", "fully_labeled", "letter_recognition", "letter_recognition.csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns="id")
    df = df.replace(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), list(range(26)))
    df = df.astype(float)

    test_mask = np.random.rand(len(df)) < config.TEST_RATIO
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # Impute missing values
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(train_df)

    train_np = imp.transform(train_df)
    test_np = imp.transform(test_df)

    x_train = train_np[:, :-1].astype(float)
    y_train = train_np[:, -1].astype(int)

    x_test = test_np[:, :-1].astype(float)
    y_test = test_np[:, -1].astype(int)

    return x_train, y_train, x_test, y_test


def prepare_data_mushroom():
    data_path = os.path.join("data_raw", "fully_labeled", "mushroom", "mushroom.csv")
    df = pd.read_csv(data_path)
    raise NotImplementedError


def prepare_data_ssl_book(sub_data_name):
    data_path = os.path.join("data_raw", "partially_labeled", sub_data_name, "data.mat")

    if sub_data_name != "sec_str":
        idxs_path = os.path.join("data_raw", "partially_labeled", sub_data_name, "idxs_100.mat")
    else:
        idxs_path = os.path.join("data_raw", "partially_labeled", sub_data_name, "idxs_10000.mat")

    data = loadmat(data_path)
    idxs = loadmat(idxs_path)

    if sub_data_name != "sec_str":
        x = data["X"]
        y = data["y"][:, 0].astype(int)
        y[y == -1] = 0
    else:
        x = data["T"]
        x = (np.arange(x.max() + 1) == x[..., None]).astype(int)
        x = x.reshape(x.shape[0], -1)
        y = data["y"][:, 0].astype(int)

    is_labeled_idxs = idxs["idxLabs"][0]

    is_labeled_array = np.zeros(y.shape[0], dtype=bool)
    is_labeled_array[is_labeled_idxs] = True

    y[~is_labeled_array] = -1  # low confidence labels

    test_mask = np.zeros(y.shape[0], dtype=bool)
    labeled_idx = np.argwhere(y >= 0)[:, 0]
    test_idx = random.sample(list(labeled_idx), int(len(labeled_idx) * config.TEST_RATIO))
    test_mask[test_idx] = True

    x_train = x[~test_mask]
    y_train = y[~test_mask]

    x_test = x[test_mask]
    y_test = y[test_mask]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    dataset_name = "cardio"
    overwrite_flag = True
    prepare_data(dataset_name, overwrite_flag)
