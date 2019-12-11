from sklearn.manifold import TSNE
from sklearn import decomposition
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import prepare_data
import config


def scatter_pca(data_id):
    data_dict = prepare_data(data_id)
    x_train, y_train, _, _ = data_dict.values()

    num_label = len(np.unique(y_train))

    pca = decomposition.PCA(n_components=2)
    x_train_ = pca.fit_transform(x_train)

    df = pd.DataFrame(np.concatenate([x_train_, y_train[:, None]], axis=1), columns=["pca-one", "pca-two", "y"])

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", num_label),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title(f"PCA Visualization of Dataset: {config.DATA_DISPATCHER[data_id]}")
    plt.show()


def scatter_tsne(data_id):
    data_dict = prepare_data(data_id)
    x_train, y_train, _, _ = data_dict.values()

    num_label = len(np.unique(y_train))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    x_train_ = tsne.fit_transform(x_train)

    df = pd.DataFrame(np.concatenate([x_train_, y_train[:, None]], axis=1), columns=["pca-one", "pca-two", "y"])

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", num_label),
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title(f"TSNE Visualization of Dataset: {config.DATA_DISPATCHER[data_id]}")
    plt.show()


if __name__ == '__main__':
    data_id = 2
    mode = "pca"

    if mode == "pca":
        scatter_pca(data_id)
    else:
        scatter_tsne(data_id)