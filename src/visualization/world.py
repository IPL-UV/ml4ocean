from src.data.world import world_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_residuals(ypred: np.ndarray, ytest: np.ndarray, dataset: str = "train"):

    # create id line
    # TODO: Make this more robust
    # min_y = np.minimum(ypred.min(), ytest.min())
    # max_y = np.minimum(ypred.max(), ytest.max())
    if dataset == "train":
        id_line = np.linspace(-3, 3, 50)
        
    elif dataset == "test":
        id_line = np.linspace(-6, -2, 50)

    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(ypred, ytest, marker="o")
    ax.plot(id_line, id_line, linewidth=3, color="black")

    return fig, ax


def plot_test_residuals(ypred: pd.DataFrame, ytest: pd.DataFrame):

    # create id line
    # TODO: Make this more robust
    # min_y = np.minimum(ypred.min(), ytest.min())
    # max_y = np.minimum(ypred.max(), ytest.max())
    id_line = np.linspace(-3, 3, 50)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(ypred.values.ravel(), ytest.values.ravel(), marker="o")
    ax.plot(id_line, id_line, linewidth=3, color="black")

    return fig, ax
