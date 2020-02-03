from typing import Optional, List
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SAVE_PATH = "/media/disk/erc/papers/2019_ML_OCN/ml4ocean/reports/figures/"
SAVE_PATH = "/home/emmanuel/figures/ml4ocn/"
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-talk")


class PlotResults:
    def __init__(self):
        pass


def get_depth_labels():

    depths = 276
    first = [*range(0, 250)]
    d1 = first[::2]

    second = [*range(250, 1001)]
    d2 = second[::5]
    return -np.concatenate((d1, d2))


def plot_mo_stats(
    df: pd.DataFrame, stat: str, color: str = "blue", save_name: Optional[str] = None
) -> None:

    # MAE plot
    fig, ax = plt.subplots(figsize=(7, 5))

    if stat.lower() == "mae":
        ylabel = "Mean Absolute Error"
    elif stat.lower() == "mse":
        ylabel = "Mean Squared Error"
    elif stat.lower() == "rmse":
        ylabel = "Root Mean Squared Error"
    elif stat.lower() == "r2":
        ylabel = "R2"

    else:
        raise ValueError(f"Unrecognized stat: {stat}")
    df["depths"] = get_depth_labels()

    df.plot(y="depths", x=stat.lower(), ax=ax, linewidth=6, color=color)
    if stat.lower() == "r2":
        ax.set_xlim([0, 1])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)
    ax.legend([])
    ax.grid()
    plt.tight_layout()

    if save_name is not None:
        fig.savefig(
            SAVE_PATH + f"mo_{save_name}_{stat}.png",
            dpi=200,
            transparent=True,
            # facecolor=False,
        )
    else:
        plt.show()


# def plot_mo_stats(ytest: np.ndarray, ypred: np.ndarray) -> None:

#     # get statsitics
#     mae_raw = mean_absolute_error(ytest, ypred, multioutput="raw_values")
#     mse_raw = mean_squared_error(ytest, ypred, multioutput="raw_values")
#     rmse_raw = np.sqrt(mse_raw)
#     r2_raw = r2_score(ytest, ypred, multioutput="raw_values")

#     plt.style.use("seaborn")

#     # Plots
#     fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

#     # R2 Values
#     ax[0, 0].plot(r2_raw)
#     ax[0, 0].set_xlabel("Depths (Pressure)")
#     ax[0, 0].set_ylabel("R2")
#     ax[0, 0].set_ylim([0, 1])

#     # MAE
#     ax[0, 1].plot(mae_raw)
#     ax[0, 1].set_xlabel("Depths (Pressure)")
#     ax[0, 1].set_ylabel("MAE")

#     # MSE
#     ax[1, 0].plot(mse_raw)
#     ax[1, 0].set_xlabel("Depths (Pressure)")
#     ax[1, 0].set_ylabel("MSE")

#     # RMSE
#     ax[1, 1].plot(rmse_raw)
#     ax[1, 1].set_xlabel("Depths (Pressure)")
#     ax[1, 1].set_ylabel("RMSE")

#     plt.tight_layout()
#     plt.show()


def plot_bbp_profile(dataframe: pd.DataFrame):

    norm = colors.LogNorm(vmin=dataframe.values.min(), vmax=dataframe.values.max())

    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(dataframe.T, cmap="viridis", norm=norm)
    plt.show()


def plot_pairplots(dataframe: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(10, 10))

    pts = sns.pairplot(dataframe)

    plt.show()


def plot_geolocations(
    gpd_dfs: List[gpd.GeoDataFrame],
    colors=List[str],
    return_plot: Optional[bool] = False,
    save_name: Optional[str] = False,
) -> None:

    # get the background map
    path = gpd.datasets.get_path("naturalearth_lowres")
    world_df = gpd.read_file(path)

    # initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # add background world map
    world_df.plot(ax=ax, color="gray", zorder=2)

    # add the locations of the dataset
    for igpd_df, icolor in zip(gpd_dfs, colors):
        igpd_df.plot(ax=ax, color=icolor, markersize=3, zorder=3)

    ax.grid(zorder=0)
    plt.tight_layout()

    if save_name is not None:
        fig.savefig(SAVE_PATH + f"geo_{save_name}.png", dpi=200, transparent=True)
    else:
        plt.show()

    if return_plot:
        return fig, ax
    else:
        return None
