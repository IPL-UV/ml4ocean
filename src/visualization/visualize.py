import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")


def plot_mo_stats(ytest: np.ndarray, ypred: np.ndarray) -> None:

    # get statsitics
    mae_raw = mean_absolute_error(ytest, ypred, multioutput="raw_values")
    mse_raw = mean_squared_error(ytest, ypred, multioutput="raw_values")
    rmse_raw = np.sqrt(mse_raw)
    r2_raw = r2_score(ytest, ypred, multioutput="raw_values")

    plt.style.use("seaborn")

    # Plots
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

    # R2 Values
    ax[0, 0].plot(r2_raw)
    ax[0, 0].set_xlabel("Depths (Pressure)")
    ax[0, 0].set_ylabel("R2")
    ax[0, 0].set_ylim([0, 1])

    # MAE
    ax[0, 1].plot(mae_raw)
    ax[0, 1].set_xlabel("Depths (Pressure)")
    ax[0, 1].set_ylabel("MAE")

    # MSE
    ax[1, 0].plot(mse_raw)
    ax[1, 0].set_xlabel("Depths (Pressure)")
    ax[1, 0].set_ylabel("MSE")

    # RMSE
    ax[1, 1].plot(rmse_raw)
    ax[1, 1].set_xlabel("Depths (Pressure)")
    ax[1, 1].set_ylabel("RMSE")

    plt.tight_layout()
    plt.show()


def plot_bbp_profile(dataframe: pd.DataFrame):

    norm = colors.LogNorm(vmin=dataframe.values.min(), vmax=dataframe.values.max())

    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(dataframe.T, cmap="viridis", norm=norm)
    plt.show()


def plot_pairplots(dataframe: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(10, 10))

    pts = sns.pairplot(dataframe)

    plt.show()


def plot_geolocations(gpd_df: gpd.GeoDataFrame, color="red") -> None:

    # get the background map
    path = gpd.datasets.get_path("naturalearth_lowres")
    world_df = gpd.read_file(path)

    # initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # add background world map
    world_df.plot(ax=ax, color="gray")

    # add the locations of the dataset
    gpd_df.plot(ax=ax, color=color, markersize=2)

    plt.show()
