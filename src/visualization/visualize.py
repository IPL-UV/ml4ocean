import geopandas as gpd
import pandas as pd
import matplotlib.colors as colors


# plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")


def plot_bbp_profile(dataframe: pd.DataFrame):

    norm = colors.LogNorm(vmin=dataframe.values.min(), vmax=dataframe.values.max())

    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(dataframe.T, cmap="viridis", norm=norm)
    plt.show()


def plot_pairplots(dataframe: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(10, 10))

    pts = sns.pairplot(dataframe)

    plt.show()


def plot_geolocations(gpd_df: gpd.GeoDataFrame) -> None:

    # get the background map
    path = gpd.datasets.get_path("naturalearth_lowres")
    world_df = gpd.read_file(path)

    # initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # add background world map
    world_df.plot(ax=ax, color="gray")

    # add the locations of the dataset
    gpd_df.plot(ax=ax, color="red", markersize=2)

    plt.show()
