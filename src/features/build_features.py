import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon


def get_geodataframe(dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
    """This function will transform the dataset from a 
    pandas.DataFrame to a geopandas.DataFrame which will
    have a special column for geometry. This will make plotting 
    a lot easier."""
    # get polygons
    geometry = [Point(xy) for xy in zip(dataframe["lon"], dataframe["lat"])]

    # coordinate systems
    crs = {"init": "epsg:4326"}

    # create dataframe
    gpd_df = gpd.GeoDataFrame(dataframe, crs=crs, geometry=geometry)
    return gpd_df
