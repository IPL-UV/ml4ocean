import numpy as np
from typing import List, Optional
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.base import BaseEstimator, TransformerMixin


class CycleTransform(BaseEstimator, TransformerMixin):
    """Converts some times to a cyclic axis of x, y using sin and cos. 
    
    
    1. Converts the times to radians 
    2. Normalizes by the maximum of the time cycle
    3. Applies the sin and cosine transformation
    4. Drops original columns

    Parameters 
    ----------
    time_types : List of , e.g. ['doy', 'month', 'hour']
        The time type to convert to a cycle
        doy - assumes 1 in 24 hours

    Example
    -------
    >> times = ['doy']
    >> X = CycleTransform(times).fit_transform(X)

    >> times = ['doy', 'month']
    >> X = CycleTransform(times).fit_transform(X)
    """

    def __init__(self, time_types: List[str] = ["doy"]):
        self.time_types = time_types

    def fit(self, X, y=None):
        """For compatibility reasons."""
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Parameters 
        ----------
        X : pd.DataFrame
            A dataframe with the values. The columns need to be one of the following
            ['doy', 'month', 'hour']
        
        y : pd.DataFrame, Optional
            Does nothing. Only for compatibility reasons.
        
        Returns
        -------
        df : pd.DataFrame
            A dataframe with the converted values.
        """
        deg2rad = 2 * np.pi

        cols = X.columns.tolist()

        if "doy" in self.time_types and "doy" in cols:

            const = 365.0  # number of days in a year

            X["doy_sin"] = np.sin(X["doy"] * deg2rad / const)
            X["doy_cos"] = np.cos(X["doy"] * deg2rad / const)

            X = X.drop("doy", axis=1)

        if "month" in self.time_types and "month" in cols:

            const = 12  # number of months in a year

            X["month_sin"] = np.sin((X["month"] - 1) * deg2rad / const)
            X["month_cos"] = np.cos((X["month"] - 1) * deg2rad / const)

            X = X.drop("month", axis=1)

        if "hour" in self.time_types and "hour" in cols:

            const = 24.0  # number of days in a year

            X["hour_sin"] = np.sin(X["hour"] * deg2rad / const)
            X["hour_cos"] = np.cos(X["hour"] * deg2rad / const)

            X = X.drop("hour", axis=1)

        # drop original column

        return X


class GeoCartTransform(BaseEstimator, TransformerMixin):
    """Transforms geo coordinates (lat, lon) to cartesian coordinates
    (x, y, z).
    
    Example
    -------
    >> df = geo_2_cartesian(df)
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """    
        Parameters 
        ----------
        df : pd.DataFrame
            A dataframe with the geo coordinates values. The columns need to 
            have the following ['lat', 'lon]
        
        Returns
        -------
        df : pd.DataFrame
            A dataframe with the converted values.
        """
        cols = X.columns.tolist()

        if "lat" not in cols or "lon" not in cols:
            print("lat,lon columns not present in X.")
            return X

        deg2rad = np.pi / 180.0

        # transform from degrees to radians
        X["lat"] *= deg2rad
        X["lon"] *= deg2rad

        # From Geo coords to cartesian coords
        X["x"] = np.cos(X["lat"]) * np.cos(X["lon"])
        X["y"] = np.cos(X["lat"]) * np.sin(X["lon"])
        X["z"] = np.sin(X["lat"])

        # drop original columns
        X = X.drop(["lat", "lon"], axis=1)

        return X


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


def geo_2_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms geo coordinates (lat, lon) to cartesian coordinates
    (x, y, z).
    
    Parameters 
    ----------
    df : pd.DataFrame
        A dataframe with the geo coordinates values. The columns need to 
        have the following ['lat', 'lon]
    
    Returns
    -------
    df : pd.DataFrame
        A dataframe with the converted values.

    Example
    -------
    >> df = geo_2_cartesian(df)
    """
    cols = df.columns.tolist()

    if "lat" not in cols or "lon" not in cols:
        print("lat,lon columns not present in df.")
        return df

    deg2rad = np.pi / 180.0

    # transform from degrees to radians
    df["lat"] *= deg2rad
    df["lon"] *= deg2rad

    # From Geo coords to cartesian coords
    df["x"] = np.cos(df["lat"]) * np.cos(df["lon"])
    df["y"] = np.cos(df["lat"]) * np.sin(df["lon"])
    df["z"] = np.sin(df["lat"])

    # drop original columns
    df = df.drop(["lat", "lon"], axis=1)

    return df


def times_2_cycles(df: pd.DataFrame, time_types: List[str] = ["doy"]) -> pd.DataFrame:
    """Converts some times to a cyclic axis of x, y using sin and cos. 
    
    
    1. Converts the times to radians 
    2. Normalizes by the maximum of the time cycle
    3. Applies the sin and cosine transformation
    4. Drops original columns

    Parameters 
    ----------
    df : pd.DataFrame
        A dataframe with the values. The columns need to be one of the following
        ['doy', 'month', 'hour']
    
    time_types : List of , e.g. ['doy', 'month', 'hour']
        The time type to convert to a cycle
        doy - assumes 1 in 24 hours
    
    Returns
    -------
    df : pd.DataFrame
        A dataframe with the converted values.

    Example
    -------
    >> times = ['doy']
    >> df = time_2_cycle(df, times)

    >> times = ['doy', 'month']
    >> df = times_2_cycles(df, times)
    """
    deg2rad = 2 * np.pi

    cols = X.columns.tolist()

    if "doy" in time_types and "doy" in cols:

        const = 365.0  # number of days in a year

        df["doy_sin"] = np.sin(df["doy"] * deg2rad / const)
        df["doy_cos"] = np.cos(df["doy"] * deg2rad / const)

        df = df.drop("doy", axis=1)

    if "month" in time_types and "month" in cols:

        const = 12  # number of months in a year

        df["month_sin"] = np.sin((df["month"] - 1) * deg2rad / const)
        df["month_cos"] = np.cos((df["month"] - 1) * deg2rad / const)

        df = df.drop("month", axis=1)

    if "hour" in time_types and "hour" in cols:

        const = 24.0  # number of days in a year

        df["hour_sin"] = np.sin(df["hour"] * deg2rad / const)
        df["hour_cos"] = np.cos(df["hour"] * deg2rad / const)

        df = df.drop("hour", axis=1)

    # drop original column

    return df
