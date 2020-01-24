# -*- coding: utf-8 -*-
import pandas as pd
from typing import Tuple

DATA_PATH = "/home/emmanuel/projects/2020_ml_ocn/data/RAW/CONTROL/"
region1 = "NORTH_ATLANTIC"
region2 = "SUBTROPICAL_GYRES"

# TODO: more documentation for dataloader


class DataLoader:
    """DataLoader for the NA data.
    
    Options:
    --------
    * region Data
        - North Atlantic
        - Subtropical Gyres (STG)
    
    Inputs
    ------
    * (sla)
    * (PAR)
    * ...
    """

    def __init__(self):
        self.core_vars = [
            "sla",
            "PAR",
            "RHO_WN_412",
            "RHO_WN_443",
            "RHO_WN_490",
            "RHO_WN_555",
            "RHO_WN_670",
            "MLD",
        ]

        self.core_outputs = ["sla"]
        self.loc_vars = ["lat", "lon", "doy"]
        self.meta_vars = ["wmo", "n_cycle"]

    def load_data(
        self, region: str = "na", drop_meta: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the region data:
        * North Atlantic Region
        * Subtropical Gyres

        Parameters
        ----------
        region : str, {'NA', 'STG'}
            the region to be extracted

        Returns
        -------
        df : pd.DataFrame
            a pandas dataframe containing the dataset

        """
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())

        # extract data
        X = pd.read_csv(f"{DATA_PATH}{region}/X_INPUT_{filename_ext}.csv")

        X = self._drop_meta(X, drop_meta)

        return X

    def load_ouputs(
        self, region: str = "na", drop_meta: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the region data:
        * North Atlantic Region
        * Subtropical Gyres
        """
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(f"{DATA_PATH}{region}/BBP_OUTPUT_{filename_ext}.csv")

        X = self._drop_meta(X, drop_meta)

        return X

    def load_temperature(
        self, region: str = "na", drop_meta: bool = False
    ) -> pd.DataFrame:
        """This loads the region data for temperature"""
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())
        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_TEMP_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        X = self._drop_meta(X, drop_meta)

        return X

    def load_density(self, region: str = "na", drop_meta: bool = False) -> pd.DataFrame:
        """This loads the region data for density"""
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_DENS_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        X = self._drop_meta(X, drop_meta)

        return X

    def load_salinity(
        self, region: str = "na", drop_meta: bool = False
    ) -> pd.DataFrame:
        """This loads the region data for salinity"""
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_PSAL_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        X = self._drop_meta(X, drop_meta)

        return X

    def load_spicy(self, region: str = "na", drop_meta: bool = False) -> pd.DataFrame:
        """This loads the region data for 'spiciness'"""
        # choose region group data
        region, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_SPICINESS_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        X = self._drop_meta(X, drop_meta)

        return X

    def _get_region_ext(self, region="na"):
        # choose region group data
        if region == "na":
            return "NORTH_ATLANTIC", "NA"
        elif region == "stg":
            return "SUBTROPICAL_GYRES", "STG"
        else:
            raise ValueError(f"Unrecognized region group: {region}")

    def _drop_meta(self, df: pd.DataFrame, drop_meta: bool = False) -> pd.DataFrame:
        if drop_meta:
            return df.drop(self.meta_vars, axis=1)
        else:
            return df


def load_standard_data(region: str = "NA"):

    # initialize dataloader
    dataloader = DataLoader()

    # load data
    X = dataloader.load_data(region=region, drop_meta=True)

    return X


def load_high_dim_data(region="NA"):

    # initialize dataloader
    dataloader = DataLoader()
    drop_meta = True

    X_temp = dataloader.load_temperature(region=region, drop_meta=drop_meta)
    X_dens = dataloader.load_density(region=region, drop_meta=drop_meta)
    X_sal = dataloader.load_salinity(region=region, drop_meta=drop_meta)
    X_spicy = dataloader.load_spicy(region=region, drop_meta=drop_meta)

    return X_temp, X_dens, X_sal, X_spicy


def load_labels(region="NA"):
    # initialize dataloader
    dataloader = DataLoader()
    drop_meta = True

    return dataloader.load_ouputs(region=region, drop_meta=drop_meta)
