# -*- coding: utf-8 -*-
import pandas as pd
from typing import Tuple

DATA_PATH = "/home/emmanuel/projects/2020_ml_ocn/data/RAW/CONTROL/"
CONTROL1 = "NORTH_ATLANTIC"
CONTROL2 = "SUBTROPICAL_GYRES"

# TODO: more documentation for dataloader


class DataLoader:
    """DataLoader for the NA data.
    
    Options:
    --------
    * Control Data
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

    def load_control_data(
        self, control: str = "na"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the control data:
        * North Atlantic Region
        * Subtropical Gyres
        """
        # choose control group data
        region, filename_ext = self._get_control_ext(control)

        return pd.read_csv(f"{DATA_PATH}{region}/X_INPUT_{filename_ext}.csv")

    def load_control_ouputs(
        self, control: str = "na"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the control data:
        * North Atlantic Region
        * Subtropical Gyres
        """
        # choose control group data
        region, filename_ext = self._get_control_ext(control)

        return pd.read_csv(f"{DATA_PATH}{region}/BBP_OUTPUT_{filename_ext}.csv")

    def load_control_temperature(self, control: str = "na") -> pd.DataFrame:
        """This loads the control data for temperature"""
        # choose control group data
        region, filename_ext = self._get_control_ext(control)
        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_TEMP_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        return X

    def load_control_density(self, control: str = "na") -> pd.DataFrame:
        """This loads the control data for density"""
        # choose control group data
        region, filename_ext = self._get_control_ext(control)

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_DENS_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})
        return X

    def load_control_salinity(self, control: str = "na") -> pd.DataFrame:
        """This loads the control data for salinity"""
        # choose control group data
        region, filename_ext = self._get_control_ext(control)

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_PSAL_{filename_ext}.txt", sep=" ", header=None
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})
        return X

    def load_control_spicy(self, control: str = "na") -> pd.DataFrame:
        """This loads the control data for 'spiciness'"""
        # choose control group data
        region, filename_ext = self._get_control_ext(control)

        X = pd.read_csv(
            f"{DATA_PATH}{region}/MATRIX_SPICINESS_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})
        return X

    def _get_control_ext(self, control="na"):
        # choose control group data
        if control == "na":
            return "NORTH_ATLANTIC", "NA"
        elif control == "stg":
            return "SUBTROPICAL_GYRES", "STG"
        else:
            raise ValueError(f"Unrecognized control group: {control}")
