# -*- coding: utf-8 -*-
import pandas as pd

DATA_PATH = "/home/emmanuel/projects/2020_ml_ocn/data/RAW/CONTROL/"
CONTROL1 = "NORTH_ATLANTIC"
CONTROL2 = "SUBTROPICAL_GYRES"


class DataLoad:
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

    def load_control_data(self, control="na"):
        """This will load the control data:
        * North Atlantic Region
        * Subtropical Gyres
        """
        # choose control group data
        if control == "na":
            region = "NORTH_ATLANTIC"
            filename_ext = "NA"
        elif control == "stg":
            region = "SUBTROPICAL_GYRES"
            filename_ext = "STG"
        else:
            raise ValueError(f"Unrecognized control group: {control}")

        # Load Data
        X = pd.read_csv(f"{DATA_PATH}{region}/X_INPUT_{filename_ext}.csv")
        y = pd.read_csv(f"{DATA_PATH}{region}/BBP_OUTPUT_{filename_ext}.csv")

        return X, y
