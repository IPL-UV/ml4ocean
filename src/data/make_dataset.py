# -*- coding: utf-8 -*-
import pandas as pd
from typing import Tuple, Optional, List

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
        self.valid_floats = {"na": [6901486, 3902123], "stg": [6901472, 3902121]}

        self.core_outputs = ["sla"]
        self.loc_vars = ["lat", "lon", "doy"]
        self.meta_vars = ["wmo", "n_cycle"]

    def load_data(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the region data:
        * North Atlantic Region
        * Subtropical Gyres

        Parameters
        ----------
        region : str, {'NA', 'STG'}
            the region to be extracted

        drop_meta : bool, default=False
            option to drop the meta data like the `n_cycles` or the 
            ARGO float number
        
        training : bool, default=True
            option to choose the training dataset or the independently 
            chosen validation dataset
        
        Returns
        -------
        df : pd.DataFrame
            a pandas dataframe containing the dataset

        """
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())

        # extract data
        X = pd.read_csv(f"{DATA_PATH}{region_name}/X_INPUT_{filename_ext}.csv")

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )
        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

    def load_ouputs(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """This will load the region data:
        * North Atlantic Region
        * Subtropical Gyres
        """
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(f"{DATA_PATH}{region_name}/BBP_OUTPUT_{filename_ext}.csv")

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )

        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

    def load_temperature(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> pd.DataFrame:
        """This loads the region data for temperature"""
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())
        X = pd.read_csv(
            f"{DATA_PATH}{region_name}/MATRIX_TEMP_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )

        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

    def load_density(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> pd.DataFrame:
        """This loads the region data for density"""
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region_name}/MATRIX_DENS_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )

        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

    def load_salinity(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> pd.DataFrame:
        """This loads the region data for salinity"""
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region_name}/MATRIX_PSAL_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )

        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

    def load_spicy(
        self, region: str = "na", drop_meta: bool = False, training: bool = True
    ) -> pd.DataFrame:
        """This loads the region data for 'spiciness'"""
        # choose region group data
        region_name, filename_ext = self._get_region_ext(region.lower())

        X = pd.read_csv(
            f"{DATA_PATH}{region_name}/MATRIX_SPICINESS_{filename_ext}.txt",
            sep=" ",
            header=None,
        )
        X = X.rename(columns={0: "wmo", 1: "n_cycle"})

        # extract training/validation dataset
        X_tr, X_val = self.extract_valid(
            X, region=region, valid_floats=self.valid_floats[region.lower()]
        )

        # drop metadata
        X_tr = self._drop_meta(X_tr, drop_meta)
        X_val = self._drop_meta(X_val, drop_meta)

        if training == True:
            return X_tr
        elif training == False:
            return X_val
        else:
            raise ValueError(f"Unrecognized boolean entry for 'training': {training}")

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

    def extract_valid(
        self,
        df: pd.DataFrame,
        region: str = "na",
        valid_floats: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """function to extract the validation dataset from
        the dataframe. Requires a column called.

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe with the dataset. Needs the column `wmo`
            to be able to extract the validation dataset

        region : str, default='na'
            the region for the validation floats
        
        valid_floats : List[str], default=None
            the list of validation floats which will override the 
            validation floats initialized within the DataLoader class.
        
        Returns
        -------
        df_train : pd.DataFrame
            the dataframe with the original dataset
        
        df_valid : pd.DataFrame
            the dataframe with the extracted validation floats
        """
        # override validation floats if given
        if valid_floats is None:
            valid_floats = self.valid_floats[region]

        # extract validation floats from
        df_valid = df[df["wmo"].isin(valid_floats)]

        # extract the training floats
        df_train = df[~df["wmo"].isin(valid_floats)]

        return df_train, df_valid


def load_standard_data(region: str = "NA", training: bool = True):

    # initialize dataloader
    dataloader = DataLoader()

    if training == True:
        drop_meta = True
    else:
        drop_meta = False

    # load data
    X = dataloader.load_data(region, training=training, drop_meta=drop_meta)

    return X


def load_high_dim_data(region="NA", training: bool = True):

    # initialize dataloader
    dataloader = DataLoader()

    if training == True:
        drop_meta = True
    else:
        drop_meta = False

    X_temp = dataloader.load_temperature(
        region=region, training=training, drop_meta=drop_meta
    )
    X_dens = dataloader.load_density(
        region=region, training=training, drop_meta=drop_meta
    )
    X_sal = dataloader.load_salinity(
        region=region, training=training, drop_meta=drop_meta
    )
    X_spicy = dataloader.load_spicy(
        region=region, training=training, drop_meta=drop_meta
    )

    return X_temp, X_dens, X_sal, X_spicy


def load_labels(region="NA", training: bool = True):
    # initialize dataloader
    dataloader = DataLoader()

    if training == True:
        drop_meta = True
    else:
        drop_meta = False

    return dataloader.load_ouputs(region=region, training=training, drop_meta=drop_meta)
