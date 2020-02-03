# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from visualization.visualize import get_depth_labels

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
        self.loc_vars = ["lat", "lon"]
        self.time_vars = ["doy"]
        self.meta_vars = ["wmo", "n_cycle"]

    def load_columns(self, region: str = "na"):

        columns = {}

        # load high dimensional datasets
        columns["temperature"] = (
            self.load_temperature(region, drop_meta=True, training=True)
            .add_prefix("temp_")
            .columns.values
        )
        columns["density"] = (
            self.load_density(region, drop_meta=True, training=True)
            .add_prefix("dens_")
            .columns.values
        )
        columns["salinity"] = (
            self.load_salinity(region, drop_meta=True, training=True)
            .add_prefix("sal_")
            .columns.values
        )
        columns["spicy"] = (
            self.load_spicy(region, drop_meta=True, training=True)
            .add_prefix("spice_")
            .columns.values
        )
        columns["core"] = self.core_vars
        columns["time"] = self.time_vars
        columns["location"] = self.loc_vars
        columns["argo_float"] = ["wmo"]
        columns["argo_time"] = ["n_cycle"]

        return columns

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

        # ren

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


class ValidationFloats:
    def __init__(self, region: str = "na"):
        self.region = region
        self.valid_floats = {"na": [6901486, 3902123], "stg": [6901472, 3902121]}
        self.meta_vars = ["wmo", "n_cycle"]
        self.depths = get_depth_labels()

    def get_validation_floats(self, region: str = "na"):
        return self.valid_floats[region]

    def _load_labels(self, region: Optional[str] = "na"):

        # get region
        if region is None:
            region = self.region

        # Load labels
        y = load_labels(region, training=False)

        # get meta columns
        self.meta_columns = y[self.meta_vars]

        columns = y.columns
        columns = y.columns

        # check that columns match depths
        assert len(columns[2:]) == len(self.depths)

        # get columns
        self.columns = np.concatenate((columns[:2].values, self.depths))
        return y

    def get_validation_res(
        self,
        ytest: np.ndarray,
        ypred: np.ndarray,
        validation_float: Optional[int] = None,
        float_num: int = 1,
    ):
        # get columns and meta Variables
        self._load_labels(self.region)

        # create numpy array with metadata columns
        print(self.meta_columns.values.shape, ypred.shape)
        ypred = np.concatenate((self.meta_columns.values, ypred), axis=1)
        ytest = np.concatenate((self.meta_columns.values, ytest), axis=1)

        # create dataframe
        ypred = pd.DataFrame(ypred, columns=self.columns)
        ytest = pd.DataFrame(ytest, columns=self.columns)

        # get validation valid_floats
        if validation_float is None:
            validation_float = self.valid_floats[self.region][float_num]

        # extract data with floats
        ypred = ypred[ypred["wmo"] == validation_float]
        ytest = ytest[ytest["wmo"] == validation_float]

        # drop float name columns
        ypred = ypred.drop(["wmo"], axis=1)
        ytest = ytest.drop(["wmo"], axis=1)

        # create time series
        ypred = pd.melt(
            ypred, id_vars=["n_cycle"], var_name="Depth", value_name="Predictions"
        )
        ytest = pd.melt(
            ytest, id_vars=["n_cycle"], var_name="Depth", value_name="Labels"
        )

        # merge into time series with depths
        y = pd.merge(ypred, ytest)

        return y


def get_data(params):

    # -------------------------------
    # Core params
    # -------------------------------

    # load training data
    X_core = load_standard_data(params.region, training=True)

    # Testing Data
    X_core_te = load_standard_data(params.region, training=False)
    X_core_te = X_core_te.iloc[:, 2:]

    # ----------------------------------
    # High Dimensional params
    # ----------------------------------
    X_temp, X_dens, X_sal, X_spicy = load_high_dim_data(params.region, training=True)

    # add prefix (Training/Validation)
    X_temp = X_temp.add_prefix("temp_")
    X_dens = X_dens.add_prefix("dens_")
    X_sal = X_sal.add_prefix("sal_")
    X_spicy = X_spicy.add_prefix("spice_")

    #
    X_temp_te, X_dens_te, X_sal_te, X_spicy_te = load_high_dim_data(
        params.region, training=False
    )

    # Subset
    X_temp_te = X_temp_te.iloc[:, 2:]
    X_dens_te = X_dens_te.iloc[:, 2:]
    X_sal_te = X_sal_te.iloc[:, 2:]
    X_spicy_te = X_spicy_te.iloc[:, 2:]

    # add prefix (Test)
    X_temp_te = X_temp_te.add_prefix("temp_")
    X_dens_te = X_dens_te.add_prefix("dens_")
    X_sal_te = X_sal_te.add_prefix("sal_")
    X_spicy_te = X_spicy_te.add_prefix("spice_")

    # --------------------------------------------
    # Load Labels
    # --------------------------------------------
    ytr = load_labels(params.region, training=True)

    yte = load_labels(params.region, training=False)

    yte = yte.iloc[:, 2:]

    # Concatenate Data
    # Training Data
    Xtr = pd.concat([X_core, X_temp, X_dens, X_sal, X_spicy], axis=1)

    # Testing Data
    Xte = pd.concat([X_core_te, X_temp_te, X_dens_te, X_sal_te, X_spicy_te], axis=1)

    dataset = {"Xtrain": Xtr, "Xtest": Xte, "ytrain": ytr, "ytest": yte}
    return dataset

