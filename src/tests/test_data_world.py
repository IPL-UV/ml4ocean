import pathlib
from typing import Tuple
from src.utils import get_paths
from src.data.world import get_meta_data, get_input_data, get_full_data, world_features
import pandas as pd

INPUT_FILE = "SOCA_GLOBAL2_20200310.csv"
META_FILE = "METADATA_20200310.csv"
PATHS = get_paths()


def test_meta_data():
    # get full path
    meta_file = PATHS.data_processed.joinpath(META_FILE)

    # assert meta file exists
    error_msg = f"File '{meta_file.name}' doesn't exist. Check name or directory."
    assert meta_file.exists(), error_msg

    # assert meta file is a file
    error_msg = f"File '{meta_file.name}' isn't a file. Check name or directory."
    assert meta_file.is_file(), error_msg


def test_get_meta_data():
    # get full path
    meta_df = get_meta_data()

    assert isinstance(meta_df, pd.DataFrame)

    # check number of samples
    n_samples = 25413
    error_msg = f"Incorrect number of samples: {meta_df.shape[0]} =/= {n_samples}"
    assert meta_df.shape[0] == n_samples, error_msg

    # check meta feature names
    meta_features = ["wmo", "n_cycle", "N", "lon", "lat", "juld", "date"]
    error_msg = f"Missing features in meta data."
    assert meta_df.columns.tolist() == meta_features, error_msg


def test_input_data():
    # get full path
    data_file = PATHS.data_processed.joinpath(INPUT_FILE)

    # assert exists
    error_msg = f"File '{data_file.name}' doesn't exist. Check name or directory."
    assert data_file.exists(), error_msg

    # assert meta file is a file
    error_msg = f"File '{data_file.name}' isn't a file. Check name or directory."
    assert data_file.is_file(), error_msg


def test_get_input_data():
    # get full path
    input_df = get_input_data()

    assert isinstance(input_df, pd.DataFrame)

    # check number of samples
    n_samples = 25413
    error_msg = f"Incorrect number of samples: {input_df.shape[0]} =/= {n_samples}"
    assert input_df.shape[0] == n_samples, error_msg

    # check data feature names
    input_meta_features = ["N", "wmo", "n_cycle"]
    input_features = [
        "sla",
        "PAR",
        "RHO_WN_412",
        "RHO_WN_443",
        "RHO_WN_490",
        "RHO_WN_555",
        "RHO_WN_670",
        "doy_sin",
        "doy_cos",
        "x_cart",
        "y_cart",
        "z_cart",
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
        "PC7",
        "PC1.1",
        "PC2.1",
        "PC3.1",
        "PC1.2",
        "PC2.2",
        "PC3.2",
        "PC4.1",
    ]
    output_features = [
        "bbp",
        "bbp.1",
        "bbp.2",
        "bbp.3",
        "bbp.4",
        "bbp.5",
        "bbp.6",
        "bbp.7",
        "bbp.8",
        "bbp.9",
        "bbp.10",
        "bbp.11",
        "bbp.12",
        "bbp.13",
        "bbp.14",
        "bbp.15",
        "bbp.16",
        "bbp.17",
        "bbp.18",
    ]
    features = input_meta_features + input_features + output_features
    error_msg = f"Missing features in input data."
    assert input_df.columns.tolist() == features, error_msg


def test_full_data():

    full_df = get_full_data()

    # checks - check indices match metadata
    error_msg = f"Missing features in input data."
    assert full_df.index.names == world_features.meta, error_msg

    # checks - check column names match feature names
    error_msg = f"Missing features in input data."
    features = world_features.input + world_features.output
    assert full_df.columns.tolist() == features, error_msg
