import pathlib
from typing import Tuple
from src.utils import get_paths
import pandas as pd
import collections

INPUT_FILE = "SOCA_GLOBAL2_20200310.csv"
META_FILE = "METADATA_20200310.csv"
PATHS = get_paths()

FEATURES = collections.namedtuple("FEATURES", ["meta", "input", "output"])

world_features = FEATURES(
    meta=["wmo", "n_cycle", "N", "lon", "lat", "juld", "date"],
    input=[
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
    ],
    output=[
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
    ],
)


def get_meta_data() -> pd.DataFrame:
    return pd.read_csv(f"{PATHS.data_processed.joinpath(META_FILE)}")


def get_input_data() -> pd.DataFrame:
    return pd.read_csv(f"{PATHS.data_processed.joinpath(INPUT_FILE)}")


def get_full_data() -> pd.DataFrame:

    return pd.merge(get_meta_data(), get_input_data()).set_index(world_features.meta)
