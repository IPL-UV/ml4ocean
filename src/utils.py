import pathlib
import sys
from collections import namedtuple


# define a container to hold all of the paths
project_paths = namedtuple(
    "project_paths",
    [
        "project",
        "code",
        "data_raw",
        "data_processed",
        "data_interim",
        "data_results",
        "models",
        "figures",
    ],
)


def get_paths():
    # define the top level directory
    PROJECT_PATH = pathlib.Path("/media/disk/erc/papers/2019_ML_OCN/")
    CODE_PATH = PROJECT_PATH.joinpath("ml4ocean", "src")

    # check if path exists and is a directory
    assert PROJECT_PATH.exists() & PROJECT_PATH.is_dir()
    assert CODE_PATH.exists() & CODE_PATH.is_dir()

    # specific paths
    FIG_PATH = PROJECT_PATH.joinpath("ml4ocean/reports/figures/global/")
    RAW_PATH = PROJECT_PATH.joinpath("data/global/raw/")
    DATA_PATH = PROJECT_PATH.joinpath("data/global/processed/")
    INTERIM_PATH = PROJECT_PATH.joinpath("data/global/interim/")
    MODEL_PATH = PROJECT_PATH.joinpath("models/global/")
    RESULTS_PATH = PROJECT_PATH.joinpath("data/global/results/")

    # check if path exists and is a directory
    assert FIG_PATH.exists() & FIG_PATH.is_dir()
    assert RAW_PATH.exists() & RAW_PATH.is_dir()
    assert DATA_PATH.exists() & DATA_PATH.is_dir()
    assert INTERIM_PATH.exists() & INTERIM_PATH.is_dir()
    assert MODEL_PATH.exists() & MODEL_PATH.is_dir()
    assert RESULTS_PATH.exists() & RESULTS_PATH.is_dir()

    # return tuple of paths
    return project_paths(
        project=PROJECT_PATH,
        code=CODE_PATH,
        data_raw=RAW_PATH,
        data_processed=DATA_PATH,
        data_interim=INTERIM_PATH,
        data_results=RESULTS_PATH,
        models=MODEL_PATH,
        figures=FIG_PATH,
    )
