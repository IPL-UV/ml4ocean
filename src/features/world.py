from typing import Tuple
import pandas as pd

SOCA2016_FLOATS = ["6901472", "6901493", "6901523", "6901496"]
ISPRS2020_FLOATS = ["6901486", "3902121"]


def subset_soca2016_floats(df: pd.DataFrame) -> pd.DataFrame:

    return df[df.index.isin(SOCA2016_FLOATS, level="wmo")]


def subset_independent_floats(
    df: pd.DataFrame, dataset: str = "soca2016"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if dataset == "soca2016":
        independent_df = df[df.index.isin(SOCA2016_FLOATS, level="wmo")]
        df = df[~df.index.isin(SOCA2016_FLOATS, level="wmo")]
        return df, independent_df
    elif dataset == "isprs2020":
        independent_df = df[df.index.isin(ISPRS2020_FLOATS, level="wmo")]
        df = df[~df.index.isin(ISPRS2020_FLOATS, level="wmo")]
        return df, independent_df
    elif dataset == "both":
        independent_df = df[
            df.index.isin(ISPRS2020_FLOATS + SOCA2016_FLOATS, level="wmo")
        ]
        df = df[~df.index.isin(ISPRS2020_FLOATS + SOCA2016_FLOATS, level="wmo")]
        return df, independent_df
