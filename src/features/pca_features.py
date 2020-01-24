from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np


def transform_all(
    dfs: List[pd.DataFrame],
    pca_model: Optional[List[pd.DataFrame]] = None,
    n_components: int = 10,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on all of the variables
    concatenated."""
    X = pd.concat(dfs, axis=1)

    # perform PCA transformation
    if pca_model == None:
        pca_model = PCA(n_components=n_components, random_state=random_state)

        # fit PCA model
        X = pca_model.fit_transform(X)

        # meta data
        columns = [f"pc_{icomponent}" for icomponent in range(n_components)]
        X = pd.DataFrame(X, columns=columns)
        pca_model.columns = columns

        return X, pca_model
    else:
        X = pca_model.transform(X)
        if hasattr(pca_model, "columns"):
            X = pd.DataFrame(X, columns=pca_model.columns)
        return X


def transform_individual(
    dfs: List[pd.DataFrame],
    n_components: int = 10,
    random_state: int = 123,
    columns: Optional[List[str]] = None,
    **kwargs: Tuple[int, str, bool, float],
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on the list of dataframes concatenated.
    
    Parameters
    ----------
    dfs: List[pd.DataFrame]
        a list of pandas dataframes to perform the PCA transformation

    random_state: int, default=123
        the random state for the PCA transformations

    columns : List[str]
        the suffix added to the column names

    kwargs: Tuple[int, str, bool, float]
        some kwargs for the PCA transformation

    Returns
    -------
    X: List[pd.DataFrame]
        a list of pandas dataframes
    

    """
    # # get column names
    # columns = [df.columns for df in dfs]
    # perform PCA transformation
    pca_models = [
        PCA(n_components=n_components, random_state=random_state, **kwargs)
    ] * len(dfs)

    # fit PCA model
    dfs = [pca_model.fit_transform(df) for df, pca_model in zip(dfs, pca_models)]

    # add metadata
    if columns is not None:
        dfs = [
            idf.add_suffix(f"{iname}_pc{pc_comp+1}")
            for pc_comp, (idf, iname) in enumerate(zip(dfs, columns))
        ]

    return dfs, pca_models
