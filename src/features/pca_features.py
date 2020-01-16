from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from typing import Tuple, Union, List, Optional
import pandas as pd
import numpy as np


def transform_all(
    dfs: List[pd.DataFrame], pca_model: Optional[List[pd.DataFrame]]=None, n_components: int = 10, random_state: int = 123
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on all of the variables
    concatenated."""
    X = pd.concat(dfs, axis=1)

    # perform PCA transformation
    if pca_model == None:
        pca_model = PCA(n_components=n_components, random_state=random_state)

        # fit PCA model
        X = pca_model.fit_transform(X)
        return X, pca_model
    else:
        X = pca_model.transform(X)

        return X


def transform_individual(
    dfs: List[pd.DataFrame],
    n_components: int = 10,
    random_state: int = 123,
    **kwargs: Tuple[int, str, bool, float]
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on the list of dataframes concatenated.
    
    Parameters
    ----------
    dfs: List[pd.DataFrame]
        a list of pandas dataframes to perform the PCA transformation

    random_state: int, default=123
        the random state for the PCA transformations

    kwargs: Tuple[int, str, bool, float]
        some kwargs for the PCA transformation

    Returns
    -------
    X: List[pd.DataFrame]
        a list of pandas dataframes
    

    """

    # perform PCA transformation
    pca_models = [
        PCA(n_components=n_components, random_state=random_state, **kwargs)
    ] * len(dfs)

    # fit PCA model
    dfs = [pca_model.fit_transform(df) for df, pca_model in zip(dfs, pca_models)]

    return dfs, pca_models
