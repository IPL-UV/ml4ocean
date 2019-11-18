from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from typing import Tuple, Union
import pandas as pd
import numpy as np


def transform_all(
    df_temp: pd.DataFrame,
    df_sal: pd.DataFrame,
    df_dens: pd.DataFrame,
    df_spicy: pd.DataFrame,
    n_components: int = 10,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on all of the variables
    concatenated."""
    X = pd.concat([df_temp, df_sal, df_dens, df_spicy], axis=1)

    # perform PCA transformation
    pca_model = PCA(n_components=n_components, random_state=random_state)

    # fit PCA model
    X = pca_model.fit_transform(X)

    return X, pca_model


def transform_individual(
    df_temp: pd.DataFrame,
    df_sal: pd.DataFrame,
    df_dens: pd.DataFrame,
    df_spicy: pd.DataFrame,
    n_components: int = 10,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, BaseEstimator]:
    """Applies a PCA transform on all of the variables
    concatenated."""

    # perform PCA transformation
    pca_models = {
        "temp": PCA(n_components=n_components, random_state=random_state),
        "dens": PCA(n_components=n_components, random_state=random_state),
        "sal": PCA(n_components=n_components, random_state=random_state),
        "spicy": PCA(n_components=n_components, random_state=random_state),
    }

    # fit PCA model

    X_temp = pca_models["temp"].fit_transform(df_temp)
    X_dens = pca_models["dens"].fit_transform(df_dens)
    X_sal = pca_models["sal"].fit_transform(df_sal)
    X_spicy = pca_models["spicy"].fit_transform(df_spicy)

    return np.hstack([X_temp, X_dens, X_sal, X_spicy]), pca_models
