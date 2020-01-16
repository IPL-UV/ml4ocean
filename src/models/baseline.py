from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Union
import numpy as np
import pandas as pd
import time


def train_glm_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
) -> BaseEstimator:
    """Train a basic Generalized Linear Model (GLM)

    Parameters
    ----------
    xtrain : np.ndarray, pd.DataFrame 
             (n_samples x d_features)
             input training data
    
    ytrain : np.ndarray, pd.DataFrame 
             (n_samples x p_outputs)
             labeled training data 
    
    verbose : int, default=0
        option to print out training messages 

    Returns 
    -------
    gl_model : BaseEstimator
        the trained model
    """
    # Initialize GLM
    gl_model = MultiTaskElasticNetCV(
        alphas=None,
        cv=3,
        random_state=123,
        n_jobs=-1,
        normalize=True,
        selection="cyclic",
        verbose=verbose,
    )

    # train GLM
    t0 = time.time()
    gl_model.fit(xtrain, ytrain)
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return gl_model


def train_rf_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
) -> BaseEstimator:
    """Train a basic Random Forest (RF) Regressor 

    Parameters
    ----------
    xtrain : np.ndarray, pd.DataFrame 
             (n_samples x d_features)
             input training data
    
    ytrain : np.ndarray, pd.DataFrame 
             (n_samples x p_outputs)
             labeled training data 
    
    verbose : int, default=0
        option to print out training messages 

    Returns 
    -------
    rf_model : BaseEstimator
        the trained model
    """
    # initialize baseline RF model
    rf_model = RandomForestRegressor(
        n_estimators=1500,
        criterion="mse",
        n_jobs=-1,
        random_state=123,
        warm_start=False,
        verbose=verbose,
    )

    # train RF model
    t0 = time.time()
    rf_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return rf_model
