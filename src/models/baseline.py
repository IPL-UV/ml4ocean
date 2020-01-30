from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF, Matern
from sklearn.linear_model import MultiTaskElasticNetCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Union, Tuple
import numpy as np
import pandas as pd
import time

import warnings

warnings.filterwarnings("ignore")


def train_lr_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 1,
) -> BaseEstimator:
    # Initialize GLM
    lr_model = LinearRegression(n_jobs=n_jobs)

    # train GLM
    t0 = time.time()
    lr_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return lr_model


def train_gp_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
) -> BaseEstimator:

    # define kernel function
    kernel = (
        ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(xtrain.shape[1]))
        + WhiteKernel()
    )

    # define GP model
    gp_model = GaussianProcessRegressor(
        kernel=kernel,  # kernel function (very important)
        normalize_y=True,  # good standard practice --> unless we have normalized before?
        random_state=123,  # reproducibility
        n_restarts_optimizer=0,  # good practice (avoids local minima)
    )

    # train GP Model
    t0 = time.time()
    gp_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return gp_model


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
        normalize=False,
        selection="random",
        verbose=verbose,
    )

    # train GLM
    t0 = time.time()
    gl_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return gl_model


def train_mlp_model(xtrain, ytrain, verbose=0, valid=0.2, tol=1e-5):

    # Initialize MLP
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 128, 128),
        activation="relu",
        solver="adam",
        batch_size=100,
        learning_rate="adaptive",
        max_iter=1_000,
        random_state=123,
        early_stopping=False,
        verbose=verbose,
        validation_fraction=valid,
        tol=tol,
    )

    # train GLM
    t0 = time.time()
    mlp_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return mlp_model


def train_rf_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 8,
    **kwargs: Tuple[int, str, bool, float],
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
        n_estimators=1_000,
        criterion="mse",
        n_jobs=n_jobs,
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


def train_mo_rf_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 8,
    mo_jobs: int = 8,
) -> BaseEstimator:
    # initialize baseline RF model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        criterion="mae",
        n_jobs=n_jobs,
        random_state=123,
        warm_start=False,
        verbose=verbose,
    )

    # initialize multioutput regressor
    mo_model = MultiOutputRegressor(estimator=rf_model, n_jobs=mo_jobs)
    # train RF model
    t0 = time.time()
    mo_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return mo_model
