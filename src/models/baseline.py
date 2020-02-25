from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, ConstantKernel, RBF, Matern, ExpSineSquared, RationalQuadratic)
from sklearn.linear_model import MultiTaskElasticNetCV, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Union, Tuple
from sklearn.ensemble import StackingRegressor
import numpy as np
import pandas as pd
import time

import warnings

warnings.filterwarnings("ignore")


def train_stack_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 1,
    order: Tuple[str, str] = ("rf", "lr"),
    lr_params: Optional[Dict]=None,
    rf_params: Optional[Dict]=None
) -> BaseEstimator:

    rf_estimator = RandomForestRegressor(
        n_estimators=1_000,
        criterion="mse",
        n_jobs=n_jobs,
        random_state=123,
        warm_start=False,
        verbose=verbose,
    )
    lr_estimator = LinearRegression()

    # Initialize GLM
    if order == ("rf", "lr"):
        stacking_regressor = StackingRegressor(
            estimators=[("Random Forest", rf_estimator)], final_estimator=lr_estimator
        )
    elif order == ("lr", "rf"):
        stacking_regressor = StackingRegressor(
            estimators=[("Linear Regression", lr_estimator)],
            final_estimator=rf_estimator,
        )
    else:
        raise ValueError()

    mo_regressor = MultiOutputRegressor(stacking_regressor, n_jobs=1)
    # train GLM
    t0 = time.time()
    mo_regressor.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return mo_regressor


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
    params,
) -> BaseEstimator:

    # define kernel function
    init_length_scale = np.ones(xtrain.shape[1])
    kernel = (
        ConstantKernel() * Matern(nu=2.5, length_scale=init_length_scale)
        + ConstantKernel() * RationalQuadratic(alpha=10, length_scale=1.0)
        + ConstantKernel() * RBF(length_scale=init_length_scale)
        + WhiteKernel(noise_level=0.01)
    )

    # define GP model
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        **params
    )

    # train GP Model
    t0 = time.time()
    gp_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if params['verbose'] > 0:
        print(f"Training time: {t1:.3f} secs.")
    return gp_model


def train_ridge_lr_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    verbose: int = 0,
    n_jobs: int = 1,
) -> BaseEstimator:
    # Initialize GLM
    lr_model = RidgeCV()

    # train GLM
    t0 = time.time()
    lr_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if verbose > 0:
        print(f"Training time: {t1:.3f} secs.")
    return lr_model


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


def train_mlp_model(xtrain, ytrain, params):

    # Initialize MLP
    mlp_model = MLPRegressor(
        **params
    )

    # train GLM
    t0 = time.time()
    mlp_model.fit(xtrain, ytrain)
    t1 = time.time() - t0
    if params['verbose'] > 0:
        print(f"Training time: {t1:.3f} secs.")
    return mlp_model


def train_rf_model(
    xtrain: Union[np.ndarray, pd.DataFrame],
    ytrain: Union[np.ndarray, pd.DataFrame],
    params
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
        **params
    )
    # train RF model
    t0 = time.time()
    rf_model.fit(xtrain, ytrain)
    t1 = time.time() - t0

    if params['verbose'] > 0:
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
        n_estimators=1_000,
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


def train_mo_gbt_model(
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
