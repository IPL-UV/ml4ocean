"""Author: Juan Emmanuel Johnson
Implementations of GP algorithms from the GPy library. These are all wrapped as 
scikit-learn estimators for convenience."""
import GPy
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, RegressorMixin


class SGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        n_inducing=10,
        random_state=123,
        max_iters=200,
        optimizer="scg",
        verbose=None,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.rng = np.random.RandomState(random_state)
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # Get inducing points
        z = self.rng.uniform(X.min(), X.max(), (self.n_inducing, d_dimensions))

        # GP Model
        gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

        # FITC for inference
        gp_model.inference_method = GPy.inference.latent_function_inference.FITC()

        # Optimize
        # gp_model.inducing_inputs.fix()
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model

    def predict(self, X, return_std=False, noiseless=True):

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        if return_std:
            return mean, np.sqrt(var)
        else:
            return mean


class SVGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        x_variance=None,
        kernel=None,
        n_inducing=10,
        random_state=123,
        max_iters=200,
        optimizer="scg",
        verbose=None,
        n_restarts=0,
    ):
        self.kernel = kernel
        self.x_variance = x_variance
        self.n_inducing = n_inducing
        self.rng = np.random.RandomState(random_state)
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_restarts = n_restarts

    def fit(self, X, y):
        # print(X)
        # print(y)
        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # Convert covariance into matrix
        if self.x_variance is not None:
            assert self.x_variance.shape[1] == d_dimensions
            x_variance = np.array(self.x_variance).reshape(1, -1)
            x_variance = np.tile(x_variance, (n_samples, 1))
        else:
            x_variance = None

        # Get inducing points
        z = self.rng.uniform(X.min(), X.max(), (self.n_inducing, d_dimensions))

        # Kernel matrix
        gp_model = GPy.models.SparseGPRegression(
            X, y, kernel=self.kernel, Z=z, X_variance=x_variance
        )

        # Optimize
        if self.n_restarts > 0:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts, robust=True, verbose=self.verbose
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model

    def predict(self, X, return_std=False, noiseless=True):

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        if return_std:
            return mean, np.sqrt(var)
        else:
            return mean

