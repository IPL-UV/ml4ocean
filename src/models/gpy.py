# """Author: Juan Emmanuel Johnson
# Implementations of GP algorithms from the GPy library. These are all wrapped as
# scikit-learn estimators for convenience."""
# import GPy
# import numpy as np
# from sklearn.utils import check_random_state
# from sklearn.utils.validation import check_X_y, check_array
# from sklearn.base import BaseEstimator, RegressorMixin
# from scipy.cluster.vq import kmeans2


# class SGP(BaseEstimator, RegressorMixin):
#     def __init__(
#         self,
#         kernel=None,
#         n_inducing=10,
#         random_state=123,
#         max_iters=200,
#         optimizer="scg",
#         verbose=None,
#     ):
#         self.kernel = kernel
#         self.n_inducing = n_inducing
#         self.rng = np.random.RandomState(random_state)
#         self.max_iters = max_iters
#         self.optimizer = optimizer
#         self.verbose = verbose

#     def fit(self, X, y):
#         X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

#         n_samples, d_dimensions = X.shape

#         # default Kernel Function
#         if self.kernel is None:
#             self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

#         # Get inducing points
#         z = self.rng.uniform(X.min(), X.max(), (self.n_inducing, d_dimensions))

#         # GP Model
#         gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

#         # FITC for inference
#         gp_model.inference_method = GPy.inference.latent_function_inference.FITC()

#         # Optimize
#         # gp_model.inducing_inputs.fix()
#         gp_model.optimize(
#             self.optimizer, messages=self.verbose, max_iters=self.max_iters
#         )

#         self.gp_model = gp_model

#         return self

#     def display_model(self):
#         return self.gp_model

#     def predict(self, X, return_std=False, noiseless=True):

#         if noiseless:
#             mean, var = self.gp_model.predict_noiseless(X)
#         else:
#             mean, var = self.gp_model.predict(X)

#         if return_std:
#             return mean, np.sqrt(var)
#         else:
#             return mean


# class SVGP(BaseEstimator, RegressorMixin):
#     def __init__(
#         self,
#         x_variance=None,
#         kernel=None,
#         n_inducing=10,
#         random_state=123,
#         max_iters=200,
#         optimizer="scg",
#         verbose=None,
#         n_restarts=0,
#     ):
#         self.kernel = kernel
#         self.x_variance = x_variance
#         self.n_inducing = int(n_inducing)
#         self.random_state = random_state
#         self.max_iters = max_iters
#         self.optimizer = optimizer
#         self.verbose = verbose
#         self.n_restarts = n_restarts

#     def fit(self, X, y):

#         self.rng = np.random.RandomState(self.random_state)
#         # print(X)
#         if np.ndim(y) < 2:
#             y = y.reshape(-1, 1)
#         n_samples, d_dimensions = X.shape

#         self.X_train = X
#         self.y_train = y

#         # default Kernel Function
#         if self.kernel is None:
#             self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

#         # Convert covariance into matrix
#         if self.x_variance is not None:
#             assert self.x_variance.shape[1] == d_dimensions
#             x_variance = np.array(self.x_variance).reshape(1, -1)
#             x_variance = np.tile(x_variance, (n_samples, 1))
#         else:
#             x_variance = None

#         # Get inducing points
#         # print(self.n_inducing)
#         z = kmeans2(X, self.n_inducing, minit="points")[0]

#         # Kernel matrix
#         gp_model = GPy.models.SparseGPRegression(
#             X, y, kernel=self.kernel, Z=z, X_variance=x_variance
#         )

#         # Optimize
#         if self.n_restarts > 0:
#             gp_model.optimize_restarts(
#                 num_restarts=self.n_restarts, robust=True, verbose=self.verbose
#             )
#         else:
#             gp_model.optimize(
#                 self.optimizer, messages=self.verbose, max_iters=self.max_iters
#             )
#         self.gp_model = gp_model

#         return self

#     def display_model(self):
#         return self.gp_model

#     def predict(self, X, return_std=False, batch_size=1000):

#         if X.shape[0] > batch_size:
#             mean, var = self.batch_predict(X, batch_size)
#         else:
#             mean, var = self.gp_model.predict_noiseless(X)

#         if return_std:
#             return mean.squeeze(), np.sqrt(var).squeeze()
#         else:
#             return mean.squeeze()

#     def predict_y(self, X, return_std=False, batch_size=1000):

#         mean, var = self.gp_model.predict(X)

#         if return_std:
#             return mean.squeeze(), np.sqrt(var).squeeze()
#         else:
#             return mean.squeeze()

#     def batch_predict(self, Xs, batchsize=1000):
#         ms, vs = [], []
#         n = max(len(Xs) / batchsize, 1)  # predict in small batches
#         for xs in np.array_split(Xs, n):
#             m, v = self.gp_model.predict_noiseless(xs)

#             ms.append(m)
#             vs.append(v)

#         return (
#             np.concatenate(ms, 1),
#             np.concatenate(vs, 1),
#         )  # num_posterior_samples, N_test, D_y
