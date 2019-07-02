import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import gen_batches


class PCATargetTransform(TransformedTargetRegressor):
    def __init__(self, regressor=None, n_components=10):

        # PCA transform
        transformer = PCA(n_components=n_components)

        # Initialize Target Transform with PCA
        super(PCATargetTransform, self).__init__(
            regressor=regressor,
            transformer=transformer,
            func=None,
            inverse_func=None,
            check_inverse=False,
        )
        self.n_components = n_components

    def predict(self, X, return_std=False):
        """Predict using the base regressor, applying inverse.
        The regressor is used to predict and the ``inverse_func`` or
        ``inverse_transform`` is applied before returning the prediction.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        y_hat : array, shape = (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "regressor_")

        # Return standard deviation
        if return_std:
            try:
                pred, std = self.regressor_.predict(X, return_std=return_std)
            except:
                pred = self.regressor_.predict(X)
        else:
            pred = self.regressor_.predict(X)

        # Inverse Transformation
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)

        # Checks
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)

        if return_std:
            return pred_trans, std
        else:
            return pred_trans


class MultiTaskGP(MultiOutputRegressor):
    def predict(self, X, return_std=False):
        """Predict multi-output variable using a model
         trained for each target variable.
        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.
        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, "estimators_")
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement" " a predict method")

        X = check_array(X, accept_sparse=True)

        if not return_std:
            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_helper)(e, "predict", X, return_std=False)
                for e in self.estimators_
            )

            return np.asarray(preds).T
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_helper)(e, "predict", X, return_std=True)
                for e in self.estimators_
            )
            preds, stds = tuple(zip(*results))
            # print(preds.shape, stds.shape)
            return np.asarray(preds).T, np.asarray(stds).T


# def batch_predict(Xs, model, batchsize=1000):
#     ms, vs = [], []
#     n = max(len(Xs) / batchsize, 1)  # predict in small batches
#     for xs in np.array_split(Xs, n):
#         m, v = model.predict(xs)

#         ms.append(m)
#         vs.append(v)

#     return (
#         np.concatenate(ms, 1),
#         np.concatenate(vs, 1),
#     )  # num_posterior_samples, N_test, D_y


def batch_predict(X, model, n_jobs=2, batch_size=100, verbose=1):

    X = check_array(X, accept_sparse=False)

    if n_jobs == 1:
        ms = []
        for (start, end) in gen_batches(X.shape[0], batch_size=batch_size):
            m = model.predict(X)
            ms.append(m)
        return np.concatenate(ms, 1)
    elif n_jobs > 1:
        # print("return std")
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(parallel_helper)(model, "predict", X[start:end])
            for (start, end) in gen_batches(X.shape[0], batch_size=batch_size)
        )
        preds = tuple(zip(*results))

        return np.asarray(preds).T
    else:
        raise ValueError(f"n_jobs '{n_jobs}' should greater than or equal to 1.")


def batch_predict_bayesian(X, model, n_jobs=2, batch_size=100, verbose=1):

    X = check_array(X, accept_sparse=False)

    if n_jobs == 1:
        ms, vs = [], []
        for (start, end) in gen_batches(X.shape[0], batch_size=batch_size):
            m, v = model.predict(X, return_std=True)
            ms.append(m)
            vs.append(v)
        return np.concatenate(ms, 1), np.concatenate(vs, 1)
    elif n_jobs > 1:
        # print("return std")
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(parallel_helper)(model, "predict", X[start:end], return_std=True)
            for (start, end) in gen_batches(X.shape[0], batch_size=batch_size)
        )
        preds, stds = tuple(zip(*results))

        return np.asarray(preds).T, np.asarray(stds).T
    else:
        raise ValueError(f"n_jobs '{n_jobs}' should greater than or equal to 1.")


#     # Perform parallel predictions using joblib
#     results = Parallel(n_jobs=n_jobs, verbose=verbose)(
#         delayed(gp_model_predictions)(
#             gp_model, x[start:end],
#             return_variance=return_variance,)
#         for (start, end) in generate_batches(x.shape[0], batch_size=batch_size)

#     )

#     # Aggregate results (predictions, derivatives, variances)
#     predictions, variance = tuple(zip(*results))
#     predictions = np.hstack(predictions)
#     variance = np.hstack(variance)


#     if return_variance and return_derivative:
#         return predictions[:, np.newaxis], derivative, variance
#     elif return_variance:
#         return predictions[:, np.newaxis], variance
#     elif return_derivative:
#         return predictions[:, np.newaxis], derivative
#     else:
# return predictions[:, np.newaxis]

