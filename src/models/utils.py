import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.validation import check_is_fitted


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
            print("Dont return std")
            preds = Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_helper)(
                    e,
                    "predict",
                    X,
                    return_std=False,
                    # e, "predict", X, return_std=return_std, noiseless=noiseless
                )
                for e in self.estimators_
            )

            return np.asarray(preds).T
        else:
            print("return std")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_helper)(
                    # e, "predict", X, return_std=return_std, noiseless=noiseless
                    e,
                    "predict",
                    X,
                    return_std=True,
                )
                for e in self.estimators_
            )
            preds, stds = tuple(zip(*results))
            # print(preds.shape, stds.shape)
            return np.asarray(preds).T, np.asarray(stds).T
