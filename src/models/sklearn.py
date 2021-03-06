# import numpy as np


# def regression_model(model):
#     class SKLWrapperRegression(object):
#         def __init__(self, is_test=False, seed=0):
#             self.model = model

#         def fit(self, X, Y):
#             self.model.fit(X, Y.flatten())
#             self.std = np.std(self.model.predict(X) - Y.flatten())

#         def predict(self, Xs):
#             pred_mean = self.model.predict(Xs)[:, None]
#             return pred_mean, np.ones_like(pred_mean) * (self.std + 1e-6) ** 2

#         def sample(self, Xs, num_samples):
#             m, v = self.predict(Xs)
#             N, D = np.shape(m)
#             m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
#             return m + np.random.randn(num_samples, N, D) * (v ** 0.5)

#     return SKLWrapperRegression
