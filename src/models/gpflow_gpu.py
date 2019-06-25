import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

class SVGP(object):
    def __init__(self, 
        num_inducing=2, 
        iterations=1, 
        small_iterations=1, 
        adam_lr=0.01, 
        gamma=0.1,
        minibatch_size=100,
        initial_likelihood_var=0.01,
        seed=0):

        self.num_inducing = num_inducing
        self.iterations = iterations 
        self.small_iterations = small_iterations 
        self.adam_lr = adam_lr 
        self.gamma = gamma 
        self.minibatch_size = minibatch_size 
        self.initial_likelihood_var = initial_likelihood_var 
        self.seed = seed
        self.model = None

    def fit(self, X, Y):
        if X.shape[0] > self.num_inducing:
            Z = kmeans2(X, self.num_inducing, minit='points')[0]
        else:
            # pad with random values
            Z = np.concatenate([X, np.random.randn(self.num_inducing - X.shape[0], X.shape[1])], 0)

        # make model if necessary
        if not self.model:
            kern = gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5)
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = self.initial_likelihood_var
            mb_size = self.minibatch_size if X.shape[0] > self.minibatch_size else None
            self.model = gpflow.models.SVGP(X, Y, kern, lik, feat=Z, minibatch_size=mb_size)

            var_list = [[self.model.q_mu, self.model.q_sqrt]]
            self.model.q_mu.set_trainable(False)
            self.model.q_sqrt.set_trainable(False)
            self.ng = gpflow.train.NatGradOptimizer(gamma=self.gamma).make_optimize_tensor(self.model, var_list=var_list)
            self.adam = gpflow.train.AdamOptimizer(self.adam_lr).make_optimize_tensor(self.model)

            self.sess = self.model.enquire_session()

            iters = self.iterations

        else:
            iters = self.small_iterations

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        self.model.feature.Z.assign(Z, session=self.sess)

        self.model.q_mu.assign(np.zeros((self.num_inducing, Y.shape[1])), session=self.sess)
        self.model.q_sqrt.assign(np.tile(np.eye(self.num_inducing)[None], [Y.shape[1], 1, 1]), session=self.sess)


        for _ in range(iters):
            self.sess.run(self.ng)
            self.sess.run(self.adam)
        self.model.anchor(session=self.sess)

    def predict(self, Xs, return_std=False):

        mean, var = self.model.predict_f(Xs, session=self.sess)

        if return_std:
            return mean, np.sqrt(var)
        else:
            return mean

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)

class MOSVGP(object):
    def __init__(self, 
        num_inducing=2, 
        iterations=1, 
        small_iterations=1, 
        adam_lr=0.01, 
        gamma=0.1,
        minibatch_size=100,
        initial_likelihood_var=0.01,
        seed=0):

        self.num_inducing = num_inducing
        self.iterations = iterations 
        self.small_iterations = small_iterations 
        self.adam_lr = adam_lr 
        self.gamma = gamma 
        self.minibatch_size = minibatch_size 
        self.initial_likelihood_var = initial_likelihood_var 
        self.seed = seed
        self.model = None

    def fit(self, X, Y):

        n_samples, d_dimensions = X.shape 
        n_samples, p_outputs = Y.shape 

        if X.shape[0] > self.num_inducing:
            Zs = [kmeans2(X, self.num_inducing, minit='points')[0] for _ in range(p_outputs)]
        else:
            # pad with random values
            Zs = [np.concatenate([X, np.random.randn(self.num_inducing - X.shape[0], X.shape[1])], 0) for _ in range(p_outputs)]
        feature_list = [gpflow.features.InducingPoints(Z) for Z in Zs]
        feature = mf.SeparateIndependentMof(feature_list)
        # make model if necessary
        if not self.model:

            # Define Kernel
            kern_list = [gpflow.kernels.RBF(X.shape[1], lengthscales=float(X.shape[1])**0.5) for _ in range(p_outputs)]
            kern = mk.SeparateIndependentMok(kern_list)

            # Define Likelihood
            lik = gpflow.likelihoods.Gaussian()
            lik.variance = self.initial_likelihood_var
            mb_size = self.minibatch_size if X.shape[0] > self.minibatch_size else None
            self.model = gpflow.models.SVGP(X, Y, kern, lik, feat=feature, minibatch_size=mb_size)

            var_list = [[self.model.q_mu, self.model.q_sqrt]]
            self.model.q_mu.set_trainable(False)
            self.model.q_sqrt.set_trainable(False)
            self.ng = gpflow.train.NatGradOptimizer(gamma=self.gamma).make_optimize_tensor(self.model, var_list=var_list)
            self.adam = gpflow.train.AdamOptimizer(self.adam_lr).make_optimize_tensor(self.model)

            self.sess = self.model.enquire_session()

            iters = self.iterations

        else:
            iters = self.small_iterations

        # we might have new data
        self.model.X.assign(X, session=self.sess)
        self.model.Y.assign(Y, session=self.sess)
        # self.model.feature.Z.assign(feature, session=self.sess)

        self.model.q_mu.assign(np.zeros((self.num_inducing, Y.shape[1])), session=self.sess)
        self.model.q_sqrt.assign(np.tile(np.eye(self.num_inducing)[None], [Y.shape[1], 1, 1]), session=self.sess)


        for _ in range(iters):
            self.sess.run(self.ng)
            self.sess.run(self.adam)
        self.model.anchor(session=self.sess)

    def predict(self, Xs, return_std=False):

        mean, var = self.model.predict_f(Xs, session=self.sess)

        if return_std:
            return mean, np.sqrt(var)
        else:
            return mean

    def sample(self, Xs, num_samples):
        m, v = self.predict(Xs)
        N, D = np.shape(m)
        m, v = np.expand_dims(m, 0), np.expand_dims(v, 0)
        return m + np.random.randn(num_samples, N, D) * (v ** 0.5)