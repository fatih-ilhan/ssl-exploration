from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from sklearn import preprocessing
from sklearn import decomposition

import math


# todo - speed up
class GMM(BaseEstimator, ClassifierMixin):

    def __init__(self, params):

        print('Hello World')

        self.max_steps = params['max_steps']
        self.PCA_dim = params['PCA_dim']
        self.stopping_epsilon = params['stopping_epsilon']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(n_components=self.PCA_dim, whiten=True)

    def fit(self, x, y):

        # Assuming there are as many mixture components as labels
        num_mixture_components = np.max(y) + 1
        num_samples = x.shape[0]
        num_features = self.PCA_dim

        # standardize input
        self.scaler.fit(x)
        x_std = self.scaler.transform(x)

        # apply PCA
        self.PCA.fit(x_std)
        x_std = self.PCA.transform(x_std)

        print("PCA explained variance_ratio:", self.PCA.explained_variance_ratio_.cumsum())

        y_hat = np.zeros(y.shape)
        is_labeled = [(y[i] != -1) for i in range(num_samples)]  # -1 means no label is given

        # Initialize parameters
        self.alpha = [0] * num_mixture_components
        self.mu = [np.zeros((num_features, 1))] * num_mixture_components
        self.sigma = [np.zeros((num_features, num_features))] * num_mixture_components

        for i in range(num_samples):
            if is_labeled[i]:
                self.alpha[y[i]] += 1
        self.alpha = self.alpha / np.sum(self.alpha)

        for j in range(num_mixture_components):
            self.mu[j] = np.mean(x_std[y==j], axis=0).reshape(num_features, 1)
            self.sigma[j] = np.cov(x_std[y==j].T)

        self.sigmai = [np.linalg.inv(self.sigma[j]) for j in range(num_mixture_components)]

        y_hat = y

        prevlikelihood = 0

        for step in range(self.max_steps):
            for i in range(num_samples):
                if not is_labeled[i]:
                    probs = [self.alpha[j] * self.compute_fnorm(x_std[i,:].reshape((-1,1)), self.mu[j], self.sigma[j], self.sigmai[j]) for j in range(num_mixture_components)]
                    y_hat[i] = np.argmax(probs)

            self.alpha = [0] * num_mixture_components
            for i in range(num_samples):
                if is_labeled[i]:
                    self.alpha[y[i]] += 1
            self.alpha = self.alpha / np.sum(self.alpha)

            for j in range(num_mixture_components):
                self.mu[j] = np.mean(x_std[y==j], axis=0).reshape(num_features,1)
                self.sigma[j] = np.cov(x_std[y==j].T)

            self.sigmai = [np.linalg.inv(self.sigma[j]) for j in range(num_mixture_components)]

            likelihood = self.compute_log_likelihood(x_std, self.alpha, self.mu, self.sigma, self.sigmai)
            print('Step: ' + str(step) + '      likelihood: ' + str(likelihood))

            if abs(likelihood - prevlikelihood) < self.stopping_epsilon:
                break

            prevlikelihood = likelihood


    def predict(self, x):

        x_std = self.scaler.transform(x)
        x_std = self.PCA.transform(x_std)

        N = x_std.shape[0]

        y = np.zeros((N,), dtype=int)

        for i in range(N):
            x_ = x_std[i, :].reshape(-1, 1)
            y[i] = int(self.compute_maximum_likelihood(x_, self.alpha, self.mu, self.sigma, self.sigmai))

        return y

    # Auxiliary Functions

    def compute_log_likelihood(self, x, alpha, mu, sigma, sigmai):
        num_mixture_components = len(alpha)
        num_samples = x.shape[0]
        num_features = x.shape[1]

        lik = 0

        for i in range(num_samples):
            x_ = x[i, :].reshape((num_features, 1))
            summand = 0
            for j in range(num_mixture_components):
                summand += alpha[j] * self.compute_fnorm(x_, mu[j], sigma[j], sigmai[j])
            lik += math.log(summand+1e-9)  # todo - summand is sometimes zero
        
        return lik

    def compute_fnorm(self, x, mu, sigma, sigmai):
        
        num_features = self.PCA_dim

        x = x.reshape((num_features, 1))
        mu = mu.reshape((num_features, 1))

        dT = (x - mu).T.reshape((1, num_features))
        dN = (x - mu).reshape((num_features, 1))
        #sigmai = np.linalg.inv(sigma)
        dist = np.matmul(dT, np.matmul(sigmai, dN))[0]
        det = np.linalg.det(sigma)
        if abs(det) == 0:
            deti = 1e6
        else:
            deti = np.linalg.det(sigmai)

        dist = np.clip(dist, 0, 1e4)
        
        deti = np.clip(deti, -1e6, 1e6)

        val = math.pow(2*math.pi, -num_features/2) * np.sqrt(deti) * math.exp(-0.5 * dist)

        return val

    def compute_maximum_likelihood(self, x, alpha, mu, sigma, sigmai):
        M = len(alpha)
        likelihoods = np.zeros((M,))

        for j in range(M):
            likelihoods[j] = alpha[j] * self.compute_fnorm(x, mu[j], sigma[j], sigmai[j])

        return np.argmax(likelihoods)

    def compute_pj(self, x, alpha, mu, sigma):

        M = len(alpha)
        p = np.zeros((M,))

        for j in range(M):
            p[j] = alpha[j] * self.compute_fnorm(x, mu[j], sigma[j])

        p = p / np.sum(p)

        return p
