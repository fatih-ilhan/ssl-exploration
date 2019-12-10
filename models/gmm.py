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
        self.PCA = decomposition.PCA(n_components=self.PCA_dim)

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
        alpha = [0] * num_mixture_components
        mu = [np.zeros((num_features, 1))] * num_mixture_components
        sigma = [np.zeros((num_features, num_features))] * num_mixture_components

        a_hat = np.zeros((num_mixture_components,))
        for i in range(num_samples):
            if is_labeled[i]:
                a_hat[y[i]] += 1
        a_hat = a_hat / np.sum(a_hat)

        for j in range(num_mixture_components):
            alpha[j] = a_hat[j]

        # seed(27)
        counts = np.zeros((num_mixture_components, 1))
        for j in range(num_mixture_components):
            for i in range(num_samples):
                if y[i] == j:
                    mu[j] += x_std[i, :].T.reshape((-1, 1))
                    counts[j] += 1
        for j in range(num_mixture_components):
            mu[j] = mu[j] / counts[j]

        counts = np.zeros((num_mixture_components, 1))
        for j in range(num_mixture_components):
            for i in range(num_samples):
                x_ = x_std[i, :].T.reshape(-1, 1)
                if y[i] == j:
                    sigma[j] = sigma[j] + np.matmul((x_ - mu[j]).reshape((num_features, 1)), (x_ - mu[j]).T.reshape((1, num_features)))
                    counts[j] += 1
        for j in range(num_mixture_components):
            sigma[j] = 1 * sigma[j] / counts[j]

        prev_likelihood = 0

        for steps in range(self.max_steps):
            # Labeling of unlabeled data
            for i in range(num_samples):
                if is_labeled[i]:
                    y_hat[i] = y[i]
                else:
                    x_ = x_std[i].T.reshape((num_features, 1))
                    y_hat[i] = self.compute_maximum_likelihood(x_, alpha, mu, sigma)

            # Standard EM
            """ ahat = np.zeros((M,))
            for i in range(M):
                x_ = x_std[i,:].T.reshape((D,1))
                pj = self.getpj(x_, alpha, mu, sigma)
                ahat = ahat + pj
            ahat = ahat / N
            for j in range(M):
                alpha[j] = ahat[j]
            """

            a_hat = np.zeros((num_mixture_components,))
            for i in range(num_samples):
                if is_labeled[i]:
                    ind = int(y_hat[i])
                    a_hat[ind] += 1
            a_hat = a_hat / np.sum(a_hat)
            for j in range(num_mixture_components):
                alpha[j] = a_hat[j]

            mu_hat = [np.zeros((num_features, 1))] * num_mixture_components
            sigma_hat = [np.zeros((num_features, num_features))] * num_mixture_components

            denom = np.zeros((num_mixture_components,))
            for i in range(num_samples):
                x_ = x_std[i, :].T.reshape((num_features, 1))
                pj = self.compute_pj(x_, alpha, mu, sigma)
                for j in range(num_mixture_components):
                    mu_hat[j] += pj[j] * x_
                    denom[j] += pj[j]
            for j in range(num_mixture_components):
                mu_hat[j] = mu_hat[j] / denom[j]

            denom = np.zeros((num_mixture_components,))
            for i in range(num_samples):
                x_ = x_std[i, :].T.reshape((num_features, 1))
                pj = self.compute_pj(x_, alpha, mu, sigma)
                for j in range(num_mixture_components):
                    muj = mu_hat[j]
                    d_ = (x_ - muj).reshape((num_features, 1))
                    sigma_hat[j] += pj[j] * np.matmul(d_, d_.T.reshape((1, num_features)))
                    denom[j] += pj[j]
            for j in range(num_mixture_components):
                sigma_hat[j] = sigma_hat[j] / denom[j]

            for j in range(num_mixture_components):
                mu[j] = mu_hat[j]
                sigma[j] = sigma_hat[j]

            likelihood = self.compute_log_likelihood(x_std, alpha, mu, sigma)

            print('Step: ' + str(steps) + '     log-likelihood: ' + str(likelihood))

            if abs(prev_likelihood - likelihood) < self.stopping_epsilon:
                break

            prev_likelihood = likelihood

        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma

    def predict(self, x):

        x_std = self.scaler.transform(x)
        x_std = self.PCA.transform(x_std)

        N = x_std.shape[0]

        y = np.zeros((N,), dtype=int)

        for i in range(N):
            x_ = x_std[i, :].reshape(-1, 1)
            y[i] = int(self.compute_maximum_likelihood(x_, self.alpha, self.mu, self.sigma))

        return y

    # Auxiliary Functions

    def compute_log_likelihood(self, x, alpha, mu, sigma):
        num_mixture_components = len(alpha)
        num_samples = x.shape[0]
        num_features = x.shape[1]

        lik = 0

        for i in range(num_samples):
            x_ = x[i, :].reshape((num_features, 1))
            summand = 0
            for j in range(num_mixture_components):
                summand += alpha[j] * self.compute_fnorm(x_, mu[j], sigma[j])
            lik += math.log(summand+1e-9)  # todo - summand is sometimes zero
        
        return lik

    def compute_fnorm(self, x, mu, sigma):
        
        num_features = self.PCA_dim

        x = x.reshape((num_features, 1))
        mu = mu.reshape((num_features, 1))

        dT = (x - mu).T.reshape((1, num_features))
        dN = (x - mu).reshape((num_features, 1))
        sigmai = np.linalg.inv(sigma)
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

    def compute_maximum_likelihood(self, x, alpha, mu, sigma):
        M = len(alpha)
        likelihoods = np.zeros((M,))

        for j in range(M):
            likelihoods[j] = alpha[j] * self.compute_fnorm(x, mu[j], sigma[j])

        return np.argmax(likelihoods)

    def compute_pj(self, x, alpha, mu, sigma):

        M = len(alpha)
        p = np.zeros((M,))

        for j in range(M):
            p[j] = alpha[j] * self.compute_fnorm(x, mu[j], sigma[j])

        p = p / np.sum(p)

        return p
