from sklearn.base import BaseEstimator, ClassifierMixin

import math
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from sklearn.utils import shuffle

import config


class GMM(BaseEstimator, ClassifierMixin):

    def __init__(self, params):

        print('Hello World')

        self.max_steps = params['max_steps']
        self.PCA_dim = params['PCA_dim']
        self.stopping_epsilon = params['stopping_epsilon']
        self.standardize_flag = params['standardize_flag']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(whiten=True)

        self.folds = 4

        self.best_val_score = 0

    def fit(self, x, y):

        # Assuming there are as many mixture components as labels
        num_mixture_components = np.max(y) + 1
        num_samples = x.shape[0]
        num_features = self.PCA_dim

        x,y = shuffle(x, y)

        # standardize input
        if self.standardize_flag:
            self.scaler.fit(x)
            x_std = self.scaler.transform(x)
        else:
            x_std = x.copy()

        # apply PCA
        if config.PCA_VAR_THR < 1:
            if self.PCA.n_components is None:
                self.PCA.n_components = x.shape[1]
                self.PCA.fit(x)
                n_components = np.where(self.PCA.explained_variance_ratio_.cumsum() > config.PCA_VAR_THR)[0][0]
                self.PCA = decomposition.PCA(n_components=n_components, whiten=True)
            self.PCA.fit(x)
            x = self.PCA.transform(x)

        y_hat = np.zeros(y.shape)
        is_labeled = [(y[i] != -1) for i in range(num_samples)]  # -1 means no label is given

        # Initial Fit
        self.alpha, self.mu, self.sigma, self.sigmai = self.fitGaussian(x_std, y, num_features, num_mixture_components)

        y_hat = y
        for i in range(num_samples):
            if not is_labeled[i]:
                probs = [self.alpha[j] * self.compute_fnorm(x_std[i,:].reshape((-1,1)), self.mu[j], self.sigma[j], self.sigmai[j]) for j in range(num_mixture_components)]
                y_hat[i] = np.argmax(probs)

        val_scores, validation_accuracies = self.validate(x_std, y_hat, num_features, num_mixture_components)
        self.best_val_score = np.sum(validation_accuracies) / self.folds

        prevlikelihood = 0
        #prev_vals = [-1e20 for j in range(self.folds)]
        prev_vals = val_scores
        prev_accuracies = validation_accuracies

        #return

        for step in range(self.max_steps):
            for i in range(num_samples):
                if not is_labeled[i]:
                    probs = [self.alpha[j] * self.compute_fnorm(x_std[i,:].reshape((-1,1)), self.mu[j], self.sigma[j], self.sigmai[j]) for j in range(num_mixture_components)]
                    y_hat[i] = np.argmax(probs)

            val_scores, validation_accuracies = self.validate(x_std, y_hat, num_features, num_mixture_components)

            #print(validation_accuracies)

            end_training = False
            for j in range(self.folds):
                end_training = end_training or (prev_vals[j] > val_scores[j])
            #end_training = (np.sum(prev_accuracies) > np.sum(validation_accuracies))

            prev_vals = val_scores
            prev_accuracies = validation_accuracies

            if end_training:
                #self.alpha, self.mu, self.sigma, self.sigmai = self.fitGaussian(x_std, y_hat, num_features, num_mixture_components)
                break

            self.best_val_score = np.sum(validation_accuracies) / self.folds
                
            self.alpha, self.mu, self.sigma, self.sigmai = self.fitGaussian(x_std, y_hat, num_features, num_mixture_components)

            likelihood = self.compute_log_likelihood(x_std, self.alpha, self.mu, self.sigma, self.sigmai)
            print('Step: ' + str(step) + '      likelihood: ' + str(likelihood))

            if abs(likelihood - prevlikelihood) < self.stopping_epsilon:
                break

            prevlikelihood = likelihood

    def predict(self, x):

        if self.standardize_flag:
            x_std = self.scaler.transform(x)
        else:
            x_std = x.copy()

        if config.PCA_VAR_THR < 1:
            x_std = self.PCA.transform(x_std)

        N = x_std.shape[0]

        y = np.zeros((N,), dtype=int)

        for i in range(N):
            x_ = x_std[i, :].reshape(-1, 1)
            y[i] = int(self.compute_maximum_likelihood(x_, self.alpha, self.mu, self.sigma, self.sigmai))

        return y

    # Auxiliary Functions

    def fitGaussian(self, x, y, num_features, num_mixture_components):
        alpha = [0] * num_mixture_components
        mu = [np.zeros((num_features, 1))] * num_mixture_components
        sigma = [np.zeros((num_features, num_features))] * num_mixture_components
        for i in range(x.shape[0]):
            #if is_labeled[i]:
            #    self.alpha[y[i]] += 1
            alpha[y[i]] += 1
        alpha = alpha / np.sum(alpha)

        for j in range(num_mixture_components):
            mu[j] = np.mean(x[y==j], axis=0).reshape(num_features,1)
            sigma[j] = np.cov(x[y==j].T)

        sigmai = [np.linalg.inv(sigma[j]) for j in range(num_mixture_components)]

        return alpha, mu, sigma, sigmai

    def validate(self, x, y, num_features, num_mixture_components):
        fold_size = math.floor(x.shape[0] / self.folds)
        masks = np.zeros((self.folds, x.shape[0]), dtype=bool)
        for j in range(self.folds):
            masks[j, j * fold_size : (j+1) * fold_size - 1] = True

        val_scores = [0 for j in range(self.folds)]

        x_train = [x[~masks[j]] for j in range(self.folds)]
        y_train = [y[~masks[j]] for j in range(self.folds)]

        x_val = [x[masks[j]] for j in range(self.folds)]
        y_val = [y[masks[j]] for j in range(self.folds)]

        validation_accuracies = [0 for j in range(self.folds)]
        validation_hits = [0 for j in range(self.folds)]
        validation_counts = [0 for j in range(self.folds)]

        for j in range(self.folds):
            alpha, mu, sigma, sigmai = self.fitGaussian(x_train[j], y_train[j], num_features, num_mixture_components)
            lik = self.compute_log_likelihood(x_val[j], alpha, mu, sigma, sigmai)
            val_scores[j] = lik

            for i in range(x_val[j].shape[0]):
                if y_val[j][i] != -1:
                    probs = [alpha[m] * self.compute_fnorm(x_val[j][i,:].reshape((-1,1)), mu[m], sigma[m], sigmai[m]) for m in range(num_mixture_components)]
                    prediction = np.argmax(probs)
                    if prediction == y_val[j][i]:
                        validation_hits[j] += 1
                    validation_counts[j] += 1

        for j in range(self.folds):
            validation_accuracies[j] = validation_hits[j] / validation_counts[j]

        return val_scores, validation_accuracies

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
