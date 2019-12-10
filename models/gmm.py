from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from sklearn import preprocessing
from sklearn import decomposition

from random import seed
from random import randint

import math


class GMM(BaseEstimator, ClassifierMixin):

    #M = 0   # mixture of components
    D = 0   # Data dimensions

    max_steps = 1000

    def __init__(self, params):

        print('Hello World')

        #self.M = params['M']

        self.PCAdim = params['PCA']

        self.scaler = preprocessing.StandardScaler()
        self.PCA = decomposition.PCA(n_components=self.PCAdim)


    def fit(self, x, y):
        
        y = np.array(y)

        print(x.shape)
        print(y.shape)

        self.scaler.fit(x)
        x_std = self.scaler.transform(x)

        self.PCA.fit(x_std)
        x_std = self.PCA.transform(x_std)

        print(self.PCA.explained_variance_ratio_.cumsum())

        yhat = np.zeros(y.shape)

        print(x_std.shape)

        print(y)

        #self.D = x_std.shape[1]
        self.D = self.PCAdim

        # Assuming there are as many mixture components as labels
        M = np.max(y) + 1
        self.M = M

        N = x.shape[0]

        M = self.M
        D = self.D
        
        islabeled = [(y[i] != -1) for i in range(N)]

        #alpha = [(1/M) for j in range(M)]
        alpha = [0 for j in range(M)]
        mu = [np.zeros((D,1)) for j in range(M)]
        sigma = [np.zeros((D,D)) for j in range(M)]

        ahat = np.zeros((M,))
        for i in range(N):
            if islabeled[i]:
                ahat[y[i]] += 1
        ahat = ahat / np.sum(ahat)
        for j in range(M):
            alpha[j] = ahat[j]

        #seed(27)
        counts = np.zeros((M,1))
        for j in range(M):
            #i = randint(1,N)
            #mu[j] = x_std[i,:].T.reshape((-1,1))
            for i in range(N):
                if y[i] == j:
                    mu[j] += x_std[i,:].T.reshape((-1,1))
                    counts[j] += 1
        for j in range(M):
            mu[j] = mu[j] / counts[j]

        counts = np.zeros((M,1))
        for j in range(M):
            for i in range(N):
                x_ = x_std[i,:].T.reshape(-1,1)
                if y[i] == j:
                    sigma[j] = sigma[j] + np.matmul((x_ - mu[j]).reshape((D,1)), (x_ - mu[j]).T.reshape((1,D)))
                    counts[j] += 1
        for j in range(M):
            sigma[j] = 1 * sigma[j] / counts[j]

        prevLikelihood = 0

        for steps in range(self.max_steps):
            
            # Labeling of unlabeled data
            for i in range(N):
                if islabeled[i]:
                    yhat[i] = y[i]
                else:
                    x_ = x_std[i].T.reshape((D,1))
                    yhat[i] = self.maximumLikelihood(x_, alpha, mu, sigma)

            # Standard EM
            """ ahat = np.zeros((M,))
            for i in range(M):
                x_ = x_std[i,:].T.reshape((D,1))
                pj = self.getpj(x_, alpha, mu, sigma)
                ahat = ahat + pj
            ahat = ahat / N
            for j in range(M):
                alpha[j] = ahat[j] """

            ahat = np.zeros((M,))
            for i in range(N):
                if islabeled[i]:
                    ind = int(yhat[i])
                    ahat[ind] += 1
            ahat = ahat / np.sum(ahat)
            for j in range(M):
                alpha[j] = ahat[j]

            muhat = [np.zeros((D,1)) for j in range(M)]
            sigmahat = [np.zeros((D,D)) for j in range(M)]

            denom = np.zeros((M,))
            for i in range(N):
                x_ = x_std[i,:].T.reshape((D,1))
                pj = self.getpj(x_,alpha,mu,sigma)
                for j in range(M):
                    muhat[j] += pj[j] * x_
                    denom[j] += pj[j]
            for j in range(M):
                muhat[j] = muhat[j] / denom[j]

            denom = np.zeros((M,))
            for i in range(N):
                x_ = x_std[i,:].T.reshape((D,1))
                pj = self.getpj(x_, alpha, mu, sigma)
                for j in range(M):
                    muj = muhat[j]
                    d_ = (x_ - muj).reshape((D,1))
                    sigmahat[j] += pj[j] * np.matmul(d_, d_.T.reshape((1,D)))
                    denom[j] += pj[j]
            for j in range(M):
                sigmahat[j] = sigmahat[j] / denom[j]

            
            for j in range(M):
                mu[j] = muhat[j]
                sigma[j] = sigmahat[j]

            likelihood = self.logLikelihood(x_std, alpha, mu, sigma)

            print('Step: ' + str(steps) + '     log-likelihood: ' + str(likelihood))

            if abs(prevLikelihood - likelihood) < 0.000000001:
                break

            prevLikelihood = likelihood


        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma



    def predict(self, x):

        x_std = self.scaler.transform(x)
        x_std = self.PCA.transform(x_std)

        N = x_std.shape[0]

        y = np.zeros((N,), dtype=int)

        for i in range(N):
            x_ = x_std[i,:].reshape(-1,1)
            y[i] = int(self.maximumLikelihood(x_, self.alpha, self.mu, self.sigma))

        return y

        

    # Auxiliary Functions

    def logLikelihood(self, x, alpha, mu, sigma):
        M = len(alpha)
        N = x.shape[0]
        D = x.shape[1]

        lik = 0

        for i in range(N):
            x_ = x[i,:].reshape((D,1))
            summand = 0
            for j in range(M):
                summand += alpha[j] * self.fnorm(x_, mu[j], sigma[j])
            lik += math.log(summand)
        
        return lik


    def fnorm(self, x, mu, sigma):
        
        D = self.D

        x = x.reshape((D,1))
        mu = mu.reshape((D,1))

        dT = (x - mu).T.reshape((1,D))
        dN = (x - mu).reshape((D,1))
        sigmai = np.linalg.inv(sigma)
        dist = np.matmul(dT, np.matmul(sigmai, dN))[0]
        det = np.linalg.det(sigma)
        deti = 0
        if abs(det) == 0:
            deti = 1e6
        else:
            deti = np.linalg.det(sigmai)

        #print(det)
        #print(deti)

        #print(dist)

        dist = np.clip(dist, 0, 1e4)
        #print(np.linalg.det(sigma))
        #print(dist)
        
        deti = np.clip(deti, -1e6, 1e6)

        #val = math.pow(2*math.pi, -D/2) * math.pow(np.linalg.det(sigma), -1/2) * math.exp( -0.5 * dist)
        val = math.pow(2*math.pi, -D/2) * np.sqrt(deti) * math.exp( -0.5 * dist)

        #print(val)

        return val


    def maximumLikelihood(self, x, alpha, mu, sigma):
        M = len(alpha)
        likelihoods = np.zeros((M,))

        for j in range(M):
            likelihoods[j] = alpha[j] * self.fnorm(x, mu[j], sigma[j])

        return np.argmax(likelihoods)

    def getpj(self, x, alpha, mu, sigma):

        M = len(alpha)
        p = np.zeros((M,))

        for j in range(M):
            p[j] = alpha[j] * self.fnorm(x, mu[j], sigma[j])

        p = p / np.sum(p)

        return p
