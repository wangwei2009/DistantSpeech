import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture


class Gaussian(object):
    def __init__(self, feature=4):
        self.feature = feature
        self.mean = np.random.rand(feature)
        self.var = np.eye(feature)

    def get_prob(self, x, diag=1e-6):
        p = self.get_log_prob(x, diag=diag)
        return np.exp(p)

    def get_log_prob(self, x, diag=1e-6):
        """compute log prob per sample

        Args:
            x (ndarray): one sample of observations, [n_feature,] or [1, n_feature]
            diag ([type], optional): [diagnal loading]. Defaults to 1e-6.

        Returns:
            [type]: [description]
        """

        if len(x.shape) == 1:
            x = x.reshape(-1, len(x))                  # in case of single sample
        assert  x.shape[1] == self.feature
        mean = self.mean[np.newaxis, :]
        var_ = self.var + np.eye((self.feature))*diag
        p = -0.5*(self.feature*np.log(2*np.pi) + np.log(np.linalg.det(var_)) + ((x-mean) @ np.linalg.inv(var_) @ (x-mean).transpose()))

        return p

    def update(self, x):
        """[summary]

        Args:
            x (ndarray): observation, [n_samples, n_features]
        """
        n_sample = x.shape[0]
        self.mean = np.squeeze(np.mean(x, axis=0))

        mean = self.mean[np.newaxis, :]
        for n in range(x.shape[0]):
            xn = x[n:n+1, :]
            self.var = self.var + ((xn-mean).transpose() @ (xn-mean))
        self.var = self.var/n_sample

        prob = 0.0
        for n in range(x.shape[0]):
            prob = prob + self.get_prob(x[n:n+1, :])

        return prob

    def get_param(self):
        pass

    def set_parm(self, mean, var):
        self.mean = mean
        self.var = var

class GaussianMixture(object):
    def __init__(self, n_components=2, n_features=2) -> None:
        self.n_features = n_features
        self.n_components = n_components
        self.mean_ = np.random.rand(n_components, n_features)
        self.covar_ = np.random.rand(n_components, n_features, n_features)
        self.weights_ = np.random.rand(n_components)

        self.gmms = [Gaussian(n_features) for n in range(n_components)]
        for n in range(n_components):
            # self.gmms.append(Gaussian(n_features))
            print(self.gmms[n])

    def get_prob(self, x):
        prob = 0.0
        for z in range(self.n_components):
            prob = prob + self.weights_[z] * self.gmms[z].get_prob(x)
        return prob

    def get_log_prob(self, x):
        prob = 0.0
        for z in range(self.n_components):
            prob = prob + np.exp(np.log(self.weights_[z]) + self.gmms[z].get_log_prob(x))
        # prob = np.maximum(prob)
        return np.log(prob)

    def score_samples(self, x):
        prob = np.zeros((x.shape[0],))
        for n in range(x.shape[0]):
            # print("x[{}, :]={}".format(n, x[n, :]))
            prob[n] = self.get_log_prob(x[n, :])
        return prob

    def get_params(self):
        for z in range(self.n_components):
            self.mean_[z] = self.gmms[z].mean
            self.covar_[z, :, :] = self.gmms[z].var

        return self.mean_, self.covar_

    def compute_posterior(self, x):
        """comppute posterior probability of latent variable given model params and obseration

        Args:
            x (ndarray): observation, [n_samples, n_features]
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]               # for one sample
        assert x.shape[1] == self.n_features
        n_samples, n_feature = x.shape
        gamma = np.zeros((n_samples, self.n_components))
        for n in range(n_samples):
            prob_den = self.get_prob(x[n, :])
            # print(prob_den)
            for z in range(self.n_components):
                gamma[n, z] = self.weights_[z] * self.gmms[z].get_prob(x[n,:])/prob_den

        return gamma

    def update_params(self, gamma, x):
        """update gmms parameter

        Args:
            gamma (ndarray): posterior probability, [n_samples, n_components]
            x (ndarray): observation, [n_samples, n_features]
        """
        N_z = np.sum(gamma, axis=0)
        n_samples = x.shape[0]
        for z in range(self.n_components):
            self.mean_[z, :] = np.sum(gamma[:, z:z+1] * x, axis=0)/N_z[z]
            for n in range(n_samples):
                self.covar_[z, :, :] = self.covar_[z, :, :] + gamma[n, z] * ((x[n:n+1, :] - self.mean_[z:z+1, :]).transpose() @ (x[n:n+1, :] - self.mean_[z:z+1, :]))
            self.covar_[z, :, :] = self.covar_[z, :, :]/N_z[z]
            self.weights_[z] = N_z[z]/n_samples
        self.set_params()


    def set_params(self):
        for z in range(self.n_components):
            self.gmms[z].set_parm(self.mean_[z, :], self.covar_[z, :, :])

    def update(self, x):

        # E step
        gamma = self.compute_posterior(x)

        # M step
        self.update_params(gamma, x)

        return np.sum(self.score_samples(x))

def test_gaussain(shifted_gaussian, iter = 5):

    # test gaussian
    gaussian = Gaussian(feature=2)
    for i in range(iter):
        prob = gaussian.update(shifted_gaussian)
        print("prob {}: {}".format(i, prob[0]))
    print(gaussian.mean)

    x = np.arange(-100., 100.,1)
    y = np.arange(-100., 100.,1)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T    # [40000, 2]
    prob = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            prob = prob + gaussian.get_prob(XX[i+j*len(y),:])
    print("cdf {}".format(prob))                                 # integration of pdf should be equal 1


def test_gmm(X_train, iter = 5):

    gmm = GaussianMixture(n_components=2, n_features=2)

    for i in range(iter):
        prob = gmm.update(X_train)
        print("prob {}: {}".format(i, prob))

    mean, var = gmm.get_params()
    print("mean:{}".format(mean))
    print("covariance:{}".format(var))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    # Z = -clf.score_samples(XX)                        # score by sklearn
    # Z = Z.reshape(X.shape)

    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                    levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    plt.savefig('pic2.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    args = parser.parse_args()

    n_samples = 300

    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(X_train)
    print(clf.get_params())
    print("mean : {}".format(clf.means_))                # estimated by sklearn
    print("var : {}".format(clf.covariances_))

    test_gaussain(stretched_gaussian)
    test_gmm(X_train)

