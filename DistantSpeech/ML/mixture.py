import argparse
import os

import numpy as np



class Gaussian(object):
    def __init__(self, feature=4):
        self.feature = feature
        self.mean = np.random.rand(feature,1)
        self.var = np.random.rand(feature, feature)

    def get_log_prob(self, x):
        assert  x.shape[0] == self.feature
        if len(x.shape) == 1:
            x = x[x.shape[0], np.newaxis]

        p = -0.5*(self.feature*np.log(2*np.pi) + np.linalg.det(self.var) + ((x-self.mean).transpose() @ np.linalg.inv(self.var) @ (x-self.mean)))

        return p

    def update(self):
        pass

    def get_param(self):
        pass

    def set_parm(self):
        pass

class GaussianMixture(object):
    def __init__(self, n_features=2, n_components=2) -> None:
        self.n_features = n_features
        self.n_components = n_components
        self.mean_ = np.random.rand(self.n_components, self.n_features)
        self.covar_ = np.random.rand(n_components, n_features, n_features)
        self.weights_ = np.random.rand(n_components)

        self.gmms = [Gaussian(n_features) for n in range(n_components)]

    def get_log_prob(self, x):
        prob = 0.0
        for n in range(self.n_components):
            prob += self.gmms[n].get_log_prob(x)




def main(args):
    feature = 1
    gmm = Gaussian(feature)
    x = np.ones((feature,1))
    p = gmm.get_log_prob(x)
    print(p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")
    args = parser.parse_args()
    main(args)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn import mixture

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

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                    levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    plt.savefig('pic.png')
