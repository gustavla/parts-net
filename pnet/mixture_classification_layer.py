from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
from pnet.bernoulli_mm import BernoulliMM
from scipy.special import logit
import amitgroup as ag


@Layer.register('mixture-classification-layer')
class MixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, min_prob=0.0001, settings={}):
        self._n_components = n_components
        self._min_prob = min_prob
        self._models = None
        self._settings = settings
        self._extra = {}

    @property
    def trained(self):
        return self._models is not None

    def reset(self):
        self._models = None

    @property
    def classifier(self):
        return True

    def extract(self, phi, data):
        X = phi(data)
        XX = X[:, np.newaxis, np.newaxis]
        theta = self._models[np.newaxis]

        S = self._settings.get('standardize')
        if S:
            llh = XX * logit(theta)
            bb = np.apply_over_axes(np.sum, llh, [-3, -2, -1])[..., 0, 0, 0]
            bb = (bb - self._means) / self._sigmas
            yhat = np.argmax(bb.max(-1), axis=1)
        else:
            llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
            bb = np.apply_over_axes(np.sum, llh, [-3, -2, -1])[..., 0, 0, 0]
            yhat = np.argmax(bb.max(-1), axis=1)
        return yhat

    def train(self, phi, data, y):
        import types
        if isinstance(data, types.GeneratorType):
            Xs = []
            c = 0
            for i, batch in enumerate(data):
                Xi = phi(batch)
                Xs.append(Xi)
                c += Xi.shape[0]
                ag.info('SVM: Has extracted', c)

            X = np.concatenate(Xs, axis=0)
        else:
            X = phi(data)

        K = y.max() + 1
        mm_models = []
        mm_comps = []
        mm_weights = []
        S = self._settings.get('standardize')
        if S:
            self._means = np.zeros((K, self._n_components))
            self._sigmas = np.zeros((K, self._n_components))

        for k in range(K):
            Xk = X[y == k]
            Xk = Xk.reshape((Xk.shape[0], -1))

            if 1:
                mm = BernoulliMM(n_components=self._n_components,
                                 n_iter=10,
                                 n_init=1,
                                 thresh=1e-5,
                                 random_state=self._settings.get('seed', 0),
                                 min_prob=self._min_prob,
                                 blocksize=200,
                                 verbose=True)
                mm.fit(Xk)

                comps = mm.predict(Xk)
                weights = mm.weights_

                mu = mm.means_.reshape((self._n_components,)+X.shape[1:])
            else:
                from pnet.bernoulli import em
                ret = em(Xk, self._n_components, 10,
                         numpy_rng=self._settings.get('seed', 0),
                         verbose=True)

                mu = ret[1].reshape((self._n_components,)+X.shape[1:])
                comps = ret[3]
                weights = np.exp(ret[0])
                print('--weights', weights)
                weights /= weights.sum()
            mm_models.append(mu)
            mm_comps.append(comps)
            mm_weights.append(weights)

            np.set_printoptions(precision=2, suppress=True)
            print('Weights:', weights)
            print('Weights sorted:', np.sort(weights)[::-1])

            # Add standardization
            if S:
                self._means[k] = np.apply_over_axes(np.sum,
                                                    mu * logit(mu),
                                                    [1, 2, 3]).ravel()
                sig = np.apply_over_axes(np.sum,
                                         logit(mu)**2 * mu * (1 - mu),
                                         [1, 2, 3]).ravel()
                self._sigmas[k] = np.sqrt(sig)

        self._extra['comps'] = mm_comps
        self._extra['weights'] = mm_weights
        self._models = np.asarray(mm_models)

    def _vzlog_output_(self, vz):
        import pylab as plt
        vz.section('Mixture model')
        vz.log(self._models.shape)

        plt.figure(figsize=(6, 4))
        X = self._models.ravel()
        plt.hist(np.log10(X[X > 0]), normed=True)
        plt.xlabel('Concentration (10^x)')
        plt.title('Model probs')
        plt.savefig(vz.impath(ext='svg'))

    def save_to_dict(self):
        d = {}
        d['n_components'] = self._n_components
        d['min_prob'] = self._min_prob
        d['models'] = self._models
        d['extra'] = self._extra

        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(n_components=d['n_components'], min_prob=d['min_prob'])
        obj._models = d['models']
        obj._extra = d['extra']
        return obj

    def __repr__(self):
        return 'MixtureClassificationLayer(n_components={})'.format(
               self._n_components)
