from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
from pnet.bernoulli_mm import BernoulliMM
from scipy.special import logit
import amitgroup as ag


@Layer.register('gmm-classification-layer')
class GMMClassificationLayer(SupervisedLayer):
    def __init__(self, settings={}):

        self._settings = dict(
                              n_components=1,
                              n_iter=10,
                              n_init=1,
                              seed=0,
                              covariance_type='diag',
                              min_covariance=0.01,
                              adaptive_min_covariance=False,
                              max_covariance_samples=None,
                              uniform_weights=True,
                              standardize=False,
                             )
        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._n_components = self._settings["n_components"]
        self._models = None

        self._extra = {}

    @property
    def trained(self):
        return self._models is not None

    def reset(self):
        self._models = None

    @property
    def classifier(self):
        return True

    def _extract(self, phi, data):
        assert self.trained, "Must train model first"
        X = phi(data)

        #ag.io.save('test-x.h5', X)

        Xflat = X.reshape(X.shape[0], -1)

        from pnet.oriented_gaussian_parts_layer import score_samples

        cov_type = self._settings['covariance_type']

        K = len(self._models)

        all_logprobs = np.zeros((K, X.shape[0]))

        for k, model in enumerate(self._models):
            means = model['means']
            means = means.reshape(means.shape[0], -1)
            logprob, log_resp, lpr = score_samples(Xflat, means,
                                              model['weights'].ravel(),
                                              model['covar'],
                                              covariance_type=cov_type)

            if self._settings['standardize']:
                logprob = (logprob - self.standarization_info[k]['mean']) / self.standarization_info[k]['std']

            #all_logprobs[k] = logprob
            all_logprobs[k]=np.max(logprob[:, np.newaxis] + log_resp, axis=1)
        # Now fetch classes
        yhat = all_logprobs.argmax(0)
        return yhat



    def _train(self, phi, data, y):
        import types

        X = phi(data)

        #ag.io.save('train-x.h5', X)
        #ag.io.save('train-y.h5', y)

        K = y.max() + 1
        mm_models = []
        mm_comps = []
        mm_weights = []

        n_init = self._settings['n_init']
        n_iter = self._settings['n_iter']
        if self._n_components == 1:
            n_iter = 1
            n_init = 1

        seed = self._settings['seed']
        covar_limit = self._settings['max_covariance_samples']
        cov_type = self._settings['covariance_type']



        self.standarization_info = []

        te = None

        for k in range(K):
            Xk = X[y == k]
            datak=data[y==k]
            Xk = Xk.reshape((Xk.shape[0], 1, -1))

            from pnet.permutation_gmm import PermutationGMM
            mm = PermutationGMM(n_components=self._n_components,
                                n_iter=n_iter,
                                n_init=n_init,
                                thresh=1e-5,
                                random_state=seed,
                                min_covar=self._settings['min_covariance'],
                                params='wmc',
                                covariance_type=cov_type,
                                covar_limit=covar_limit,
                                target_entropy=te)
            print('Fitting')
            mm.fit(Xk)
            print('Done.')



            print('Predicting')
            comps = mm.predict(Xk)


            def mean0(x):
                if x.shape[0] == 0:
                    return np.zeros(x.shape[1:])
                else:
                    return np.mean(x, axis=0)

            visparts=[]
            for kk in range(mm.n_components):
                visparts.append(np.mean(datak[comps[:, 0] == kk],axis=0))

            visparts=np.array(visparts)
            #ii = mm.predict_flat(X)

            logprob, log_resp, lpr = mm.score_block_samples(Xk)
            ii = log_resp.reshape((log_resp.shape[0], -1)).argmax(-1)

            # Standardize the logprob
            if self._settings['standardize']:
                mean = logprob.mean(0)
                std = logprob.std(0)
                info = dict(mean=mean, std=std)
                self.standarization_info.append(info)
                print('info', info)

            sh = (mm.n_components, len(mm.permutations))
            comps = np.vstack(np.unravel_index(ii, sh)).T
            print('Predicting done.')

            weights = mm.weights_
            mu = mm.means_.reshape((self._n_components,)+X.shape[1:])
            covar = mm.covars_

            if cov_type == 'diag':
                covar = mm.covars_.reshape(-1, mm.covars_.shape[-1])

            if self._settings['uniform_weights']:
                # Set weights to uniform
                self._weights = np.ones(weights.shape)
                self._weights /= np.sum(self._weights)
            else:
                self._weights = weights

            mm_models.append(dict(means=mu,
                                  weights=weights,
                                  covar=covar,
                                  visparts=visparts))

            np.set_printoptions(precision=2, suppress=True)
            print('Weights:', weights)
            print('Weights sorted:', np.sort(weights)[::-1])



        self._models = mm_models

    def _vzlog_output_(self, vz):
        import pylab as plt
        vz.section('GMM model')

        #for i, model in enumerate(self._models):
            #vz.text('Class', i)
            #grid = ag.plot.ImageGrid(model['means'])

        #vz.log(self._models.shape)

        #plt.figure(figsize=(6, 4))
        #X = self._models.ravel()
        #plt.hist(np.log10(X[X > 0]), normed=True)
        #plt.xlabel('Concentration (10^x)')
        #plt.title('Model probs')
        #plt.savefig(vz.impath(ext='svg'))

    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        d['n_components'] = self._n_components
        d['models'] = self._models
        d['extra'] = self._extra

        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        obj._models = d['models']
        obj._extra = d['extra']
        return obj

    def __repr__(self):
        return 'MixtureClassificationLayer(n_components={})'.format(
               self._n_components)
