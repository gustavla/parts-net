import numpy as np
import itertools as itr
import amitgroup as ag
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
import time

_COV_TYPES = ['ones', 'tied', 'diag', 'diag-perm',
              'full', 'full-perm', 'full-full']


class PermutationGMM(BaseEstimator):
    """

    """
    def __init__(self, n_components=1, permutations=1,
                 covariance_type='tied', min_covar=1e-3, n_iter=20,
                 n_init=1, random_state=0, thresh=1e-2, covar_limit=None,
                 reg_covar=0.0):

        # TODO: Possibly remove reg_covar

        assert covariance_type in _COV_TYPES, "Covariance type not supported"
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self._covtype = covariance_type
        self.random_state = random_state
        self.n_components = n_components

        if isinstance(permutations, int):
            # Cycle through them
            P = permutations
            self.permutations = np.zeros((P, P))
            for p1, p2 in itr.product(range(P), range(P)):
                self.permutations[p1, p2] = (p1 + p2) % P
        else:
            self.permutations = np.asarray(permutations)

        self.n_iter = n_iter
        self.n_init = n_init
        self.thresh = thresh
        self.min_covar = min_covar
        self._covar_limit = covar_limit
        self._reg_covar = reg_covar

        self.weights_ = None
        self.means_ = None
        self.covars_ = None

    def score_block_samples(self, X):
        """
        Score complete samples according to the full model. This means that
        each sample has all its blocks with the different transformations for
        each permutation.

        Parameters
        ----------
        X : ndarray
            Array of samples. Must have shape `(N, P, D)`, where `N` are number
            of samples, `P` number of permutations and `D` number of dimensions
            (flattened if multi-dimensional).

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each full data point in X.
        log_responsibilities : array_like,
                               shape (n_samples, n_components, n_permutations)
            Log posterior probabilities of each mixture component and
            permutation for each observation.

        """
        from scipy.stats import multivariate_normal
        N = X.shape[0]
        K = self.n_components
        P = len(self.permutations)

        unorm_log_resp = np.empty((N, K, P))
        unorm_log_resp[:] = np.log(self.weights_[np.newaxis])

        for p in range(P):
            for shift in range(P):
                p0 = self.permutations[shift, p]
                for k in range(K):
                    if self._covtype == 'ones':
                        cov = np.diag(self.covars_)
                    elif self._covtype == 'tied':
                        cov = self.covars_
                    elif self._covtype == 'diag-perm':
                        cov = np.diag(self.covars_[p])
                    elif self._covtype == 'diag':
                        cov = np.diag(self.covars_[k, p])
                    elif self._covtype == 'full':
                        cov = self.covars_[k]
                    elif self._covtype == 'full-perm':
                        cov = self.covars_[p]
                    elif self._covtype == 'full-full':
                        cov = self.covars_[k, p]

                    unorm_log_resp[:, k, p] += multivariate_normal.logpdf(
                        X[:, p0],
                        mean=self.means_[k, shift],
                        cov=cov)

        unorm_reshaped = unorm_log_resp.reshape((unorm_log_resp.shape[0], -1))
        logprob = logsumexp(unorm_reshaped, axis=-1)
        log_resp = unorm_log_resp - logprob[..., np.newaxis, np.newaxis]
        log_resp = log_resp.clip(min=-500)

        return logprob, log_resp

    def fit(self, X):
        """
        Estimate model parameters with the expectation-maximization algorithm.

        Parameters are set when constructing the estimator class.

        Parameters
        ----------
        X : array_like, shape (n, n_permutations, n_features)
            Array of samples, where each sample has been transformed
            `n_permutations` times.

        """

        assert X.ndim == 3
        N, P, F = X.shape

        assert P == len(self.permutations)
        K = self.n_components

        XX = X.reshape((-1, X.shape[-1]))
        self._bkg_cov = np.cov(XX.T)

        max_log_prob = -np.inf

        loglikelihoods = []
        for trial in range(self.n_init):
            self.weights_ = np.ones((K, P)) / (K * P)

            # Initialize by picking K components at random.
            repr_samples = X[self.random_state.choice(N, K, replace=False)]
            self.means_ = repr_samples

            flatX = X.reshape((-1, F))

            # Initialize to covariance matrix of all samples
            cv = np.cov(flatX.T) + self.min_covar * np.eye(F)

            if self._covtype == 'ones':
                self.covars_ = np.ones(cv.shape[0])
            elif self._covtype == 'tied':
                self.covars_ = cv
            elif self._covtype == 'diag':
                self.covars_ = np.tile(np.diag(cv).copy(), (K, P, 1))
            elif self._covtype == 'diag-perm':
                self.covars_ = np.tile(np.diag(cv).copy(), (P, 1))
            elif self._covtype == 'full':
                self.covars_ = np.tile(cv, (K, 1, 1))
            elif self._covtype == 'full-perm':
                self.covars_ = np.tile(cv, (P, 1, 1))
            elif self._covtype == 'full-full':
                self.covars_ = np.tile(cv, (K, P, 1, 1))

            self.converged_ = False
            for loop in range(self.n_iter):
                start = time.clock()

                # E-step
                logprob, log_resp = self.score_block_samples(X)

                # TODO
                hh = np.histogram(np.exp(log_resp.max(-1).max(-1)),
                                  bins=np.linspace(0, 1, 11))
                print(hh[0])

                sh = (-1, log_resp.shape[1])
                resp = np.exp(log_resp)
                lresp = log_resp.transpose((0, 2, 1)).reshape(sh)
                log_dens = logsumexp(lresp, axis=0)[np.newaxis, :, np.newaxis]
                dens = np.exp(log_dens)

                # M-step
                for p in range(P):
                    v = 0.0
                    for shift in range(P):
                        p0 = self.permutations[shift, p]
                        v += np.dot(resp[:, :, shift].T, X[:, p0])

                    self.means_[:, p, :] = v
                self.means_ /= dens.ravel()[:, np.newaxis, np.newaxis]
                ww = (ag.apply_once(np.sum, resp, [0], keepdims=False) / N)

                self.weights_[:] = ww.clip(0.0001, 1 - 0.0001)

                if self._covtype == 'ones':
                    # Do not update
                    pass
                elif self._covtype == 'tied':
                    from pnet.cyfuncs import calc_new_covar
                    self.covars_[:] = calc_new_covar(X[:self._covar_limit],
                                                     self.means_,
                                                     resp,
                                                     self.permutations)

                    # Now make sure the diagonal is not overfit
                    dd = np.diag(self.covars_)
                    D = self.covars_.shape[0]
                    self.covars_ += np.eye(D) * self.min_covar

                elif self._covtype == 'diag':
                    from pnet.cyfuncs import calc_new_covar_diag as calc
                    self.covars_[:] = calc(X[:self._covar_limit],
                                           self.means_,
                                           resp,
                                           self.permutations)

                    self.covars_[:] += self.min_covar

                elif self._covtype == 'diag-perm':
                    from pnet.cyfuncs import calc_new_covar_diagperm as calc
                    self.covars_[:] = calc(X[:self._covar_limit],
                                           self.means_,
                                           resp,
                                           self.permutations)

                    self.covars_[:] = self.covars_.clip(min=self.min_covar)

                elif self._covtype == 'full':
                    from pnet.cyfuncs import calc_new_covar_full as calc
                    self.covars_[:] = calc(X[:self._covar_limit],
                                           self.means_,
                                           resp,
                                           self.permutations)

                    for k in range(K):
                        dd = np.diag(self.covars_[k])
                        clipped_dd = dd.clip(min=self.min_covar)
                        self.covars_[k] += np.diag(clipped_dd - dd)

                elif self._covtype == 'full-perm':
                    from pnet.cyfuncs import calc_new_covar_fullperm as calc
                    self.covars_[:] = calc(X[:self._covar_limit],
                                           self.means_,
                                           resp,
                                           self.permutations)

                    for p in range(P):
                        dd = np.diag(self.covars_[p])
                        clipped_dd = dd.clip(min=self.min_covar)
                        self.covars_[p] += np.diag(clipped_dd - dd)

                elif self._covtype == 'full-full':
                    from pnet.cyfuncs import calc_new_covar_fullfull as calc
                    self.covars_[:] = calc(X[:self._covar_limit],
                                           self.means_,
                                           resp,
                                           self.permutations)

                    D = self.covars_.shape[2]
                    for k, p in itr.product(range(K), range(P)):
                        # Regularize covariance
                        self.covars_[k, p] *= (1 - self._reg_covar)
                        self.covars_[k, p] += self._reg_covar * self._bkg_cov

                        dd = np.diag(self.covars_[k, p])
                        clipped_dd = dd.clip(min=self.min_covar)
                        #self.covars_[k, p] += np.diag(clipped_dd - dd)
                        #self.covars_[k, p] += np.diag(clipped_dd - dd)
                        self.covars_[k, p] += np.eye(D) * self.min_covar

                # Calculate log likelihood
                loglikelihoods.append(logprob.sum())

                ag.info("Trial {trial}/{n_trials}  Iteration {iter}  "
                        "Time {time:.2f}s  Log-likelihood {llh}".format(
                            trial=trial+1,
                            n_trials=self.n_init,
                            iter=loop+1,
                            time=time.clock() - start,
                            llh=loglikelihoods[-1]))

                if trial > 0:
                    absdiff = abs(loglikelihoods[-1] - loglikelihoods[-2])
                    if absdiff/abs(loglikelihoods[-2]) < self.thresh:
                        self.converged_ = True
                        break

            if loglikelihoods[-1] > max_log_prob:
                ag.info("Updated best log likelihood to {}"
                        .format(loglikelihoods[-1]))
                max_log_prob = loglikelihoods[-1]
                best_params = {'weights': self.weights_,
                               'means': self.means_,
                               'covars': self.covars_,
                               'converged': self.converged_}
            else:
                ag.info("Did not updated best")

        self.weights_ = best_params['weights']
        self.means_ = best_params['means']
        self.covars_ = best_params['covars']
        self.converged_ = best_params['converged']

    def predict_flat(self, X):
        """
        Returns an array of which mixture component each data entry is
        associate with the most. This is similar to `predict`, except it
        collapses component and permutation to a single index.

        Parameters
        ----------
        X : ndarray
            Data array to predict.

        Returns
        -------
        components: list
            An array of length `num_data`  where `components[i]` indicates the
            argmax of the posteriors. The permutation EM gives two indices, but
            they have been flattened according to ``index * component +
            permutation``.
        """
        logprob, log_resp = self.score_block_samples(X)
        ii = log_resp.reshape((log_resp.shape[0], -1)).argmax(-1)
        return ii

    def predict(self, X):
        """
        Returns a 2D array of which mixture component each data entry is
        associate with the most.

        Parameters
        ----------
        X : ndarray
            Data array to predict.

        Returns
        -------
        components: list
            An array of shape `(num_data, 2)`  where `components[i]` indicates
            the argmax of the posteriors. For each sample, we have two values,
            the first is the part and the second is the permutation.
        """
        ii = self.predict_flat(X)
        sh = (self.n_components, len(self.permutations))
        return np.vstack(np.unravel_index(ii, sh)).T
