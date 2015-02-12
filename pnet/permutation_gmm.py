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
    def __init__(self, n_components=1, permutations=1, covariance_type='tied',
                 min_covar=1e-3, n_iter=20, n_init=1, params='wmc',
                 random_state=0, thresh=1e-2, covar_limit=None, target_entropy=None):

        assert covariance_type in _COV_TYPES, "Covariance type not supported"
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self._covtype = covariance_type
        self.random_state = random_state
        self.n_components = n_components

        if isinstance(permutations, int):
            # Cycle through them
            P = permutations
            self.permutations = np.zeros((P, P), dtype=np.int64)
            for p1, p2 in itr.product(range(P), range(P)):
                self.permutations[p1, p2] = (p1 + p2) % P
        else:
            self.permutations = np.asarray(permutations)

        self.n_iter = n_iter
        self.n_init = n_init
        self.thresh = thresh
        self.min_covar = min_covar
        self._covar_limit = covar_limit
        self._params = params
        self._target_entropy = target_entropy

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
        logprob = logsumexp(unorm_reshaped.clip(min=-500), axis=-1)
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
        def diff_entropy(cov):
            sign, logdet = np.linalg.slogdet(cov)
            return 0.5 * cov.shape[0] * np.log(2 * np.pi * np.e) + logdet

        def reg_covar(cov0, mcov, target_entropy):
            def regularize_cov(reg_val):
                return cov0 * (1 - reg_val) + np.eye(cov0.shape[0]) * reg_val

            lo, hi = self.min_covar * (1 + np.array([-0.95, 2.95]))
            ent = None
            for d in range(15):
                mi = np.mean([lo, hi])

                c = regularize_cov(mi)
                ent = diff_entropy(c)
                print('ent', ent)
                if ent > target_entropy:
                    hi = mi
                else:
                    lo = mi

            mcov1 = np.mean([lo, hi])
            print('mcov multiple', mcov1 / mcov)
            return regularize_cov(mcov1)


        assert X.ndim == 3
        N, P, F = X.shape

        assert P == len(self.permutations)
        K = self.n_components

        if K == 1 and P == 1 and self._covtype == 'diag':
            self.weights_ = np.ones(1)
            self.means_ = ag.apply_once(np.mean, X, [0])
            c0 = ag.apply_once(np.var, X, [0])
            self.covars_ = c0 + self.min_covar
            self.converged_ = True
            def diff_entropy(cov):
                return 0.5 * cov.shape[2] * np.log(2 * np.pi * np.e) + np.sum(np.log(np.fabs(cov)))

            if self._target_entropy is None:
                c = c0 + self.min_covar
                ent = diff_entropy(c)
                self._target_entropy = ent
                self.covars_ = c
                self._entropy = ent
            else:
                lo, hi = self.min_covar * (1 + np.array([-0.25, 0.25]))
                self._entropy = diff_entropy(self.covars_)
                for d in range(10):
                    mi = np.mean([lo, hi])

                    c = c0 + mi
                    ent = diff_entropy(c)
                    print('ent', ent)
                    if ent > self._target_entropy:
                        hi = mi
                    else:
                        lo = mi

                mcov = np.mean([lo, hi])
                print('mcov', mcov)

                print('target_entropy', self._target_entropy)
                print('diff', np.fabs(ent - self._target_entropy))

                self.covars_ = c0 + mcov
                self._entropy = diff_entropy(self.covars_)
            return
        if K == 1 and P == 1 and self._covtype == 'tied':
            cov0 = np.cov(X[:,0].T)

            U, S, V = np.linalg.svd(cov0)
            def regularize_cov(reg_val):
                #return cov0 + np.eye(cov.shape[0]) * reg_val
                return cov0 * (1 - reg_val) + np.eye(cov0.shape[0]) * reg_val
                #regS = S.clip(min=reg_val)
                #regS = (S + 0.0001) * (reg_val / self.min_covar)
                #return np.dot(np.dot(U, np.diag(regS)), V)

            self.weights_ = np.ones(1)
            self.means_ = ag.apply_once(np.mean, X, [0])
            self.converged_ = True
            if self._target_entropy is None:
                c = regularize_cov(self.min_covar)
                ent = diff_entropy(c)
                self.covars_ = c
                self._entropy = ent
            else:
                lo, hi = self.min_covar * (1 + np.array([-0.95, 1.95]))
                ent = None
                for d in range(15):
                    mi = np.mean([lo, hi])

                    c = regularize_cov(mi)
                    ent = diff_entropy(c)
                    print('ent', ent)
                    if ent > self._target_entropy:
                        hi = mi
                    else:
                        lo = mi

                mcov = np.mean([lo, hi])
                print('mcov', mcov)

                print('target_entropy', self._target_entropy)
                print('diff', np.fabs(ent - self._target_entropy))

                self.covars_ = regularize_cov(mcov) #np.cov(X[:,0].T) + np.diag(np.ones(F)*mcov)

                #print('diff entropy', diff_entropy(self.covars_))

                self._entropy = diff_entropy(self.covars_)
            return

        #N34 = 3 * N // 4
        #HX = X[N34:]
        #X = X[:N34]

        #HN = N - N34
        #N = N34
        print('HERE')

        XX = X.reshape((-1, X.shape[-1]))

        max_log_prob = -np.inf

        for trial in range(self.n_init):
            loglikelihoods = []
            self.weights_ = np.ones((K, P)) / (K * P)

            flatX = X.reshape((-1, F))

            # Initialize to covariance matrix of all samples
            if self._covtype == 'diag':
                pass
            elif 0:
                cv = np.eye(F)
            elif 1:
                print('cov')
                cv = (1 - self.min_covar) * np.cov(flatX.T) + self.min_covar * np.eye(F)
                print('cov done')
            else:
                cv = ag.io.load('/var/tmp/cov.h5')

            # Initialize by picking K components at random.
            if self._covtype == 'diag':
                repr_samples = X[self.random_state.choice(N, K, replace=False)]
                self.means_ = repr_samples
            elif 0:
                # Initialize by running kmeans
                assert P == 1
                from sklearn.cluster import KMeans
                clf = KMeans(n_clusters=K)
                XX2 = np.dot(cv, flatX.T).T

                clf.fit(XX2)

                means = clf.means_

                self.means_ = clf.means_.reshape((K,) + X.shape[1:])
            else:
                # TODO: Does not initialize permutations in a coherent way, but
                # this might not be needed anyway
                rs = np.random.RandomState(trial)  # TODO: Insert seed
                mm = rs.multivariate_normal(np.zeros(F), cv, size=K * P)
                self.means_ = mm.reshape((K,) + X.shape[1:])

            if self._covtype == 'ones':
                self.covars_ = np.ones(cv.shape[0])
            elif self._covtype == 'tied':
                self.covars_ = cv
            elif self._covtype == 'diag':
                self.covars_ = np.tile(np.ones(F), (K, P, 1))
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

                #test_logprob, _ = self.score_block_samples(HX)
                #test_loglikelihood = test_logprob.sum()

                # TODO
                hh = np.histogram(np.exp(log_resp.max(-1).max(-1)),
                                  bins=np.linspace(0, 1, 11))

                sh = (-1, log_resp.shape[1])
                resp = np.exp(log_resp)
                lresp = log_resp.transpose((0, 2, 1)).reshape(sh)
                log_dens = logsumexp(lresp, axis=0)[np.newaxis, :, np.newaxis]
                dens = np.exp(log_dens)

                # M-step

                if 'm' in self._params:
                    for p in range(P):
                        v = 0.0
                        for shift in range(P):
                            p0 = self.permutations[shift, p]
                            v += np.dot(resp[:, :, shift].T, X[:, p0])

                        self.means_[:, p, :] = v
                    self.means_ /= dens.ravel()[:, np.newaxis, np.newaxis]

                if 'w' in self._params:
                    ww = (ag.apply_once(np.sum, resp, [0], keepdims=False) / N)
                    self.weights_[:] = ww.clip(0.0001, 1 - 0.0001)

                if 'c' in self._params:
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
                        self.covars_ = (self.covars_ * (1 - self.min_covar) +
                                        np.eye(D) * self.min_covar)

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
                            #dd = np.diag(self.covars_[k])
                            #clipped_dd = dd.clip(min=self.min_covar)
                            #self.covars_[k] += np.diag(clipped_dd - dd)

                            self.covars_[k] = reg_covar(self.covars_[k],
                                                        self.min_covar,
                                                        -9000.0)

                            #c = self.covars_[k]
                            #c = (1 - mcov) * c + mcov * np.eye(c.shape[0])
                            #self.covars_[k] = c

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
                            dd = np.diag(self.covars_[k, p])
                            clipped_dd = dd.clip(min=self.min_covar)
                            #self.covars_[k, p] += np.diag(clipped_dd - dd)
                            #self.covars_[k, p] += np.diag(clipped_dd - dd)
                            self.covars_[k, p] += np.eye(D) * self.min_covar

                # Calculate log likelihood
                loglikelihoods.append(logprob.sum())

                ag.info("Trial {trial}/{n_trials}  Iteration {iter}  "
                        "Time {time:.2f}s  Log-likelihood {llh:.2f} "
                        #"Test log-likelihood {tllh:.2f}"
                        "".format(
                            trial=trial+1,
                            n_trials=self.n_init,
                            iter=loop+1,
                            time=time.clock() - start,
                            llh=loglikelihoods[-1] / N,
                            #tllh=test_loglikelihood / HN,
                            ))

                if loop > 0:
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
        if self.means_.shape[0] == 1:
            return np.zeros(X.shape[0], dtype=np.int64)
        else:
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
