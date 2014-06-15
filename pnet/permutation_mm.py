from __future__ import division, print_function, absolute_import
import numpy as np
import itertools as itr
import amitgroup as ag
from scipy.special import logit
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
import time

class PermutationMM(BaseEstimator):
    """
    A Bernoulli mixture model with the option of a latent permutation. Each
    sample gets transformed a number of times into a set of blocks. The
    parameter space is similarly divided into blocks, which when trained
    will represent the same transformations. The latent permutation
    dictates what parameter block a sample block should be tested against.

    n_components : int, optional
        Number of mixture components. Defaults to 1.

    permutations : int or array, optional
        If integer, the number of permutations should be specified and a cyclic
        permutation will be automatically built.  If an array, each row is a
        permutation of the blocks.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    min_prob : float, optional
        Floor for the minimum probability

    thresh : float, optional
        Convergence threshold.

    n_iter : float, optional
        Number of EM iterations to perform

    n_init : int, optional
        Number of random initializations to perform with
        the best kept.

    Attributes
    ----------
    `weights_` : array, shape (`n_components`,)
        Stores the mixing weights for each component

    `means_` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    """
    def __init__(self, n_components=1, permutations=1, n_iter=20, n_init=1, random_state=0, min_probability=0.05, thresh=1e-8):


        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state
        self.n_components = n_components

        if isinstance(permutations, int):
            # Cycle through them
            P = permutations
            self.permutations = np.zeros((P, P))
            for p1, p2 in itr.product(range(P), range(P)):
                self.permutations[p1,p2] = (p1 + p2) % P
        else: 
            self.permutations = np.asarray(permutations)

        self.n_iter = n_iter
        self.n_init = n_init
        self.min_probability = min_probability
        self.thresh = thresh

        self.weights_ = None
        self.means_ = None

    def score_block_samples(self, X):
        """
        Score complete camples according to the full model. This means that each sample
        has all its blocks with the different transformations for each permutation.

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
        log_responsibilities : array_like, shape (n_samples, n_components, n_permutations)
            Log posterior probabilities of each mixture component and
            permutation for each observation.

        """
        N = X.shape[0]
        K = self.n_components
        P = len(self.permutations)

        unorm_log_resp = np.empty((N, K, P))
        unorm_log_resp[:] = np.log(self.weights_[np.newaxis])
        for p in range(P):
            for shift in range(P):
                p0 = self.permutations[shift,p]
                unorm_log_resp[:,:,p] += np.dot(X[:,p0], logit(self.means_[:,shift]).T)

        unorm_log_resp+= np.log(1 - self.means_[:,self.permutations]).sum(2).sum(2)

        logprob = logsumexp(unorm_log_resp.reshape((unorm_log_resp.shape[0], -1)), axis=-1)
        log_resp = (unorm_log_resp - logprob[...,np.newaxis,np.newaxis]).clip(min=-500)

        return logprob, log_resp 


    def fit(self, X):
        """
        Estimate model parameters with the expectation-maximization algorithm.

        Parameters are set when constructing the estimator class.

        Parameters
        ----------
        X : array_like, shape (n, n_permutations, n_features)
            Array of samples, where each sample has been transformed `n_permutations` times.

        """

        assert X.ndim == 3
        N, P, F = X.shape

        assert P == len(self.permutations) 
        K = self.n_components
        eps = self.min_probability

        max_log_prob = -np.inf

        for trial in range(self.n_init):
            self.weights_ = np.ones((K, P)) / (K * P)

            # Initialize by picking K components at random.
            repr_samples = X[self.random_state.choice(N, K, replace=False)]
            self.means_ = repr_samples.clip(eps, 1 - eps)

            #self.q = np.empty((N, K, P))
            loglikelihoods = []
            self.converged_ = False
            for loop in range(self.n_iter):
                start = time.clock()
                
                # E-step
                logprob, log_resp = self.score_block_samples(X)

                resp = np.exp(log_resp)
                log_dens = logsumexp(log_resp.transpose((0, 2, 1)).reshape((-1, log_resp.shape[1])), axis=0)[np.newaxis,:,np.newaxis]
                dens = np.exp(log_dens)

                # M-step
                for p in range(P):
                    v = 0.0
                    for shift in range(P):
                        p0 = self.permutations[shift,p]
                        v += np.dot(resp[:,:,shift].T, X[:,p0])

                    self.means_[:,p,:] = v
                self.means_ /= dens.ravel()[:,np.newaxis,np.newaxis]
                self.means_[:] = self.means_.clip(eps, 1 - eps)
                self.weights_[:] = (np.apply_over_axes(np.sum, resp, [0])[0,:,:] / N).clip(0.0001, 1 - 0.0001)


                # Calculate log likelihood
                loglikelihoods.append(logprob.sum())

                ag.info("Trial {trial}/{n_trials}  Iteration {iter}  Time {time:.2f}s  Log-likelihood {llh}".format(trial=trial+1,
                                                                                                                 n_trials=self.n_init,
                                                                                                                 iter=loop+1,
                                                                                                                 time=time.clock() - start,
                                                                                                                 llh=loglikelihoods[-1]))

                if trial > 0 and abs(loglikelihoods[-1] - loglikelihoods[-2])/abs(loglikelihoods[-2]) < self.thresh:
                    self.converged_ = True
                    break


            if loglikelihoods[-1] > max_log_prob: 
                ag.info("Updated best log likelihood to {0}".format(loglikelihoods[-1]))
                max_log_prob = loglikelihoods[-1]
                best_params = {'weights': self.weights_,
                               'means' : self.means_,
                               'converged': self.converged_}


        self.weights_ = best_params['weights']
        self.means_ = best_params['means']
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
            An array of length `num_data`  where `components[i]` indicates
            the argmax of the posteriors. The permutation EM gives two indices, but
            they have been flattened according to ``index * component + permutation``.
        """
        logprob, log_resp = self.score_block_samples(X)
        ii = log_resp.reshape((log_resp.shape[0], -1)).argmax(-1)
        return ii

    def predict(self, X):
        """
        Returns a 2D array of which mixture component each data entry is associate with the most. 

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
        return np.vstack(np.unravel_index(ii, (self.n_components, len(self.permutations)))).T

