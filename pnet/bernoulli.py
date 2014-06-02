from __future__ import division, print_function, absolute_import
"""
A collection of routines for multivariate independent Bernoulli random vector.
"""

import math
import numpy as np
from scipy.special import logit
from scipy.misc import logsumexp
import time

def log_likelihood(data, *args):
    """
    Compute log likelihood of each of the n data point under each of the m
    Bernoulli models.
     
    Parameters
    ----------
    data : (n,d)-shape NumPy array
        The n data point of d-dimensional binary data.
    *args : (mu) or (logit_mu, sum_log_one_mu)
        Here mu is a (m,d)-shape NumPy array of the m multivariate independent
        Bernoulli model parameters, and sum_log_one_mu is a (m,)-shape NumPy
        array.
     
    Returns
    -------
    loglike : (n,m)-shape NumPy array
        The log likelihood.
    """
    if len(args) == 1:
        logit_mu = logit(args[0])
        sum_log_one_mu = np.log(1 - args[0]).sum(axis=1)
    else:
        logit_mu = args[0]
        sum_log_one_mu = args[1]
    n, d = data.shape
    m = sum_log_one_mu.size

    # Do computation in blocks to avoid out-of-memory.
    memory_limit = 500 # In MB.
    b = max(math.floor(memory_limit * 1024 * 1024 / (64 * d) - m), 1) # Block size.
    num_b = math.ceil(n / b) # Number of blocks.
    
    loglike = np.empty((n,m))
    for i in range(num_b):
        loglike[i*b:(i+1)*b] = np.dot(data[i*b:(i+1)*b], logit_mu.transpose()) + sum_log_one_mu
    return loglike

def em(data, num_mixture_component, max_num_iteration=25,
       loglike_tolerance=1e-3, mu_truncation=(1,1), additional_mu=None,
       permutation=None, numpy_rng=None, verbose=False):
    """
    Fit a mixture model of multivariate independent Bernoulli random vector
    using EM, with an optional latent permutation.
    
    Parameters
    ----------
    data : (n,d)-shape NumPy array
        The n data points of d-dimensional binary data.
    num_mixture_component : positive integer
        The number m1 of mixture components.
    
    Optional Parameters
    -------------------
    max_num_iteration : positive integer
        The maximum number of EM iterations.
    loglike_tolerance : number in (0,1)
        The EM iteration stops when the percentage increase in the log
        likelihood is below this tolerance.
    mu_truncation : number or 2-tuple or (2,)-shape NumPy array
        If it is a number i.e. epsilon in (0,0.5), then truncate mu to within
        [epsilon,1-epsilon].  If it is two numbers, then it is the parameter of
        a Beta prior distribution, which acts as pseudocount for each dimension
        of the Bernoulli parameter.
    additional_mu : (m2,d)-shape np.float64 NumPy array
        Additional mixture components in the mixture model whose Bernoulli
        parameters are fixed and are not updated during the EM.    
    permutation : (p,d)-shape np.int_ NumPy array with values in {0,...,d-1}
        Each row is a permutation of the indices 0,...,d-1.  It permutes the
        entries of mu.
    numpy_rng : int or numpy.random.RandomState object
        Specify the random state to initialize the EM.
    verbose : bool
        If True, then print the time used for and the log likelihood at each
        EM iteration.
        
    Returns
    -------
    log_weight : (m1+m2,)-shape np.float64 NumPy array
        The log weights of the mixture components, sorted in decreasing order.
        However, the weights of the additional mixture components (if any) are
        always in the last m2 entries in their given order.
    mu : (m1,d)-shape np.float64 NumPy array with values in (0,1)
        The Bernoulli parameters of the m1 mixture components.
    loglike : (num_iteration,)-shape np.float64 NumPy array
        The expected log joint likelihood of the data.  The EM algorithm tries
        to maximize this.
    data_label : (n,) or (n,2)-shape np.int_ NumPy array
        The label (and permutation, if it exists) of mixture component for each
        data point.
    """
    if verbose:
        print('amit.bernoulli.em() started.')
        
    n, d = data.shape
    m1 = num_mixture_component
    # Truncate with epsilon or beta prior?
    if isinstance(mu_truncation, float):
        use_epsilon = True
        epsilon = mu_truncation
    else:
        use_epsilon = False
        beta_prior = mu_truncation
    
    # Additional mu?
    m = m1 + (0 if additional_mu is None else additional_mu.shape[0])
    # Permutation?
    if permutation is None:
        permutation = np.arange(d).reshape((1,d))
    p = permutation.shape[0]
    inverse_permutation = np.argsort(permutation)   

    # Initialize log weight and mu.
    if numpy_rng is None:
        numpy_rng = np.random.RandomState()
    elif not isinstance(numpy_rng, np.random.RandomState):
        numpy_rng = np.random.RandomState(numpy_rng) # A seed is given.

    log_w = np.empty((p,m)) # Log weight.
    log_w.fill(-np.log(p*m))
    
    # Use a random sample of data to initialize mu.  If the data value is 1,
    # then generate mu randomly from density f(mu) = (p+1) mu^p where p > 0 is
    # the purity level.
    purity_level = 2
    mu = numpy_rng.uniform(size=(m1,d)) ** (1 / (purity_level + 1))
    is_flip = np.logical_not(data[numpy_rng.choice(n, m1, replace=False)])
    mu[is_flip] = 1 - mu[is_flip]
#     mu = data[numpy_rng.choice(n, m1, replace=False)].astype(np.float64)
    if use_epsilon:
        mu[mu < epsilon] = epsilon
        mu[mu > 1 - epsilon] = 1 - epsilon
    else:
        mu *= (n / m1) / ((n / m1) + np.sum(beta_prior)) # Adjusted by beta prior.
        mu += beta_prior[0] / ((n / m1) + np.sum(beta_prior))
    # TODO: Include permtuation when initialize mu.  If m = 1 and p = 1, then
    # immediately get to the final mu.
    
    # Pre-allocation.
    log_odd = np.empty((d,m))
    sum_log_one_mu = np.empty(m)
    log_q = np.empty((p,n,m))
    if additional_mu is not None:
        # Compute once for the fixed additional mu.
        log_odd[:,m1:] = logit(additional_mu).transpose()
        sum_log_one_mu[m1:] = np.log(1 - additional_mu).sum(axis=1)

    # Do EM.
    loglike = []
    t = 0
    while t < max_num_iteration:
        if verbose:
            clock_start = time.clock()
            
        # E-step: Compute log responsiblity.
        log_odd[:,:m1] = logit(mu).transpose()
        sum_log_one_mu[:m1] = np.log(1 - mu).sum(axis=1)
        for i in range(p):
            log_q[i] = np.dot(data, log_odd[permutation[i]])
        log_q += (log_w + sum_log_one_mu).reshape((p,1,m))
        norm_log_q = logsumexp(logsumexp(log_q, axis=2), axis=0) # Normalizing constants.
        log_q -= norm_log_q.reshape((1,n,1)) # Now log q is normalized.
        
        # M-step: Compute log weight and mu.
        log_w = logsumexp(log_q, axis=1)
        log_qm1 = logsumexp(log_w[:,:m1], axis=0)
        log_w -= logsumexp(log_w) # Now log weight is normalized.
        
        q = np.exp(log_q[...,:m1])
        mu = np.zeros((m1,d))
        for i in range(p):
            mu += np.dot(q[i].transpose(), data)[:,inverse_permutation[i]]
        if use_epsilon:
            eps = np.finfo(np.float_).eps
            mu[mu < eps] = eps # So it is safe to take log.
            mu = np.exp(np.log(mu) - log_qm1.reshape((m1,1)))
            mu[mu < epsilon] = epsilon
            mu[mu > 1 - epsilon] = 1 - epsilon
        else:
            mu += beta_prior[0]
            mu /= (np.exp(log_qm1) + np.sum(beta_prior)).reshape((m1,1))

        # Compute expected log joint likelihood.
        loglike.append(norm_log_q.sum())
        if verbose:
            print('  Iter {}: {:.3f} seconds.  Log-likelihood: {:.1f}'.format(t+1, time.clock() - clock_start, loglike[-1]))
        # Exit?
        if t >= 1 and loglike[-1] - loglike[-2] < loglike_tolerance * -loglike[-2]:
            break
        t += 1
    loglike = np.asarray(loglike, dtype=np.float64)
        
    # Sort the mixture components.
    log_weight = log_w.sum(axis=0)
    ordering = np.argsort(log_weight[:m1])[::-1]
    log_weight[:m1] = log_weight[ordering]
    mu = mu[ordering]
    
    # Compute data label.
    data_m = logsumexp(log_q, axis=0).argmax(axis=1)
    if p > 1:
        idx = np.ravel_multi_index((np.arange(n), data_m), (n,m))
        data_p = log_q.reshape((p,n*m))[:,idx].argmax(axis=0)
    inverse_ordering = np.argsort(ordering) # Re-order the data label.
    data_m[data_m < m1] = inverse_ordering[data_m[data_m < m1]]
    if p == 1:
        data_label = data_m
    else:
        data_label = np.hstack((data_m.reshape((-1,1)), data_p.reshape((-1,1))))
    
    if verbose:
        print('amit.bernoulli.em() finished.')
    return log_weight, mu, loglike, data_label

# TODO: Write a function for EM with "diff versions", not permutation.
