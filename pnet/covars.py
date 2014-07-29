from __future__ import division, print_function, absolute_import

import numpy as np

def distances_for_patch_covariance(cov):
    """
    For a covariance matrix 
    """
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1]

    D = int(np.sqrt(cov.shape[0]))
    dists = np.zeros((D**2, D**2)) 
    for d1 in range(D**2):
        for d2 in range(D**2):
            d1_x, d1_y = divmod(d1, D)
            d2_x, d2_y = divmod(d2, D)
            dist = np.sqrt((d1_x - d2_x)**2 + (d1_y - d2_y)**2)
            dists[d1,d2] = dist

    return dists

def corr(cov):
    dd = np.diag(np.diag(cov)**(-1/2))
    return np.dot(dd, np.dot(cov, dd))
