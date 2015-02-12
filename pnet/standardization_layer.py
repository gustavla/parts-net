from __future__ import division, print_function, absolute_import
from pnet.layer import Layer
import numpy as np


@Layer.register('standardization-layer')
class StandardizationLayer(Layer):
    def __init__(self, epsilon=0.1):
        self._means = None
        self._vars = None
        self._epsilon = epsilon

    @property
    def trained(self):
        return self._means is not None and self._vars is not None

    def _train(self, phi, data, y=None):
        X = phi(data)
        Xflat = X.reshape((X.shape[0], -1)).astype(np.float64)

        self._means = Xflat.mean(0)
        self._vars = Xflat.var(0)

    def _extract(self, phi, data):
        X = phi(data)
        Xflat = X.reshape((X.shape[0], -1)).astype(np.float64)
        Xflat = (Xflat - self._means) / np.sqrt(self._vars + self._epsilon)

        return Xflat.reshape(X.shape)

    def save_to_dict(self):
        d = {}
        d['means'] = self._means
        d['vars'] = self._vars
        d['epsilon'] = self._epsilon
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(epsilon=d['epsilon'])
        obj._means = d['means']
        obj._vars = d['vars']
