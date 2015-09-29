from __future__ import division, print_function, absolute_import 

import amitgroup as ag
import numpy as np
from pnet.layer import Layer

@Layer.register('intensity-threshold-layer')
class IntensityThresholdLayer(Layer):
    def __init__(self, threshold=0.5, whiten=False):
        self._threshold = threshold
        self._whiten=whiten

    def _extract(self, phi, data):
        X = phi(data)
        if (self._whiten):
            XX=np.copy(X)
            XXa=np.reshape(XX,XX.shape[0:1]+(XX.shape[1]*XX.shape[2]*XX.shape[3],))
            mm=np.mean(XXa,axis=1)[...,np.newaxis]
            st=np.std(XXa,axis=1)[...,np.newaxis]
            XXa=(XXa-mm)/st
            Xout=XXa.reshape(X.shape)
        else:
            Xout=(X > self._threshold).astype(np.uint8)[...,np.newaxis]
        return Xout

    def save_to_dict(self):
        return dict(threshold=self._threshold,whiten=self._whiten)

    @classmethod
    def load_from_dict(cls, d):
        return cls(threshold=d['threshold'],whiten=d['whiten'])

