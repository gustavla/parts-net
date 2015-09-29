from __future__ import division, print_function, absolute_import
from pnet.layer import Layer
import numpy as np


@Layer.register('rectifier-layer')
class RectifierLayer(Layer):
    def __init__(self, rectifier='relu'):
        self._rectifier = rectifier

    def _extract(self, phi, data):
        X = phi(data)
        print('Percentage above threshold:',np.float(np.sum(X>0))/X.size)
        if self._rectifier == 'relu':
            return X.clip(min=0)
        else:
            raise ValueError("Unknown recitifer")

    def save_to_dict(self):
        d = {}
        d['rectifier'] = self._rectifier
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(rectifier=d['rectifier'])
        return obj