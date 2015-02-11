from __future__ import division, print_function, absolute_import 

import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('edge-layer')
class EdgeLayer(Layer):
    def __init__(self, **kwargs):
        self._edge_settings = kwargs 

    @property
    def pos_matrix(self):
        if self._edge_settings.get('preserve_size', True):
            return pnet.matrix.identity()
        else:
            return pnet.matrix.translation(-2, -2)

    def _extract(self, phi, data):
        X = phi(data)
        if isinstance(X, list):
            return [ag.features.bedges(Xi, **self._edge_settings) for Xi in X]
        else:
            return ag.features.bedges(X, **self._edge_settings)

    def save_to_dict(self):
        d = {}
        d['edge_settings'] = self._edge_settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(**d['edge_settings'])       
        return obj

    def __repr__(self):
        return 'EdgeLayer(bedges_settings={})'.format(self._edge_settings)
