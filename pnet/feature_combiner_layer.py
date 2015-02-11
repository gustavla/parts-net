from __future__ import division, print_function, absolute_import

import numpy as np
from pnet.layer import Layer


@Layer.register('feature-combiner-layer')
class FeatureCombinerLayer(Layer):
    def __init__(self, layers):
        self._layers = layers
        self._trained = False

    @property
    def layers(self):
        return self._layers

    @property
    def trained(self):
        return self._trained

    def _train(self, phi, data, y=None):
        X = phi(data)
        for layer in self._layers:
            layer.train(X, y)

        self._trained = True

    def _extract(self, phi, data):
        X = phi(data)
        features = []
        for layer in self._layers:
            feat = layer.extract(X)
            features.append(feat)

            # This only works if it's outputting 4D features
            assert feat.ndim == 4

        return np.concatenate(features, axis=3)

    def save_to_dict(self):
        d = {}
        d['layers'] = []
        for layer in self._layers:
            layer_dict = layer.save_to_dict()
            layer_dict['name'] = layer.name
            d['layers'].append(layer_dict)
        return d

    @classmethod
    def load_from_dict(cls, d):
        layers = [
            Layer.getclass(layer_dict['name']).load_from_dict(layer_dict)
            for layer_dict in d['layers']
        ]
        obj = cls(layers)
        return obj

    def __repr__(self):
        return 'FeatureCombinerLayer({})'.format(num_parts=self._layers)
