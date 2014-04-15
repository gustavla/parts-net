from __future__ import division, print_function, absolute_import 

import numpy as np
import amitgroup as ag
from pnet.layer import Layer

@Layer.register('parts-net')
class PartsNet(Layer):
    def __init__(self, layers):
        self._layers = layers

    @property
    def layers(self):
        return self._layers

    def train(self, X):
        curX = X
        for l, layer in enumerate(self._layers):
            if not layer.trained:
                ag.info('Training layer {}...'.format(l))
                layer.train(curX)
                ag.info('Done.')

            curX = layer.extract(curX) 

            if isinstance(curX, tuple):
                layer.TMP_output_shape = curX[1].shape
            else:
                layer.TMP_output_shape = curX.shape

    def extract(self, X):
        curX = X
        for layer in self._layers:
            curX = layer.extract(curX) 
        return curX

    def infoplot(self, vz):
        vz.title('Layers')

        vz.text('Layer shapes')
        for i, layer in enumerate(self.layers):
            vz.log(i, layer.TMP_output_shape)

        # Plot some information
        for i, layer in enumerate(self.layers):
            vz.section('Layer #{} : {}'.format(i, layer.name))
            layer.infoplot(vz)

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
        layers = []
        for layer_dict in d['layers']:
            layer = Layer.getclass(layer_dict['name']).load_from_dict(layer_dict)
            layers.append(layer)
        obj = cls(layers)
        return obj
