from __future__ import division, print_function, absolute_import 

import amitgroup as ag
from pnet.layer import Layer

class PartsNet(Layer):
    def __init__(self, layers):
        self._layers = layers

    def train(self, X):
        curX = X
        for l, layer in enumerate(self._layers):
            if not layer.trained:
                layer.train(curX)
            ag.info('Training layer {}... Done.'.format(l))
            curX = layer.extract(curX) 

    def extract(self, X):
        curX = X
        for layer in self._layers:
            curX = layer.extract(curX) 
        return curX


    def save_to_dict(self):
        d = {}
        d['layers'] = []
        for layer in self._layers:
            d['layers'] = layer.save_to_dict() 

        return d

    @classmethod
    def load_from_dict(self, d):
        pass
        #self._layers = []
        #for layer_dict in d['layers']:
            #layer = Layer.construct
            #self._layers.append(layer)

