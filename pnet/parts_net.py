from __future__ import division, print_function, absolute_import 

from pnet.layer import Layer

class PartsNet(Layer):
    def __init__(self, layers):
        self._layers = layers

    def train(self, X):
        curX = X
        for layer in self._layers:
            if not layer.trained:
                layer.train(curX)
            curX = layer.extract(curX) 

        #return curX
        
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
        #self._layers = []
        #for layer_dict in d['layers']:
            #layer = Layer.construct
            #self._layers.append(layer)

