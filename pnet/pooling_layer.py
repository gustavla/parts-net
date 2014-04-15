from __future__ import division, print_function, absolute_import 

from pnet.layer import Layer

@Layer.register('pooling-layer')
class PoolingLayer(Layer):

    def __init__(self, shape=(1, 1), strides=(1, 1)):
        self._shape = shape
        self._strides = strides

    def extract(self, F_X):
        F, X = F_X 
        from pnet.cyfuncs import index_map_pooling_multi
        return index_map_pooling_multi(X, F, self._shape, self._strides) 

    def save_to_dict(self):
        d = {}
        d['shape'] = self._shape
        d['strides'] = self._strides
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['shape'], d['strides'])
        return obj
        
