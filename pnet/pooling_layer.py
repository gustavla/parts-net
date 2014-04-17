from __future__ import division, print_function, absolute_import 

from pnet.layer import Layer
import numpy as np

@Layer.register('pooling-layer')
class PoolingLayer(Layer):

    def __init__(self, shape=(1, 1), strides=(1, 1)):
        self._shape = shape
        self._strides = strides
        self._extract_info = {}

    def extract(self, X_F):
        X, F = X_F 
        from pnet.cyfuncs import index_map_pooling_multi


        feature_map = index_map_pooling_multi(X, F, self._shape, self._strides) 

        self._extract_info['concentration'] = np.apply_over_axes(np.mean, feature_map, [0, 1, 2])[0,0,0]
        return feature_map

    def infoplot(self, vz):
        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._extract_info['concentration'], label='concentration')
        plt.savefig(vz.generate_filename(ext='svg'))

        plt.figure(figsize=(4, 4))
        cc = self._extract_info['concentration']
        plt.hist(np.log10(cc[cc>0]), normed=True)
        plt.xlabel('Concentration (10^x)')
        plt.title('Concentration')
        plt.savefig(vz.generate_filename(ext='svg'))

        #vz.log('concentration', self._extract_info['concentration'])
        vz.log('mean concentration: {:.1f}%'.format(100*self._extract_info['concentration'].mean()))


    def save_to_dict(self):
        d = {}
        d['shape'] = self._shape
        d['strides'] = self._strides
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['shape'], d['strides'])
        return obj
        
