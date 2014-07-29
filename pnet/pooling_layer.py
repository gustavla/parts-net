from __future__ import division, print_function, absolute_import 

from pnet.layer import Layer
import pnet
import numpy as np

@Layer.register('pooling-layer')
class PoolingLayer(Layer):

    def __init__(self, shape=(1, 1), strides=(1, 1), operation='max', settings={}):
        self._shape = shape
        self._strides = strides
        self._settings = settings
        self._operation = operation
        self._extract_info = {}

    def extract(self, phi, data):
        X_F = phi(data)
        X = X_F[0]
        F = X_F[1]

        #if X.ndim == 3:
            #from pnet.cyfuncs import index_map_pooling as poolf
        #else:

        if self._operation == 'max':
            from pnet.cyfuncs import index_map_pooling_multi as poolf
            feature_map = poolf(X, F, self._shape, self._strides) 

        elif self._operation == 'sum':
            from pnet.cyfuncs import index_map_sum_pooling_multi as poolf
            feature_map = poolf(X, F, self._shape, self._strides) 
        else:
            raise ValueError('Unknown pooling operation: {}'.format(self._operation))

        self._extract_info['concentration'] = np.apply_over_axes(np.mean, feature_map, [0, 1, 2])[0,0,0]

        return feature_map

    @property
    def pos_matrix(self):
        S = pnet.matrix.scale(1 / self._strides[0], 1 / self._strides[1])
        T = pnet.matrix.translation(-(self._shape[0] - 1) / 2,
                                    -(self._shape[1] - 1) / 2)
        return np.dot(S, T)

    def _vzlog_output_(self, vz):
        import pylab as plt

        if 'concentration' in self._extract_info:
            plt.figure(figsize=(6, 3))
            plt.plot(self._extract_info['concentration'], label='concentration')
            plt.savefig(vz.impath(ext='svg'))

            plt.figure(figsize=(4, 4))
            cc = self._extract_info['concentration']
            plt.hist(np.log10(cc[cc>0]), normed=True)
            plt.xlabel('Concentration (10^x)')
            plt.title('Concentration')
            plt.savefig(vz.impath(ext='svg'))

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

    def __repr__(self):
        return 'PoolingLayer(shape={shape}, strides={strides}, has_support_mask={has_support_mask})'.format(
                    shape=self._shape,
                    strides=self._strides,
                    has_support_mask='support_mask' in self._settings,
                    settings=self._settings)
