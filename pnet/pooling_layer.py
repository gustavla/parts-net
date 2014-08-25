from __future__ import division, print_function, absolute_import

from pnet.layer import Layer
import pnet
import numpy as np

@Layer.register('pooling-layer')
class PoolingLayer(Layer):

    def __init__(self, shape=None, strides=None, final_shape=None,
                 operation='max', settings={}):
        assert (shape is None) != (final_shape is None)
        self._shape = shape
        self._final_shape = final_shape

        self._strides = strides
        self._settings = settings
        self._operation = operation
        self._extract_info = {}

    def calc_shape(self, input_shape):
        if self._shape is not None:
            return self._shape
        else:
            return [input_shape[i] // self._final_shape[i] for i in range(2)]

    def calc_strides(self, input_shape):
        if self._strides is None:
            return self.calc_shape(input_shape)
        else:
            return self._strides

    def extract(self, phi, data):
        X_F = phi(data)
        X = X_F[0]
        F = X_F[1]

        #if X.ndim == 3:
            #from pnet.cyfuncs import index_map_pooling as poolf
        #else:

        shape = self.calc_shape(X.shape[1:3])
        strides = self.calc_strides(X.shape[1:3])

        if self._operation == 'max':
            from pnet.cyfuncs import index_map_pooling_multi as poolf
            feature_map = poolf(X, F, shape, strides)

        elif self._operation == 'sum':
            from pnet.cyfuncs import index_map_sum_pooling_multi as poolf
            feature_map = poolf(X, F, shape, strides)

        else:
            raise ValueError('Unknown pooling operation: {}'.format(
                             self._operation))

        self._extract_info['concentration'] = np.apply_over_axes(np.mean, feature_map, [0, 1, 2])[0,0,0]

        output_dtype = self._settings.get('output_dtype')
        if output_dtype is not None:
            return feature_map.astype(output_dtype)
        else:
            return feature_map

    @property
    def pos_matrix(self):
        if self._final_shape is not None:
            # TODO: This doesn't work
            return pnet.matrix.scale(1.0, 1.0)
        else:
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
        d['final_shape'] = self._final_shape
        d['strides'] = self._strides
        d['operation'] = self._operation
        d['settings'] = self._settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(shape=d['shape'], strides=d['strides'],
                  final_shape=d['final_shape'], operation=d['operation'],
                  settings=d['settings'])
        return obj

    def __repr__(self):
        if self._shape is not None:
            shape_type = 'shape'
            shape = self._shape
        else:
            shape_type = 'final_shape'
            shape = self._final_shape

        return ('PoolingLayer({shape_type}={shape}, strides={strides}').format(
                    shape_type=shape_type,
                    shape=shape,
                    strides=self._strides)
