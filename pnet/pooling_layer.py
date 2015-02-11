from __future__ import division, print_function, absolute_import

from pnet.layer import Layer
import pnet
import numpy as np
import itertools as itr
import amitgroup as ag

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

    def _extract(self, phi, data):
        X_F = phi(data)
        output_dtype = self._settings.get('output_dtype')
        if isinstance(X_F, tuple):
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

            c = ag.apply_once(np.mean, feature_map, [0, 1, 2], keepdims=False)
            self._extract_info['concentration'] = c

            if output_dtype is not None:
                return feature_map.astype(output_dtype)
            else:
                return feature_map

        else:
            X = X_F

            if self._operation == 'max':
                op = np.max
            elif self._operation == 'sum':
                op = np.sum
            elif self._operation == 'avg':
                op = np.mean
            else:
                raise ValueError('Unknown pooling operation: {}'.format(
                                 self._operation))

            if self._final_shape is not None:
                fs = self._final_shape
                x_steps = np.arange(fs[0]+1) * X.shape[1] / fs[0]
                x_bounds = np.round(x_steps).astype(np.int_)

                y_steps = np.arange(fs[1]+1) * X.shape[2] / fs[1]
                y_bounds = np.round(y_steps).astype(np.int_)

                xs = enumerate(zip(x_bounds[:-1], x_bounds[1:]))
                ys = enumerate(zip(y_bounds[:-1], y_bounds[1:]))

                N = X.shape[0]
                F = X.shape[-1]

                feature_map = np.zeros((N,) + fs + (F,), dtype=output_dtype)

                for (i, (x0, x1)), (j, (y0, y1)) in itr.product(xs, ys):
                    patch = X[:, x0:x1, y0:y1]
                    feat = ag.apply_once(op, patch, [1, 2], keepdims=False)
                    feature_map[:, i, j] = feat
                return feature_map

            else:
                raise NotImplementedError('Not yet')


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
