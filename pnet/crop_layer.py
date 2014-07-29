from __future__ import division, print_function, absolute_import
from pnet.layer import Layer
import pnet


@Layer.register('crop-layer')
class CropLayer(Layer):
    def __init__(self, frame=0):
        self._frame = frame

    def extract(self, phi, data):
        X = phi(data)

        def remove_frame(Y):
            return Y[:, self._frame:-self._frame, self._frame:-self._frame]

        if isinstance(X, tuple):
            return (remove_frame(X[0]),) + X[1:]
        else:
            return remove_frame(X)

    @property
    def pos_matrix(self):
        T = pnet.matrix.translation(-self._frame, -self._frame)
        return T

    def save_to_dict(self):
        d = {}
        d['frame'] = self._frame
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['frame'])
        return obj

    def __repr__(self):
        return 'CropLayer(frame={})'.format(self._frame)
