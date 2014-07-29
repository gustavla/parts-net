from __future__ import division, print_function, absolute_import

import numpy as np
from pnet.layer import Layer
import pnet
from skimage.transform.pyramids import pyramid_reduce, pyramid_expand


def resize(im, f):
    if f < 1:
        im2 = pyramid_reduce(im, downscale=1/f)
    elif f > 1:
        im2 = pyramid_expand(im, upscale=f)
    else:
        im2 = im
    return im2


@Layer.register('resize-layer')
class ResizeLayer(Layer):
    def __init__(self, factor=1.0):
        self._factor = factor

    @property
    def pos_matrix(self):
        return pnet.matrix.scale(1 / self._factor)

    def extract(self, phi, data):
        X = phi(data)
        if self._factor == 1.0:
            if X.dtype == np.uint8:
                return X.astype(np.float64) / 255
            else:
                return X
        else:
            return np.asarray([resize(im, self._factor) for im in X])

    def __repr__(self):
        return "ResizeLayer(factor={})".format(self._factor)

    def save_to_dict(self):
        d = {}
        d['factor'] = self._factor
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['factor'])
        return obj
