from __future__ import division, print_function, absolute_import
import amitgroup as ag
from pnet.layer import Layer

@Layer.register('crop-layer')
class CropLayer(Layer):
    def __init__(self, frame=0):
        self._frame = frame 

    def extract(self, X):
        return X[:,self._frame:-self._frame,self._frame:-self._frame]

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


