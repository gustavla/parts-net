from __future__ import division, print_function, absolute_import 

from pnet.layer import Layer
import numpy as np
import amitgroup as ag

@Layer.register('spreading-layer')
class SpreadingLayer(Layer):

    def __init__(self, radius=1, spread='box'):
        self._radius = radius
        self._spread = spread
        self._extract_info = {}

    def extract(self, phi, data):
        X = phi(data)

        feature_map = ag.features.bspread(X, spread=self._spread,
                                          radius=self._radius)
        return feature_map

    def _vzlog_output_(self, vz):
        pass

    def save_to_dict(self):
        d = {}
        d['radius'] = self._radius
        d['spread'] = self._spread
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['radius'], d['spread'])
        return obj

    def __repr__(self):
        return 'SpreadingLayer(radius={radius}, spread={spread})'.format(
                    radius=self._radius,
                    spread=self._spread)
