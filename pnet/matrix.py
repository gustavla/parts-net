from __future__ import division, print_function, absolute_import
import numpy as np


def translation(dx, dy):
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotation(a):
    return np.matrix([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a),  0],
                     [0,         0,          1]])

def scale(sx, sy=None):
    if sy is None:
        sy = sx
    return np.matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

def identity():
    return np.matrix(np.eye(3))
