from __future__ import division, print_function, absolute_import 

import amitgroup as ag
import numpy as np

def load_data(filename):
    if filename[0] == ':':
        if filename == ':small-norb-training':
            data = ag.io.load_small_norb('training', ret='x', x_dtype=np.float32)
            data = data.transpose(0, 2, 3, 1).copy()
        elif filename == ':cifar-10-training':
            data = np.concatenate([ag.io.load_cifar_10('training', ret='x',
                                   offset=10000*b) for b in range(5)], axis=0)
            data = data.transpose(0, 2, 3, 1).copy()
        else:
            raise ValueError('Unknown data: {}'.format(filename))
    else:
        data = ag.io.load(filename)


    return data
