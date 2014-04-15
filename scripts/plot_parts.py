from __future__ import division, print_function, absolute_import
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default=None, type=argparse.FileType('wb'), help='Filename of output file')

args = parser.parse_args()

import gv
import pnet
import os
import numpy as np
from pylab import cm
net = pnet.PartsNet.load('parts-net.npy')

p = net._layers[1]._parts[...,0]
shape = p.shape[-2:]

grid = gv.plot.ImageGrid(10, 10, p.shape[1:])

for i in xrange(p.shape[0]):
    grid.set_image(p[i], i//10, i%10, vmin=0, vmax=1, cmap=cm.RdBu_r)

if 0:
    print('shape', shape)
    print( np.asarray(list(gv.multirange(*[2]*np.prod(shape)))).shape)
    types = np.asarray(list(gv.multirange(*[2]*np.prod(shape)))).reshape((-1,) + shape)

    t = p[:,np.newaxis]
    x = types[np.newaxis]

    llh = x * np.log(t) + (1 - x) * np.log(1 - t)

    counts = np.bincount(np.argmax(llh.sum(-1).sum(-1), 0), minlength=p.shape[0])

    print('counts', counts)

    #import pdb; pdb.set_trace()

if args.output is None:
    plt.imshow(grid.image, interpolation='nearest')
    plt.show()
else:
    fn = args.output
    grid.save(args.output.name, scale=10)
    os.chmod(args.output.name, 0644)

