from __future__ import division, print_function, absolute_import
import gv
import pnet
import os
from pylab import cm
net = pnet.PartsNet.load('parts-net.npy')

p = net._layers[1]._parts[...,0]

grid = gv.plot.ImageGrid(10, 10, p.shape[1:])

for i in xrange(p.shape[0]):
    grid.set_image(p[i], i//10, i%10, vmin=0, vmax=1, cmap=cm.RdBu_r)

fn = '/home/larsson/html/plot.png'
grid.save(fn, scale=10)
os.chmod(fn, 0644)
