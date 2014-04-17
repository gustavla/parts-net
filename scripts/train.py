from __future__ import division, print_function, absolute_import 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true')

args = parser.parse_args()

if args.log:
    from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import sys
ag.set_verbose(True)

from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC 


layers = [
    #pnet.IntensityThresholdLayer(),
    pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
    #pnet.IntensityThresholdLayer(),
    pnet.PartsLayer(40, (3, 3), settings=dict(outer_frame=0, 
                                              threshold=8, 
                                              samples_per_image=40, 
                                              max_samples=10000, 
                                              min_prob=0.0005)),
    pnet.PoolingLayer(shape=(3, 3), strides=(1, 1)),
] + [
    pnet.PartsLayer(100, (2, 2), settings=dict(outer_frame=0,
                                              threshold=5,
                                              samples_per_image=40,
                                              max_samples=10000,
                                              min_prob=0.0005,
                                              min_llh=-40,
                                              n_coded=1
                                              )),
    pnet.PoolingLayer(shape=(4, 4), strides=(2, 2)),
]

net = pnet.PartsNet(layers)

digits = range(10)
ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)

#print(net.sizes(X[[0]]))

net.train(ims)

X = net.extract(ims)
#print('X mean', X.mean())
#print('X', np.apply_over_axes(np.mean, X, [0, 1, 2]).ravel())

YY = []
labels = []


mm_models = []

for d in digits:
    ims0 = ag.io.load_mnist('training', [d], selection=slice(100), return_labels=False)
    Z = net.extract(ims0)
    Yflat = Z.reshape((Z.shape[0], -1))
    YY.append(Yflat)
    labels.append(np.ones(Yflat.shape[0]) * d)

         

    if 1:
        min_prob = 0.0005
        M = 1
        mm = BernoulliMM(n_components=M, n_iter=10, n_init=1, random_state=0, min_prob=min_prob)
        mm.fit(Yflat)
        mm_models.append(mm.means_.reshape((M,)+Z.shape[1:]))

if 0:
    YY = np.concatenate(YY, axis=0)
    labels = np.concatenate(labels, axis=0)

    svc = LinearSVC()
    svc.fit(YY, labels)
    np.save('svc.npy', svc)
else:
    mm_models = np.asarray(mm_models)
    np.save('mm.npy', mm_models)

net.save('parts-net.npy')

if args.log:
    net.infoplot(vz)
    vz.finalize()
