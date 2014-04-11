from __future__ import division, print_function, absolute_import 
import numpy as np
import amitgroup as ag

import pnet

# Example of how to define the model

net = pnet.PartsNet([
    pnet.PartsLayer(10, (4, 4), settings=dict(outer_frame=0, threshold=1, samples_per_image=15)),
    #pnet.PoolingLayer(shape=(1, 1), strides=(1, 1)),
    #pnet.PartsLayer(20, (5, 5), settings=dict(outer_frame=1, threshold=1, samples_per_image=50)),
    #pnet.PoolingLayer(shape=(3, 3), strides=(3, 3)),
])

X = ag.io.load_mnist('training', selection=slice(1000), return_labels=False)
X = (X[...,np.newaxis] > 0.5).astype(np.uint8)


print(X.shape)
net.train(X)

Y = net.extract(X)
print(Y.shape)

print(Y[0])
#z = net.extract_features(X)



