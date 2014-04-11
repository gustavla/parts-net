from __future__ import division, print_function, absolute_import 
import numpy as np
import amitgroup as ag

import pnet
from sklearn.svm import LinearSVC 

# Example of how to define the model

net = pnet.PartsNet([
    pnet.PartsLayer(10, (5, 5), settings=dict(outer_frame=0, threshold=1, samples_per_image=15)),
    pnet.PoolingLayer(shape=(2, 2), strides=(2, 2)),
    pnet.PartsLayer(20, (4, 4), settings=dict(outer_frame=1, threshold=1, samples_per_image=20)),
    pnet.PoolingLayer(shape=(3, 3), strides=(3, 3)),
])

digits = range(10)

ims = ag.io.load_mnist('training', selection=slice(2000), return_labels=False)

X = ag.features.bedges(ims, k=5, radius=1, spread='orthogonal', minimum_contrast=0.05) 

print(X.shape)
net.train(X)

YY = []
labels = []

for d in digits:
    ims0 = ag.io.load_mnist('training', [d], selection=slice(100), return_labels=False)
    X0 = ag.features.bedges(ims0, k=5, radius=1, spread='orthogonal', minimum_contrast=0.05) 

    Z = net.extract(X0)
    Yflat = Z.reshape((Z.shape[0], -1))
    YY.append(Yflat)
    labels.append(np.ones(Yflat.shape[0]) * d)

YY = np.concatenate(YY, axis=0)
labels = np.concatenate(labels, axis=0)

svc = LinearSVC()
svc.fit(YY, labels)

np.save('svc.npy', svc)

#X = (X[...,np.newaxis] > 0.5).astype(np.uint8)


#z = net.extract_features(X)

#print("Layers")
#for i, layer in enumerate(net._layers):
    #print(i, layer.TMP_output_shape)

