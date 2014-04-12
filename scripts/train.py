from __future__ import division, print_function, absolute_import 
import numpy as np
import amitgroup as ag
ag.set_verbose(True)

from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC 

# Example of how to define the model
if 1:
    image_layer = [
        pnet.IntensityThresholdLayer(threshold=0.5),
        pnet.PartsLayer(10, (2, 2), settings=dict(outer_frame=0, 
                                                  threshold=1, 
                                                  samples_per_image=200, 
                                                  max_samples=100000, 
                                                  min_prob=0.001)),
        pnet.PoolingLayer(shape=(3, 3), strides=(1, 1)),
    ]
else:
    image_layer = [
        pnet.EdgeLayer(k=5, radius=1, spread='box', minimum_contrast=0.05),
    ]


layers = image_layer 

layers += [
    pnet.PartsLayer(100, (5, 5), settings=dict(outer_frame=1, 
                                              threshold=15, 
                                              samples_per_image=40, 
                                              max_samples=50000, 
                                              min_prob=0.0005)),
    pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
]

net = pnet.PartsNet(layers)

    #pnet.PartsLayer(20, (1, 1), settings=dict(outer_frame=0, threshold=1, samples_per_image=100)),
    #pnet.PoolingLayer(shape=(2, 2), strides=(2, 2)),

digits = range(10)

ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)

#print(net.sizes(X[[0]]))

net.train(ims)

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

#X = (X[...,np.newaxis] > 0.5).astype(np.uint8)


#z = net.extract_features(X)

print("Layers")
for i, layer in enumerate(net._layers):
    print(i, layer.TMP_output_shape)

