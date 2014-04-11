#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import amitgroup as ag
import os
import pnet

def test(ims, labels, net, svc):
    #model = np.load(settings['detector']['file']).flat[0]
    #descriptor = gv.load_descriptor(settings)
    #descriptor = ag.features.PartsDescriptor.load(settings['parts']['file'])

    X = ag.features.bedges(ims, k=5, radius=1, spread='orthogonal', minimum_contrast=0.05) 

    Z = net.extract(X)

    return svc.predict(Z.reshape((Z.shape[0], -1))) == labels


if pnet.parallel.main(__name__): 
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    #parser.add_argument('--modify-settings', type=str, default='', help='Overwrite settings')

    ims, labels = ag.io.load_mnist('testing')#, selection=slice(0, 1000))

    ims_batches = np.array_split(ims, 500)
    labels_batches = np.array_split(labels, 500)
    #model = test(ims, labels, settings) 
    #np.save(output_file, model)

    net = pnet.PartsNet.load('parts-net.npy')
    svc = np.load('svc.npy').flat[0]

    args = [tup+(net, svc,) for tup in zip(ims_batches, labels_batches)]

    corrects = 0
    total = 0

    def format_error_rate(pr):
        return "{:.2f}%".format(100*(1-pr))

    print("Starting...")
    for res in pnet.parallel.starmap_unordered(test, args):
        corrects += res.sum()
        total += res.size

        pr = corrects / total

        print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims), format_error_rate(pr)))

    print("Final error rate:", format_error_rate(pr)) 
