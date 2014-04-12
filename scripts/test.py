#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import amitgroup as ag
import os
import pnet

def test(ims, labels, net, mm):
    #model = np.load(settings['detector']['file']).flat[0]
    #descriptor = gv.load_descriptor(settings)
    #descriptor = ag.features.PartsDescriptor.load(settings['parts']['file'])

    Z = net.extract(ims)

    if 0:
        return svc.predict(Z.reshape((Z.shape[0], -1))) == labels
    else:
        #ZZ = Z.reshape((Z.shape[0], -1))

        Xs = Z[:,np.newaxis,np.newaxis]
        theta = mm[np.newaxis]
        #print('Z', Z.shape)
        #print('mm', mm.shape)

        llh = Xs * np.log(theta) + (1 - Xs) * np.log(1 - theta)
        yhat = np.argmax(np.apply_over_axes(np.sum, llh, [-3, -2, -1])[...,0,0,0].max(-1), axis=1)

        return yhat == labels


if pnet.parallel.main(__name__): 
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    #parser.add_argument('--modify-settings', type=str, default='', help='Overwrite settings')

    ims, labels = ag.io.load_mnist('testing')#, selection=slice(0, 1000))

    ims_batches = np.array_split(ims, 200)
    labels_batches = np.array_split(labels, 200)
    #model = test(ims, labels, settings) 
    #np.save(output_file, model)

    net = pnet.PartsNet.load('parts-net.npy')
    #svc = np.load('svc.npy').flat[0]
    mm = np.load('mm.npy')

    args = [tup+(net, mm,) for tup in zip(ims_batches, labels_batches)]

    corrects = 0
    total = 0

    def format_error_rate(pr):
        return "{:.2f}%".format(100*(1-pr))

    print("Starting...")
    for i, res in enumerate(pnet.parallel.starmap_unordered(test, args)):
        if i != 0 and i % 20 == 0:
            print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims), format_error_rate(pr)))

        corrects += res.sum()
        total += res.size

        pr = corrects / total


    print("Final error rate:", format_error_rate(pr)) 
