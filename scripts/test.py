#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
import amitgroup as ag
import os
import pnet

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__): 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', metavar='<parts net file>', type=argparse.FileType('rb'), help='Filename of model file')

    args = parser.parse_args()

    ims, labels = ag.io.load_mnist('testing')#, selection=slice(0, 1000))

    ims_batches = np.array_split(ims, 200)
    labels_batches = np.array_split(labels, 200)

    net = pnet.PartsNet.load(args.model)

    args = [tup+(net,) for tup in zip(ims_batches, labels_batches)]

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
