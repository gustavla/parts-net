from __future__ import division, print_function, absolute_import 

#from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os

import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__): 
    ag.set_verbose(True)
    print("1")
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    #parser.add_argument('param', metavar='<param>', type=string)
    
    parser.add_argument('model',metavar='<model file>',type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('classifier',metavar='<classifier>', type=str, choices=('mixture', 'svm-mixture', 'rot-mixture'), help='num Of Class Model')
    parser.add_argument('samples',metavar='<samples>', type=int, help='Samples per class')
    parser.add_argument('numOfClassModel',metavar='<numOfClassModel>', type=str, help='num Of Class Model')

    args = parser.parse_args()

    numOfClassModel = args.numOfClassModel
    digits = range(10)
    
    N = args.samples

    if numOfClassModel == 'adaptive':
        num_class_models = [min(20, int(np.ceil(N / 500)))]
    elif numOfClassModel == 'many':
        num_class_models = [1, 2, 5, 10, 15, 20]
    elif numOfClassModel == 'afew':
        num_class_models = [1, 2, 5, 10]
    else:
        num_class_models = [int(numOfClassModel)]

    if args.classifier == 'svm-mixture':
        classifier_names = ['mixture', 'svm']
    else:
        classifier_names = [args.classifier]

    #data = np.load(args.data)
    #label = np.load(args.label)
    
    if N >= 6000:
        N = None

    net = pnet.PartsNet.load(args.model)

    #images, labels = ag.io.load_mnist('training') 

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    all_num_parts = []
    #ims10k = data[:10000]
    #label10k = np.array(label[:10000]) 
    #np.save('a.npy',label10k)
    #ims2k = data[10000:12000]
    #label2k = np.array(label[10000:12000])

    ims2k, label2k = ag.io.load_mnist('testing')

    #np.save('b.npy',label2k) 
    print(ims2k.shape)
    if 0:
        sup_ims = []
        sup_labels = []
        # Load supervised training data
        for d in digits:
            ims0 = ims10k[label10k == d]
            sup_ims.append(ims0)
            sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

        sup_ims = np.concatenate(sup_ims, axis=0)
        sup_labels = np.concatenate(sup_labels, axis=0)
    print("=================")
    #print(sup_labels)

    if net.layers[0].name == 'oriented-parts-layer':
        rotspreads = [0]
    else:
        rotspreads = [0]

    all_images, all_labels = ag.io.load_mnist('training') 

    for classifier in classifier_names:
        num_class_models0 = num_class_models
        if classifier == 'svm':
            num_class_models0 = [1]

        # per class: 30 100 200 500 1000 2000 full

        # 1 1 2 5 10 20 20

        for n_classes in num_class_models0:
            for rotspread in rotspreads:
                trials = 3
                if N == 6000:
                    trials = 1 

                rs = np.random.RandomState(0)
                error_rates = []
                print('Classifier:', classifier, 'Components:', n_classes, 'Rotational spreading:', rotspread)
                for trial in xrange(trials):
                    if N == 60000:
                        sup_ims = all_images
                        sup_labels = all_labels
                    else:
                        II = np.arange(len(all_images))
                        rs.shuffle(II)

                        perm_images = all_images[II]
                        perm_labels = all_labels[II]

                        image_slices = []
                        label_slices = []
                        for d in digits:
                            X = perm_images[perm_labels == d][:N]
                            image_slices.append(X)
                            label_slices.append(np.ones(len(X), dtype=np.int64) * d)

                        sup_ims = np.concatenate(image_slices)
                        sup_labels = np.concatenate(label_slices)


                    if net.layers[0].name == 'oriented-parts-layer':
                        net.layers[0]._settings['rotation_spreading_radius'] = rotspread

                    if classifier == 'mixture':
                        layers = [
                            pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                            pnet.MixtureClassificationLayer(n_components=n_classes, min_prob=1e-5, settings=dict(seed=trial))
                        ]
                    elif classifier == 'svm':
                        layers = [
                            pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                            pnet.SVMClassificationLayer(C=None, settings=dict(seed=trial)),
                        ]
                    elif classifier == 'rot-mixture':
                        n_orientations = 16
                        layers = [
                            pnet.RotationMixtureClassificationLayer(n_components=n_classes, n_orientations=n_orientations, min_prob=1e-5, pooling_settings=dict(
                                    shape=(4, 4),
                                    strides=(4, 4),
                                    rotation_spreading_radius=rotspread,
                                ))
                        ]

                    clnet = pnet.PartsNet([net] + layers)
                            
                    start1 = time.time()
                    clnet.train(sup_ims, sup_labels)
                    end1 = time.time()

                    #print("Now testing...")
                    ### Test ######################################################################

                    corrects = 0
                    total = 0
                    test_ims = ims2k
                    test_labels = label2k


                    ims_batches = np.array_split(test_ims, 200)
                    labels_batches = np.array_split(test_labels, 200)

                    def format_error_rate(pr):
                        return "{:.2f}%".format(100*(1-pr))

                    #with gv.Timer('Testing'):
                    start2 = time.time()
                    args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
                    for i, res in enumerate(pnet.parallel.starmap(test, args)):
                        corrects += res.sum()
                        total += res.size
                        pr = corrects / total
                    end2 = time.time()

                    error_rate = 1.0 - pr 


                    error_rates.append(error_rate)
                    print(trial, 'error rate', error_rate * 100)
            
                print('Classifier:', classifier, 'Components:', n_classes, 'Rotational spreading:', rotspread)
                print("error rates", error_rates)
                print("mean error rate", 100 * np.mean(error_rates))

