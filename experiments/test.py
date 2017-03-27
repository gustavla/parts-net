from __future__ import division, print_function, absolute_import

#from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
import deepdish as dd


import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels


def data_path(which):
    return os.path.expandvars('$MNIST_VARIATIONS_DIR/mnist-rot-{which}.h5'.format(which))


if pnet.parallel.main(__name__):
    ag.set_verbose(True)
    print("1")
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    #parser.add_argument('param', metavar='<param>', type=string)

    parser.add_argument('model',metavar='<model file>',type=str, help='Filename of model file')
    #parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'),help='Filename of data file')
    #parser.add_argument('label',metavar='<mnist data file>',type=argparse.FileType('rb'),help='Filename of data file')
    #parser.add_argument('test_data',metavar='<mnist test data file>',type=argparse.FileType('rb'),help='Filename of data file')
    #parser.add_argument('test_label',metavar='<mnist test data file>',type=argparse.FileType('rb'),help='Filename of data file')
    parser.add_argument('classifier',metavar='<classifier>', type=str, choices=('mixture', 'svm-mixture', 'rot-mixture'), help='num Of Class Model')
    parser.add_argument('numOfClassModel',metavar='<numOfClassModel>', type=str, help='num Of Class Model')
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('saveFile', metavar='<part Model file>', type=str,help='Filename of savable model file')
    parser.add_argument('-c', '--count', type=int)

    args = parser.parse_args()

    numOfClassModel = args.numOfClassModel
    saveFile = args.saveFile

    if numOfClassModel == 'many':
        num_class_models = [1, 2, 5, 10, 20]
    else:
        num_class_models = [int(numOfClassModel)]

    if args.classifier == 'svm-mixture':
        classifier_names = ['mixture', 'svm']
    else:
        classifier_names = [args.classifier]

    #data = ag.io.load(args.data)
    #label = ag.io.load(args.label)
    #test_data0 = ag.io.load(args.test_data)
    #test_label0 = ag.io.load(args.test_label)

    data, label = dd.io.load(data_path('train'), ['/data', '/labels'], sel=np.s_[:10000])
    data = data.astype(np.float64)
    test_data0, test_label0 = dd.io.load(data_path('test'), ['/data', '/labels'])
    test_data0 = test_data0.astype(np.float64)

    seed = args.seed
    rs = np.random.RandomState(seed)

    net = pnet.PartsNet.load(args.model)

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    if args.count is not None:
        iis = []
        for c in range(10):
            ii = np.where(label == c)[0]
            rs.shuffle(ii)
            iis.append(ii[:args.count//10])

        iis = np.concatenate(iis)
        rs.shuffle(iis)

        data = data[iis]
        label = label[iis]

    ims10k = data
    label10k = np.array(label).astype(np.int_)
    ims2k = test_data0
    label2k = test_label0.astype(np.int_)
    digits = range(10)
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
    else:
        sup_ims = ims10k
        sup_labels = label10k

    if net.layers[0].name == 'oriented-parts-layer':
        rotspreads = [0] # TODO, temporary
    else:
        rotspreads = [0]

    for classifier in classifier_names:
        num_class_models0 = num_class_models
        if classifier == 'svm':
            num_class_models0 = [1]

        for n_classes in num_class_models0:
            for rotspread in rotspreads:
                if net.layers[0].name == 'oriented-parts-layer':
                    net.layers[0]._settings['rotation_spreading_radius'] = rotspread

                if classifier == 'mixture':
                    layers = [
                        pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                        pnet.MixtureClassificationLayer(n_components=n_classes, min_prob=1e-5, settings=dict(seed=seed))
                    ]
                elif classifier == 'svm':
                    layers = [
                        pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                        pnet.SVMClassificationLayer(C=None, settings=dict(seed=seed)),
                    ]
                elif classifier == 'rot-mixture':
                    n_orientations = 16
                    layers = [
                        pnet.RotationMixtureClassificationLayer(n_components=n_classes,
                                                                n_orientations=n_orientations,
                                                                min_prob=1e-5,
                                                                settings=dict(seed=seed),
                                                                pooling_settings=dict(
                                shape=(4, 4),
                                strides=(4, 4),
                                rotation_spreading_radius=rotspread,
                            ))
                    ]

                clnet = pnet.PartsNet([net] + layers)

                start1 = time.time()
                print('Training supervised...')
                print(sup_ims.shape)
                clnet.train(sup_ims, sup_labels)
                print('Done.')
                end1 = time.time()

                clnet.save(saveFile)
                #clnet = pnet.PartsNet.load(saveFile)

                #print("Now testing...")
                ### Test ######################################################################

                corrects = 0
                total = 0
                test_ims = ims2k
                test_labels = label2k


                ims_batches = np.array_split(test_ims, 2500)
                labels_batches = np.array_split(test_labels, 2500)

                def format_error_rate(pr):
                    return "{:.2f}%".format(100*(1-pr))

                start2 = time.time()
                args = (tup+(clnet,) for tup in zip(ims_batches, labels_batches))
                for i, res in enumerate(pnet.parallel.starmap(test, args)):
                    corrects += res.sum()
                    total += res.size
                    pr = corrects / total
                end2 = time.time()

                error_rate = 1.0 - pr

                error_rates.append(error_rate)
                print('Classifier:', classifier, 'Components:', n_classes, 'Rotational spreading:', rotspread, 'Seed:', seed)
                print('error rate', error_rate * 100)
