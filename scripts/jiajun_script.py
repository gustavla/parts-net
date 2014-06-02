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
    
    parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts') 
    parser.add_argument('patchSize', metavar='<patch size>', type=int, help='size of patches')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('label',metavar='<mnist label file>', type = argparse.FileType('rb'), help = 'Filename of label file')
    parser.add_argument('saveFile', metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file')
    parser.add_argument('classifier',metavar='<classifier>', type=str, choices=('svm-mixture', 'rot-mixture'), help='num Of Class Model')
    parser.add_argument('numOfClassModel',metavar='<numOfClassModel>', type=str, help='num Of Class Model')
    
    args = parser.parse_args()
    numParts = args.numParts
   
    numOfClassModel = args.numOfClassModel

    if numOfClassModel == 'many':
        num_class_models = [1, 2, 5, 10, 20]
    else:
        num_class_models = [int(numOfClassModel)]

    if args.classifier == 'svm-mixture':
        classifier_names = ['mixture', 'svm']

    classifier_name = args.classifier
    param = args.data
    saveFile = args.saveFile
    
    patchSize = args.patchSize
    
    data = np.load(param)
    label = np.load(args.label)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []
    

    for training_seed in xrange(2):
        print("Inside")
        layers = [
            pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.08),#, pre_blurring=1.0),
       
            pnet.PartsLayer(numParts, (patchSize, patchSize), settings=dict(outer_frame=2, 
                                                      em_seed=training_seed,
                                                      threshold=2, 
                                                      samples_per_image=10, 
                                                      max_samples=100000,
                                                      min_prob=0.00005,
                                                      )),
        ]

        net = pnet.PartsNet(layers)

        digits = range(10)
        print('Extracting subsets...')
        
        ims10k = data[:10000]
        label10k = label[:10000]

        ims2k = data[10000:12000]
        label2k = label[10000:12000]
        print('Done.')


        start0 = time.time()
        print('Training unsupervised...')
        net.train(ims10k)
        print('Done.')
        end0 = time.time()

        net.save(saveFile)
   
 
        digits = range(10)
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
        print(sup_ims.shape)
        print(sup_labels)

        for classifier in classifier_names:
            num_class_models0 = num_class_models
            if classifier == 'svm':
                num_class_models0 = [1]

            for n_classes in num_class_models0:
                print('Classifier:', classifier, 'Components:', n_classes)
                if classifier == 'mixture':
                    layers = [
                        pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                        pnet.MixtureClassificationLayer(n_components=n_classes, min_prob=1e-5)
                    ]
                elif classifier == 'svm':
                    layers = [
                        pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                        pnet.SVMClassificationLayer(C=None),
                    ]

                clnet = pnet.PartsNet([net] + layers)
                        
                start1 = time.time()
                print('Training supervised...')
                print(sup_ims.shape)
                clnet.train(sup_ims, sup_labels)
                print('Done.')
                end1 = time.time()

                corrects = 0
                total = 0
                if 0:
                    test_ims, test_labels = mnist_data['test_image'], mnist_data['test_label']
                else:
                    test_ims = ims2k
                    test_labels = label2k


                ims_batches = np.array_split(test_ims, 10)
                labels_batches = np.array_split(test_labels, 10)

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
                print(training_seed, 'error rate', error_rate * 100, 'num parts', numParts)

                unsup_training_times.append(end0 - start0)
                sup_training_times.append(end1 - start1)
                testing_times.append(end2 - start2)

                print('times', end0-start0, end1-start1, end2-start2)

