from __future__ import division, print_function, absolute_import 

#from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
#import gv

import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__): 
    print("1")
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    parser.add_argument('param', metavar='<param>', type=float)

    args0 = parser.parse_args()
    param = args0.param

    #for i in xrange(1, 7):
    #    print(make_support(i, 4).astype(np.uint8))

    #params = randomize_layers_parameters(args0.seed)
    #print(params)

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    maxdepth = 7 

    print("2")
    # Switch which experiment here
    #from pnet.mnist_danny import parse_background_random as loadf
    from pnet.mnist_danny import parse_background_images as loadf
    print("Loading...")
    mnist_data = loadf()
    print("Done.")

    for training_seed in xrange(1):
        layers = [
            #pnet.IntensityThresholdLayer(),
            pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),#, pre_blurring=1.0),
            #pnet.IntensityThresholdLayer(),
            #pnet.IntensityThresholdLayer(),
       
            pnet.PartsLayer(250, (7, 7), settings=dict(outer_frame=1, 
                                                      em_seed=training_seed,
                                                      threshold=2, 
                                                      samples_per_image=60, 
                                                      max_samples=200000,
                                                      train_limit=10000,
                                                      min_prob=0.00005,
                                                      )),
            pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
        ]


        if 0:
            layers += [
                pnet.RandomForestPartsLayer(256, (5, 5), settings=dict(outer_frame=1, 
                                                          em_seed=training_seed,
                                                          threshold=2, 
                                                          samples_per_image=40, 
                                                          max_samples=200000,
                                                          train_limit=10000,
                                                          min_prob=0.0005,
                                                          trees=10,  
                                                          max_depth=3,
                                                          )),
                pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ]
        elif 0:
            layers += [
                pnet.GaussianPartsLayer(100, (5, 5), settings=dict(
                                                          em_seed=training_seed,
                                                          samples_per_image=40, 
                                                          max_samples=200000,
                                                          train_limit=100000,
                                                          )),
                pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ]
        elif 0:
            layers += [
                pnet.PartsLayer(1000, (5, 5), settings=dict(outer_frame=1, 
                                                          em_seed=training_seed,
                                                          threshold=2, 
                                                          samples_per_image=40, 
                                                          max_samples=200000,
                                                          train_limit=100000,
                                                          min_prob=0.0005,
                                                          )),
                pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ]
        elif 0:
            layers += [
                pnet.BinaryTreePartsLayer(maxdepth, (5, 5), settings=dict(outer_frame=1, 
                                                          em_seed=training_seed,
                                                          threshold=2, 
                                                          samples_per_image=40, 
                                                          max_samples=200000,
                                                          train_limit=10000,
                                                          min_prob=0.005,
                                                          #keypoint_suppress_radius=1,
                                                          min_samples_per_part=50,
                                                          split_criterion='IG',
                                                          split_entropy=0.2,
                                                          min_information_gain=0.01, 
                                                          )),
                pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ] 
        elif 0:
            layers += [
                pnet.BinaryTreePartsLayer(maxdepth, (5, 5), settings=dict(outer_frame=1, 
                                                          em_seed=training_seed,
                                                          threshold=2, 
                                                          samples_per_image=40, 
                                                          max_samples=200000,
                                                          train_limit=10000,
                                                          min_prob=0.0005,
                                                          #keypoint_suppress_radius=1,
                                                          min_samples_per_part=50,
                                                          split_criterion=split_criterion,
                                                          split_entropy=split_entropy,
                                                          min_information_gain=split_entropy, 
                                                          )),
                pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ] 
            [
                pnet.BinaryTreePartsLayer(10, (1, 1), settings=dict(outer_frame=0, 
                                                          em_seed=training_seed+1,
                                                          threshold=1, 
                                                          samples_per_image=200, 
                                                          max_samples=1000000, 
                                                          train_limit=10000,
                                                          #min_information_gain=0.05,
                                                          split_entropy=0.05,
                                                          min_prob=0.0005
                                                          )),
                pnet.PoolingLayer(shape=(1, 1), strides=(1, 1)),
            ] 

        layers += [
            pnet.MixtureClassificationLayer(n_components=1, min_prob=1e-5),
            #pnet.SVMClassificationLayer(C=None),
        ]

        net = pnet.PartsNet(layers)
        TRAIN_SAMPLES = 10000 
        #TRAIN_SAMPLES = 1200 
        print(training_seed)
        digits = range(10)
        #ims = ag.io.load_mnist('training', selection=slice(0 + 3000 * training_seed, TRAIN_SAMPLES + 3000 * training_seed), return_labels=False)
        ims = mnist_data['training_image'][0 + 1000 * training_seed : TRAIN_SAMPLES + 1000 * training_seed]
        ims_label = mnist_data['training_label'][0 + 1000 * training_seed : TRAIN_SAMPLES + 1000 * training_seed]
        validation_ims = mnist_data['training_image'][10000:12000]
        validation_label = mnist_data['training_label'][10000:12000]
        #print(net.sizes(X[[0]]))
        print(ims.shape)
        start0 = time.time()
        net.train(ims)
        end0 = time.time()

        N = 1000

        sup_ims = []
        sup_labels = []
        # Load supervised training data
        for d in digits:
            if N is None:
                ims0 = ims[ims_label == d]
            else:
                #ims0 = ag.io.load_mnist('training', [d], selection=slice(N*training_seed, N*(1+training_seed)), return_labels=False)
                ims0 = ims[ims_label == d]
            sup_ims.append(ims0)
            sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

        sup_ims = np.concatenate(sup_ims, axis=0)
        sup_labels = np.concatenate(sup_labels, axis=0)

        #print('labels', np.bincount(sup_labels, minlength=10))

        start1 = time.time()
        net.train(sup_ims, sup_labels)
        end1 = time.time()

        #print("Now testing...")
        ### Test ######################################################################

        corrects = 0
        total = 0
        test_ims, test_labels = mnist_data['test_image'], mnist_data['test_label']
        test_ims = validation_ims
        test_labels = validation_label
        # TEMP
        if 0:
            test_ims = test_ims[:1000]
            test_labels = test_labels[:1000]

        #with gv.Timer("Split to batches"):
        ims_batches = np.array_split(test_ims, 200)
        labels_batches = np.array_split(test_labels, 200)

        def format_error_rate(pr):
            return "{:.2f}%".format(100*(1-pr))

        #import gv
        #with gv.Timer('Testing'):
        start2 = time.time()
        args = (tup+(net,) for tup in itr.izip(ims_batches, labels_batches))
        for i, res in enumerate(pnet.parallel.starmap(test, args)):
            corrects += res.sum()
            total += res.size
            pr = corrects / total
        end2 = time.time()

        error_rate = 1.0 - pr 


        num_parts = 0#net.layers[1].num_parts

        error_rates.append(error_rate)
        print(training_seed, 'error rate', error_rate * 100, 'num parts', num_parts)#, 'num parts 2', net.layers[3].num_parts)

        unsup_training_times.append(end0 - start0)
        sup_training_times.append(end1 - start1)
        testing_times.append(end2 - start2)

        #print('times', end0-start0, end1-start1, end2-start2)

        all_num_parts.append(num_parts)

        #vz.section('MNIST')
        #gv.img.save_image(vz.generate_filename(), test_ims[0])
        #gv.img.save_image(vz.generate_filename(), test_ims[1])
        #gv.img.save_image(vz.generate_filename(), test_ims[2])

        # Vz
        #net.infoplot(vz)

        #vz.flush()
        net.save('tmp{}.npy'.format(training_seed))

    print(r"{ppl} & {depth} & {num_parts} & {unsup_time:.1f} & {test_time:.1f} & ${rate:.2f} \pm {std:.2f}$ \\".format(
        ppl=2,
        depth=maxdepth,
        num_parts=r'${:.0f} \pm {:.0f}$'.format(np.mean(all_num_parts), np.std(all_num_parts)),
        unsup_time=np.median(unsup_training_times) / 60,
        #sup_time=np.median(sup_training_times),
        test_time=np.median(testing_times) / 60,
        rate=100*np.mean(error_rates),
        std=100*np.std(error_rates)))

    print(r"{ppl} {depth} {num_parts} {unsup_time} {test_time} {rate} {std}".format(
        ppl=2,
        depth=maxdepth,
        num_parts=r'${:.0f} \pm {:.0f}$'.format(np.mean(all_num_parts), np.std(all_num_parts)),
        unsup_time=np.median(unsup_training_times) / 60,
        #sup_time=np.median(sup_training_times),
        test_time=np.median(testing_times) / 60,
        rate=100*np.mean(error_rates),
        std=100*np.std(error_rates)))


    #np.savez('gdata2-{}-{}-{}.npz'.format(maxdepth, split_criterion, split_entropy), all_num_parts=all_num_parts, unsup_time=unsup_training_times, test_time=testing_times, rates=error_rates)

    print('mean error rate', np.mean(error_rates) * 100)
    #net.save(args.model)
