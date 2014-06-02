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
    
    parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts') 
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('saveFile'), metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file'
    
    args = parser.parse_args()
    numParts = args.numParts
    param = args.data
    saveFile = arg.saveFile

    data = np.load(param)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []
    

    for training_seed in xrange(1):
        print("Inside")
        layers = [
            pnet.OrientedPartsLayer(numParts, 32, (6, 6), settings=dict(outer_frame=1, 
                                                      em_seed=training_seed,
                                                      n_iter=15,
                                                      n_init=1,
                                                      threshold=2, 
                                                      samples_per_image=10, 
                                                      max_samples=10000,
                                                      #rotation_spreading_radius=1,
                                                      min_prob=0.0005,
                                                      bedges=dict(
                                                            k=5,
                                                            minimum_contrast=0.05,
                                                            spread='orthogonal',
                                                            radius=1,
                                                            #pre_blurring=1,
                                                            contrast_insensitive=False,
                                                          ),
                                                      )),
        ]

        net = pnet.PartsNet(layers)
        #TRAIN_SAMPLES = 12000 
        #TRAIN_SAMPLES = 1200 

        digits = range(10)
        #ims = ag.io.load_mnist('training', selection=slice(0 + 3000 * training_seed, TRAIN_SAMPLES + 3000 * training_seed), return_labels=False)
        #ims = mnist_data['training_image'][0 + 1000 * training_seed : TRAIN_SAMPLES + 1000 * training_seed]
        print('Extracting subsets...')
        
        ims10k = data[0][:10000]
        label10k = data[1][:10000]

        ims2k = data[0][10000:12000]
        label2k = data[0][10000:12000]
        print('Done.')

        #print(net.sizes(X[[0]]))

        start0 = time.time()
        print('Training unsupervised...')
        net.train(ims10k)
        print('Done.')
        end0 = time.time()

        net.save(saveFile)

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


            for classifier in 'mixture', 'svm':
                for rotspread in [0, 1]:
                    net.layers[0]._settings['rotation_spreading_radius'] = rotspread

                    print('Classifier:', classifier, 'Rotational spreading:', rotspread)
                    if classifier == 'mixture':
                        cl = pnet.MixtureClassificationLayer(n_components=1, min_prob=1e-5)
                    elif classifier == 'svm':
                        cl = pnet.SVMClassificationLayer(C=None)

                    clnet = pnet.PartsNet([net, cl])
                            
                    start1 = time.time()
                    print('Training supervised...')
                    clnet.train(sup_ims, sup_labels)
                    print('Done.')
                    end1 = time.time()

                    #print("Now testing...")
                    ### Test ######################################################################

                    corrects = 0
                    total = 0
                    if 0:
                        test_ims, test_labels = mnist_data['test_image'], mnist_data['test_label']
                    else:
                        test_ims = ims2k
                        test_labels = label2k


                    with gv.Timer("Split to batches"):
                        ims_batches = np.array_split(test_ims, 10)
                        labels_batches = np.array_split(test_labels, 10)

                    def format_error_rate(pr):
                        return "{:.2f}%".format(100*(1-pr))

                    import gv
                    #with gv.Timer('Testing'):
                    start2 = time.time()
                    args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
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


    if 0:
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

