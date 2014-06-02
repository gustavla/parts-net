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
    print("ohhh")
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'),help='Filename of data file')
    parser.add_argument('label',metavar='<mnist data file>',type=argparse.FileType('rb'),help='Filename of data file')
    parser.add_argument('numOfClassModel',metavar='<numOfClassModel>', type=int, help='num Of Class Model')

    args = parser.parse_args()

    param = args.model
    numOfClassModel = args.numOfClassModel
    param = args.data

    data = np.load(param)
    label = np.load(args.label)

    net = pnet.PartsNet.load(args.model)

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []
    ims10k = data[:10000]
    label10k = np.array(label[:10000]) 
    np.save('a.npy',label10k)
    ims2k = data[10000:12000]
    label2k = np.array(label[10000:12000])
    np.save('b.npy',label2k) 
    print(ims2k.shape)
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

    for classifier in 'mixture', 'svm':
        for rotspread in [0, 1]:
            net.layers[0]._settings['rotation_spreading_radius'] = rotspread

            print('Classifier:', classifier, 'Rotational spreading:', rotspread)
            if classifier == 'mixture':
                cl = pnet.MixtureClassificationLayer(n_components=numOfClassModel, min_prob=1e-5)
            elif classifier == 'svm':
                cl = pnet.SVMClassificationLayer(C=None)

            clnet = pnet.PartsNet([net, cl])
                    
            start1 = time.time()
            print('Training supervised...')
            print(sup_ims.shape)
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

