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

print("0 pid", os.getpid())

if pnet.parallel.main(__name__): 
    ag.set_verbose(True)
    print("1")
    sys.stdout.flush()
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    #parser.add_argument('param', metavar='<param>', type=string)
    
    parser.add_argument('size', metavar='<part size>', type=int)
    parser.add_argument('orientations', metavar='<num orientations>', type=int)
    parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts') 
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('saveFile', metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file')
    args = parser.parse_args()
    print("2")
    sys.stdout.flush()
    part_size = args.size
    num_orientations = args.orientations
    numParts = args.numParts
    param = args.data
    saveFile = args.saveFile
    seed = args.seed

    data = np.load(param)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []
    

    for training_seed in [seed]:
        print("Inside")
        print("3")
        sys.stdout.flush()
        if 1:
            layers = [
                pnet.OrientedPartsLayer(numParts, num_orientations, (part_size, part_size), settings=dict(outer_frame=2, 
                                                          em_seed=training_seed,
                                                          n_iter=5,
                                                          n_init=1,
                                                          threshold=2, 
                                                          #samples_per_image=20, 
                                                          samples_per_image=50, 
                                                          max_samples=80000,
                                                          #max_samples=5000,
                                                          #max_samples=100000,
                                                          #max_samples=2000,
                                                          rotation_spreading_radius=0,
                                                          min_prob=0.0005,
                                                          bedges=dict(
                                                                k=5,
                                                                minimum_contrast=0.05,
                                                                spread='orthogonal',
                                                                #spread='box',
                                                                radius=1,
                                                                #pre_blurring=1,
                                                                contrast_insensitive=False,
                                                              ),
                                                          )),
            ]
        else:
            layers = [
                pnet.EdgeLayer(
                                k=5,
                                minimum_contrast=0.05,
                                spread='orthogonal',
                                #spread='box',
                                radius=1,
                                #pre_blurring=1,
                                contrast_insensitive=False,
                ),
                pnet.PartsLayer(numParts, (6, 6), settings=dict(outer_frame=1, 
                                                          em_seed=training_seed,
                                                          threshold=2, 
                                                          samples_per_image=40, 
                                                          max_samples=100000,
                                                          #max_samples=30000,
                                                          train_limit=10000,
                                                          min_prob=0.00005,
                                                          )),
            ]


        net = pnet.PartsNet(layers)
        #TRAIN_SAMPLES = 12000 
        #TRAIN_SAMPLES = 1200 

        digits = range(10)
        #ims = ag.io.load_mnist('training', selection=slice(0 + 3000 * training_seed, TRAIN_SAMPLES + 3000 * training_seed), return_labels=False)
        #ims = mnist_data['training_image'][0 + 1000 * training_seed : TRAIN_SAMPLES + 1000 * training_seed]
        print('Extracting subsets...')
        
        ims10k = data[:10000]

        #data1 = np.load('data/allDATA/mnistIMG.npy')
        #data2 = np.load('data/allDATA/mnistROT.npy')


        #ims10k = np.concatenate([data1, data2])
        #rs = np.random.RandomState(0)
        #rs.shuffle(ims10k)


        start0 = time.time()
        print('Training unsupervised...')
        sys.stdout.flush()
        net.train(ims10k)
        print('Done.')
        sys.stdout.flush()
        end0 = time.time()

        net.save(saveFile)        

