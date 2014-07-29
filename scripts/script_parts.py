from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.rc('font', size=8)

from vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
from amitgroup.plot import ImageGrid

import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__):
    ag.set_verbose(True)
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    #parser.add_argument('param', metavar='<param>', type=string)

    parser.add_argument('size', metavar='<part size>', type=int)
    parser.add_argument('orientations', metavar='<num orientations>', type=int)
    parser.add_argument('num_parts', metavar='<number of parts>', type=int, help='number of parts')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('saveFile', metavar='<output file>', type=argparse.FileType('wb'),help='Filename of savable model file')
    parser.add_argument('--p1', type=int, default=10)
    parser.add_argument('--p2', type=int, default=4)
    parser.add_argument('--p3', type=int, default=3)
    args = parser.parse_args()
    part_size = args.size
    num_orientations = args.orientations
    num_parts = args.num_parts
    param = args.data
    saveFile = args.saveFile
    seed = args.seed

    param1 = args.p1
    param2 = args.p2
    param3 = args.p3


    data = ag.io.load(param)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    SHORT = True#False

    for training_seed in [seed]:
        if 1:
            settings=dict(n_iter=10,
                          seed=0,
                          n_init=5,
                          standardize=True,
                          samples_per_image=100,
                          max_samples=10000,
                          uniform_weights=True,
                          max_covariance_samples=None,
                          covariance_type='diag',
                          min_covariance=0.0025,
                          logratio_thresh=-np.inf,
                          std_thresh=0.05,
                          std_thresh_frame=0,
                          #rotation_spreading_radius=0,
                          )

            layers = [
                pnet.OrientedGaussianPartsLayer(n_parts=8, n_orientations=1,
                                                part_shape=(3, 3),
                                                settings=settings),
            ]
            if not SHORT:
                layers += [
                    pnet.PoolingLayer(shape=(1, 1), strides=(1, 1)),
                    pnet.OrientedPartsLayer(n_parts=num_parts,
                                            n_orientations=num_orientations,
                                            part_shape=(part_size, part_size),
                                            settings=dict(outer_frame=1,
                                                          seed=training_seed,
                                                          threshold=2,
                                                          samples_per_image=20,
                                                          max_samples=30000,
                                                          #max_samples=30000,
                                                          #train_limit=10000,
                                                          min_prob=0.00005,)),

                ]
        elif 1:
            layers = [
                pnet.OrientedGaussianPartsLayer(num_parts,
                                                num_orientations,
                                                (part_size, part_size),
                                                settings=dict(n_iter=10,
                                                              n_init=1,
                                                              samples_per_image=50,
                                                              max_samples=50000,
                                                              max_covariance_samples=None,
                                                              standardize=False,
                                                              covariance_type='diag',
                                                              min_covariance=0.0025,
                                                              logratio_thresh=-np.inf,
                                                              std_thresh=-0.05,
                                                              std_thresh_frame=1,
                                                              #rotation_spreading_radius=0,
                                                              ),
                                                ),
                #pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
            ]
        elif 0:
            layers = [
                pnet.OrientedPartsLayer(num_parts,
                                        num_orientations,
                                        (part_size, part_size),
                                        settings=dict(outer_frame=2,
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
                                minimum_contrast=0.08,
                                spread='orthogonal',
                                #spread='box',
                                radius=1,
                                #pre_blurring=0,
                                contrast_insensitive=False,
                ),
                pnet.PartsLayer(num_parts, (6, 6), settings=dict(outer_frame=1,
                                                          seed=training_seed,
                                                          threshold=2,
                                                          samples_per_image=40,
                                                          max_samples=100000,
                                                          #max_samples=30000,
                                                          train_limit=10000,
                                                          min_prob=0.00005,
                                                          )),
            ]


        net = pnet.PartsNet(layers)

        print('Extracting subsets...')

        ims10k = data[:10000]

        start0 = time.time()
        print('Training unsupervised...')
        net.train(lambda x: x, ims10k)
        print('Done.')
        end0 = time.time()


        net.save(saveFile)

        if 1:
            N = 10
            data = ims10k

            feat = net.extract(lambda x: x, data[:10], layer=0)

            pooling = pnet.PoolingLayer(strides=(1, 1), shape=(1, 1))

            Y = pooling.extract(lambda x: x, feat)

            grid4 = ImageGrid(Y.transpose((0, 3, 1, 2)))
            grid4.save(vz.impath(), scale=2)

            #import pdb; pdb.set_trace()
            grid5 = ImageGrid(data[:10])
            grid5.save(vz.impath(), scale=2)

        vz.output(net)
        if SHORT:
            import sys; sys.exit(1)
