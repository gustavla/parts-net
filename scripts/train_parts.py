import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
import pnet
import time

if __name__ == '__main__':
    ag.set_verbose(True)
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')
    #parser.add_argument('param', metavar='<param>', type=string)

    parser.add_argument('size', metavar='<part size>', type=int)
    #parser.add_argument('orientations', metavar='<num orientations>', type=int)
    #parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts')
    parser.add_argument('data', metavar='<data file>',
                        type=str,
                        help='Filename of data file')
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('save_file', metavar='<output file>',
                        type=argparse.FileType('wb'),
                        help='Filename of savable model file')
    parser.add_argument('--factor', '-f', type=float)
    args = parser.parse_args()

    part_size = args.size
    #num_orientations = args.orientations
    #numParts = args.numParts
    #saveFile = args.saveFile
    save_file = args.save_file
    seed = args.seed

    if args.data[0] == ':':
        if args.data == ':small-norb-training':
            data, _ = ag.io.load_small_norb('training')
            data = data.transpose(0, 2, 3, 1)
        else:
            raise ValueError('Unknown data: {}'.format(args.data))
    else:
        data = ag.io.load(args.data)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []


    for training_seed in [seed]:

        layers = []

        if args.factor is not None:
            layers = [
                pnet.ResizeLayer(factor=args.factor),
            ]

        if 0:
            layers += [
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
        elif 0:
            layers += [
                pnet.GaussianPartsLayer(1000, (part_size, part_size), settings=dict(
                                                          seed=0,
                                                          covariance_type='tied',
                                                          n_per_image=20,
                                                          #n_samples=50000,
                                                          n_samples=40000,
                                                          )),
            ]
        elif 1:
            layers += [
                pnet.KMeansPartsLayer(1000, (part_size, part_size), settings=dict(
                                                          seed=training_seed,
                                                          n_per_image=100,
                                                          #n_samples=600000,
                                                          n_samples=200000,
                                                          #n_samples=20000,
                                                          #n_samples=3000,
                                                          n_init=1,
                                                          max_iter=300,
                                                          n_jobs=1,
                                                          random_centroids=False,
                                                          )),
            ]

        elif 1:
            layers += [
                pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
                pnet.BinaryTreePartsLayer(8, (part_size, part_size), settings=dict(outer_frame=1,
                                                          em_seed=training_seed,
                                                          threshold=2,
                                                          samples_per_image=40,
                                                          max_samples=100000,
                                                          train_limit=10000,
                                                          min_prob=0.005,
                                                          #keypoint_suppress_radius=1,
                                                          min_samples_per_part=50,
                                                          split_criterion='IG',
                                                          min_information_gain=0.01,
                                                          )),
            ]

        else:
            layers += [
                pnet.EdgeLayer(
                                k=5,
                                minimum_contrast=0.05,
                                spread='orthogonal',
                                radius=1,
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

        print(net)

        print('Training parts')
        net.train(lambda _: _, data)
        print('Done.')

        net.save(save_file)

