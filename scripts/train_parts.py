from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag
import pnet
import pnet.data


if __name__ == '__main__':
    ag.set_verbose(True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('size', metavar='<part size>', type=int)
    parser.add_argument('data', metavar='<data file>',
                        type=str,
                        help='Filename of data file')
    parser.add_argument('seed', type=int, default=1)
    parser.add_argument('save_file', metavar='<output file>',
                        type=str,
                        help='Filename of savable model file')
    parser.add_argument('--factor', '-f', type=float)
    parser.add_argument('--count', '-c', type=int)
    args = parser.parse_args()

    part_size = args.size
    save_file = args.save_file
    seed = args.seed

    data = pnet.data.load_data(args.data)
    data = data[:args.count]
    if data.ndim == 3:
        data = data[..., np.newaxis]
    print('Using', args.count)
    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    for training_seed in [seed]:

        layers = []

        if args.factor is not None:
            layers = [
                pnet.ResizeLayer(factor=args.factor, make_gray=False),
            ]

        if 1:
            settings = dict(n_iter=10,
                            seed=0,
                            n_init=1,
                            standardize=True,
                            samples_per_image=100,
                            #max_samples=300000,
                            max_samples=200000,
                            uniform_weights=True,
                            max_covariance_samples=20000,
                            covariance_type='tied',
                            logratio_thresh=-np.inf,
                            std_thresh_frame=0,
                            channel_mode='together',
                            coding='soft',

                            normalize_globally=False,
                            min_covariance=0.0075,
                            std_thresh=0.01,
                            standardization_epsilon=10 / 255,
                            code_bkg=False,

                            whitening_epsilon=None,
                            min_count=0,
                            )

            layers += [
                pnet.OrientedGaussianPartsLayer(n_parts=1000, n_orientations=1,
                                                part_shape=(6, 6),
                                                settings=settings),
            ]
        elif 0:
            if 0:
                settings = dict(n_iter=20,
                                seed=0,
                                n_init=5,
                                standardize=True,
                                samples_per_image=50,
                                max_samples=20000,
                                uniform_weights=True,
                                #max_covariance_samples=10000,
                                covariance_type='diag',
                                logratio_thresh=-np.inf,
                                std_thresh_frame=0,
                                channel_mode='separate',

                                normalize_globally=False,
                                min_covariance=0.1,
                                std_thresh=0.01,
                                standardization_epsilon=10 / 255,
                                code_bkg=False,
                                whitening_epsilon=None,
                                )

                layers += [
                    pnet.OrientedGaussianPartsLayer(n_parts=24, n_orientations=1,
                                                    part_shape=(3, 3),
                                                    settings=settings),
                ]
            else:
                layers += [
                    pnet.KMeansPartsLayer(24, (4, 4), settings=dict(
                                                              seed=0,
                                                              n_per_image=50,
                                                              n_samples=20000,
                                                              n_init=5,
                                                              max_iter=300,
                                                              n_jobs=1,
                                                              random_centroids=False,
                                                              code_bkg=False,
                                                              std_thresh=0.01,
                                                              whitening_epsilon=0.1,
                                                              coding='triangle',
                                                              )),
                ]

            if 0:
                layers += [
                    pnet.PoolingLayer(final_shape=(14, 14), operation='sum'),
                    pnet.OrientedGaussianPartsLayer(n_parts=400,
                                            n_orientations=1,
                                            part_shape=(4, 4),
                                            settings=dict(
                                                            n_iter=20,
                                                            seed=training_seed,
                                                            n_init=1,
                                                            standardize=False,
                                                            samples_per_image=100,
                                                            max_samples=10000,
                                                            max_covariance_samples=1000,
                                                            uniform_weights=True,
                                                            covariance_type='tied',
                                                            logratio_thresh=-np.inf,
                                                            std_thresh_frame=0,
                                                            channel_mode='together',

                                                            min_covariance=0.1,
                                                            std_thresh=0.01,
                                                            code_bkg=False,
                                                            whitening_epsilon=None,
                                                          )),

                ]
            elif 0:
                layers += [
                    pnet.PoolingLayer(shape=(3, 3), strides=(1, 1)),
                    pnet.OrientedPartsLayer(n_parts=500,
                                            n_orientations=1,
                                            part_shape=(6, 6),
                                            settings=dict(outer_frame=1,
                                                          seed=training_seed,
                                                          threshold=2,
                                                          samples_per_image=100,
                                                          max_samples=300000,
                                                          min_covariance=0.1,
                                                          std_thresh=0.01,
                                                          standardization_epsilon=10 / 255,
                                                          standardize=False,
                                                          min_prob=0.0005,)),

                ]
            else:
                layers += [
                    #pnet.PoolingLayer(shape=(2, 2), strides=(1, 1), settings=dict(output_dtype='float32')),
                    pnet.KMeansPartsLayer(400, (6, 6), settings=dict(
                                                              seed=0,
                                                              n_per_image=100,
                                                              n_samples=100000,
                                                              n_init=1,
                                                              max_iter=300,
                                                              n_jobs=1,
                                                              random_centroids=False,
                                                              code_bkg=False,
                                                              std_thresh=0.01,
                                                              whitening_epsilon=0.1,
                                                              standardize=False,
                                                              whiten=False,
                                                              coding='triangle',
                                                              )),
                ]
        elif 0:
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
                pnet.KMeansPartsLayer(400, (part_size, part_size), settings=dict(
                                                          seed=training_seed,
                                                          n_per_image=100,
                                                          n_samples=100000,
                                                          n_init=1,
                                                          max_iter=300,
                                                          n_jobs=1,
                                                          random_centroids=False,
                                                          code_bkg=False,
                                                          std_thresh=0.01,
                                                          whitening_epsilon=0.1,
                                                          coding='soft',
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
