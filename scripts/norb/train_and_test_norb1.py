from __future__ import division, print_function, absolute_import
import amitgroup as ag
import pnet
import pnet.norb
import numpy as np

ag.set_verbose(True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parts', metavar='<parts file>',
                        type=argparse.FileType('rb'),
                        help='Filename of parts file')
    args = parser.parse_args()

    feat_net = pnet.Layer.load(args.parts)

    layers = feat_net.layers

    S = 11

    layers += [
        pnet.PoolingLayer(shape=(S, S), strides=(S, S), operation='sum'),
        # pnet.MixtureClassificationLayer(n_components=10, min_prob=1e-5,
        # settings=dict( standardize=False,),)
        pnet.SVMClassificationLayer(C=100.0, settings=dict(standardize=True)),
    ]

    net = pnet.PartsNet(layers)

    limit = None
    error_rate, conf_mat = pnet.norb.train_and_test(net,
                                                    samples_per_class=None,
                                                    seed=0, limit=limit)

    print('Error rate: {:.02f}'.format(error_rate * 100))
    np.set_printoptions(precision=2, suppress=True)
    print('Confusion matrix:')

    norm_conf = conf_mat / np.apply_over_axes(np.sum, conf_mat, [1])
    print(norm_conf)

if pnet.parallel.main(__name__):
    main()
