from __future__ import division, print_function, absolute_import
import amitgroup as ag
import pnet
import pnet.cifar
import numpy as np

ag.set_verbose(True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parts',metavar='<parts file>',
                        type=argparse.FileType('rb'),
                        help='Filename of parts file')
    args = parser.parse_args()

    feat_net = pnet.Layer.load(args.parts)

    training_seed = 0

    layers = feat_net.layers

    S = 13

    layers += [
        #pnet.PoolingLayer(final_shape=(2, 2), operation='sum'),
        #pnet.PoolingLayer(shape=(29, 29), strides=(29, 29), operation='sum'),
        pnet.PoolingLayer(shape=(S, S), strides=(S, S), operation='sum'),
        #pnet.MixtureClassificationLayer(n_components=10, min_prob=1e-5, settings=dict( standardize=False,),)
        pnet.SVMClassificationLayer(C=10**np.linspace(-3, -5, 10), settings=dict(standardize=True)),
    ]

    net = pnet.PartsNet(layers)

    limit = None #15000
    error_rate, conf_mat = pnet.cifar.train_and_test(net,
                                                     samples_per_class=None,
                                                     seed=0, limit=limit)

    print('Error rate: {:.02f}'.format(error_rate * 100))
    np.set_printoptions(precision=2, suppress=True)
    print('Confusion matrix:')

    norm_conf = conf_mat / np.apply_over_axes(np.sum, conf_mat, [1])
    print(norm_conf)

    from vzlog.default import vz
    vz.output(net)

if pnet.parallel.main(__name__):
    main()
