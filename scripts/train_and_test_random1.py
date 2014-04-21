from __future__ import division, print_function, absolute_import 

import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
ag.set_verbose(True)

import pnet

def randomize_layers_parameters(seed):
    params = {} 

    rs = np.random.RandomState(seed)

    layer_options = dict(
        #part_side=[1,1,2,2,3,3,4,5,6],
        pooling_size_above=np.arange(0, 3),
        threshold=np.arange(1, 6),
        max_samples=[5000, 10000, 50000],
        min_prob=[0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        support_index=[0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],
        pooling_support_index=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],
        n_coded=[1,1,1,2,3,4],
        min_percentile=np.arange(0, 30, 5),
    )

    joint_options = dict(
        standardize=[0,1,2,3],
        layer0_part_side=[2,2,2,3,3,4],
        layer1_part_side=[1,2,3,4,5],
        layer0_num_parts=np.arange(10, 60, 10), 
        layer1_num_parts=np.arange(10, 110, 10), 
        layer0_pooling_stride=np.arange(1, 3),
        layer1_pooling_stride=np.arange(1, 5),
    )

    for layer in xrange(2):
        prefix = 'layer{}_'.format(layer)

        for key, values in layer_options.items():
            params[prefix + key] = values[rs.randint(len(values))]

    for key, values in joint_options.items():
        params[key] = values[rs.randint(len(values))]
    
    #if params['standardize'] == 0:
    #   params['min_percentile'] = 0.0

    return params

def make_support(side, support_index):
    if support_index == 0:
        return None
    else:
        support_mask = np.zeros((side, side), dtype=np.bool)

        if support_index == 1: # Circle
            for i, j in itr.product(xrange(side), xrange(side)):
                if (i+0.5-side/2)**2 + (j+0.5-side/2)**2 < (side/2)**2:
                    support_mask[i,j] = True
        elif support_index == 2: # Checker board
            for i, j in itr.product(xrange(side), xrange(side)):
                if (i+j)%2 == 0: 
                    support_mask[i,j] = True
        elif support_index == 3: # Subsample 2x2
            for i, j in itr.product(xrange(side), xrange(side)):
                if i%2 == 0 and j%2 == 0: 
                    support_mask[i,j] = True
        elif support_index == 4: # Subsample 3x3
            for i, j in itr.product(xrange(side), xrange(side)):
                if i%3 == 0 and j%3 == 0: 
                    support_mask[i,j] = True

        return support_mask

def construct_layers(params):
    layers = [
        pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
    ]

    for layer in xrange(2):
        prefix = 'layer{}_'.format(layer)
        part_side = params[prefix+'part_side']
        pooling_stride = params[prefix+'pooling_stride']
        pooling_size = pooling_stride +  params[prefix+'pooling_size_above']
        layers += [
            pnet.PartsLayer(params[prefix+'num_parts'], (part_side, part_side), settings=dict(
                                                      outer_frame=0,
                                                      threshold=params[prefix+'threshold'],
                                                      samples_per_image=100,
                                                      max_samples=params[prefix+'max_samples'],
                                                      min_prob=params[prefix+'min_prob'],
                                                      standardize=params['standardize'],
                                                      min_percentile=params[prefix+'min_percentile'],
                                                      support_mask=make_support(part_side, params[prefix+'support_index'])
                                                      )),
            pnet.PoolingLayer(shape=(pooling_size, pooling_size), strides=(pooling_stride, pooling_stride), settings=dict(
                                                                                  support_mask=make_support(pooling_size, params[prefix+'pooling_support_index']))),
        ]

    layers += [
        pnet.MixtureClassificationLayer(n_components=1),
    ]
    return layers

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')

    args = parser.parse_args()


    #for i in xrange(1, 7):
    #    print(make_support(i, 4).astype(np.uint8))

    params = randomize_layers_parameters(args.seed)
    #print(params)

    error_rates = []

    for training_seed in xrange(5):
        layers = construct_layers(params) 

        for l, layer in enumerate(layers):
            print(l, layer)
        #import sys; sys.exit(0)

        net = pnet.PartsNet(layers)

        digits = range(10)
        ims = ag.io.load_mnist('training', selection=slice(0 + 3000 * training_seed, 10000 + 3000 * training_seed), return_labels=False)

        #print(net.sizes(X[[0]]))

        net.train(ims)

        sup_ims = []
        sup_labels = []
        # Load supervised training data
        for d in digits:
            ims0 = ag.io.load_mnist('training', [d], selection=slice(100*training_seed, 100*(1+training_seed)), return_labels=False)
            sup_ims.append(ims0)
            sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

        sup_ims = np.concatenate(sup_ims, axis=0)
        sup_labels = np.concatenate(sup_labels, axis=0)

        net.train(sup_ims, sup_labels)

        print("Now testing...")
        ### Test ######################################################################

        corrects = 0
        total = 0
        test_ims, test_labels = ag.io.load_mnist('training', selection=slice(30000, 40000))
        ims_batches = np.array_split(test_ims, 200)
        labels_batches = np.array_split(test_labels, 200)

        def format_error_rate(pr):
            return "{:.2f}%".format(100*(1-pr))

        args = [tup+(net,) for tup in zip(ims_batches, labels_batches)]
        for i, res in enumerate(itr.starmap(test, args)):
            if i != 0 and i % 20 == 0:
                print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims), format_error_rate(pr)))

            corrects += res.sum()
            total += res.size
            pr = corrects / total

        error_rate = 1.0 - pr 


        error_rates.append(error_rate)
        print('error rate', error_rate)

    with open(os.path.join('output', '{:06}-{:06.3f}.txt'.format(args.seed, np.mean(error_rates)*100)), 'w') as f:
        print(np.mean(error_rates), file=f)
        print(error_rates, file=f)
    #net.save(args.model)
