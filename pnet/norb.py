from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag
import pnet
import itertools as itr

def _test(Xs, ys, net, n_classes):
    yhats = net.classify(Xs)
    return pnet.rescalc.confusion_matrix(ys, yhats, n_classes)

def train_and_test(net, samples_per_class=None, seed=0, limit=None):
    # Load training data
    print('Loading data')
    X, y = ag.io.load_small_norb('training', selection=slice(1000))
    X = X.astype(np.float64) / 255.0
    #X, y = ag.io.load_small_norb('training')
    print('Loading data... Done')
    #X = X.astype(np.float64)/255.0

    n_classes = y.max() + 1

    print('Shape of training data', X.shape)

    print('Training unsupervised')
    # Train unsupervised
    net.train(lambda x: x, X)
    print('Training unsupervised... Done')

    rs = np.random.RandomState(seed)
    if samples_per_class is not None:
        indices = []

        for k in range(y.max() + 1):
            II = np.where(y == k)[0]
            rs.shuffle(II)

            indices.append(II[:samples_per_class])

        indices = np.concatenate(indices)

        X = X[indices]
        y = y[indices]

    print('Training supervised')
    # Train supervised (avoid twins)
    net.train(lambda x: x, X[::2], y[::2])
    print('Training supervised... Done')

    from vzlog.default import vz
    vz.output(net)
    vz.flush()
    net.save('model.h5')

    # Load testing data

    S = 5000
    corrects = 0
    total = 0

    confusion_matrix = np.zeros((n_classes, n_classes))

    #from multiprocessing import Pool
    #p = Pool(4)

    for n in itr.count(0):
        test_X, test_y = ag.io.load_small_norb('testing', selection=slice(S * n, S * (n + 1)))
        if test_X.size == 0:
            break

        test_X = test_X.astype(np.float64) / 255.0

        BATCHES = len(test_X) // 2

        test_X_batches = np.array_split(test_X, BATCHES)
        test_y_batches = np.array_split(test_y, BATCHES)

        # Test
        params = [tup+(net, n_classes) for tup in zip(test_X_batches, test_y_batches)]
        #for i, conf in enumerate(p.starmap(_test, params)):
        #for i, conf in enumerate(itr.starmap(_test, params)):
        for i, conf in enumerate(pnet.parallel.starmap_unordered(_test, params)):
            corrects += np.trace(conf)
            total += np.sum(conf)

            confusion_matrix += conf

            pr = corrects / total
            error_rate = 1.0 - pr
            ag.info('{} {} ({}) Current error rate {:.02f}% ({})'.format(n, i, total, 100*error_rate, limit))

            if limit is not None and total >= limit:
                break

        if limit is not None and total >= limit:
            break

    pr = corrects / total
    error_rate = 1.0 - pr

    return error_rate, confusion_matrix
