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
    S = 2500
    def batch_loader(dataset, count):
        for n in range(count // S):
            if count is None:
                count0 = S
            else:
                count0 = max(min(S, count - S * n), 0)

            test_X = ag.io.load_cifar_10(dataset, offset=S * n,
                                         count=count0, ret='x')
            if test_X.size == 0:
                break
            test_X = test_X.transpose(0, 2, 3, 1).copy()
            yield test_X

    count = 50000
    #count_sel = slice(count) if count is not None else None

    X = batch_loader('training', count)
    if count is 50000:
        y = np.concatenate([ag.io.load_cifar_10('training', ret='y', offset=10000*b) for b in range(5)])
    elif count <= 10000:
        y = ag.io.load_cifar_10('training', ret='y', count=count)
    else:
        raise NotImplementedError("Not yet")

    print('Loading data... Done')

    n_classes = y.max() + 1

    print('Training unsupervised')
    # Train unsupervised
    #net.train(lambda x: x, X)
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
    net.train(lambda x: x, X, y)
    print('Training supervised... Done')

    from vzlog.default import vz
    vz.output(net)
    vz.flush()
    net.save('model.h5')

    # Load testing data

    corrects = 0
    total = 0

    confusion_matrix = np.zeros((n_classes, n_classes))

    #from multiprocessing import Pool
    #p = Pool(4)

    for n in range(10000 // S):
        test_X, test_y = ag.io.load_cifar_10('testing',
                                             offset=n * S, count=S)
        if test_X.size == 0:
            break

        test_X = test_X.transpose(0, 2, 3, 1).copy()

        BATCHES = max(1, len(test_X) // 5000)

        test_X_batches = np.array_split(test_X, BATCHES)
        test_y_batches = np.array_split(test_y, BATCHES)

        # Test
        zipped = zip(test_X_batches, test_y_batches)
        params = [tup+(net, n_classes) for tup in zipped]
        #for i, conf in enumerate(p.starmap(_test, params)):
        #for i, conf in enumerate(itr.starmap(_test, params)):
        for i, conf in enumerate(pnet.parallel.starmap_unordered(_test,
                                 params)):
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
