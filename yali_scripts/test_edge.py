from __future__ import division, print_function, absolute_import
import numpy as np
import deepdish as dd
import pnet
import amitgroup as ag
import multiprocessing as mp
ag.set_verbose(True)
#ag.set_parallel(8)
num_pool=8

pool=None
if (num_pool>1):
    pool=mp.Pool(num_pool)

def test_edge(color=False,marg=5):
    num_train_labeled=5000
    print('hello')

    net = pnet.PartsNet([
        #pnet.IntensityThresholdLayer(threshold=0.2,whiten=False),
        pnet.EdgeLayer(settings=dict(k=5, radius=0, spread='orthogonal', minimum_contrast=0.1, color_gradient_thresh=.2),num_pool=num_pool,pool=pool),
        # pnet.OrientedPartsLayer(settings=dict(num_parts=20, num_orientations=8, part_shape=(3,3), outer_frame=0,
        #                                        max_samples=10000, n_iter=10, threshold=20, raw=False, rot_spread=True, num_pool=num_pool, var_explained=None,
        #                                        min_prob=.05, lperc=10, by_class=False, seed=42654, lbias=1.5, shape=(3,3), strides=(2,2), allocate=True, grad=False),pool=pool),
        # pnet.OrientedPartsLayer(settings=dict(num_parts=40, num_orientations=1, part_shape=(3,3), outer_frame=0,
        #                                         max_samples=2000, threshold=1, raw=False, num_pool=num_pool, var_explained=None,
        #                                         min_prob=.05, by_class=False, seed=None, lbias=6, shape=(3,3), strides=(2,2), allocate=True),pool=pool),
        #pnet.OrientedGaussianPartsLayer(settings=dict(num_parts=20, num_orientations=8, part_shape=(3,3), seed=21765,
        #                                                                max_samples=10000, by_class=False, n_iter=10, std_thresh=0.005, num_pool=num_pool,
        #                                                                covariance_type='diag',lbias=0, min_covariance=0.001, coding='hard', allocate=True), pool=pool),
        #pnet.PoolingLayer(settings=dict(shape=(3, 3), strides=(2, 2), operation='max')),
        #pnet.MixtureClassificationLayer(settings=dict(num_components=20,min_prob=.0001,classify=True, subclassify=False, allocate=True, num_train=num_train_labeled, standardize=False, seed=None)),
        #pnet.FeatureCombinerLayer(settings=dict(sub_region=(20,20),strides=(2,2),C=.001,grain=0,num_pool=1, num_train=num_train_labeled),pool=pool),
        pnet.SVMClassificationLayer(settings=dict(C=.001,standardize=False, num_components=1, min_count=2, num_train=num_train_labeled, seed=432, num_sgd_iters=0))
        ]
    )


    if (not color):
        tr_x, tr_y = dd.io.load_mnist('training', count=60000,marg=marg)
        tr_x = tr_x[...,np.newaxis]
    else:
        tr_x, tr_y = dd.io.load_cifar_10('training', count=50000, marg=marg)
    net.train(tr_x, tr_y)

    # if (not color):
    #     tr_x, tr_y = dd.io.load_mnist('training', count=num_train,marg=marg)
    #     tr_x = tr_x[...,np.newaxis]
    # else:
    #     tr_x, tr_y = dd.io.load_cifar_10('training', count=num_train, marg=marg)
    #
    # net.train(tr_x, tr_y)




    Dname='_Enets'
    ll=find_next_net_index(Dname)
    netname=Dname+'/net'+str(ll)+'.h5'
    net.save(netname,num_train_labeled)

    return(netname)

def find_next_net_index(Dname,str='net'):
        import os
        ll=list()
        ls=len(str)
        aa=os.listdir(Dname);
        if aa==[]:
            return(1)
        for s in aa:
            if (str in s):
                u=s.index('.')
                ll.append(int(s[ls:u]))
        if (ll==[]):
            return(1)
        return(max(ll)+1)