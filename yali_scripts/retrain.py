__author__ = 'amit'


import pylab as py
import numpy as np
import deepdish as dd
import pnet
from test_edge import find_next_net_index
import net_test
import multiprocessing as mp
import shutil
import amitgroup as ag
num_test=1000
debug=0
marg=0
test=False
num_pool=8
num_train=50000
if debug:
    num_pool=1
    num_train=500
pool=None
if (num_pool>1):
    pool=mp.Pool(num_pool)
netname='_Enets/net58.h5' #PARTS6x6/P6_S3_T20_nc1_RS0/net_20K.h5'
keeplayers=1
up_layer=3
fully_connected=False
color=True


if color:
    tr_x, tr_y = dd.io.load_cifar_10('training', offset=0, count=num_train, marg=marg)
else:
    tr_x, tr_y = dd.io.load_mnist('training', offset=0, count=num_train, marg=marg)

if (netname != []):
    net=pnet.Layer.load(netname)
    print(netname)
    print('Error before retraining',net.error)
    for la in net._layers:
        print(la.name)
    if (not test):
        for l in range(np.minimum(keeplayers+1,len(net._layers))):
            net._layers[l].num_pool=num_pool
            net._layers[l].pool=pool
        if (len(net._layers)>keeplayers+1):
            for l in np.arange(len(net._layers)-1,keeplayers,-1):
                del net._layers[l]
    else:
        for la in net._layers:
            la.num_pool=num_pool
            la.pool=pool
else:
   test=False
   keeplayers=-1
   net=pnet.PartsNet([])


with ag.Timer('Training'):
    if (not test):
        grains=(-3,-3,-3)
        pool_strides=(2,2,2)
        strides=(32,16,8)
        deltas=(.0001,.0001,.001)
        #Cs=(.00001,.00001,.00001)
        Cs=(.001,.001,.001)
        num_pool=(8,8,8)
        PEN=('L2','L2','L2')
        copies=(1,3,3)
        keep=(1000,1000,1000)
        MINE=(True,False, False)

        prune=(False,False,False)
        num_rounds=(5,5,5)
        nsrs=(9,9,9)
        if (debug):
            num_pool=(1,1,1)
        boost=(None,None,None,None)
        lower_layer=keeplayers+1+1
        srange=np.arange(lower_layer,up_layer+1)
        #lower_layer=keeplayers+2+1
        for i,s in enumerate(srange):

            ii=keeplayers+1+i
            print('s',s,'strides',strides)
            net._layers.append(pnet.FeatureCombinerLayer(settings=dict(sub_region=(3,3), num_class=10, num_copies_per_layer=copies[ii],
                                                num_keep_per_layer=keep[ii],  boost=boost[ii], strides=strides[ii], C=Cs[ii], rectify=True, seed=54321,
                                                num_sgd_iters=1, max_pool=False, initial_delta=deltas[ii], num_sub_region_samples=nsrs[ii], smooth_score_output=1., grain=grains[ii], test_prop=0,
                                                num_pool=num_pool[ii], num_train=num_train, num_components=1, pool=3, pool_stride=pool_strides[ii],
                                                sgd_batch_size=100, num_sgd_rounds=num_rounds[ii], num_bfgs_iters=1, mine=MINE[ii], penalty=PEN[ii], prune=prune[ii]),pool=pool))

        if (fully_connected):
            stride=4
            print('Strides', strides)
            net._layers.append(pnet.FeatureCombinerLayer(settings=dict(sub_region=(stride,stride), num_class=10, num_copies_per_layer=500,
                                                num_keep_per_layer=1000, strides=stride, C=10, rectify=True, seed=None,
                                                num_sgd_iters=1, smooth_score_output=1., grain=-5, test_prop=0,
                                                num_pool=num_pool, num_train=num_train, num_components=1, pool=1, pool_stride=1,
                                                sgd_batch_size=20000, num_sgd_rounds=50, num_bfgs_iters=100),pool=pool))

        SC=.00001
        #SC=1
        net._layers.append(pnet.SVMClassificationLayer(settings=dict(C=SC,standardize=False, smooth_score_output=1., simple_aggregate=False,
                                                                 allocate=True, initial_delta=.0001, seed=77144525, num_components=1,
                                                                 min_count=2, num_train=num_train, num_sgd_iters=1, test_prop=0,
                                                                 sgd_batch_size=100, num_sgd_rounds=400, num_bfgs_iters=1, prune=False)))
        net.train(tr_x, tr_y)

error=net_test.net_test(net,marg,1000,num_test,color=color)

f=open('RESULTS.txt','a')
import pprint
new_netname=''
out_string=''
if (not test):
    Dname='_Enets'
    ll=find_next_net_index(Dname)
    new_netname=Dname+'/net'+str(ll)+'.h5'
    out_string=new_netname+' ERROR: '+str(error)
if (netname):
    out_string=netname+' to '+out_string

f.write(out_string+'\n')
for i,la in enumerate(net._layers):
        f.write('Layer '+str(i)+' '+la.name+'\n')
        f.write(pprint.pformat(la._settings))
        f.write('\n')
f.close()

# Write the actual network
if (not test):
    Dname='_Enets'
    ll=find_next_net_index(Dname)
    netname=Dname+'/net'+str(ll)+'.h5'
    net.save(netname,num_train,num_test,error)

ll=find_next_net_index('_OUTPUT','OUTPUT')
Cname='_OUTPUT/'+'OUTPUT'+str(ll)+'.txt'
shutil.copy('_OUTPUT/OUTPUT0.txt',Cname)


def other_layers():

    print('other')
    #net._layers.append(pnet.EdgeLayer(settings=dict(k=5, radius=0, spread='orthogonal',
    #                                                minimum_contrast=0.1, color_gradient_thresh=.2),num_pool=num_pool,pool=pool))


    ##net._layers.append(pnet.OrientedPartsLayer(settings=dict(num_parts=80, num_orientations=8, part_shape=(3,3),
    #                                   outer_frame=0,
    #                                   max_samples=50000, threshold=1, raw=True, num_pool=num_pool, var_explained=None,
    #                                   min_prob=.05, by_class=False, lperc=10, seed=None, lbias=6, shape=(3,3), strides=(2,2), allocate=True),pool=pool))
    #net._layers.append(pnet.MixtureClassificationLayer(settings=dict(num_components=20,min_prob=.01,
    #                                             classify=False, subclassify=True, allocate=True,
    #                                            num_train=num_train, standardize=False, seed=None)))