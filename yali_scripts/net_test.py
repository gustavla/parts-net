__author__ = 'amit'
import pnet
import numpy as np
import deepdish as dd
#import multiprocessing as mp
num_pool=1

def net_test(net,marg=7,count=1000,numtest=10000, color=False):

    #pool=mp.Pool(num_pool)

    nnet=list()
    for l in range(num_pool):
            nnet.append(net)
    if count < numtest:
        count=numtest
    if numtest<count:
        count=numtest
    rates=[];
    for offs in np.arange(0,numtest,count):
        print(offs)
        if (not color):
            te_x, te_y = dd.io.load_mnist('testing',offset=offs, count=count,marg=marg)
            te_xl=list(np.array_split(te_x,num_pool))
        else:
            te_x, te_y = dd.io.load_cifar_10('testing', offset=offs, count=count,marg=marg)
            te_xl=list(np.array_split(te_x,num_pool))


        print('Starting test')
        output=map(classify,zip(nnet,te_xl))
        te_yl=zip(*output)
        te_yhat=np.array(te_yl).T
        te_yhat=te_yhat.reshape(te_yhat.size)
#        te_yhat = net.classify(te_x)
        numcl=np.max(te_yhat)+1
        conf=np.zeros((numcl,numcl))

        for c in range(numcl):
            for d in range(numcl):
                conf[c,d]=np.sum(te_yhat[te_y==c]==d)
        np.set_printoptions(precision=3, suppress=True)
        print(conf)
        rate=(te_yhat != te_y).mean()
        print(offs,rate)
        rates.append(rate)

    rr=np.mean(np.array(rates))
    print('Error rate: {:.2f}%'.format(100*rr))
    return(rr)


def classify((net,te_x)):
    te_y=net.classify(te_x)
    return(te_y)