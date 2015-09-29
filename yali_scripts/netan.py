__author__ = 'amit'
import pnet
import numpy as np
import pylab as py
import shutil
import test_edge
import net_test
#import test_col
import Draw_means
from pnet.oriented_gaussian_parts_layer import lognormalized

#a=-np.random.rand(3,10)*2000
#lognormalized(a,1)

#net = pnet.Layer.load("net_col.h5")
from collections import OrderedDict
num_runs=1
marg=0
color=True
num_test=1000
rates=np.zeros(num_runs)
for i in range(num_runs):
  netname=test_edge.test_edge(color,marg)
  #netname='Enets/net376.h5'
  net=pnet.Layer.load(netname)
  #net._layers[0].nu0_pool=1
  #net._layers[1].num_pool=1
  error=net_test.net_test(net,marg,1000,num_test,color=color)
  net.save(netname,net.num_train,num_test,error)
  ll=test_edge.find_next_net_index('_OUTPUT','OUTPUT')
  Cname='_OUTPUT/'+'OUTPUT'+str(ll)+'.txt'
  shutil.copy('OUTPUT0.txt',Cname)

Draw_means.draw_means('_Enets',None,10,True)



