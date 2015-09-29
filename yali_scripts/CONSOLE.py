__author__ = 'amit'


# import pylab as py
# import numpy as np
# import deepdish as dd
import pnet
#import test_edge
#import shutil
import os
# import pylab as py
# from test_edge import find_next_net_index
# import net_test
# from pnet.cyfuncs import index_map_pooling
# #from sklearn.decomposition import PCA
num_test=1000
marg=0
color=True
import pylab as py
count=10
# c=0
#te_x, te_y = dd.io.load_cifar_10('training', offset=0, count=count,marg=marg)
# for c in range(10):
#     numdat=np.sum(te_y==c)
#     X = te_x[te_y==c]
#     M = np.mean(X,axis=0)
#     X=X-M
#     X=np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
#     S=np.dot(X.T,X)/numdat
#     [ee,vv]=np.linalg.eig(S)
#     vv=np.float64(vv)
#     ee=np.float64(ee)
#     # ZZ=np.dot(X,vv)
#     # ZZ=np.dot(np.diag(1./np.sqrt(ee)),ZZ.T)
#     if (c==0):
#       py.figure(2)
#       py.plot(np.cumsum(ee[0:50])/np.sum(ee))
#       py.show()
#     nump=5
#     print('variance explained by '+str(nump)+'components',np.sum(ee[0:nump])/np.sum(ee))
#
#     #pca = PCA(n_components=5)
#     #pca.fit(X)
#
#     for p in range(nump):
#         py.subplot(10,nump,c*nump+p+1)
#         py.imshow(np.reshape(vv[:,p],(28,28)))
#         py.axis('off')
#
# py.show()
# print('done')

# for c in range(10):
#     grid = dd.plot.ColorImageGrid(te_x[te_y==c,:,:,:],cols=20)
#     grid.save('cfar'+str(c)+'.png', scale=5)
# te_x=np.zeros((1,100,100,3))
#
# for x in range(100):
#     for y in range(100):
#         if ((x-50)**2+(y-50)**2 < 200):
#             te_x[0,x,y,0]=1
import numpy as np
rng=np.arange(18,20) #(50,) #(14,) #np.arange(20,30)
if len(rng):
    names=[]
    for n in rng:
        names.append('_Enets/net'+str(n)+'.h5')
else:
    Dname='_Enets/'
    nms=os.listdir(Dname)
    names=[]
    for nn in nms:
        if ('net' in nn):
            names.append(Dname+'/'+nn)
f=open('RESULTS_9_21_2015.txt','a')
import pprint
for netname in names:
    net=pnet.Layer.load(netname)
    f.write(netname+' ERROR: '+str(net.error)+'\n')
    for i,la in enumerate(net._layers):
        f.write('Layer '+str(i)+' '+la.name+'\n')
        f.write(pprint.pformat(la._settings))
        f.write('\n')
f.close()

    # ll=find_next_net_index('_OUTPUT','OUTPUT')
    # Cname='_OUTPUT/'+'OUTPUT'+str(ll)+'.txt'
    # shutil.copy('_OUTPUT/OUTPUT0.txt',Cname)

#
# ll=test_edge.find_next_net_index('_OUTPUT','OUTPUT')
# Cname='_OUTPUT/'+'OUTPUT'+str(ll)+'.txt'
# shutil.copy('OUTPUT0.txt',Cname)