from __future__ import division, print_function, absolute_import 

import amitgroup as ag
from pnet.layer import Layer
import pnet
import numpy as np
from pnet.cyfuncs import index_map_pooling
import multiprocessing as mp

def extract_batch((X,color_gradient_thresh,edge_settings)):
        print("Extracting edges",X.shape)
        if X.ndim==4 and X.shape[3]>1:
            X=np.float64(X.transpose(3,0,1,2))
            Xe=[]
            for i in range(X.shape[0]):
                Xe.append(ag.features.bedges(X[i,:,:,:], **edge_settings))
            Xe=np.array(Xe).transpose(1,2,3,4,0)
            Xe=Xe.reshape(Xe.shape[0:3]+(-1,))
            # Add color gradient information

            if (color_gradient_thresh<1):
                X=X.transpose(1,2,3,0)
                ip=((0,1),(0,2),(1,2))
                for iip in ip:
                    Xd=X[:,:,:,iip[0]]-X[:,:,:,iip[1]]
                    for j in (-1,1):
                        Xc=np.uint8((j*Xd>color_gradient_thresh))
                        Xcs=index_map_pooling(Xc,(3,3),(1,1))
                        Xcs=Xcs[...,np.newaxis]
                        #print(ip,j)
                        Xe=np.concatenate((Xe,Xcs), axis=3)
            return(Xe)
        else:
            return ag.features.bedges(X, **edge_settings)


@Layer.register('edge-layer')
class EdgeLayer(Layer):
    def __init__(self, settings={}, num_pool=1, pool=None):
        self._settings=dict(k=5,
                            radius=1,
                            spread='orthogonal',
                            minimum_contrast=0.03,
                            color_gradient_thresh=2.,
                            )
        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._edge_settings = self._settings
        self._color_gradient_thresh=self._settings['color_gradient_thresh']
        self.pool=pool
        self.num_pool=num_pool

    @property
    def pos_matrix(self):
        if self._edge_settings.get('preserve_size', True):
            #return pnet.matrix.translation(-1,-1)
            return pnet.matrix.identity()
        else:
            return pnet.matrix.translation(-2, -2)

    def _extract(self, phi, data):
        X = phi(data)
        if (self.num_pool>1 and self.pool is None):
            self.pool=mp.Pool(self.num_pool)
        sett=(self._color_gradient_thresh,self._edge_settings)
        if (X.shape[0]<100):
            Xe=extract_batch((X,self._color_gradient_thresh,self._edge_settings))
        else:
            X_subbatches=np.array_split(X,self.num_pool)
            args = ((X_b,) + sett for X_b in X_subbatches)
            if (self.pool is not None):
                output=self.pool.map(extract_batch,args)
            else:
                output=map(extract_batch,args)
            Xe=np.concatenate(output)

        print('Ended edge extraction',Xe.shape)
        return(Xe)

    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        return obj

    def __repr__(self):
        return 'EdgeLayer(bedges_settings={})'.format(self._edge_settings)
