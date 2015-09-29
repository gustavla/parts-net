from __future__ import division, print_function, absolute_import
from pnet.layer import Layer
import pnet
import numpy as np


@Layer.register('standardization-layer')
class StandardizationLayer(Layer):
    def __init__(self, settings={}):
        self._settings = dict(
                              bias=0.0,
                              epsilon=0.000001,
                              num_images=None
                            )
        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._means = None
        self._vars = None
        self._epsilon = self._settings['epsilon']
        self._bias = self._settings['bias']
        self._num_images=self._settings['num_images']


    @property
    def trained(self):
        return self._means is not None and self._vars is not None

    def _train(self, phi, data, y=None):
        nd=data.shape[0]
        if (self._num_images is not None):
            nd=self._num_images
        X = phi(data[:nd,])
        if X.__class__==tuple:
            XA=X[0]
        else:
            XA=X
        num_clust=XA.shape[3]
        means=np.zeros(num_clust)
        vars=np.zeros(num_clust)
        Xflat = XA.reshape((XA.shape[0]*XA.shape[1]*XA.shape[2],XA.shape[3])).astype(np.float64)
        XX=Xflat[Xflat[:,0]>-100000]
        KK=np.argmax(XX,axis=1)
        for k in range(num_clust):
            IK=np.where(KK==k)[0]
            print('feature',k,len(IK))
            if (len(IK)>10):
                XXk=XX[IK][:,k]
                means[k]=XXk.mean()
                vars[k]=XXk.var()
                if (vars[k]>.0001):
                    print(np.min((XXk-means[k])/np.sqrt(vars[k])),
                        np.max((XXk-means[k])/np.sqrt(vars[k])))
                else:
                    print('Zero variance')
        self._means=means
        self._vars=vars
        self._vars[self._vars<.0001]=0

    def _extract(self, phi, data):
        X = phi(data)
        if X.__class__==tuple:
            XA=X[0]
        else:
            XA=X
        Xflat = XA.reshape((XA.shape[0]*XA.shape[1]*XA.shape[2],XA.shape[3])).astype(np.float64)
        TT=np.where(Xflat[:,0]>-100000)
        VV=np.where(self._vars==0)
        Xflat[TT,:]=(Xflat[TT,:]-self._means)/np.sqrt(self._vars+self._epsilon)
        Xflat[TT][:,VV]=-1-self._bias
        Xflat[Xflat==-100000]=-1-self._bias
        return Xflat.reshape(XA.shape) + self._bias



    @property
    def pos_matrix(self):
        return pnet.matrix.scale(1.0, 1.0)

    def save_to_dict(self):
        d = {}
        d['settings']=self._settings
        d['means'] = self._means
        d['vars'] = self._vars
        d['epsilon'] = self._epsilon
        d['bias'] = self._bias
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        obj._means = d['means']
        obj._vars = d['vars']
        return obj

    def __repr__(self):
        return ('StandardizationLayer(epsilon={epsilon}, bias={bias}').format(
                    epsilon=self._epsilon,
                    bias=self._bias)
