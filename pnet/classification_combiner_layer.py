from __future__ import division, print_function, absolute_import

import numpy as np
import pnet
from pnet.layer import Layer
from pnet.layer import SupervisedLayer
import multiprocessing as mp
from scipy.stats import itemfreq

def _do_extract((X,layers)):

    print('In do extract - number of layers',len(layers))
    Y=[]
    for r in range(len(layers)):
        Y.append(layers[r]._extract(None,X))
    Y=np.array(Y)
    return Y

@Layer.register('classification-combiner-layer')
class ClassificationCombinerLayer(SupervisedLayer):
    def __init__(self, settings={},pool=None):

        self._settings = dict(
                              num_pool=1,
                              Dir=None,
                              cnums=None,
                              layer_num=None
                            )
        for k, v in settings.items():
            if k not in self._settings:
                    raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self.num_pool=self._settings['num_pool']
        self.pool=pool
        if (self._settings['Dir'] is not None):
            self._layers=[]
            Dir=self._settings['Dir']
            cnums=self._settings['cnums']
            layer_num=self._settings['layer_num']
            for ix in cnums:
                name=Dir+'/net'+str(ix)+'.h5'
                net=pnet.Layer.load(name)
                self._layers.append(net._layers[layer_num])
        self._trained = True

    @property
    def layers(self):
        return self._layers

    @property
    def trained(self):
        return self._trained


    def _train(self, phi, data, y):

        self._trained = True


    def _extract(self, phi, data):
        X = phi(data)
        Yss = []
        if (self.pool is None and self.num_pool>1):
               print('Creating pool in classification combiner layer')
               self.pool=mp.Pool(self.num_pool)
        num_subs=np.maximum(8,self.num_pool)
        X_subs=np.array_split(X,num_subs)
        print('Extract',self.num_pool, self.pool)
        args=((X_s,)+(self._layers,) for X_s in X_subs)
        if (self.pool is not None):
            pYs=self.pool.map(_do_extract,args)
        else:
            pYs=map(_do_extract,args)
        Ys=np.concatenate(pYs,axis=1)
        Ys=Ys.T
        # for pY in pYs:
        #     Yss.append()
        #Ys=np.concatenate(Yss, axis=1)
        yhat=-np.ones(Ys.shape[0])
        for i in range(Ys.shape[0]):
            freqs=itemfreq(Ys[i,:])
            k=np.argmax(freqs[:,1])
            yhat[i]=freqs[k,0]

        return(yhat)


    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        d['layers'] = []
        for layer in self._layers:
            layer_dict = layer.save_to_dict()
            layer_dict['name'] = layer.name
            d['layers'].append(layer_dict)
        return d

    @classmethod
    def load_from_dict(cls, d):
        layers = [
            Layer.getclass(layer_dict['name']).load_from_dict(layer_dict)
            for layer_dict in d['layers']
        ]
        obj = cls(d['settings'],layers)
        return obj

    def __repr__(self):
        return 'FeatureCombinerLayer({})'.format(num_parts=self._layers)
