from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from pnet.bernoullimm import BernoulliMM

@Layer.register('mixture-classification-layer')
class MixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, min_prob=0.0001, block_size=0, settings={}):
        self._n_components = n_components
        self._min_prob = min_prob
        self._models = None
        self._block_size = block_size
    @property
    def trained(self):
        return self._models is not None

    @property
    def classifier(self):
        return True

    def extract(self, X_all):
        print "mixture classification extract started"
        theta = self._models[np.newaxis]
        Yhat = np.zeros(X_all.shape[0])
        if 1:
            #use blocksize 50
            for i in range(0,X_all.shape[0],50):
                blockend = min(X_all.shape[0],i + 50) 
                X = X_all[i:blockend]
                XX =  X[:,np.newaxis,np.newaxis]
                llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
                Yhat[i:blockend] = np.argmax(np.apply_over_axes(np.sum, llh, [-3, -2, -1])[...,0,0,0].max(-1), axis=1)
        print "mixture classification extract finished"
        print(Yhat.shape)
        return Yhat 
    
    def train(self, X, Y, OriginalX = None):
        K = Y.max() + 1
        print(np.max(X))
        mm_models = []
        for k in xrange(K):
            Xk = X[Y == k]
            #import pdb ; pdb.set_trace()
            print(Xk.shape)
            print(X.shape)
            print(Y.shape)
            Xk = Xk.reshape((Xk.shape[0], -1))
            mm = BernoulliMM(n_components=self._n_components, n_iter=10, n_init=1, random_state=0, min_prob=self._min_prob,blocksize = self._block_size)
            mm.fit(Xk)
            mm_models.append(mm.means_.reshape((self._n_components,)+X.shape[1:]))

        self._models = np.asarray(mm_models)
        print "mixtureclassification-layer training finished"
    
    def save_to_dict(self):
        d = {}
        d['n_components'] = self._n_components
        d['min_prob'] = self._min_prob
        d['models'] = self._models
        d['block_size'] = self._block_size
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(n_components=d['n_components'], min_prob=d['min_prob'])
        obj._models = d['models']
        if 'block_size' in d:
            obj._block_size = d['block_size']
        else:
            obj._block_size = 0
        return obj
