from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from pnet.bernoullimm import BernoulliMM

@Layer.register('mixture-classification-layer')
class MixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, min_prob=0.0001, settings={}):
        self._n_components = n_components
        self._min_prob = min_prob
        self._models = None

    @property
    def trained(self):
        return self._models is not None

    @property
    def classifier(self):
        return True

    def extract(self, X):
        XX =  X[:,np.newaxis,np.newaxis]
        theta = self._models[np.newaxis]
        #print('Z', Z.shape)
        #print('mm', mm.shape)

        llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
        Yhat = np.argmax(np.apply_over_axes(np.sum, llh, [-3, -2, -1])[...,0,0,0].max(-1), axis=1)
        return Yhat 
    
    def train(self, X, Y):
        K = Y.max() + 1

        mm_models = []
        for k in xrange(K):
            Xk = X[Y == k]
            import pdb ; pdb.set_trace()
            Xk = Xk.reshape((Xk.shape[0], -1))
            mm = BernoulliMM(n_components=self._n_components, n_iter=10, n_init=1, random_state=0, min_prob=self._min_prob)
            mm.fit(Xk)
            mm_models.append(mm.means_.reshape((self._n_components,)+X.shape[1:]))

        self._models = np.asarray(mm_models)

    def save_to_dict(self):
        d = {}
        d['n_components'] = self._n_components
        d['min_prob'] = self._min_prob
        d['models'] = self._models 
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(n_components=d['n_components'], min_prob=d['min_prob'])
        obj._models = d['models']
        return obj
