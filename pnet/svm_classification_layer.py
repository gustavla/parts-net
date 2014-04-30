from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from sklearn.svm import LinearSVC

@Layer.register('svm-classification-layer')
class SVMClassificationLayer(SupervisedLayer):
    def __init__(self, C=1.0, settings={}):
        self._penalty = C
        self._svm = None

    @property
    def trained(self):
        return self._svm is not None

    @property
    def classifier(self):
        return True;

    def extract(self,X):
        Xflat = X.reshape((X.shape[0], -1))
        return self._svm.predict(Xflat) 

    def train(self, X, Y, OriginalX = None):
        Xflat = X.reshape((X.shape[0], -1))
        svc = LinearSVC(C=self._penalty)
        svc.fit(Xflat, Y)
        self._svm = svc

    def save_to_dict(self):
        d = {}
        d['svm'] = self._svm
        d['penalty'] = self._penalty
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(C=d['penalty'])
        obj._svm = d['svm']
        return obj

