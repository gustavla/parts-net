from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from sklearn.svm import LinearSVC
from sklearn import cross_validation

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

    def train(self, X, Y):
        Xflat = X.reshape((X.shape[0], -1))
        if self._penalty is None:
            Cs = 10**np.linspace(-1, -4, 10)
            avg_scores = np.zeros(len(Cs))
            for i, C in enumerate(Cs):
                clf = LinearSVC(C=C)
                scores = cross_validation.cross_val_score(clf, Xflat, Y, cv=5)
                avg_scores[i] = np.mean(scores)
                print('C', C, 'scores', scores, 'avg', np.mean(scores))

            Ci = np.argmax(avg_scores)
            C = Cs[Ci]

            clf = LinearSVC(C=C)
            clf.fit(Xflat, Y)

        else:
            # Cross-validate the penalty
            clf = LinearSVC(C=self._penalty)
            clf.fit(Xflat, Y)

        self._svm = clf 

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

