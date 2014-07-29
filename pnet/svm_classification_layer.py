from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from sklearn.svm import LinearSVC
from sklearn import cross_validation


@Layer.register('svm-classification-layer')
class SVMClassificationLayer(SupervisedLayer):
    def __init__(self, C=None, settings={}):
        self._penalty = C
        self._settings = settings
        self._extra = {}
        self._svm = None

    @property
    def trained(self):
        return self._svm is not None

    @property
    def classifier(self):
        return True

    def extract(self, phi, data):
        X = phi(data)
        Xflat = X.reshape((X.shape[0], -1))

        if self._settings.get('standardize'):
            means = self._extra['means']
            stds = self._extra['stds'].clip(min=1e-10)
            Xflat = (Xflat - means) / stds

        return self._svm.predict(Xflat)

    def train(self, phi, data, y):
        X = phi(data)
        ag.info('Entering SVM training with X.shape = ', X.shape)

        Xflat = X.reshape((X.shape[0], -1)).astype(np.float64)
        seed = self._settings.get('seed', 0)

        # Standardize features
        if self._settings.get('standardize'):
            self._extra['means'] = Xflat.mean(0)
            self._extra['stds'] = Xflat.std(0)

            means = self._extra['means']
            stds = self._extra['stds'].clip(min=1e-10)
            Xflat = (Xflat - means) / stds

        if self._penalty is None:
            Cs = 10**np.linspace(0, -4, 10)
            avg_scores = np.zeros(len(Cs))
            for i, C in enumerate(Cs):
                clf = LinearSVC(C=C)
                scores = cross_validation.cross_val_score(clf, Xflat, y, cv=5)
                avg_scores[i] = np.mean(scores)
                print('C', C, 'scores', scores, 'avg', np.mean(scores))

            Ci = np.argmax(avg_scores)
            C = Cs[Ci]

            clf = LinearSVC(C=C, random_state=seed)
            clf.fit(Xflat, y)

        else:
            # Cross-validate the penalty
            clf = LinearSVC(C=self._penalty, random_state=seed)
            clf.fit(Xflat, y)

        self._svm = clf

    def save_to_dict(self):
        d = {}
        d['svm'] = self._svm
        d['settings'] = self._settings
        d['extra'] = self._extra
        d['penalty'] = self._penalty
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(C=d['penalty'])
        obj._settings = d['settings']
        obj._extra = d['extra']
        obj._svm = d['svm']
        return obj
