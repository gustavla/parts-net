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

        self.epsilon = 0.01

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
            variances = self._extra['vars']
            Xflat = (Xflat - means) / np.sqrt(variances + self.epsilon)

        return self._svm.predict(Xflat)

    def train(self, phi, data, y):
        import types
        if isinstance(data, types.GeneratorType):
            Xs = []
            c = 0
            for i, Xi in enumerate(data):
                X_ = phi(Xi)
                Xs.append(X_)
                c += X_.shape[0]
                ag.info('SVM: Has extracted', c)
            X = np.concatenate(Xs, axis=0)
        else:
            X = phi(data)

        ag.info('Entering SVM training with X.shape = ', X.shape)

        Xflat = X.reshape((X.shape[0], -1)).astype(np.float64)
        seed = self._settings.get('seed', 0)

        with ag.Timer('SVM training'):
            # Standardize features
            if self._settings.get('standardize'):
                self._extra['means'] = Xflat.mean(0)
                self._extra['vars'] = Xflat.var(0)

                means = self._extra['means']
                variances = self._extra['vars']

                Xflat = (Xflat - means) / np.sqrt(variances + self.epsilon)

            if self._penalty is None or isinstance(self._penalty, np.ndarray):
                # Cross-validate the penalty
                if isinstance(self._penalty, np.ndarray):
                    Cs = self._penalty
                else:
                    Cs = 10**np.linspace(0, -4, 10)
                avg_scores = np.zeros(len(Cs))
                for i, C in enumerate(Cs):
                    clf = LinearSVC(C=C, random_state=seed)

                    # This contains some temporary code for better cross
                    # validation on NORB
                    if 1:
                        cv = 5
                    elif 0:
                        from sklearn.cross_validation import LeaveOneLabelOut

                        cats = ag.io.load_small_norb('testing', ret='i',
                                                     count=Xflat.shape[0])
                        cv = LeaveOneLabelOut(cats[:, 0])
                    else:
                        rs = np.random.RandomState(0)
                        p = 4
                        from sklearn.cross_validation import LeavePLabelOut
                        cats = ag.io.load_small_norb('training', ret='i',
                                                     count=Xflat.shape[0])
                        ll = ag.io.load_small_norb('training',
                                                   ret='y')[:Xflat.shape[0]]
                        assert (ll == y).all()

                        cv_gen = LeavePLabelOut(cats[:, 0], p)
                        cv0 = list(cv_gen)
                        rs.shuffle(cv0)
                        cv = cv0[:5]

                    scores = cross_validation.cross_val_score(clf, Xflat, y,
                                                              cv=cv)
                    avg_scores[i] = np.mean(scores)
                    print('C', C, 'scores', scores, 'avg', np.mean(scores))

                Ci = np.argmax(avg_scores)
                C = Cs[Ci]

                clf = LinearSVC(C=C, random_state=seed)
                clf.fit(Xflat, y)

            else:
                clf = LinearSVC(C=self._penalty, random_state=seed)
                clf.fit(Xflat, y)

        # Check training error rate
        yhat = clf.predict(Xflat)
        training_error_rate = 1 - (y == yhat).mean()
        ag.info('Training error rate: {}%'.format(100 * training_error_rate))

        self._svm = clf

    def _vzlog_output_(self, vz):
        vz.log('means', self._extra['means'])
        vz.log('vars', self._extra['vars'])
        vz.log('min var', np.min(self._extra['vars']))

        import pylab as plt

        plt.figure()
        plt.hist(self._extra['means'])
        plt.title('Means')
        plt.savefig(vz.impath())

        plt.figure()
        plt.hist(self._extra['vars'])
        plt.title('Variances')
        plt.savefig(vz.impath())

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
