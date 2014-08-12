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

        # Binocular
        #X = X.reshape((X.shape[1] // 2, 2,) + X.shape[1:])

        Xflat = X.reshape((X.shape[0], -1))

        if self._settings.get('standardize'):
            means = self._extra['means']
            variances = self._extra['vars']#.clip(min=1e-10)
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
            #X = np.concatenate([phi(Xi) for Xi in data], axis=0)
            X = np.concatenate(Xs, axis=0)
        else:
            X = phi(data)

        # Reshape to make a sample both binocular images
        #X = X.reshape((X.shape[0] // 2, 2,) + X.shape[1:])
        #assert (y[::2] == y[1::2]).all()
        #y = y[::2]


        ag.info('Entering SVM training with X.shape = ', X.shape)

        Xflat = X.reshape((X.shape[0], -1)).astype(np.float64)
        seed = self._settings.get('seed', 0)

        with ag.Timer('SVM training'):
            # Standardize features
            if self._settings.get('standardize'):
                self._extra['means'] = Xflat.mean(0)
                self._extra['vars'] = Xflat.var(0)

                means = self._extra['means']
                variances = self._extra['vars']#.clip(min=1e-10)

                Xflat = (Xflat - means) / np.sqrt(variances + self.epsilon)

            if self._penalty is None:
                # Cross-validate the penalty
                Cs = 10**np.linspace(0, -4, 10)
                avg_scores = np.zeros(len(Cs))
                for i, C in enumerate(Cs):
                    clf = LinearSVC(C=C, random_state=seed)

                    if 0:
                        cv = 5
                    elif 0:
                        from sklearn.cross_validation import LeaveOneLabelOut

                        cats = ag.io.load_small_norb('testing', ret='i', count=Xflat.shape[0])[:,0]
                        cv = LeaveOneLabelOut(cats)
                    else:
                        rs = np.random.RandomState(0)
                        p = 4
                        from sklearn.cross_validation import LeavePLabelOut
                        cats = ag.io.load_small_norb('training', ret='i', count=Xflat.shape[0])[:,0]
                        ll = ag.io.load_small_norb('training', ret='y')[:Xflat.shape[0]]
                        assert (ll == y).all()


                        cv_gen = LeavePLabelOut(cats, p)
                        cv0 = list(cv_gen)
                        rs.shuffle(cv0)
                        cv = cv0[:5]

                        #import IPython
                        #from vzlog.default import vz
                        #IPython.embed()

                        if 0:
                            from vzlog.default import vz
                            XX = ag.io.load_small_norb('training', ret='x', count=Xflat.shape[0])
                            XX = (XX.astype(np.float64) / 255).transpose(0, 2, 3, 1)

                            import pylab as plt
                            for ci in range(5):
                                vz.section('ci', ci)
                                print('ci', ci)
                                vv = np.bincount(cv[ci][0], minlength=y.size).astype(np.bool_)
                                for c in range(5):
                                    vz.title('class', c)
                                    print('class', c)
                                    #X_ = Xflat[vv & (y == c)]
                                    grid = ag.plot.ColorImageGrid(XX[vv & (y == c)][:10], cols=10)
                                    grid.save(vz.impath())

                                    fig = plt.figure()
                                    #cc = ag.apply_once_over_axes(Xflat[vv & (y == c)], [1, 2, 3], remove_axes=True)
                                    plt.hist(cc)
                                    plt.savefig(vz.impath(ext='svg'))
                                    fig.close()

                                    vz.flush()


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
        ag.info('Confusion matrix')
        import pnet
        ag.info(pnet.rescalc.confusion_matrix(y, yhat))

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
