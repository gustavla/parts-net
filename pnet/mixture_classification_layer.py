from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
from pnet.bernoulli_mm import BernoulliMM
from Draw_means import show_visparts
import amitgroup as ag


@Layer.register('mixture-classification-layer')
class MixtureClassificationLayer(SupervisedLayer):
    def __init__(self, settings={}):
        self._settings=dict(num_components=1,
                            min_prob=.0001,
                            seed=None,
                            standardize=False,
                            classify=True,
                            subclassify=False,
                            allocate=True,
                            num_train=None,
                            min_count=0)
        for k, v in settings.items():
            if k not in self._settings:
                    raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._n_components = self._settings['num_components']
        self._min_prob = self._settings['min_prob']
        self._models = None
        self._extra = {}

    @property
    def trained(self):
        return self._models is not None

    def reset(self):
        self._models = None

    @property
    def classifier(self):
        return True

    def _extract(self, phi, data):
        X = phi(data)
        #XX = X[:, np.newaxis, np.newaxis]
        Xflat=X.reshape((X.shape[0],np.prod(X.shape[1:])))
        C=len(self._models)


        abb=np.zeros((Xflat.shape[0],C))
        ysub=np.zeros((Xflat.shape[0],C))
        bbout=[]
        for c in range(C):
            ltheta = np.log(self._models[c])
            ltheta=ltheta.reshape((ltheta.shape[0],np.prod(ltheta.shape[1:])))
            oltheta= np.log(1-self._models[c])
            oltheta=oltheta.reshape((ltheta.shape[0],np.prod(ltheta.shape[1:])))
            bb=np.dot(Xflat,ltheta.T)+np.dot(1-Xflat,oltheta.T)
            if c>0:
                bbout=np.concatenate((bbout,bb),axis=1)
            else:
                bbout=bb
            abb[:,c]=np.max(bb,axis=1)
            ysub[:,c]=np.argmax(bb,axis=1)
        if self._settings.get('subclassify'):
                return X, ysub
        if self._settings.get('classify'):
            yhat = np.argmax(abb, axis=1)
        else:
            return np.exp(bbout/1000)
        return yhat

    def _train(self, phi, data, y):
        import types
        if self._settings['num_train'] is not None:
            data=data[0:self._settings['num_train'],:]
            y=y[0:self._settings['num_train']]

        X = phi(data)

        K = y.max() + 1
        mm_models = []
        mm_comps = []
        mm_weights = []
        mm_visparts=[]
        quant=np.float32(X.shape[-1])
        min_prob=1/(2*quant)
        for k in range(K):
            Xk = X[y == k]
            Xk = Xk.reshape((Xk.shape[0], -1))
            mm = BernoulliMM(n_components=self._n_components,
                             n_iter=10,
                             n_init=1,
                             thresh=1e-5,
                             random_state=self._settings['seed'],
                             min_prob=min_prob,
                             blocksize=200,
                             allocate=self._settings['allocate'],
                             verbose=True)
            mm.fit(Xk)
            mm=self.organize_parts(Xk,mm,data[y==k],self._n_components, X.shape[1:])
            mm.mu=np.floor(mm.mu*quant)/quant+1./(2*quant)
            show_visparts(mm.visparts,1,10, True,k)


            mm_models.append(mm.mu)
            mm_weights.append(mm.weights_)
            mm_visparts.append(mm.visparts)
            np.set_printoptions(precision=2, suppress=True)
            print('Weights:', mm.weights_)

        self._extra['weights'] = mm_weights
        self._extra['vis_parts'] = mm_visparts
        self._models = mm_models



    def organize_parts(self,Xflat,mm,raw_originals,ml,sh):
        # For each data point returns part index and angle index.
        comps = mm.predict(Xflat)
        counts = np.bincount(comps, minlength=ml)
        # Average all images beloinging to true part k at selected rotation given by second coordinate of comps
        visparts = np.asarray([
            raw_originals[comps == k,:].mean(0)
            for k in range(ml)
        ])
        ww = counts / counts.sum()
        ok = counts >= self._settings['min_count']
        mu = mm.means_.reshape((ml,)+sh)
        # Reject some parts
        #ok = counts >= self._settings['min_count']
        II = np.argsort(counts)[::-1]
        print('Counts',np.sort(counts))
        #II = range(len(counts))
        II = np.asarray([ii for ii in II if ok[ii]])
        ag.info('Keeping', len(II), 'out of', ok.size, 'parts')
        mm._num_true_parts = len(II)
        mm.means_ = mm.means_[II]
        mm.weights_ = mm.weights_[II]
        mm.counts = counts[II]
        mm.visparts = visparts[II]
        mm.mu=mu[II,:]
        return(mm)


    def _vzlog_output_(self, vz):
        import pylab as plt
        vz.section('Mixture model')
        vz.log(self._models.shape)

        plt.figure(figsize=(6, 4))
        X = self._models.ravel()
        plt.hist(np.log10(X[X > 0]), normed=True)
        plt.xlabel('Concentration (10^x)')
        plt.title('Model probs')
        plt.savefig(vz.impath(ext='svg'))

    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        d['models'] = self._models
        d['extra'] = self._extra

        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        obj._models = d['models']
        obj._extra = d['extra']
        return obj

    def __repr__(self):
        return 'MixtureClassificationLayer(n_components={})'.format(
               self._n_components)
