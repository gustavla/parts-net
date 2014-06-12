from __future__ import division, print_function, absolute_import 

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
from pnet.parts_layer import PartsLayer
import pnet

@Layer.register('random-forest-parts-layer')
class RandomForestPartsLayer(PartsLayer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._random_forest = None
        self._settings = settings

    @property
    def num_parts(self):
        return self._num_parts

    def extract(self, X):
        
        dims = (X.shape[1]-self._part_shape[0]+1, X.shape[2]-self._part_shape[1]+1)

        feats = -np.ones((X.shape[0],) + dims + (1,), dtype=np.int64)

        th = self._settings['threshold']

        for i, j in itr.product(range(dims[0]), range(dims[1])):
            Xij = X[:,i:i+self._part_shape[0],j:j+self._part_shape[1]]

            Xij = Xij.reshape((Xij.shape[0], -1))

            #edge_counts = np.apply_over_axes(np.sum, Xij, [1, 2, 3])
            edge_counts = Xij.sum(1)
            ok = (edge_counts >= th)

            parts = self._random_forest.predict(Xij)

            feats[:,i,j,0] = ok * parts + (1 - ok) * (-1)

        return (feats, self._num_parts)

    @property
    def trained(self):
        return self._random_forest is not None 

    def train(self, X, Y=None):
        assert Y is None
        patches = self._get_patches(X)
        comps = self.train_from_samples(patches)

        # Now, train random forests to do the coding
        from sklearn.ensemble import RandomForestClassifier

        patches = patches.reshape((patches.shape[0], -1))
         
        clf = RandomForestClassifier(n_estimators=self._settings.get('trees', 10), max_depth=self._settings.get('max_depth', 10))
        import gv
        with gv.Timer('Forest training'):
            clf = clf.fit(patches, comps)

        self._random_forest = clf


    def train_from_samples(self, patches):
        from pnet.bernoulli_mm import BernoulliMM
        min_prob = self._settings.get('min_prob', 0.01)

        support_mask = self._settings.get('support_mask')
        if support_mask is not None:
            kp_patches = patches[:,support_mask]
        else:
            kp_patches = patches.reshape((patches.shape[0], -1, patches.shape[-1]))

        if self._settings.get('kp') == 'funky':
            patches = patches[:,::2,::2]

            # Only patches above the threshold
            print('patches', patches.shape)
            patches = patches[np.apply_over_axes(np.sum, patches.astype(np.int64), [1, 2, 3]).ravel() >= self._settings['threshold']]
            print('patches', patches.shape)

        flatpatches = kp_patches.reshape((kp_patches.shape[0], -1))

        mm = BernoulliMM(n_components=self._num_parts, 
                         n_iter=10, 
                         tol=1e-15,
                         n_init=1, 
                         random_state=self._settings.get('em_seed', 0), 
                         min_prob=min_prob, 
                         verbose=False)
        mm.fit(flatpatches)
        logprob, resp = mm.eval(flatpatches)
        comps = resp.argmax(-1)


        Hall = (mm.means_ * np.log(mm.means_) + (1 - mm.means_) * np.log(1 - mm.means_))
        H = -Hall.mean(-1)

        if support_mask is not None:
            self._parts = 0.5 * np.ones((self._num_parts,) + patches.shape[1:])
            self._parts[:,support_mask] = mm.means_.reshape((self._parts.shape[0], -1, self._parts.shape[-1]))
        else:
            self._parts = mm.means_.reshape((self._num_parts,)+patches.shape[1:])
        self._weights = mm.weights_

        # Calculate entropy of parts

        # Sort by entropy
        II = np.argsort(H)

        self._parts = self._parts[II]

        self._num_parts = II.shape[0]
        #self._train_info['entropy'] = H[II]

        return comps


    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['random_forest'] = self._random_forest
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._random_forest = d['random_forest']
        return obj

    def __repr__(self):
        return 'RandomForestPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
