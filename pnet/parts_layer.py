from __future__ import division, print_function, absolute_import 

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer

# TODO: Use later
if 0:
    def _threshold_in_counts(self, threshold, num_edges):
        size = self.settings['part_size']
        frame = self.settings['patch_frame']
        return max(1, int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges))

@Layer.register('parts-layer')
class PartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        #, outer_frame=1, threshold=1):
        self._num_parts = num_parts
        self._part_shape = part_shape
        #self._outer_frame = outer_frame
        #self._threshold = threshold
        self._settings = settings

        self._parts = None
        self._weights = None

    def extract(self, X):
        assert self._parts is not None, "Must be trained before calling extract"

        th = self._settings['threshold']
        part_logits = np.rollaxis(logit(self._parts).astype(np.float64), 0, 4)
        constant_terms = np.apply_over_axes(np.sum, np.log(1-self._parts).astype(np.float64), [1, 2, 3]).ravel()

        from pnet.cyfuncs import code_index_map

        feature_map = code_index_map(X, part_logits, constant_terms, th,
                                     outer_frame=self._settings['outer_frame'])
        return (self._num_parts, feature_map)
    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X):
        ag.info('Extracting patches')
        patches = self._get_patches(X)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    def train_from_samples(self, patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        from pnet.bernoullimm import BernoulliMM
        min_prob = self._settings.get('min_prob', 0.01)

        mm = BernoulliMM(n_components=self._num_parts, n_iter=1000, tol=1e-15,n_init=4, random_state=0, min_prob=min_prob)
        mm.fit(patches.reshape((patches.shape[0], -1)))

        #import pdb; pdb.set_trace()
        self._parts = mm.means_.reshape((self._num_parts,)+patches.shape[1:])
        self._weights = mm.weights_

    def _get_patches(self, X):
        assert X.ndim == 4

        samples_per_image = self._settings.get('samples_per_image', 20) 
        fr = self._settings['outer_frame']
        patches = []

        rs = np.random.RandomState(self._settings.get('patch_extraction_seed', 0))

        th = self._settings['threshold']

        for Xi in X:

            # How many patches could we extract?
            w, h = [Xi.shape[i]-self._part_shape[i]+1 for i in xrange(2)]

            # TODO: Maybe shuffle an iterator of the indices?
            indices = list(itr.product(xrange(w-1), xrange(h-1)))
            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            for sample in xrange(samples_per_image):
                N = 200
                for tries in xrange(N):
                    x, y = i_iter.next()
                    selection = [slice(x, x+self._part_shape[0]), slice(y, y+self._part_shape[1])]

                    patch = Xi[selection]
                    #edgepatch_nospread = edges_nospread[selection]
                    if fr == 0:
                        tot = patch.sum()
                    else:
                        tot = patch[fr:-fr,fr:-fr].sum()

                    if th <= tot: 
                        patches.append(patch)
                        if len(patches) >= self._settings.get('max_samples', np.inf):
                            return np.asarray(patches)
                        break

                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))

        return np.asarray(patches)

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._weights = d['weights']
        return obj
