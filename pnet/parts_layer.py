from __future__ import division, print_function, absolute_import 

import numpy as np
import itertools as itr
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
        
        from pnet.cyfuncs import code_index_map

        th = self._settings['threshold']


        return code_index_map(X, weights, constants, th,
                              outer_frame=self._settings['outer_frame']) 


    def train(self, X):
        patches = self._get_patches(X)
        return self.train_from_samples(patches)

    def train_from_samples(self, patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        from pnet.bernoullimm import BernoulliMM

        mm = BernoulliMM(n_components=self._num_parts, n_iter=20, n_init=1)
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
            i_iter = iter(indices)

            for sample in xrange(samples_per_image):
                for tries in xrange(50):
                    #x, y = random.randint(0, w-1), random.randint(0, h-1)
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
                        break

        return np.asarray(patches)

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        d['weights'] = self._weights

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._weights = d['weights']
        return obj
