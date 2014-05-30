from __future__ import division, print_function, absolute_import 

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('gaussian-parts-layer')
class GaussianPartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._keypoints = None

        self._clf = None

    @property
    def num_parts(self):
        return self._num_parts

    def extract(self, X):
        assert self.trained, "Must be trained before calling extract"

        #th = self._settings['threshold']
        #n_coded = self._settings.get('n_coded', 1)

        ps = self._part_shape

        #patches = np.asarray([X[:,i:i+ps[0],j:j+ps[1]] for i, j in itr.product(xrange(X.shape[0]-ps[0]+1), xrange(X.shape[1]-ps[1]+1))])

        dim = (X.shape[1]-ps[0]+1, X.shape[2]-ps[1]+1)

        feature_map = np.zeros((X.shape[0],) + dim, dtype=np.int64)

        for i, j in itr.product(xrange(dim[0]), xrange(dim[1])):
            feature_map[:,i,j] = self._clf.predict(X[:,i:i+ps[0],j:j+ps[1]].reshape((X.shape[0], -1)))

        return (feature_map[...,np.newaxis], self._num_parts)

    @property
    def trained(self):
        return self._clf is not None 

    def train(self, X, Y=None):
        assert Y is None
        ag.info('Extracting patches')
        patches = self._get_patches(X)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    def train_from_samples(self, patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        #min_prob = self._settings.get('min_prob', 0.01)

        kp_patches = patches.reshape((patches.shape[0], -1, patches.shape[-1]))

        flatpatches = kp_patches.reshape((kp_patches.shape[0], -1))

        from sklearn.mixture import GMM
        self._clf = GMM(n_components=self._num_parts,
                        n_iter=20,
                        n_init=1,
                        random_state=self._settings.get('em_seed', 0),
                        covariance_type=self._settings.get('covariance_type', 'diag'),
                        )

        self._clf.fit(flatpatches)

        #Hall = (mm.means_ * np.log(mm.means_) + (1 - mm.means_) * np.log(1 - mm.means_))
        #H = -Hall.mean(-1)

        #self._parts = mm.means_.reshape((self._num_parts,)+patches.shape[1:])
        #self._weights = mm.weights_

        # Calculate entropy of parts

        # Sort by entropy
        #II = np.argsort(H)

        #self._parts = self._parts[II]

        #self._num_parts = II.shape[0]
        #self._train_info['entropy'] = H[II]


        # Reject some parts

    def _get_patches(self, X):
        assert X.ndim == 3

        samples_per_image = self._settings.get('samples_per_image', 20) 
        patches = []

        rs = np.random.RandomState(self._settings.get('patch_extraction_seed', 0))

        consecutive_failures = 0

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

                    if True: 
                        patches.append(patch)
                        if len(patches) >= self._settings.get('max_samples', np.inf):
                            return np.asarray(patches)
                        consecutive_failures = 0
                        break

                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))
                        ag.info('cons', consecutive_failures)
                        consecutive_failures += 1

                    if consecutive_failures >= 10:
                        # Just give up.
                        raise ValueError("FATAL ERROR: Threshold is probably too high.")

        return np.asarray(patches)

    def infoplot(self, vz):
        from pylab import cm
        mu = self._clf.means_.reshape((self._num_parts,) + self._part_shape) 

        side = int(np.ceil(np.sqrt(self._num_parts)))

        grid = pnet.plot.ImageGrid(side, side, self._part_shape, border_color=(0.6, 0.2, 0.2))

        for n in xrange(self._num_parts):
            grid.set_image(mu[n], n//side, n%side, vmin=0, vmax=1, cmap=cm.gray)

        grid.save(vz.generate_filename(ext='png'))

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['clf'] = self._clf
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._clf = d['clf']
        return obj

    def __repr__(self):
        return 'GaussianPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
