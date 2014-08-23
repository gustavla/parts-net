from __future__ import division, print_function, absolute_import 

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import sys
import pnet

@Layer.register('gaussian-parts-layer')
class GaussianPartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._keypoints = None
        self._extra = {}

        self._clf = None

    @property
    def num_parts(self):
        return self._num_parts

    @property
    def pos_matrix(self):
        return self.conv_pos_matrix(self._part_shape)

    def extract(self, phi, data):
        assert self.trained, "Must be trained before calling extract"
        X = phi(data)

        ps = self._part_shape

        dim = (X.shape[1]-ps[0]+1, X.shape[2]-ps[1]+1)

        feature_map = -np.ones((X.shape[0],) + dim, dtype=np.int64)

        for i, j in itr.product(range(dim[0]), range(dim[1])):
            Xij_patch = X[:,i:i+ps[0],j:j+ps[1]]
            flatXij_patch = Xij_patch.reshape((X.shape[0], -1))

            flatXij_patch = self._standardize_patches(flatXij_patch) 

            feature_map[:,i,j] = self._clf.predict(flatXij_patch)

            #not_ok = (flatXij_patch.std(-1) <= 0.2)
            #feature_map[not_ok,i,j] = -1

        return (feature_map[...,np.newaxis], self._num_parts)

    @property
    def trained(self):
        return self._clf is not None 

    def train(self, phi, data, y=None):
        assert y is None
        X = phi(data)
        ag.info('Extracting patches')

        n_samples = self._settings.get('n_samples', 100000)
        n_per_image = self._settings.get('n_per_image', 200)

        gen = ag.image.extract_patches(X, self._part_shape, 
                                       samples_per_image=n_per_image,
                                       seed=self._settings.get('seed', 0))

        # Filter
        #gen = filter(lambda x: x.std() > self.threshold, gen)

        patches = np.asarray(list(itr.islice(gen, n_samples)))

        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    def _standardize_patches(self, flat_patches):
        #means = ag.apply_once(np.mean, patches, [1, 2])
        #stds = ag.apply_once(np.std, patches, [1, 2])

        means = ag.apply_once(np.mean, flat_patches, [1])
        stds = ag.apply_once(np.std, flat_patches, [1])

        epsilon = 0.025 / 2

        return (flat_patches - means) / (stds + epsilon)

    def train_from_samples(self, patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        #min_prob = self._settings.get('min_prob', 0.01)

        kp_patches = patches.reshape((patches.shape[0], -1, patches.shape[-1]))

        pp = kp_patches.reshape((kp_patches.shape[0], -1))

        pp = self._standardize_patches(pp) 

    
        # TODO: Temporary for easy comparison with whitening
        sigma = np.dot(pp.T, pp) / len(pp)

        self._extra['sigma'] = sigma

        # Remove the ones with too little activity

        #ok = pp.std(-1) > 0.2

        #pp = pp[ok]


        from sklearn.mixture import GMM
        self._clf = GMM(n_components=self._num_parts,
                        n_iter=self._settings.get('n_iter', 15),
                        n_init=self._settings.get('n_init', 1),
                        random_state=self._settings.get('seed', 0),
                        covariance_type=self._settings.get('covariance_type', 'diag'),
                        )

        self._clf.fit(pp)

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

    def _vzlog_output_(self, vz):
        from pylab import cm
        mu = self._clf.means_.reshape((self._num_parts,) + self._part_shape) 

        side = int(np.ceil(np.sqrt(self._num_parts)))

        grid = ag.plot.ImageGrid(mu, border_color=1)
        grid.save(vz.impath(), scale=5)

        cov = self._clf.covars_
        if cov.ndim == 3: 
            grid2 = ag.plot.ImageGrid(cov, border_color=1)
            grid2.save(vz.impath(), scale=2)
        elif cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            grid2 = ag.plot.ImageGrid(cov[np.newaxis], border_color=1)
            grid2.save(vz.impath(), scale=5)

        #ag.image.save(vz.impath(), self._clf.covars_)

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['extra'] = self._extra

        d['clf'] = self._clf
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._clf = d['clf']
        obj._extra = d['extra']
        return obj

    def __repr__(self):
        return 'GaussianPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
