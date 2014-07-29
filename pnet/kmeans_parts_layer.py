from __future__ import division, print_function, absolute_import

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet

@Layer.register('kmeans-parts-layer')
class KMeansPartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._extra = {}
        self._parts = None
        self._whitening_matrix = None

    @property
    def num_parts(self):
        return self._num_parts

    @property
    def part_shape(self):
        return self._part_shape

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

            #if flatXij_patch.dtype == np.uint8:
                #flatXij_patch = flatXij_patch.astype(np.float64) / 255

            # Whiten the patches

            ok = (flatXij_patch.std(-1) > self.threshold)

            flatXij_patch[ok] = self._standardize_patches(flatXij_patch[ok])

            flatXij_patch[ok] = self.whiten_patches(flatXij_patch[ok])

            feature_map[ok,i,j] = self._extra['classifier'].predict(flatXij_patch[ok])

            #feature_map[~ok,i,j] = -1

        return (feature_map[...,np.newaxis], self._num_parts)

    @property
    def trained(self):
        return self._parts is not None

    @property
    def threshold(self):
        return -1  # No threshold

    def _standardize_patches(self, flat_patches):
        #means = ag.apply_once_over_axes(np.mean, patches, [1, 2])
        #stds = ag.apply_once_over_axes(np.std, patches, [1, 2])

        means = np.apply_over_axes(np.mean, flat_patches, [1])
        stds = np.apply_over_axes(np.std, flat_patches, [1])

        epsilon = 0.025 / 2

        return (flat_patches - means) / (stds + epsilon)

    def whiten_patches(self, flat_patches):
        return np.dot(self._whitening_matrix, flat_patches.T).T

    def train(self, phi, data, y=None):
        assert y is None
        X = phi(data)
        ag.info('Extracting patches')
        #patches = self._get_patches(X)

        n_samples = self._settings.get('n_samples', 100000)
        n_per_image = self._settings.get('n_per_image', 200)

        # Get patches
        gen = ag.image.extract_patches(X, self._part_shape,
                                       samples_per_image=n_per_image,
                                       seed=self._settings.get('seed', 0))


        # Filter
        #gen = filter(lambda x: x.std() > self.threshold, gen)

        rs = np.random.RandomState(0)

        if 0:
            def func_ok(x):
                if rs.uniform() < 0.2:
                    return True
                else:
                    return x.std() > 0.02

            gen = filter(func_ok, gen)

        # Now request the patches and convert them to floats
        #patches = np.asarray(list(itr.islice(gen, n_samples)), dtype=np.float64) / 255
        patches = np.asarray(list(itr.islice(gen, n_samples)))

        from pnet.vzlog import default as vz

        # Flatten the patches
        pp = patches.reshape((patches.shape[0], -1))

        def plot(title):
            vz.section(title)
            ag.plot.ImageGrid(pp[:200].reshape((-1, 6, 6)), rows=5).save(vz.impath(), scale=3)

        plot('Original patches')

        # Standardize the patches
        pp = self._standardize_patches(pp)

        plot('Standardized patches')

        # Determine whitening coefficients
        sigma = np.dot(pp.T, pp) / len(pp)

        self._extra['sigma'] = sigma

        U, S, _ = np.linalg.svd(sigma)

        epsilon = 1e-40
        shrinker = np.diag(1 / np.sqrt(S + epsilon))
        #shrinker = np.diag(1 / np.sqrt(S.clip(min=epsilon)))

        #self._whitening_matrix = U @ shrinker @ U.T
        self._whitening_matrix = np.dot(U, np.dot(shrinker, U.T))

        # Now whiten the training patches
        pp = self.whiten_patches(pp)

        plot('Whitened patches')

        # Run K-means
        from sklearn.cluster import KMeans, MiniBatchKMeans

        #cl = MiniBatchKMeans(
        cl = KMeans(
                    n_clusters=self._num_parts,
                    n_init=self._settings.get('n_init', 1),
                    random_state=self._settings.get('seed', 0),
                    #batch_size=50000,
                    n_jobs=self._settings.get('n_jobs', 1),
                    )

        ag.info('Training', self._num_parts, 'K-means parts')
        cl.fit(pp)
        ag.info('Done.')

        self._parts = cl.cluster_centers_.reshape((-1,) + patches.shape[1:])
        self._extra['classifier'] = cl

        vz.section('Parts')

    def _vzlog_output_(self, vz):
        from pylab import cm
        grid = ag.plot.ImageGrid(self._parts)
        grid.save(vz.impath(), scale=4)

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['extra'] = self._extra
        d['whitening_matrix'] = self._whitening_matrix

        d['parts'] = self._parts
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._extra = d['extra']
        obj._whitening_matrix = d['whitening_matrix']
        return obj

    def __repr__(self):
        return 'KmeansPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
