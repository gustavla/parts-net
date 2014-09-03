from __future__ import division, print_function, absolute_import

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet


@Layer.register('kmeans-parts-layer')
class KMeansPartsLayer(Layer):
    def __init__(self, n_parts=100, part_shape=(6, 6), settings={}):
        self._num_parts = n_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._parts = None
        self._whitening_matrix = None
        self._min_prob = -8

        self._settings = dict(whitening_epsilon=0.1,
                              standardization_epsilon=10 / 255,
                              n_init=1,
                              max_iter=300,
                              seed=0,
                              n_jobs=1,
                              n_samples=100000,
                              n_per_image=200,
                              code_bkg=False,
                              random_centroids=False,
                              std_thresh=0,
                              standardize=True,
                              whiten=True,
                              coding='hard',
                              )
        self._extra = {}

        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v

        # Whitening and standardization epsilons
        self.w_epsilon = self._settings['whitening_epsilon']
        self.epsilon = self._settings['standardization_epsilon']

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

        if self._settings['code_bkg']:
            bkg_part = self.num_parts
            n_features = self.num_parts + 1
        else:
            bkg_part = -1
            n_features = self.num_parts

        coding = self._settings['coding']
        if coding == 'hard':
            feature_map = np.empty((X.shape[0],) + dim, dtype=np.int64)
            feature_map[:] = bkg_part

            with ag.Timer('extracting'):
                for i, j in itr.product(range(dim[0]), range(dim[1])):
                    Xij_patch = X[:, i:i+ps[0], j:j+ps[1]]
                    flatXij_patch = Xij_patch.reshape((X.shape[0], -1))

                    ok = (flatXij_patch.std(-1) >= self._settings['std_thresh'])

                    XX = flatXij_patch[ok]
                    if len(XX) > 0:
                        if XX.dtype != self._dtype:
                            XX = XX.astype(self._dtype)
                        feature_map[ok, i, j] = self._extract_func(XX)

            return (feature_map[..., np.newaxis], n_features)
        else:
            feature_map = np.empty((X.shape[0],) + dim + (n_features,),
                                   dtype=self._dtype)
            if coding == 'soft':
                feature_map[:] = self._min_prob
            else:
                feature_map[:] = 0.0

            with ag.Timer('extracting'):
                for i, j in itr.product(range(dim[0]), range(dim[1])):
                    Xij_patch = X[:, i:i+ps[0], j:j+ps[1]]
                    flatXij_patch = Xij_patch.reshape((X.shape[0], -1))

                    ok = (flatXij_patch.std(-1) >= self._settings['std_thresh'])

                    XX = flatXij_patch[ok]
                    if len(XX) > 0:
                        if XX.dtype != self._dtype:
                            XX = XX.astype(self._dtype)
                        feature_map[ok, i, j] = self._extract_func(XX)



            return feature_map


    @property
    def trained(self):
        return self._parts is not None

    def _standardize_patches(self, flat_patches):
        means = ag.apply_once(np.mean, flat_patches, [1])
        variances = ag.apply_once(np.var, flat_patches, [1])

        return (flat_patches - means) / np.sqrt(variances + self.epsilon)

    def whiten_patches(self, flat_patches):
        return np.dot(self._whitening_matrix, flat_patches.T).T

    def train(self, phi, data, y=None):
        assert y is None
        n_samples = self._settings['n_samples']
        n_per_image = self._settings['n_per_image']

        n_images = n_samples // n_per_image

        # TODO: This always processes all the images
        X = phi(data[:n_images])
        ag.info('Extracting patches')
        #patches = self._get_patches(X)

        # Get patches
        gen = ag.image.extract_patches(X, self._part_shape,
                                       samples_per_image=n_per_image,
                                       seed=self._settings['seed'])


        ag.info('Extracting patches 2')
        # Filter
        th = self._settings['std_thresh']
        if th > 0:
            gen = (x for x in gen if x.std() >= th)

        rs = np.random.RandomState(0)

        # Now request the patches and convert them to floats
        #patches = np.asarray(list(itr.islice(gen, n_samples)), dtype=np.float64) / 255
        patches = np.asarray(list(itr.islice(gen, n_samples)))
        ag.info('Extracting patches 3')

        from vzlog.default import vz

        # Flatten the patches
        pp = patches.reshape((patches.shape[0], -1))

        C = X.shape[-1]
        sh = (-1,) + self._part_shape + (C,)

        if C <= 3:
            def plot(title):
                vz.section(title)
                grid = ag.plot.ColorImageGrid(pp[:1000].reshape(sh), rows=15)
                grid.save(vz.impath(), scale=3)
        else:
            def plot(title): return

        plot('Original patches')

        # Standardize the patches
        if self._settings['standardize']:
            pp = self._standardize_patches(pp)

        plot('Standardized patches')

        # Determine whitening coefficients
        sigma = np.dot(pp.T, pp) / len(pp)

        self._extra['sigma'] = sigma

        if self._settings['whiten']:
            U, S, _ = np.linalg.svd(sigma)

            shrinker = np.diag(1 / np.sqrt(S + self.w_epsilon))

            #self._whitening_matrix = U @ shrinker @ U.T
            self._whitening_matrix = np.dot(U, np.dot(shrinker, U.T))

            # Now whiten the training patches
            pp = self.whiten_patches(pp)
        else:
            self._whitening_matrix = None

        plot('Whitened patches')

        if self._settings['random_centroids']:
            rs = np.random.RandomState(self._settings['seed'])
            sh = (self._num_parts,) + self._part_shape
            self._parts = rs.normal(0, 1, size=sh)
            #self._parts /= ag.apply_once(np.mean, self._parts, [1, 2])
            return
        else:
            # Run K-means
            from sklearn.cluster import KMeans, MiniBatchKMeans

            #cl = MiniBatchKMeans(
            cl = KMeans(
                        n_clusters=self._num_parts,
                        n_init=self._settings['n_init'],
                        max_iter=self._settings['max_iter'],
                        random_state=self._settings['seed'],
                        #batch_size=50000,
                        n_jobs=self._settings['n_jobs'],
                        )

            ag.info('Training', self._num_parts, 'K-means parts')
            cl.fit(pp)
            ag.info('Done.')

            counts = np.bincount(cl.labels_, minlength=self._num_parts)
            ww = counts / counts.sum()
            HH = np.sum(-ww * np.log(ww))
            print('entropy', HH)

            II = np.argsort(counts)[::-1]
            cl.cluster_centers_ = cl.cluster_centers_[II]
            counts = counts[II]

            ag.info('counts', counts)

            self._parts = cl.cluster_centers_.reshape((-1,) + patches.shape[1:])

            vz.section('Parts')

        self._preprocess()

    def _create_extract_func(self):
        import theano.tensor as T
        import theano
        # Create Theano convolution function
        s_input = T.matrix(name='input')
        x = s_input

        # Standardize input
        s_means = s_input.mean(1, keepdims=True)
        s_vars = s_input.var(1, keepdims=True)
        if self._settings['standardize']:
            x = (x - s_means) / T.sqrt(s_vars + self.epsilon)
            # Standardize, end

        # Whiten
        if self._settings['whiten']:
            s_whiten = theano.shared(self._whitening_matrix.astype(s_input.dtype),
                                     name='whitening_matrix')
            x = T.dot(s_whiten, x.T).T

        mu = (self._parts
              .reshape((self._parts.shape[0], -1))
              .astype(s_input.dtype))
        alphas = np.sum(mu ** 2, 1) / 2

        s_mu = theano.shared(mu, name='mu')
        s_alphas = theano.shared(alphas, name='alphas').dimshuffle('x', 0)

        sums = T.tensordot(x, s_mu, [[1], [1]])

        coding = self._settings['coding']
        if coding == 'hard':
            xx = (s_alphas - sums)
            s_output = xx.argmin(1)
        elif coding == 'triangle':
            x2 = T.sum(x ** 2, 1, keepdims=True) / 2
            dist2 = (x2 - sums + s_alphas)
            dist = T.sqrt(dist2)
            mean_dists = dist.mean(1, keepdims=True)
            # TODO CHANGE BACK!:
            #s_output = (1 + T.tanh(mean_dists - dist)) / 2 #T.maximum(0, mean_dists - dist)
            s_output = T.maximum(0, mean_dists - dist)
        elif coding == 'soft':
            x2 = T.sum(x ** 2, 1, keepdims=True) / 2
            dist2 = (x2 - sums + s_alphas)
            unorm_logs = -dist2
            A = unorm_logs.max(axis=1, keepdims=True)
            Z_logs = A + T.log(T.sum(T.exp(unorm_logs - A), axis=1, keepdims=True))
            log_posts = unorm_logs - Z_logs

            dist = T.sqrt(dist2)
            mean_dists = dist.mean(1, keepdims=True)
            #s_output = T.maximum(0, mean_dists - dist)
            s_output = T.maximum(self._min_prob, log_posts)


        self._dtype = s_input.dtype
        f = theano.function([s_input], s_output)
        return f

    def _preprocess(self):
        self._extract_func = self._create_extract_func()

    def _vzlog_output_(self, vz):
        from pylab import cm
        vz.log('parts shape', self._parts.shape)

        C = self._parts.shape[-1]

        if C == 1:
            #grid = ag.plot.ImageGrid(self._parts[...,0])
            vz.log('span', ag.span(self._parts))
            grid = ag.plot.ImageGrid(self._parts[...,0], vsym=True)
            grid.save(vz.impath(), scale=4)
        elif C in [2, 3]:
            grid = ag.plot.ColorImageGrid(cpp, vmin=None, vmax=None, vsym=True)
            grid.save(vz.impath(), scale=4)
        else:
            grid = ag.plot.ImageGrid(self._parts.transpose(0, 3, 1, 2), cmap=cm.jet)
            grid.save(vz.impath(), scale=2)

        #grid = ag.plot.ImageGrid(self._parts.transpose(0, 3, 1, 2))
        #grid.save(vz.impath(), scale=4)

        if self._whitening_matrix is not None:
            vz.text('Whitening matrix')
            grid = ag.plot.ImageGrid(self._whitening_matrix, vsym=True, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=4)

        if self._extra['sigma'].shape[0] < 100:
            vz.text('Sigma matrix')
            grid = ag.plot.ImageGrid(self._extra['sigma'], vsym=True, cmap=cm.RdBu_r)
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
        obj._preprocess()
        return obj

    def __repr__(self):
        return 'KmeansPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
