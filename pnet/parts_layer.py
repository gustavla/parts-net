from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
from amitgroup.plot import ImageGrid


@Layer.register('parts-layer')
class PartsLayer(Layer):
    def __init__(self, num_parts, part_shape, settings={}):
        self._num_parts = num_parts
        self._part_shape = part_shape
        self._settings = settings
        self._train_info = {}
        self._keypoints = None

        # TODO Temporary
        self._min_llhs = None

        self._parts = None
        self._weights = None

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
        assert self._parts is not None, \
            "Must be trained before calling extract"
        X = phi(data)

        th = self._settings['threshold']
        n_coded = self._settings.get('n_coded', 1)

        support_mask = self._settings.get('support_mask')
        if support_mask is None:
            support_mask = np.ones(self._part_shape, dtype=np.bool)

        outer_frame = self._settings.get('outer_frame', 0)
        standardize = self._settings.get('standardize', 0)
        min_percentile = self._settings.get('min_percentile', 0.0)

        from pnet.cyfuncs import code_index_map_general
        feature_map = code_index_map_general(X,
                                             self._parts,
                                             support_mask.astype(np.uint8),
                                             th,
                                             outer_frame=outer_frame,
                                             n_coded=n_coded,
                                             standardize=standardize,
                                             min_percentile=min_percentile)

        return (feature_map, self._num_parts)

    @property
    def trained(self):
        return self._parts is not None

    def train(self, phi, data, y=None):
        assert y is None
        X = phi(data)
        ag.info('Extracting patches')
        patches = self._get_patches(X)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    def train_from_samples(self, patches):
        min_prob = self._settings.get('min_prob', 0.01)

        support_mask = self._settings.get('support_mask')
        if support_mask is not None:
            kp_patches = patches[:, support_mask]
        else:
            sh = (patches.shape[0], -1, patches.shape[-1])
            kp_patches = patches.reshape(sh)

        if self._settings.get('kp') == 'funky':
            patches = patches[:, ::2, ::2]

            # Only patches above the threshold
            print('patches', patches.shape)
            sums = np.apply_over_axes(np.sum,
                                      patches.astype(np.int64),
                                      [1, 2, 3]).ravel()
            patches = patches[sums >= self._settings['threshold']]
            print('patches', patches.shape)

        flatpatches = kp_patches.reshape((kp_patches.shape[0], -1))

        from pnet.bernoulli_mm import BernoulliMM
        mm = BernoulliMM(n_components=self._num_parts,
                         n_iter=self._settings.get('n_iter', 15),
                         tol=1e-3,
                         n_init=1,
                         random_state=self._settings.get('seed', 0),
                         min_prob=min_prob,
                         verbose=True)
        mm.fit(flatpatches)
        comps = mm.predict(flatpatches)

        counts = np.bincount(comps, minlength=self._num_parts)

        means = mm.means_
        weights = mm.weights_
        if support_mask is not None:
            means = 0.5 * np.ones((self._num_parts,) + patches.shape[1:])
            sh = (self._parts.shape[0], -1, self._parts.shape[-1])
            means[:, support_mask] = mm.means_.reshape(sh)
        else:
            means = mm.means_.reshape((self._num_parts,)+patches.shape[1:])

        self._parts = means

        Hall = (means * np.log(means) + (1 - means) * np.log(1 - means))
        H = -Hall.mean(-1).mean(-1).mean(-1)

        self._weights = weights

        # Calculate entropy of parts

        # Sort by entropy
        # II = np.argsort(H)

        # Sort by counts
        II = np.argsort(counts)[::-1]

        # Re-order everything
        H = H[II]
        counts = counts[II]
        self._parts = self._parts[II]

        self._num_parts = II.shape[0]

        self._train_info['entropy'] = H[II]
        self._train_info['counts'] = counts

        # Reject some parts

    def reject_parts(self):
        ok = self

        """
        Hall = self._parts * np.log(self._parts) + \
            (1 - self._parts) * np.log(1 - self._parts)
        H = -np.apply_over_axes(np.mean, Hall, [1, 2, 3]).ravel()
        """

        th1 = self._parts
        th0 = np.apply_over_axes(np.mean, self._parts, [1, 2])

        M1 = th1 * np.log(th1 / th0) +\
            (1 - th1) * np.log((1 - th1) / (1 - th0))
        S1 = np.log((th1 / (1 - th1) * ((1 - th0) / th0)))**2 * th1 * (1 - th1)
        mu1 = np.apply_over_axes(np.sum, M1, [1, 2, 3]).ravel()
        sigma1 = np.sqrt(np.apply_over_axes(np.sum, S1, [1, 2, 3])).ravel()

        M1 = th0 * np.log(th1 / th0) +\
            (1 - th1) * np.log((1 - th0) / (1 - th0))
        S1 = np.log((th1 / (1 - th1) * ((1 - th0) / th0)))**2 * th0 * (1 - th0)
        mu0 = np.apply_over_axes(np.sum, M1, [1, 2, 3]).ravel()
        # sigma0 = np.sqrt(np.apply_over_axes(np.sum, S1, [1, 2, 3])).ravel()

        ok = ((mu1 - mu0) / sigma1) > self._settings.get('reject_entropy', 1.0)

        print(ok.shape)
        print(ok.sum())

        self._parts = self._parts[ok]
        self._num_parts = self._parts.shape[0]

    def _get_patches(self, X):
        assert X.ndim == 4

        samples_per_image = self._settings.get('samples_per_image', 20)
        fr = self._settings.get('outer_frame', 0)
        patches = []

        rs = np.random.RandomState(self._settings.get('seed', 0))

        th = self._settings['threshold']
        max_th = self._settings.get('max_threshold', np.inf)
        support_mask = self._settings.get('support_mask')

        consecutive_failures = 0
        max_samples = self._settings.get('max_samples', np.inf)

        for Xi in X:

            # How many patches could we extract?
            w, h = [Xi.shape[i]-self._part_shape[i]+1 for i in range(2)]

            # TODO: Maybe shuffle an iterator of the indices?
            indices = list(itr.product(range(w-1), range(h-1)))
            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            for sample in range(samples_per_image):
                TRIES = 200
                for tries in range(TRIES):
                    x, y = next(i_iter)
                    selection = [slice(x, x+self._part_shape[0]),
                                 slice(y, y+self._part_shape[1])]

                    patch = Xi[selection]
                    if support_mask is not None:
                        tot = patch[support_mask].sum()
                    elif fr == 0:
                        tot = patch.sum()
                    else:
                        tot = patch[fr:-fr, fr:-fr].sum()

                    if th <= tot <= max_th:
                        patches.append(patch)
                        if len(patches) >= max_samples:
                            return np.asarray(patches)
                        consecutive_failures = 0
                        break

                    if tries == TRIES-1:
                        ag.info('WARNING: {} tries'.format(TRIES))
                        ag.info('cons', consecutive_failures)
                        consecutive_failures += 1

                    if consecutive_failures >= 10:
                        # Just give up.
                        raise ValueError(
                            "FATAL ERROR: "
                            "Threshold is probably too high (in {})".format(
                                self.__class__.__name__))

        return np.asarray(patches)

    def _vzlog_output_(self, vz):
        from pylab import cm

        grid = ImageGrid(self._parts.transpose((0, 3, 1, 2)),
                         cmap=cm.jet, vmin=0, vmax=1)
        grid.save(vz.impath(), scale=3)

        vz.log('Counts:')
        vz.log(self._train_info.get('counts'))

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._weights, label='weight')
        plt.savefig(vz.impath(ext='svg'))

        vz.log('weights span', self._weights.min(), self._weights.max())

        if 'entropy' in self._train_info:
            import pylab as plt
            plt.figure(figsize=(6, 3))
            plt.plot(self._train_info['entropy'])
            plt.savefig(vz.impath(ext='svg'))

            vz.log('median entropy', np.median(self._train_info['entropy']))

        llhs = self._train_info.get('llhs')
        if llhs is not None:
            vz.log('llhs', llhs.shape)

            plt.figure(figsize=(6, 3))
            plt.errorbar(np.arange(self._num_parts), llhs.mean(0),
                         yerr=llhs.std(0), fmt='--o')
            plt.title('log likelihood means')
            plt.savefig(vz.impath(ext='svg'))

            parts = np.argmax(llhs, 1)

            means = np.zeros(self._num_parts)
            sigmas = np.zeros(self._num_parts)
            for f in range(self._num_parts):
                llh0 = llhs[parts == f, f]
                means[f] = llh0.mean()
                sigmas[f] = llh0.std()
                vz.log('llh', f, ':', means[f], sigmas[f])

            plt.figure(figsize=(6, 3))
            plt.errorbar(np.arange(self._num_parts), means,
                         yerr=sigmas, fmt='--o')

            from scipy.special import logit

            anal_means = -np.prod(self._parts.shape[1:]) * \
                self._train_info['entropy']
            L = logit(self._parts)**2 * self._parts * (1 - self._parts)
            anal_sigmas = np.sqrt(np.apply_over_axes(np.sum,
                                                     L,
                                                     [1, 2, 3]).ravel())

            plt.errorbar(np.arange(self._num_parts), anal_means,
                         yerr=anal_sigmas, fmt='--o')
            plt.title('coded log likelihood means')
            plt.savefig(vz.impath(ext='svg'))

            plt.figure(figsize=(6, 3))
            plt.hist(llhs.ravel(), 50)
            plt.title('Log likelihood distribution')
            plt.savefig(vz.impath(ext='svg'))

    def save_to_dict(self):
        d = {}
        d['num_parts'] = self._num_parts
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        d['keypoints'] = self._keypoints
        d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._keypoints = d.get('keypoints')
        obj._weights = d['weights']
        return obj

    def __repr__(self):
        return ('PartsLayer(num_parts={num_parts}, part_shape={part_shape}, '
                'settings={settings})').format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
