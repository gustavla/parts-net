from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
import pnet.matrix
from scipy import linalg
from sklearn.utils.extmath import logsumexp, pinvh
from amitgroup.plot import ImageGrid


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model"""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model"""
    n_samples, n_dim = X.shape
    icv = pinvh(covars)
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(linalg.det(covars))
                  + np.sum(X * np.dot(X, icv), 1)[:, np.newaxis]
                  - 2 * np.dot(np.dot(X, icv), means.T)
                  + np.sum(means * np.dot(means, icv), 1))
    return lpr


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices.
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))

    assert len(means) == len(covars)

    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def score_samples(X, means, weights, covars, covariance_type='diag'):
    lpr = (log_multivariate_normal_density(X, means, covars,
                                           covariance_type)
           + np.log(weights))
    logprob = logsumexp(lpr, axis=1)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])
    return logprob, responsibilities


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)              if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in X under
        each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)


@Layer.register('oriented-gaussian-parts-layer')
class OrientedGaussianPartsLayer(Layer):
    def __init__(self, n_parts=1, n_orientations=1, part_shape=(6, 6),
                 settings={}):
        self._num_true_parts = n_parts
        self._num_parts = n_parts * n_orientations
        self._n_orientations = n_orientations
        self._part_shape = part_shape
        self._settings = dict(polarities=1,
                              n_iter=10,
                              n_init=1,
                              seed=0,
                              standardize=True,
                              samples_per_image=40,
                              max_samples=np.inf,
                              max_covariance_samples=None,
                              logratio_thresh=-np.inf,
                              std_thresh=0.2,
                              std_thresh_frame=0,
                              covariance_type='tied',
                              min_covariance=1e-3,
                              uniform_weights=True,
                              )
        self._extra = {}

        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v

        self._train_info = None
        self._keypoints = None

        self._means = None
        self._covar = None
        self._weights = None

        self._visparts = None

    @property
    def num_parts(self):
        return self._num_parts

    @property
    def part_shape(self):
        return self._part_shape

    def _prepare_covariance(self):
        cov_type = self._settings['covariance_type']

        if cov_type == 'tied':
            covar = self._covar
        elif cov_type == 'full-perm':
            covar = np.concatenate([
                np.tile(cov, (self._num_true_parts, 1, 1))
                for cov in self._covar
            ])
        elif cov_type == 'diag':
            covar = self._covar.reshape((-1,) + self._covar.shape[-1:])
        elif cov_type == 'diag-perm':
            covar = np.concatenate([
                np.tile(cov, (self._num_true_parts, 1))
                for cov in self._covar
            ])
        else:
            covar = self._covar.reshape((-1,) + self._covar.shape[-2:])

        return covar

    def preprocess(self):
        self._extra['loglike_thresh'] = -np.inf

        X0 = 0.0 * np.ones((1, np.prod(self._part_shape)))

        covar = self._prepare_covariance()

        K = self._means.shape[0]
        logprob, _ = score_samples(X0,
                                   self._means.reshape((K, -1)),
                                   self._weights.ravel(),
                                   covar,
                                   self.gmm_cov_type,
                                   )

        if 0:
            bkg_means = self._extra['bkg_mean'].ravel()[np.newaxis]
            bkg_covars = self._extra['bkg_covar'][np.newaxis]
            bkg_weights = np.ones(1)

            bkg_logprob, _ = score_samples(X0,
                                           bkg_means,
                                           bkg_weights,
                                           bkg_covars,
                                           'full',
                                           )

    @property
    def pos_matrix(self):
        return self.conv_pos_matrix(self._part_shape)

    def extract(self, phi, data):
        assert self.trained, "Must be trained before calling extract"
        X = phi(data)

        ps = self._part_shape
        dim = (X.shape[1]-ps[0]+1, X.shape[2]-ps[1]+1)

        feature_map = -np.ones((X.shape[0],) + dim, dtype=np.int64)

        EX_N = min(10, len(X))
        ex_log_probs = np.zeros((EX_N,) + dim)
        ex_log_probs2 = []

        covar = self._prepare_covariance()

        for i, j in itr.product(range(dim[0]), range(dim[1])):
            Xij_patch = X[:, i:i+ps[0], j:j+ps[1]]
            flatXij_patch = Xij_patch.reshape((X.shape[0], -1))

            if self._settings['standardize']:
                flatXij_patch = self._standardize_patches(flatXij_patch)

            K = self._means.shape[0]
            logprob, resp = score_samples(flatXij_patch,
                                          self._means.reshape((K, -1)),
                                          self._weights.ravel(),
                                          covar,
                                          self.gmm_cov_type,
                                          )

            bkg_means = self._extra['bkg_mean'].ravel()[np.newaxis]
            bkg_covars = self._extra['bkg_covar'][np.newaxis]
            bkg_weights = np.ones(1)

            bkg_logprob, _ = score_samples(flatXij_patch,
                                           bkg_means,
                                           bkg_weights,
                                           bkg_covars,
                                           'full',
                                           )

            logratio = logprob - bkg_logprob

            not_ok = (logratio < self._settings['logratio_thresh'])
            C = resp.argmax(-1)

            ex_log_probs[:, i, j] = logprob[:EX_N] - bkg_logprob[:EX_N]
            feature_map[:, i, j] = C

            not_ok2 = (flatXij_patch.std(-1) <= self._settings['std_thresh'])
            not_ok |= not_ok2

            feature_map[not_ok, i, j] = -1

            ex_log_probs2.append(logratio[~not_ok2])

        self.__TEMP_ex_log_probs = ex_log_probs
        self.__TEMP_ex_log_probs2 = np.concatenate(ex_log_probs2)

        return (feature_map[..., np.newaxis], self._num_parts)

    @property
    def trained(self):
        return self._means is not None

    # TODO: Needs work
    def _standardize_patches(self, flat_patches):
        means = np.apply_over_axes(np.mean, flat_patches, [1])
        stds = np.apply_over_axes(np.std, flat_patches, [1])

        epsilon = 0.025 / 2

        return (flat_patches - means) / (stds + epsilon)

    def train(self, phi, data, y=None):
        X = phi(data)
        raw_originals, the_rest = self._get_patches(X)
        self._train_info = {}
        self._train_info['example_patches2'] = raw_originals[:10]

        # Standardize them
        # TODO
        if self._settings['standardize']:
            mu = ag.apply_once_over_axes(np.mean, raw_originals, [1, 2, 3])
            sigma = ag.apply_once_over_axes(np.std, raw_originals, [1, 2, 3])
            epsilon = 0.025 / 2
            raw_originals = (raw_originals - mu) / (sigma + epsilon)

        self.train_from_samples(raw_originals, the_rest)
        self.preprocess()

    @property
    def gmm_cov_type(self):
        """The covariance type that should be used for sklearn's GMM"""
        c = self._settings['covariance_type']
        if c in ['full-full', 'full-perm']:
            return 'full'
        elif c == 'diag-perm':
            return 'diag'
        else:
            return c

    def rotate_indices(self, mm, ORI):
        """
        """
        import scipy.signal
        kern = np.array([[-1, 0, 1]]) / np.sqrt(2)
        orientations = 8

        for p in range(self._num_true_parts):
            main_rot = None
            for i, part_ in enumerate(mm.means_[p]):
                part = part_.reshape(self._part_shape)

                gr_x = scipy.signal.convolve(part, kern, mode='same')
                gr_y = scipy.signal.convolve(part, kern.T, mode='same')

                a = np.arctan2(gr_y[1:-1, 1:-1], gr_x[1:-1, 1:-1])
                ori_index = orientations * (a + 1.5*np.pi) / (2 * np.pi)
                indices = np.round(ori_index).astype(np.int32)
                theta = (orientations - indices) % orientations

                # This is not the most robust way, but it works
                counts = np.bincount(theta.ravel(), minlength=8)
                print(':', i, counts)
                if counts.argmax() == 0:
                    main_rot = i
                    break

            II = np.roll(np.arange(ORI), -main_rot)

            mm.means_[p, :] = mm.means_[p, II]
            mm.covars_[p, :] = mm.covars_[p, II]
            mm.weights_[p, :] = mm.weights_[p, II]

    def train_from_samples(self, raw_originals, the_rest):
        ORI = self._n_orientations
        POL = self._settings['polarities']
        P = ORI * POL

        def cycles(X):
            c = [np.concatenate([X[i:], X[:i]]) for i in range(len(X))]
            return np.asarray(c)

        RR = np.arange(ORI)
        PP = np.arange(POL)
        II = [list(itr.product(PPi, RRi))
              for PPi in cycles(PP)
              for RRi in cycles(RR)]
        lookup = dict(zip(itr.product(PP, RR), itr.count()))
        n_init = self._settings['n_init']
        n_iter = self._settings['n_iter']
        seed = self._settings['seed']
        covar_limit = self._settings['max_covariance_samples']

        num_angle = ORI
        d = np.prod(raw_originals.shape[2:])
        permutation = np.empty((num_angle, num_angle * d), dtype=np.int_)
        for a in range(num_angle):
            if a == 2:
                permutation[a] = np.arange(num_angle * d)
            else:
                permutation[a] = np.roll(permutation[a-1], d)

        permutations = np.asarray([[lookup[ii] for ii in rows] for rows in II])

        from pnet.permutation_gmm import PermutationGMM
        mm = PermutationGMM(n_components=self._num_true_parts,
                            permutations=permutations,
                            n_iter=n_iter,
                            n_init=n_init,
                            random_state=seed,
                            reg_covar=0.0,
                            covariance_type=self._settings['covariance_type'],
                            covar_limit=covar_limit,
                            min_covar=self._settings['min_covariance'],
                            )

        Xflat = raw_originals.reshape(raw_originals.shape[:2] + (-1,))
        mm.fit(Xflat)

        comps = mm.predict(Xflat)
        ag.info('Samples:', np.bincount(comps[:, 0]))

        # Example patches - initialize to NaN if component doesn't fill it up
        EX_N = 50
        ex_shape = (self._num_true_parts, EX_N) + self._part_shape
        ex_patches = np.empty(ex_shape)
        ex_patches[:] = np.nan

        for k in range(self._num_true_parts):
            XX = raw_originals[comps[:, 0] == k]
            rot = comps[comps[:, 0] == k, 1]
            for n in range(min(EX_N, len(XX))):
                ex_patches[k, n] = XX[n, rot[n]]

        self._train_info['example_patches'] = ex_patches

        # Rotate the parts into the canonical rotation
        if POL == 1 and False:
            self.rotate_indices(mm, ORI)

        means_shape = (mm.n_components * P,) + raw_originals.shape[2:]
        self._means = mm.means_.reshape(means_shape)
        self._covar = mm.covars_
        if self._settings['uniform_weights']:
            # Set weights to uniform
            self._weights = np.ones(mm.weights_.shape)
            self._weights /= np.sum(self._weights)
        else:
            self._weights = mm.weights_

        co = np.bincount(comps[:, 0], minlength=self._num_true_parts)
        self._train_info['counts'] = co

        ag.info('Training counts', self._train_info['counts'])

        self._visparts = np.asarray([
            raw_originals[comps[:, 0] == k,
                          comps[comps[:, 0] == k][:, 1]].mean(0)
            for k in range(self._num_true_parts)
        ])

        def collapse12(Z):
            return Z.reshape((-1,) + Z.shape[2:])

        if len(the_rest) > 0:
            XX = np.concatenate([collapse12(the_rest),
                                 collapse12(raw_originals)])
        else:
            XX = collapse12(raw_originals)

        def regularize_covar(cov, min_covar=0.001):
            dd = np.diag(cov)
            return cov + np.diag(dd.clip(min=min_covar) - dd)

        self._extra['bkg_mean'] = XX.mean(0)
        sample_cov = np.cov(XX.reshape((XX.shape[0], -1)).T)
        self._extra['bkg_covar'] = np.diag(np.diag(sample_cov))
        min_cov = self._settings['min_covariance']
        self._extra['bkg_covar'] = regularize_covar(self._extra['bkg_covar'],
                                                    min_cov)

    def _get_patches(self, X):
        samples_per_image = self._settings['samples_per_image']
        the_originals = []
        the_rest = []
        ag.info("Extracting patches from")
        ps = self._part_shape

        ORI = self._n_orientations
        POL = self._settings['polarities']
        assert POL in (1, 2), "Polarities must be 1 or 2"

        # LEAVE-BEHIND
        # Make it square, to accommodate all types of rotations
        size = X.shape[1:3]
        new_side = np.max(size)

        new_size = [new_side + (new_side - X.shape[1]) % 2,
                    new_side + (new_side - X.shape[2]) % 2]

        from skimage import transform

        for n, img in enumerate(X):

            img_padded = ag.util.pad_to_size(img, (new_size[0], new_size[1]))
            pad = [(new_size[i]-size[i])//2 for i in range(2)]

            angles = np.arange(0, 360, 360/ORI)
            radians = angles*np.pi/180
            all_img = np.asarray([
                transform.rotate(img_padded,
                                 angle,
                                 resize=False,
                                 mode='nearest')
                for angle in angles])

            # Add inverted polarity too
            if POL == 2:
                all_img = np.concatenate([all_img, 1-all_img])

            rs = np.random.RandomState(0)

            # Set up matrices that will translate a position in the canonical
            # image to the rotated iamges. This way, we're not rotating each
            # patch on demand, which will end up slower.

            center_adjusts = [ps[0] % 2,
                              ps[1] % 2]

            offset = (np.asarray(new_size) - center_adjusts) / 2
            matrices = [pnet.matrix.translation(offset[0], offset[1]) *
                        pnet.matrix.rotation(a) *
                        pnet.matrix.translation(-offset[0], -offset[1])
                        for a in radians]

            # Add matrices for the polarity flips too, if applicable
            matrices *= POL

            # This avoids hitting the outside of patches, even after rotating.
            # The 15 here is fairly arbitrary
            avoid_edge = int(1 + np.max(ps)*np.sqrt(2))

            # These indices represent the center of patches
            range_x = range(pad[0]+avoid_edge, pad[0]+img.shape[0]-avoid_edge)
            range_y = range(pad[1]+avoid_edge, pad[1]+img.shape[1]-avoid_edge)

            indices = list(itr.product(range_x, range_y))

            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            minus_ps = [-(ps[i]//2) for i in range(2)]
            plus_ps = [minus_ps[i] + ps[i] for i in range(2)]

            max_samples = self._settings['max_samples']
            consecutive_failures = 0

            # We want rotation of 90 deg to have exactly the same pixels. For
            # this, we need adjust the center of the patch a bit before
            # rotating.

            std_thresh = self._settings['std_thresh']

            for sample in range(samples_per_image):
                TRIES = 200
                for tries in range(TRIES):
                    x, y = next(i_iter)

                    fr = self._settings['std_thresh_frame']
                    sel0_inner = [0,
                                  slice(x+minus_ps[0]+fr, x+plus_ps[0]-fr),
                                  slice(y+minus_ps[1]+fr, y+plus_ps[1]-fr)]

                    XY = np.array([x, y, 1])[:, np.newaxis]

                    # Now, let's explore all orientations
                    vispatch = np.zeros((ORI * POL,) + ps)

                    br = False
                    for ori in range(ORI * POL):
                        p = np.dot(matrices[ori], XY)

                        # The epsilon makes the truncation safer
                        ip = [int(round(float(p[i]))) for i in range(2)]

                        selection = [ori,
                                     slice(ip[0] + minus_ps[0],
                                           ip[0] + plus_ps[0]),
                                     slice(ip[1] + minus_ps[1],
                                           ip[1] + plus_ps[1])]

                        orig = all_img[selection]
                        try:
                            vispatch[ori] = orig
                        except:
                            br = True
                            break

                    if br:
                        continue

                    # Randomly rotate this patch, so that we don't bias
                    # the unrotated (and possibly unblurred) image
                    shift = rs.randint(ORI)

                    vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)

                    if POL == 2:
                        vispatch[ORI:] = np.roll(vispatch[ORI:], shift, axis=0)

                    if all_img[sel0_inner].std() > std_thresh:
                        the_originals.append(vispatch)
                        if len(the_originals) % 500 == 0:
                            ag.info('Samples {}/{}'.format(len(the_originals),
                                                           max_samples))

                        if len(the_originals) >= max_samples:
                            return (np.asarray(the_originals),
                                    np.asarray(the_rest))

                        consecutive_failures = 0
                        break
                    else:
                        the_rest.append(vispatch)

                    if tries == TRIES-1:
                        ag.info('WARNING: {} tries'.format(TRIES))
                        ag.info('cons', consecutive_failures)
                        consecutive_failures += 1

                    if consecutive_failures >= 10:
                        # Just give up.
                        raise ValueError('FATAL ERROR: Threshold is '
                                         'probably too high (in {})'
                                         .format(self.__class__.__name__))

        return np.asarray(the_originals), np.asarray(the_rest)

    def _vzlog_output_(self, vz):
        from pylab import cm

        if 'example_patches2' in self._train_info:
            vz.text('Training samples')
            grid = ImageGrid(self._train_info['example_patches2'],
                             vmin=0, vmax=1,
                             border_color=(1, 1, 0))
            grid.save(vz.impath(), scale=3)

        vz.log('Loglikelihood threshold:', self._extra.get('loglike_thresh'))

        if self._train_info is not None:
            XX = self._train_info['example_patches']
            vz.log(XX.shape)
            concats = [self._means[::self._n_orientations, np.newaxis], XX]
            means_and_exs = np.concatenate(concats, axis=1)
            grid = ImageGrid(means_and_exs, border_color=0.8)
            grid.highlight(col=0, color=(1.0, 1.0, 0.5))
            grid.save(vz.impath(), scale=4)

            vz.log('Training counts', self._train_info.get('counts'))

            cc = self._train_info['counts']
            vz.log('A', cc / cc.sum())

            covs = [np.cov(X.reshape((X.shape[0], -1)).T) for X in XX]
            grid = ImageGrid(covs, cmap=cm.RdBu_r, vsym=True)
            grid.save(vz.impath(), scale=4)

        if 0:
            log_probs = self.__TEMP_ex_log_probs
            grid = ImageGrid(log_probs, cmap=cm.jet)
            grid.save(vz.impath(), scale=2)

            import pylab as plt
            plt.figure()
            plt.hist(log_probs.ravel(), 100)
            plt.title('Log ratio, all')
            plt.savefig(vz.impath('svg'))

        if 0:
            log_probs2 = self.__TEMP_ex_log_probs2
            import pylab as plt
            plt.figure()
            plt.hist(log_probs2.ravel(), 100)
            plt.title('Log ratio, above std thresh')
            plt.savefig(vz.impath('svg'))

        # Weights
        vz.log('Parts weights: ')
        vz.log(self._weights)

        # The canonical parts
        vz.text('Canonical parts')
        grid = ImageGrid(self._means[::self._n_orientations])
        grid.save(vz.impath(), scale=10)

        # Parts (means)
        if self._n_orientations > 1:
            vz.text('Parts')
            grid = ImageGrid(self._means,
                             cols=self._n_orientations,
                             )

            grid.save(vz.impath(), scale=10)

        # Covariance matrix
        vz.text('Covariance matrix')
        grid = ImageGrid(self._covar, vsym=True,
                         border_color=0.5, cmap=cm.RdBu_r)
        grid.save(vz.impath(), scale=5)

        def diff_entropy(cov):
            return 1/2 * np.linalg.slogdet(2 * np.pi * np.e * cov)[1]

        # Variance
        if self._settings['covariance_type'] == 'diag':
            vz.text('Variance')
            variances = np.asarray([
                [
                    self._covar[k, p].reshape(self._part_shape)
                    for p in range(self._n_orientations)
                ] for k in range(self._num_true_parts)
            ])
            grid = ImageGrid(variances, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

        elif self._settings['covariance_type'] == 'diag-perm':
            vz.text('Variance')
            variances = np.asarray([
                self._covar[p].reshape(self._part_shape)
                for p in range(self._n_orientations)
            ])
            grid = ImageGrid(variances, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

        elif self._covar.ndim == 2:
            vz.text('Variance')
            var = np.diag(self._covar).reshape(self._part_shape)
            grid = ImageGrid(var, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)
            vz.log('Average variance:', np.mean(var))

        elif self._settings['covariance_type'] == 'full':
            vz.text('Variance')
            variances = np.asarray([
                np.diag(self._covar[k]).reshape(self._part_shape)
                for k in range(self._num_true_parts)
            ])
            grid = ImageGrid(variances, rows=1, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

            vz.log('Average variance:', np.mean(variances))

            Hs = [
                diff_entropy(cov) for cov in self._covar
            ]

            vz.text('Differential entropies:')
            vz.log(Hs)

        elif self._covar.ndim == 4:
            vz.text('Variance')
            variances = np.asarray([
                [
                    np.diag(self._covar[k, p]).reshape(self._part_shape)
                    for p in range(self._n_orientations)
                ] for k in range(self._num_true_parts)
            ])
            grid = ImageGrid(variances, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

            vz.log('Average variance:', np.mean(variances))

            vz.section('Entropy')
            Hs = np.asarray([
                [
                    diff_entropy(self._covar[k, p])
                    for p in range(self._n_orientations)
                ] for k in range(self._num_true_parts)
            ])
            vz.log('Differential entropies:', Hs)
            grid1 = ImageGrid(Hs, cmap=cm.cool, border_width=0)
            grid1.save(vz.impath(), scale=10)
            vz.log('Min:', Hs.min(), 'Max:', Hs.max())

        # TODO: Temp
        if 'bkg_mean' in self._extra:
            vz.section('Background model')
            bkg_mean = self._extra['bkg_mean']
            bkg_covar = self._extra['bkg_covar']

            grid = ImageGrid(bkg_mean, vmin=0, vmax=1)
            grid.save(vz.impath(), scale=10)

            grid = ImageGrid(bkg_covar, cmap=cm.RdBu_r, vsym=1)
            grid.save(vz.impath(), scale=5)

            vz.log('Average intensity:', self._extra['bkg_mean'].mean())
            vz.log('Variances:')
            vz.log(np.diag(bkg_covar))
            vz.log('S.D.s:')
            vz.log(np.sqrt(np.diag(bkg_covar)))
            vz.log('Entropy:', diff_entropy(bkg_covar))

    def save_to_dict(self):
        d = {}
        d['num_true_parts'] = self._num_true_parts
        d['num_orientations'] = self._n_orientations
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['train_info'] = self._train_info
        d['extra'] = self._extra

        d['means'] = self._means
        d['covar'] = self._covar
        d['weights'] = self._weights
        d['visparts'] = self._visparts
        return d

    @classmethod
    def load_from_dict(cls, d):
        # This code is just temporary {
        num_true_parts = d.get('num_true_parts')
        if num_true_parts is None:
            num_true_parts = int(d['num_parts'] // d['num_orientations'])
        # }
        obj = cls(num_true_parts,
                  d['num_orientations'],
                  d['part_shape'],
                  settings=d['settings'])
        obj._means = d['means']
        obj._covar = d['covar']
        obj._weights = d.get('weights')
        obj._visparts = d.get('visparts')
        obj._train_info = d.get('train_info')
        obj._extra = d.get('extra', {})

        #
        obj.preprocess()
        return obj

    def __repr__(self):
        return 'OrientedPartsLayer(num_parts={num_parts}, '\
               'part_shape={part_shape}, '\
               'settings={settings})'.format(num_parts=self._num_parts,
                                             part_shape=self._part_shape,
                                             settings=self._settings)
