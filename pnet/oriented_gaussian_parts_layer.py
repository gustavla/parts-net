from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
import pnet.matrix
from scipy import linalg
from sklearn.utils.extmath import logsumexp, pinvh
from amitgroup.plot import ImageGrid, ColorImageGrid


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
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(linalg.det(covars) + 1e-5)
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

    logprob = logsumexp(lpr.clip(min=-300), axis=1)
    #responsibilities = np.exp(lpr - logprob[:, np.newaxis])
    log_resp = lpr - logprob[:, np.newaxis]
    return logprob, log_resp


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
        self._n_orientations = n_orientations
        self._part_shape = part_shape
        self._settings = dict(polarities=1,
                              n_iter=10,
                              n_init=1,
                              seed=0,
                              standardize=True,
                              standardization_epsilon=0.001,
                              samples_per_image=40,
                              max_samples=np.inf,
                              max_covariance_samples=None,
                              logratio_thresh=-np.inf,
                              std_thresh=0.2,
                              std_thresh_frame=0,
                              covariance_type='tied',
                              min_covariance=1e-3,
                              uniform_weights=True,
                              channel_mode='together',
                              normalize_globally=False,
                              code_bkg=False,
                              whitening_epsilon=None,
                              min_count=5,
                              coding='hard',
                              )
        self._min_log_prob = np.log(0.0005)
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
        self._whitening_matrix = None

        self.w_epsilon = self._settings['whitening_epsilon']

    @property
    def num_parts(self):
        return self._num_true_parts * self._n_orientations

    @property
    def part_shape(self):
        return self._part_shape

    @property
    def visualized_parts(self):
        return self._visparts

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

        if 0:
            X0 = 0.0 * np.ones((1, np.prod(self._part_shape)))

            covar = self._prepare_covariance()

            K = self._means.shape[0]
            logprob, _ = score_samples(X0,
                                       self._means.reshape((K, -1)),
                                       self._weights.ravel(),
                                       covar,
                                       self.gmm_cov_type,
                                       )

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

    def __extract_helper(self, X, covar, img_stds):
        flatXij_patch = X.reshape((X.shape[0], -1))
        if self._settings['normalize_globally']:
            not_ok = (flatXij_patch.std(-1) / img_stds <= self._settings['std_thresh'])
        else:
            not_ok = (flatXij_patch.std(-1) <= self._settings['std_thresh'])

        if self._settings['standardize']:
            flatXij_patch = self._standardize_patches(flatXij_patch)

        if self._settings['whitening_epsilon'] is not None:
            flatXij_patch = self.whiten_patches(flatXij_patch)

        K = self._means.shape[0]
        logprob, log_resp = score_samples(flatXij_patch,
                                          self._means.reshape((K, -1)),
                                          self._weights.ravel(),
                                          covar,
                                          self.gmm_cov_type,
                                          )

        coding = self._settings['coding']
        if coding == 'hard':
            C = log_resp.argmax(-1)

            if self._settings['code_bkg']:
                bkg_part = self.num_parts
            else:
                bkg_part = -1

            C[not_ok] = bkg_part
            return C
        elif coding == 'triangle':
            #means = ag.apply_once(np.mean, resp, [1])
            dist = -log_resp
            means = ag.apply_once(np.mean, dist, [1])

            f = np.maximum(means - dist, 0)
            f[not_ok] = 0.0

            return f
        elif coding == 'soft':
            f = np.maximum(self._min_log_prob, log_resp)
            f[not_ok] = self._min_log_prob
            return f
        elif coding == 'raw':
            return logprob[:, np.newaxis] + log_resp

    def _extract(self, phi, data):
        assert self.trained, "Must be trained before calling extract"
        X = phi(data)

        channel_mode = self._settings['channel_mode']
        ps = self._part_shape

        if channel_mode == 'together':
            C = 1
        elif channel_mode == 'separate':
            C = X.shape[-1]
        dim = (X.shape[1]-ps[0]+1, X.shape[2]-ps[1]+1, C)


        EX_N = min(10, len(X))
        #ex_log_probs = np.zeros((EX_N,) + dim)
        #ex_log_probs2 = []

        covar = self._prepare_covariance()

        img_stds = ag.apply_once(np.std, X, [1, 2, 3], keepdims=False)
        #img_stds = None

        coding = self._settings['coding']
        if coding == 'hard':
            feature_map = -np.ones((X.shape[0],) + dim, dtype=np.int64)
        elif coding == 'triangle':
            feature_map = np.zeros((X.shape[0],) + dim + (self.num_parts,), dtype=np.float32)
        elif coding in ('soft', 'raw'):
            feature_map = np.empty((X.shape[0],) + dim + (self.num_parts,), dtype=np.float32)
            feature_map[:] = self._min_log_prob

        for i, j in itr.product(range(dim[0]), range(dim[1])):
            Xij_patch = X[:, i:i+ps[0], j:j+ps[1]]
            if channel_mode == 'together':
                feature_map[:, i, j, 0] = self.__extract_helper(Xij_patch, covar,
                                                                img_stds)
            elif channel_mode == 'separate':
                for c in range(C):
                    f = self.__extract_helper(Xij_patch[...,c], covar, img_stds)
                    assert f.dtype in [np.int64, np.int32]
                    f[f != -1] += c * self.num_parts
                    feature_map[:, i, j, c] = f

        #self.__TEMP_ex_log_probs = ex_log_probs
        #self.__TEMP_ex_log_probs2 = np.concatenate(ex_log_probs2)

        if coding == 'hard':
            num_features = self.num_parts * C
            if self._settings['code_bkg']:
                num_features += 1
            return (feature_map, num_features)
        else:
            return feature_map.reshape(feature_map.shape[:3] + (-1,))

    @property
    def trained(self):
        return self._means is not None

    # TODO: Needs work
    def _standardize_patches(self, flat_patches):
        means = np.apply_over_axes(np.mean, flat_patches, [1])
        variances = np.apply_over_axes(np.var, flat_patches, [1])

        epsilon = self._settings['standardization_epsilon']
        return (flat_patches - means) / np.sqrt(variances + epsilon)

    def whiten_patches(self, flat_patches):
        return np.dot(self._whitening_matrix, flat_patches.T).T

    def _train(self, phi, data, y=None):
        X = phi(data)
        raw_originals, the_rest = self._get_patches(X)
        self._train_info = {}
        self._train_info['example_patches2'] = raw_originals[:10]

        # Standardize them
        old_raw_originals = raw_originals.copy()
        if self._settings['standardize']:
            mu = ag.apply_once(np.mean, raw_originals, [1, 2, 3, 4])
            variances = ag.apply_once(np.var, raw_originals, [1, 2, 3, 4])
            epsilon = self._settings['standardization_epsilon']
            raw_originals = (raw_originals - mu) / np.sqrt(variances + epsilon)

        pp = raw_originals.reshape((np.prod(raw_originals.shape[:2]), -1))
        sigma = np.dot(pp.T, pp) / len(pp)
        self._extra['sigma'] = sigma
        if self.w_epsilon is not None:
            U, S, _ = np.linalg.svd(sigma)

            shrinker = np.diag(1 / np.sqrt(S + self.w_epsilon))

            #self._whitening_matrix = U @ shrinker @ U.T
            self._whitening_matrix = np.dot(U, np.dot(shrinker, U.T))
        else:
            self._whitening_matrix = np.eye(sigma.shape[0])

        pp = self.whiten_patches(pp)
        raw_originals = pp.reshape(raw_originals.shape)

        self.train_from_samples(raw_originals, the_rest)

        # TODO
        if 0:
            f = self.extract(lambda x: x, old_raw_originals[:,0])
            feat = f[0].ravel()
            ag.info('bincounts', np.bincount(feat[feat!=-1], minlength=f[1]))

        self.preprocess()

    @property
    def gmm_cov_type(self):
        """The covariance type that should be used for sklearn's GMM"""
        c = self._settings['covariance_type']
        if c in ['full-full', 'full-perm']:
            return 'full'
        elif c in ['diag-perm', 'ones']:
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
                indices = np.round(ori_index).astype(np.int64)
                theta = (orientations - indices) % orientations

                # This is not the most robust way, but it works
                counts = np.bincount(theta.ravel(), minlength=8)
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
                            thresh=1e-5,
                            covariance_type=self._settings['covariance_type'],
                            covar_limit=covar_limit,
                            min_covar=self._settings['min_covariance'],
                            params='wmc',
                            )

        Xflat = raw_originals.reshape(raw_originals.shape[:2] + (-1,))
        mm.fit(Xflat)

        comps = mm.predict(Xflat)

        # Floating counts
        from scipy.misc import logsumexp
        logprob, log_resp = mm.score_block_samples(Xflat)
        fcounts = np.exp(logsumexp(log_resp[...,0], axis=0))

        def mean0(x):
            if x.shape[0] == 0:
                return np.zeros(x.shape[1:])
            else:
                return np.mean(x, axis=0)

        visparts = np.asarray([
            mean0(raw_originals[comps[:, 0] == k,
                                comps[comps[:, 0] == k][:, 1]])
            for k in range(self._num_true_parts)
        ])

        if 1:
            EX_N = 50
            ex_shape = ((self._num_true_parts, EX_N) + self._part_shape +
                        raw_originals.shape[4:])
            ex_patches = np.empty(ex_shape)
            ex_patches[:] = np.nan

            for k in range(self._num_true_parts):
                XX = raw_originals[comps[:, 0] == k]
                rot = comps[comps[:, 0] == k, 1]
                for n in range(min(EX_N, len(XX))):
                    ex_patches[k, n] = XX[n, rot[n]]


        counts = np.bincount(comps[:, 0])
        II = np.argsort(counts)[::-1]

        ok = counts >= self._settings['min_count']
        II = np.asarray([ii for ii in II if ok[ii]])

        counts = counts[II]
        fcounts = fcounts[II]
        means = mm.means_[II]
        weights = mm.weights_[II]
        self._visparts = visparts[II]
        self._train_info['example_patches'] = ex_patches[II]


        np.seterr(all='raise')

        n_components = len(II)
        self._num_true_parts = len(II)

        print('Kept', n_components, 'out of', mm.n_components)

        # Covariance types that will need resorting
        covtypes = ['diag', 'diag-perm', 'full', 'full-full']
        if self._settings['covariance_type'] in covtypes:
            covars = mm.covars_[II]
        else:
            covars = mm.covars_

        ag.info(':counts', list(zip(counts, fcounts)))

        # Example patches - initialize to NaN if component doesn't fill it up
        # Rotate the parts into the canonical rotation
        #if POL == 1 and False:
            #self.rotate_indices(mm, ORI)

        means_shape = (n_components * P,) + raw_originals.shape[2:]
        ag.info('means_shape', means_shape)
        self._means = means.reshape(means_shape)
        self._covar = covars
        if self._settings['uniform_weights']:
            # Set weights to uniform
            self._weights = np.ones(weights.shape)
            self._weights /= np.sum(self._weights)
        else:
            self._weights = weights

        self._train_info['counts'] = counts

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

        channel_mode = self._settings['channel_mode']

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

            img_padded = ag.util.pad_to_size(img, (new_size[0], new_size[1],) + X.shape[3:])
            pad = [(new_size[i]-size[i])//2 for i in range(2)]

            angles = np.arange(0, 360, 360 / ORI)
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

            img_std = np.std(img_padded)

            ag.info('Image #{}, collected {} patches and rejected {} (std={}'.format(
                n, len(the_originals), len(the_rest), img_std))

            for sample in range(samples_per_image):
                TRIES = 10000
                for tries in range(TRIES):
                    x, y = next(i_iter)

                    fr = self._settings['std_thresh_frame']
                    sel0_inner = [0,
                                  slice(x+minus_ps[0]+fr, x+plus_ps[0]-fr),
                                  slice(y+minus_ps[1]+fr, y+plus_ps[1]-fr)]
                    if channel_mode == 'separate':
                        ii = rs.randint(X.shape[3])
                        sel0_inner += [ii]

                    from copy import copy
                    sel1_inner = copy(sel0_inner)
                    sel1_inner[0] = slice(None)

                    XY = np.array([x, y, 1])[:, np.newaxis]

                    # Now, let's explore all orientations
                    if channel_mode == 'together':
                        vispatch = np.zeros((ORI * POL,) + ps + X.shape[3:])
                    elif channel_mode == 'separate':
                        vispatch = np.zeros((ORI * POL,) + ps + (1,))

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

                        if channel_mode == 'separate':
                            selection += [slice(ii, ii+1)]

                        orig = all_img[selection]
                        try:
                            vispatch[ori] = orig
                        except ValueError:
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

                    #if all_img[sel0_inner].std() > std_thresh:
                    all_stds = ag.apply_once(np.std,
                                             all_img[sel1_inner],
                                             [1, 2],
                                             keepdims=False)
                    #if np.median(all_stds) > std_thresh:
                    #if np.median(all_stds) > std_thresh:

                    if self._settings['normalize_globally']:
                        ok = np.median(all_stds) / img_std > std_thresh
                    else:
                        ok = np.median(all_stds) > std_thresh

                    if ok:
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
            grid = ColorImageGrid(self._train_info['example_patches2'],
                                  border_color=(1, 1, 0))
            grid.save(vz.impath(), scale=3)

        vz.log('Loglikelihood threshold:', self._extra.get('loglike_thresh'))

        vz.log('Part shape:', self._means.shape)

        if self._train_info is not None:
            if 'example_patches' in self._train_info:
                XX = self._train_info['example_patches']
                vz.log(XX.shape)
                concats = [self._means[::self._n_orientations, np.newaxis], XX]
                means_and_exs = np.concatenate(concats, axis=1)
                grid = ColorImageGrid(means_and_exs, border_color=0.8, vmin=None, vmax=None, vsym=True)
                grid.highlight(col=0, color=(1.0, 1.0, 0.5))
                grid.save(vz.impath(), scale=4)

            vz.log('Training counts', self._train_info.get('counts'))

            cc = self._train_info['counts']
            vz.log('A', cc / cc.sum())

            #covs = [np.cov(X.reshape((X.shape[0], -1)).T) for X in XX]
            #grid = ImageGrid(covs, cmap=cm.RdBu_r, vsym=True)
            #grid.save(vz.impath(), scale=4)

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
        vz.log(self._weights.ravel())

        # The canonical parts
        vz.text('Canonical parts')
        vz.log('span', ag.span(self._means))
        grid = ColorImageGrid(self._means[::self._n_orientations], vsym=True)
        grid.save(vz.impath(), scale=4)

        # Parts (means)
        if self._n_orientations > 1:
            vz.text('Parts')
            grid = ColorImageGrid(self._means, cols=self._n_orientations,
                                  vmin=None, vmax=None, vsym=True)

            grid.save(vz.impath(), scale=4)

        # Covariance matrix
        #vz.text('Covariance matrix')
        #grid = ImageGrid(self._covar, vsym=True,
                         #border_color=0.5, cmap=cm.RdBu_r)
        #grid.save(vz.impath(), scale=5)

        def diff_entropy(cov):
            return 1/2 * np.linalg.slogdet(2 * np.pi * np.e * cov)[1]

        # Variance
        if self._settings['covariance_type'] == 'diag':
            vz.text('Variance')
            sh = self._covar.shape[:-1] + self._part_shape
            variances = self._covar.reshape(sh)
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

        elif self._settings['covariance_type'] == 'full-perm':
            vz.text('Variance')
            grid = ImageGrid(self._covar, vsym=True, rows=1,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

        elif self._covar.ndim == 2:
            vz.text('Variance')
            try:
                var = np.diag(self._covar).reshape(self._part_shape)
                grid = ImageGrid(var, vsym=True,
                                 border_color=0.5, cmap=cm.RdBu_r)
                grid.save(vz.impath(), scale=5)
                vz.log('Average variance:', np.mean(var))

                vz.text('Covariance')
                grid = ImageGrid(self._covar, vsym=True, cmap=cm.RdBu_r)
                grid.save(vz.impath(), scale=5)
            except:
                pass

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

        elif self._settings['covariance_type'] == 'full-full':
            vz.text('Covariances')
            grid = ImageGrid(self._covar, vsym=True,
                             border_color=0.5, cmap=cm.RdBu_r)
            grid.save(vz.impath(), scale=5)

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
        if 'bkg_mean' in self._extra and False:
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
        d['whitening_matrix'] = self._whitening_matrix

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
        obj._whitening_matrix = d['whitening_matrix']

        #
        obj.preprocess()
        return obj

    def __repr__(self):
        return ('OrientedPartsLayer(num_true_parts={num_parts}, '
                'n_orientations={n_orientations}, '
                'part_shape={part_shape}, '
                'settings={settings})'
               ).format(num_parts=self._num_true_parts,
                        n_orientations=self._n_orientations,
                        part_shape=self._part_shape,
                        settings=self._settings)
