from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
import pnet.matrix
from amitgroup.plot import ImageGrid

if 0:
    def _get_patches(self, X):
        assert X.ndim == 4

        samples_per_image = self._settings.get('samples_per_image', 20)
        fr = self._settings.get('outer_frame', 0)
        patches = []

        pseed = self._settings.get('patch_extraction_seed', 0)
        rs = np.random.RandomState(pseed)

        th = self._settings['threshold']
        max_th = self._settings.get('max_threshold', np.inf)
        support_mask = self._settings.get('support_mask')

        consecutive_failures = 0

        for Xi in X:

            # How many patches could we extract?
            w, h = [Xi.shape[i]-self._part_shape[i]+1 for i in range(2)]

            # TODO: Maybe shuffle an iterator of the indices?
            indices = list(itr.product(range(w-1), range(h-1)))
            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            for sample in range(samples_per_image):
                N = 200
                for tries in range(N):
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
                        max_samples = self._settings.get('max_samples', np.inf)
                        if len(patches) >= max_samples:
                            return np.asarray(patches)
                        consecutive_failures = 0
                        break

                    if tries == N-1:
                        ag.info('WARNING: {} tries'.format(N))
                        ag.info('cons', consecutive_failures)
                        consecutive_failures += 1

                    if consecutive_failures >= 10:
                        # Just give up.
                        raise ValueError("FATAL ERROR: "
                                         "Threshold is probably too high.")

        return np.asarray(patches)


def _extract_many_edges(bedges_settings, settings, images,
                        must_preserve_size=False):
    """Extract edges of many images (must be the same size)"""
    sett = bedges_settings.copy()
    sett['radius'] = 0
    sett['preserve_size'] = False or must_preserve_size

    edge_type = settings.get('edge_type', 'yali')
    if edge_type == 'yali':
        X = ag.features.bedges(images, **sett)
        return X
    else:
        raise RuntimeError("No such edge type")


def _extract_batch(X, settings, num_parts, num_orientations, part_shape,
                   parts):
    assert parts is not None, "Must be trained before calling extract"

    th = settings['threshold']
    n_coded = settings.get('n_coded', 1)

    support_mask = np.ones(part_shape, dtype=np.bool)

    from pnet.cyfuncs import code_index_map_general

    standardize = settings.get('standardize', 0)
    min_percentile = settings.get('min_percentile', 0.0)
    feature_map = code_index_map_general(X,
                                         parts,
                                         support_mask.astype(np.uint8),
                                         th,
                                         outer_frame=settings['outer_frame'],
                                         n_coded=n_coded,
                                         standardize=standardize,
                                         min_percentile=min_percentile)

    # Rotation spreading?
    rotspread = settings.get('rotation_spreading_radius', 0)
    if rotspread > 0:
        between_feature_spreading = np.zeros((num_parts, rotspread*2 + 1),
                                             dtype=np.int64)
        ORI = num_orientations

        for f in range(num_parts):
            thepart = f // ORI
            ori = f % ORI
            for i in range(rotspread*2 + 1):
                v = thepart * ORI + (ori - rotspread + i) % ORI
                between_feature_spreading[f, i] = v

        neg_ones = -np.ones((1, rotspread*2 + 1), dtype=np.int64)
        bb = np.concatenate([between_feature_spreading, neg_ones], 0)

        # Update feature map
        feature_map = bb[feature_map[..., 0]]

    return feature_map


@Layer.register('oriented-parts-layer')
class OrientedPartsLayer(Layer):
    def __init__(self, n_parts=1, n_orientations=1, part_shape=(6, 6),
                 settings={}):
        self._num_true_parts = n_parts
        self._num_parts = n_parts * n_orientations
        self._num_orientations = n_orientations
        self._part_shape = part_shape
        self._settings = dict(outer_frame=0,
                              n_init=1,
                              n_iter=8,
                              threshold=0.0,
                              samples_per_image=40,
                              max_samples=np.inf,
                              seed=0,
                              min_prob=0.005,
                              )

        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v

        self._train_info = {}
        self._keypoints = None

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
        im = phi(data)
        b = int(im.shape[0] // 100)
        sett = (self._settings,
                self.num_parts,
                self._num_orientations,
                self._part_shape,
                self._parts)
        if b == 0:

            feat = _extract_batch(im, *sett)
        else:
            im_batches = np.array_split(im, b)

            args = ((im_b,) + sett for im_b in im_batches)

            res = pnet.parallel.starmap_unordered(_extract_batch, args)
            feat = np.concatenate([batch for batch in res])

        return (feat, self._num_parts, self._num_orientations)

    @property
    def trained(self):
        return self._parts is not None

    def train(self, phi, data, y=None):
        raw_patches, raw_originals = self._get_patches(phi, data)

        from vzlog import default as vz
        grid = ImageGrid(raw_originals[:100])
        grid.save(vz.impath(), scale=3)
        vz.flush()

        vz.section('The edges')

        from pylab import cm

        for i in range(5):
            vz.log(i)
            grid = ImageGrid(raw_patches[i].transpose((0, 3, 1, 2)),
                             cmap=cm.cool)
            grid.save(vz.impath(), scale=3)
        vz.flush()

        return self.train_from_samples(raw_patches, raw_originals)

    def train_from_samples(self, raw_patches, raw_originals):
        min_prob = self._settings['min_prob']

        ORI = self._num_orientations
        POL = self._settings.get('polarities', 1)
        P = ORI * POL

        def cycles(X):
            return np.asarray([np.concatenate([X[i:], X[:i]])
                               for i in range(len(X))])

        RR = np.arange(ORI)
        PP = np.arange(POL)
        II = [list(itr.product(PPi, RRi))
              for PPi in cycles(PP)
              for RRi in cycles(RR)]
        lookup = dict(zip(itr.product(PP, RR), itr.count()))
        n_init = self._settings.get('n_init', 1)
        n_iter = self._settings.get('n_iter', 10)
        seed = self._settings.get('em_seed', 0)

        num_angle = ORI
        d = np.prod(raw_patches.shape[2:])
        permutation = np.empty((num_angle, num_angle * d), dtype=np.int_)
        for a in range(num_angle):
            if a == 0:
                permutation[a] = np.arange(num_angle * d)
            else:
                permutation[a] = np.roll(permutation[a-1], d)

        from pnet.bernoulli import em

        X = raw_patches.reshape((raw_patches.shape[0], -1))

        if 0:
            ret = em(X, self._num_true_parts, n_iter,
                     mu_truncation=min_prob,
                     permutation=permutation, numpy_rng=seed,
                     verbose=True)

            comps = ret[3]
            sh = (self._num_true_parts * P,) + raw_patches.shape[2:]
            self._parts = ret[1].reshape(sh)

            if comps.ndim == 1:
                zeros = np.zeros(len(comps), dtype=np.int_)
                comps = np.vstack([comps, zeros]).T

        else:
            permutations = [[lookup[ii] for ii in rows] for rows in II]
            permutations = np.asarray(permutations)

            from pnet.permutation_mm import PermutationMM
            mm = PermutationMM(n_components=self._num_true_parts,
                               permutations=permutations,
                               n_iter=n_iter,
                               n_init=n_init,
                               random_state=seed,
                               min_probability=min_prob)

            Xflat = raw_patches.reshape(raw_patches.shape[:2] + (-1,))
            mm.fit(Xflat)

            comps = mm.predict(Xflat)

            sh = (mm.n_components * P,) + raw_patches.shape[2:]
            self._parts = mm.means_.reshape(sh)

        if 0:
            # Reject some parts
            pp = self._parts[::self._num_orientations]
            Hall = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp))
            H = np.apply_over_axes(np.mean, Hall, [1, 2, 3]).ravel()

            from scipy.stats import scoreatpercentile

            Hth = scoreatpercentile(H, 50)
            ok = H <= Hth

            blocks = []
            for i in range(self._num_true_parts):
                if ok[i]:
                    blocks.append(self._parts[i*self._num_orientations:
                                              (i+1)*self._num_orientations])

            self._parts = np.concatenate(blocks)
            self._num_parts = len(self._parts)
            self._num_true_parts = self._num_parts // self._num_orientations

        ml = self._num_true_parts
        self._train_info['counts'] = np.bincount(comps[:, 0], minlength=ml)

        print(self._train_info['counts'])

        self._visparts = np.asarray([
            raw_originals[comps[:, 0] == k,
                          comps[comps[:, 0] == k][:, 1]].mean(0)
            for k in range(self._num_true_parts)
        ])

    def _get_patches(self, phi, data):
        samples_per_image = self._settings['samples_per_image']
        fr = self._settings.get('outer_frame', 0)
        the_patches = []
        the_originals = []
        ag.info("Extracting patches from")
        ps = self._part_shape

        ORI = self._num_orientations
        POL = self._settings.get('polarities', 1)
        assert POL in (1, 2), "Polarities must be 1 or 2"

        from skimage import transform

        for n, img in enumerate(data):
            size = img.shape[:2]
            # Make it square, to accommodate all types of rotations
            new_size = np.max(size)

            new_size0 = new_size + (new_size - img.shape[0]) % 2
            new_size1 = new_size + (new_size - img.shape[1]) % 2

            img_padded = ag.util.pad_to_size(img, (new_size0, new_size1))
            pad = [(new_size-size[i])//2 for i in range(2)]

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
                all_img = np.concatenate([all_img, 1 - all_img])

            rs = np.random.RandomState(0)

            # Set up matrices that will translate a position in the canonical
            # image to the rotated images. This way, we're not rotating each
            # patch on demand, which will be much slower.
            offset = new_size / 2
            matrices = [pnet.matrix.translation(offset, offset) *
                        pnet.matrix.rotation(a) *
                        pnet.matrix.translation(-offset, -offset)
                        for a in radians]

            # Add matrices for the polarity flips too, if applicable
            matrices *= POL
            X = phi(all_img)

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

            max_samples = self._settings.get('max_samples', np.inf)
            E = X.shape[-1]
            th = self._settings['threshold']

            center_adjusts = [(ps[0] % 2) / 2,
                              (ps[1] % 2) / 2]

            for sample in range(samples_per_image):
                for tries in range(100):
                    x, y = next(i_iter)

                    selection0 = [0,
                                  slice(x+minus_ps[0], x+plus_ps[0]),
                                  slice(y+minus_ps[1], y+plus_ps[1])]

                    # Return grayscale patch and edges patch
                    patch0 = X[selection0]
                    if fr == 0:
                        tot = patch0.sum()
                    else:
                        tot = patch0[fr:-fr, fr:-fr].sum()

                    if th <= tot:
                        XY = np.array([x + center_adjusts[0],
                                       y + center_adjusts[1],
                                       1])[:, np.newaxis]
                        # Now, let's explore all orientations

                        patch = np.zeros((ORI * POL,) + ps + (E,))
                        vispatch = []

                        br = False
                        for ori in range(ORI * POL):
                            p = np.dot(matrices[ori], XY)

                            A = np.asarray(phi.pos_matrix)
                            p2 = np.dot(A, p)

                            H = np.matrix([ps[0] / 2, ps[1] / 2, 0]).T

                            invA = np.linalg.inv(A)

                            lower = np.asarray(np.dot(invA, p2 - H))
                            upper = np.asarray(np.dot(invA, p2 + H))

                            eps = 1e-5
                            # ip = [int((float(p[i])) + eps) for i in range(2)]
                            ip2 = [int((float(p2[i])) + eps) for i in range(2)]

                            def safe_round(x):
                                return int(float(x) + eps)

                            selection_orig = [ori,
                                              slice(safe_round(lower[0]),
                                                    safe_round(upper[0])),
                                              slice(safe_round(lower[1]),
                                                    safe_round(upper[1]))]

                            selection = [ori,
                                         slice(ip2[0] + minus_ps[0],
                                               ip2[0] + plus_ps[0]),
                                         slice(ip2[1] + minus_ps[1],
                                               ip2[1] + plus_ps[1])]

                            try:
                                patch[ori] = X[selection]
                            except:
                                br = True
                                break

                            orig = all_img[selection_orig]
                            vispatch.append(orig)

                        if br:
                            continue

                        # Randomly rotate this patch, so that we don't bias
                        # the unrotated (and possibly unblurred) image

                        vispatch = np.asarray(vispatch)

                        shift = rs.randint(ORI)

                        patch[:ORI] = np.roll(patch[:ORI], shift, axis=0)
                        vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)

                        if POL == 2:
                            patch[ORI:] = np.roll(patch[ORI:], shift, axis=0)
                            vispatch[ORI:] = np.roll(vispatch[ORI:], shift,
                                                     axis=0)

                        the_patches.append(patch)
                        the_originals.append(vispatch)

                        if len(the_patches) >= max_samples:
                            return (np.asarray(the_patches),
                                    np.asarray(the_originals))

                        break

                    if tries == 99:
                        print("100 tries!")

        return np.asarray(the_patches), np.asarray(the_originals)

    def _vzlog_output_(self, vz):
        from pylab import cm

        vz.log(self._train_info['counts'])

        grid1 = ImageGrid(self._visparts, vmin=0, vmax=1)
        grid1.save(vz.impath(), scale=4)

        # Plot all the parts
        vz.log('parts', self._parts.shape)
        grid2 = ImageGrid(self._parts.transpose((0, 3, 1, 2)),
                          vmin=0, vmax=1, cmap=cm.jet,
                          border_color=(0.6, 0.2, 0.2))
        grid2.save(vz.impath(), scale=5)

    def save_to_dict(self):
        d = {}
        d['num_true_parts'] = self._num_true_parts
        d['num_orientations'] = self._num_orientations
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        d['visparts'] = self._visparts
        d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        num_true_parts = d.get('num_true_parts')
        if num_true_parts is None:
            num_true_parts = int(d['num_parts'] // d['num_orientations'])

        obj = cls(num_true_parts, d['num_orientations'], d['part_shape'],
                  settings=d['settings'])
        obj._parts = d['parts']
        obj._visparts = d.get('visparts')
        obj._weights = d.get('weights')
        return obj

    def __repr__(self):
        return ('OrientedPartsLayer(num_parts={num_parts}, '
                'part_shape={part_shape}, settings={settings})').format(
            num_parts=self._num_parts,
            part_shape=self._part_shape,
            settings=self._settings)
