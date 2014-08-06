from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
import pnet.matrix
from amitgroup.plot import ImageGrid


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


def _extract_batch2(im, settings, num_parts, num_orientations, part_shape,
                    parts, extract_func):
    with ag.Timer('transpose'):
        # TODO: Change to self._dtype
        gpu_im = im.transpose((0, 3, 1, 2)).astype(np.float32)

    res = extract_func(gpu_im)

    #print('res size', ag.memsize(res))

    imp = ag.util.pad(im, (0, 1, 1, 0), value=0)[:, :-1, :-1]

    cum_im = imp.sum(-1).cumsum(1).cumsum(2)

    ps = part_shape

    counts = (cum_im[:, ps[0]:, ps[1]:] -
              cum_im[:, :-ps[0], ps[1]:] -
              cum_im[:, ps[0]:, :-ps[1]] +
              cum_im[:, :-ps[0], :-ps[1]])

    th = settings['threshold']
    res[counts < th] = -1

    return res[..., np.newaxis]


def _extract_batch(X, settings, num_parts, num_orientations, part_shape,
                   parts, extract_func):
    assert parts is not None, "Must be trained before calling extract"

    th = settings['threshold']
    n_coded = settings.get('n_coded', 1)

    support_mask = np.ones(part_shape, dtype=np.bool)

    from pnet.cyfuncs import code_index_map_general

    standardize = settings.get('standardize', 0)
    min_percentile = settings.get('min_percentile', 0.0)
    rotspread = settings.get('rotation_spreading_radius', 0)

    feature_map = code_index_map_general(X,
                                         parts,
                                         support_mask.astype(np.uint8),
                                         th,
                                         outer_frame=settings['outer_frame'],
                                         n_coded=n_coded,
                                         standardize=standardize,
                                         min_percentile=min_percentile)

    # Rotation spreading?
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
                              min_count=20,
                              std_thresh=0.05,  # TODO: Temp
                              circular=False,
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
        return self._num_true_parts * self._num_orientations

    @property
    def part_shape(self):
        return self._part_shape

    @property
    def pos_matrix(self):
        return self.conv_pos_matrix(self._part_shape)

    def extract(self, phi, data):
        im = phi(data)

        # Guess an appropriate batch size.
        output_shape = (im.shape[1] - self._part_shape[0] + 1,
                        im.shape[2] - self._part_shape[1] + 1)

        # The empirical factor adjusts the projected size with an empirically
        # measured size.
        empirical_factor = 2.0
        bytesize = (self.num_parts * np.prod(output_shape) *
                    np.dtype(self._dtype).itemsize * empirical_factor)

        MEMORY = 4 * 1024**3

        n_batches = int(bytesize * data.shape[0] / MEMORY)

        #print('output_shape', ag.humanize_bytesize(num_parts * gpu_im.shape[0] *
        #                                           np.prod(output_shape) * 32/8))

        sett = (self._settings,
                self.num_parts,
                self._num_orientations,
                self._part_shape,
                self._parts,
                self._extract_func)

        if n_batches == 0:
            feat = _extract_batch2(im, *sett)
        else:
            im_batches = np.array_split(im, n_batches)

            args = ((im_b,) + sett for im_b in im_batches)

            res = pnet.parallel.starmap_unordered(_extract_batch2, args)
            feat = np.concatenate([batch for batch in res])

        return (feat, self.num_parts, self._num_orientations)

    @property
    def trained(self):
        return self._parts is not None

    def train(self, phi, data, y=None):
        raw_patches, raw_originals = self._get_patches(phi, data)

        assert len(raw_patches), "No patches found"

        if 0:
            from vzlog import default as vz
            grid = ImageGrid(raw_originals[:100])
            grid.save(vz.impath(), scale=3)
            vz.flush()

            vz.sction('The edges')

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
        ml = self._num_true_parts
        counts = np.bincount(comps[:, 0], minlength=ml)

        ag.info('Training counts:', counts)

        # Reject some parts
        ok = counts >= self._settings['min_count']

        ag.info('Keeping', ok.sum(), 'out of', ok.size, 'parts')

        self._num_true_parts = ok.sum()
        mm.means_ = mm.means_[ok]
        mm.weights_ = mm.weights_[ok]
        counts_final = counts[ok]

        # Store info
        sh = (self._num_true_parts * P,) + raw_patches.shape[2:]
        self._parts = mm.means_.reshape(sh)
        self._parts_vis = self._parts.copy()

        if self._settings['circular']:
            sh = raw_patches.shape[2:4]
            assert sh[0] == sh[1], 'Must use square parts with circular'
            side = sh[0]

            off = -(side - 1) / 2

            # Remove edges and make circular
            x, y = np.meshgrid(np.arange(side) + off, np.arange(side) + off)

            mask = (x ** 2) + (y ** 2) <= (side / 2) ** 2

            mask0 = mask[np.newaxis, ..., np.newaxis]

            # We'll set them to any value (0.1). This could use better handling
            # if so that the likelihoods aren't ruined.
            self._parts = mask0 * self._parts + ~mask0 * 0.1
            self._parts_vis[:, ~mask, :] = np.nan

            from vzlog import default as vz
            from pylab import cm
            grid = ag.plot.ImageGrid(mask, cmap=cm.jet)
            grid.save(vz.impath(), scale=10)

        self._train_info['counts_initial'] = counts
        self._train_info['counts'] = counts_final

        # Visualize parts : we iterate only over 'ok' ones
        self._visparts = np.asarray([
            raw_originals[comps[:, 0] == k,
                          comps[comps[:, 0] == k][:, 1]].mean(0)
            for k in np.where(ok)[0]
        ])

        self._preprocess()

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

        size = data.shape[1:3]
        # Make it square, to accommodate all types of rotations
        new_side = np.max(size)

        new_size = [new_side + (new_side - data.shape[1]) % 2,
                    new_side + (new_side - data.shape[2]) % 2]

        pad = [(new_size[i]-size[i])//2 for i in range(2)]

        data_padded = ag.util.pad_to_size(data, (-1, new_size[0], new_size[1]))

        angles = np.arange(0, 360, 360/ORI)
        radians = angles*np.pi/180

        all_data = np.asarray([
            transform.rotate(data_padded.transpose((1, 2, 0)),
                             angle,
                             resize=False,
                             mode='nearest')
            for angle in angles]).transpose((3, 0, 1, 2))

        # Add inverted polarity too
        if POL == 2:
            all_data = np.concatenate([all_data, 1 - all_data], axis=1)

        # Set up matrices that will translate a position in the canonical
        # image to the rotated images. This way, we're not rotating each
        # patch on demand, which will be much slower.
        offset = (np.asarray(new_size) - 1) / 2
        matrices = [pnet.matrix.translation(offset[0], offset[1]) *
                    pnet.matrix.rotation(a) *
                    pnet.matrix.translation(-offset[0], -offset[1])
                    for a in radians]

        # Add matrices for the polarity flips too, if applicable
        matrices *= POL

        # This avoids hitting the outside of patches, even after rotating.
        # The 15 here is fairly arbitrary
        avoid_edge = int(1 + np.max(ps)/2)

        rs = np.random.RandomState(0)

        sh_samples = all_data.shape[:2]
        sh_image = all_data.shape[2:]

        all_X = phi(all_data.reshape((-1,) + sh_image))
        all_X = all_X.reshape(sh_samples + all_X.shape[1:])

        X_shape = all_X.shape[2:4]

        # These indices represent the center of patches
        range_x = range(pad[0]+avoid_edge, pad[0]+X_shape[0]-avoid_edge)
        range_y = range(pad[1]+avoid_edge, pad[1]+X_shape[1]-avoid_edge)

        center_adjusts = [ps[0] % 2,
                          ps[1] % 2]

        offset = (np.asarray(X_shape) - center_adjusts) / 2
        mmatrices = [pnet.matrix.translation(offset[0], offset[1]) *
                     pnet.matrix.rotation(a) *
                     pnet.matrix.translation(-offset[0], -offset[1])
                     for a in radians]

        for n in range(len(data)):
            all_img = all_data[n]
            X = all_X[n]

            indices = list(itr.product(range_x, range_y))

            rs.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            minus_ps = [-(ps[i]//2) for i in range(2)]
            plus_ps = [minus_ps[i] + ps[i] for i in range(2)]

            max_samples = self._settings.get('max_samples', np.inf)
            E = X.shape[-1]
            th = self._settings['threshold']
            # std_thresh = self._settings['std_thresh']

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
                        # std = patch0.std()
                    else:
                        tot = patch0[fr:-fr, fr:-fr].sum()
                        # std = patch0[fr:-fr, fr:-fr].std()

                    # if std_thresh <= std:
                    if th <= tot:
                        XY = np.array([x,
                                       y,
                                       1])[:, np.newaxis]
                        # Now, let's explore all orientations

                        patch = np.zeros((ORI * POL,) + ps + (E,))
                        vispatch = []

                        br = False
                        sels = []
                        for ori in range(ORI * POL):
                            p = np.dot(mmatrices[ori], XY)

                            A = np.asarray(phi.pos_matrix)
                            p2 = p

                            H = np.matrix([ps[0] / 2, ps[1] / 2, 0]).T

                            invA = np.linalg.inv(A)

                            lower = np.asarray(np.dot(invA, p2 - H))
                            upper = np.asarray(np.dot(invA, p2 + H))

                            ip2 = [int(round(float(p2[i]))) for i in range(2)]

                            def safe_round(x):
                                return int(float(x) + 1e-5)

                            selection_orig = [ori,
                                              slice(safe_round(lower[0]),
                                                    safe_round(upper[0])),
                                              slice(safe_round(lower[1]),
                                                    safe_round(upper[1]))]

                            slice1 = slice(ip2[0] + minus_ps[0],
                                           ip2[0] + plus_ps[0])
                            slice2 = slice(ip2[1] + minus_ps[1],
                                           ip2[1] + plus_ps[1])

                            selection = [ori, slice1, slice2]

                            try:
                                patch[ori] = X[selection]
                            except:
                                br = True
                                break

                            orig = all_img[selection_orig]
                            vispatch.append(orig)
                            sels.append(selection)

                        if br:
                            continue

                        def assert_equal(t, x, y):
                            assert x == y, '{}: {} != {}'.format(t, x, y)

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

    def _preprocess(self):
        import theano.tensor as T
        import theano
        from theano.tensor.nnet import conv

        # Create Theano convolution function
        s_input = T.tensor4(name='input')

        gpu_parts = self._parts.transpose((0, 3, 1, 2)).astype(s_input.dtype)
        gpu_parts = (gpu_parts)[:, :, ::-1, ::-1]

        gpu_logits = np.log(gpu_parts / (1 - gpu_parts))
        gpu_rest = np.log(1 - gpu_logits).sum(1).sum(1).sum(1)

        s_logits = theano.shared(gpu_logits, name='logits')
        s_rest = theano.shared(gpu_rest, name='rest')

        s_conv = (conv.conv2d(s_input, s_logits) +
                  s_rest.dimshuffle('x', 0, 'x', 'x'))

        s_output = s_conv.argmax(1)

        self._dtype = s_input.dtype
        self._extract_func = theano.function([s_input], s_output)

    def _vzlog_output_(self, vz):
        from pylab import cm

        if self._train_info:
            vz.log('Initial counts:', self._train_info['counts_initial'])
            vz.log('Final counts:', self._train_info['counts'])

        grid1 = ImageGrid(self._visparts, vmin=0, vmax=1)
        grid1.save(vz.impath(), scale=4)

        # Plot all the parts
        vz.log('parts', self._parts.shape)
        grid2 = ImageGrid(self._parts_vis.transpose((0, 3, 1, 2)),
                          vmin=0, vmax=1, cmap=cm.jet,
                          border_color=1)
        grid2.save(vz.impath(), scale=3)

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
        obj = cls(num_true_parts, d['num_orientations'], d['part_shape'],
                  settings=d['settings'])
        obj._parts = d['parts']
        obj._visparts = d.get('visparts')
        obj._weights = d.get('weights')
        obj._preprocess()
        return obj

    def __repr__(self):
        return ('OrientedPartsLayer(n_parts={n_parts}, '
                'n_orientations={n_orientations}, '
                'part_shape={part_shape}').format(
            n_parts=self._num_true_parts,
            n_orientations=self._num_orientations,
            part_shape=self._part_shape)
