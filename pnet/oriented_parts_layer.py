from __future__ import division, print_function, absolute_import 

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
import pnet.matrix
import random

# TODO: Use later
def _threshold_in_counts(settings, num_edges, contrast_insensitive, part_shape):
    threshold = settings['threshold']
    frame = settings['outer_frame']
    if not contrast_insensitive:
        num_edges //= 2
    th = max(1, int(threshold * (part_shape[0] - 2*frame) * (part_shape[1] - 2*frame) * num_edges))
    return th

def _extract_many_edges(bedges_settings, settings, images, must_preserve_size=False):
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

@Layer.register('oriented-parts-layer')
class OrientedPartsLayer(Layer):
    def __init__(self, num_parts, num_orientations, part_shape, settings={}):
        #, outer_frame=1, threshold=1):
        self._num_true_parts = num_parts
        self._num_parts = num_parts * num_orientations
        self._num_orientations = num_orientations
        self._part_shape = part_shape
        #self._outer_frame = outer_frame
        #self._threshold = threshold
        self._settings = settings
        self._train_info = {}
        self._keypoints = None

        self._parts = None
        self._weights = None

    @property
    def num_parts(self):
        return self._num_parts

    def extract(self, im):
        assert self._parts is not None, "Must be trained before calling extract"

        X = _extract_many_edges(self._settings['bedges'], self._settings, im, must_preserve_size=True) 

        th = self._settings['threshold']
        n_coded = self._settings.get('n_coded', 1)
        
        support_mask = np.ones(self._part_shape, dtype=np.bool)

        from pnet.cyfuncs import code_index_map_general

        feature_map = code_index_map_general(X, 
                                             self._parts, 
                                             support_mask.astype(np.uint8),
                                             th,
                                             outer_frame=self._settings['outer_frame'], 
                                             n_coded=n_coded,
                                             standardize=self._settings.get('standardize', 0),
                                             min_percentile=self._settings.get('min_percentile', 0.0))



        # Rotation spreading?
        rotspread = self._settings.get('rotation_spreading_radius', 0)
        if rotspread == 0:
            between_feature_spreading = None
        else:
            between_feature_spreading = np.zeros((self.num_parts, rotspread*2 + 1), dtype=np.int64)
            ORI = self._num_orientations

            for f in xrange(self.num_parts):
                thepart = f // ORI
                ori = f % ORI 
                for i in xrange(rotspread*2 + 1):
                    between_feature_spreading[f,i] = thepart * ORI + (ori - rotspread + i) % ORI

            bb = np.concatenate([between_feature_spreading, -np.ones((1, rotspread*2 + 1), dtype=np.int64)], 0)

            # Update feature map
            feature_map = bb[feature_map[...,0]]
        
        return (feature_map, self._num_parts, self._num_orientations)

    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None):
        raw_patches, raw_originals = self._get_patches(X)


        return self.train_from_samples(raw_patches, raw_originals)

    def train_from_samples(self, raw_patches, raw_originals):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        #from pnet.bernoullimm import BernoulliMM
        from pnet.permutation_mm import PermutationMM
        min_prob = self._settings.get('min_prob', 0.01)

        ORI = self._num_orientations
        POL = self._settings.get('polarities', 1)
        P = ORI * POL

        def cycles(X):
            return np.asarray([np.concatenate([X[i:], X[:i]]) for i in xrange(len(X))])

        RR = np.arange(ORI)
        PP = np.arange(POL)
        II = [list(itr.product(PPi, RRi)) for PPi in cycles(PP) for RRi in cycles(RR)]
        lookup = dict(zip(itr.product(PP, RR), itr.count()))
        permutations = np.asarray([[lookup[ii] for ii in rows] for rows in II])
        print(permutations)
        n_init = self._settings.get('n_init', 1)
        n_iter = self._settings.get('n_iter', 10)
        seed = self._settings.get('em_seed', 0)

        mm = PermutationMM(n_components=self._num_true_parts, permutations=permutations, n_iter=n_iter, n_init=n_init, random_state=seed, min_probability=min_prob)
        #import IPython
        #IPython.embed()

        mm.fit(raw_patches.reshape(raw_patches.shape[:2] + (-1,)))

        comps = mm.mixture_components()

        self._parts = mm.means_.reshape((mm.n_components * P,) + raw_patches.shape[2:])

        self._train_info['counts'] = np.bincount(comps[:,0], minlength=mm.n_components)

        mcomps = comps.copy()
        #mcomps[:,1] = self._num_orientations - mcomps[:,1] - 1


        print(self._train_info['counts'])

        #raw_originals[tuple(mcomps[mcomps[:,0]==k].T)].mean(0) for k in xrange(mm.n_components)             

        self._visparts = np.asarray([
            raw_originals[comps[:,0]==k,comps[comps[:,0]==k][:,1]].mean(0) for k in xrange(mm.n_components)             
        ])

        XX = [
            raw_originals[comps[:,0]==k,comps[comps[:,0]==k][:,1]] for k in xrange(mm.n_components)             
        ]

        #import pdb; pdb.set_trace()

        N = 100

        from pnet.vzlog import default as vz
        from pylab import cm

        m = self._train_info['counts'].argmax()
        mcomps = comps[comps[:,0] == m]

        raw_originals_m = raw_originals[comps[:,0] == m]

        if 0:
            grid0 = pnet.plot.ImageGrid(N, self._num_orientations, raw_originals.shape[2:], border_color=(1, 1, 1))
            for i in xrange(min(N, raw_originals_m.shape[0])): 
                for j in xrange(self._num_orientations):
                    grid0.set_image(raw_originals_m[i,(mcomps[i,1]+j)%self._num_orientations], i, j, vmin=0, vmax=1, cmap=cm.gray)

            grid0.save(vz.generate_filename(), scale=3)

        grid0 = pnet.plot.ImageGrid(mm.n_components, N, raw_originals.shape[2:], border_color=(1, 1, 1))
        for m in xrange(mm.n_components):
            for i in xrange(min(N, XX[m].shape[0])):
                grid0.set_image(XX[m][i], m, i, vmin=0, vmax=1, cmap=cm.gray)
                #grid0.set_image(XX[i][j], i, j, vmin=0, vmax=1, cmap=cm.gray)

        grid0.save(vz.generate_filename(), scale=3)

        # Reject some parts

    def _get_patches(self, X):
        bedges_settings = self._settings['bedges']
        samples_per_image = self._settings['samples_per_image']
        fr = self._settings.get('outer_frame', 0)
        the_patches = []
        the_originals = []
        ag.info("Extracting patches from")
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)
        setts = self._settings['bedges'].copy()
        radius = setts['radius']
        setts2 = setts.copy()
        setts['radius'] = 0
        ps = self._part_shape


        ORI = self._num_orientations
        POL = self._settings.get('polarities', 1)
        assert POL in (1, 2), "Polarities must be 1 or 2"
        #assert ORI%2 == 0, "Orientations must be even, so that opposite can be collapsed"

        # LEAVE-BEHIND

        from skimage import transform

        for n, img in enumerate(X):
            size = img.shape[:2]
            # Make it square, to accommodate all types of rotations
            new_size = int(np.max(size) * np.sqrt(2))
            img_padded = ag.util.zeropad_to_shape(img, (new_size, new_size))
            pad = [(new_size-size[i])//2 for i in xrange(2)]

            angles = np.arange(0, 360, 360/ORI)
            radians = angles*np.pi/180
            all_img = np.asarray([transform.rotate(img_padded, angle, resize=False, mode='mirror') for angle in angles])
            # Add inverted polarity too
            if POL == 2:
                all_img = np.concatenate([all_img, 1-all_img])


            # Set up matrices that will translate a position in the canonical image to
            # the rotated iamges. This way, we're not rotating each patch on demand, which
            # will end up slower.
            matrices = [pnet.matrix.translation(new_size/2, new_size/2) * pnet.matrix.rotation(a) * pnet.matrix.translation(-new_size/2, -new_size/2) for a in radians]

            # Add matrices for the polarity flips too, if applicable
            matrices *= POL 

            #inv_img = 1 - img
            all_unspread_edges = _extract_many_edges(bedges_settings, self._settings, all_img, must_preserve_size=True)

            # Spread the edges
            all_edges = ag.features.bspread(all_unspread_edges, spread=setts['spread'], radius=radius)
            #inv_edges = ag.features.bspread(inv_unspread_edges, spread=setts['spread'], radius=radius)

            # How many patches could we extract?

            # This avoids hitting the outside of patches, even after rotating.
            # The 15 here is fairly arbitrary
            avoid_edge = int(0 + np.max(ps)*np.sqrt(2))

            # This step assumes that the img -> edge process does not down-sample any

            # TODO: Maybe shuffle an iterator of the indices?

            # These indices represent the center of patches
            indices = list(itr.product(xrange(pad[0]+avoid_edge, pad[0]+img.shape[0]-avoid_edge), xrange(pad[1]+avoid_edge, pad[1]+img.shape[1]-avoid_edge)))

            #print(indices)
            random.seed(0)
            random.shuffle(indices)
            i_iter = itr.cycle(iter(indices))

            minus_ps = [-ps[i]//2 for i in xrange(2)]
            plus_ps = [minus_ps[i] + ps[i] for i in xrange(2)]

            E = all_edges.shape[-1]
            #th = _threshold_in_counts(self._settings, E, self._settings['bedges']['contrast_insensitive'], self._part_shape)
            th = self._settings['threshold']

            max_samples = self._settings.get('max_samples', np.inf)

            rs = np.random.RandomState(0)

            for sample in xrange(samples_per_image):
                for tries in xrange(100):
                    #selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                    #x, y = random.randint(0, w-1), random.randint(0, h-1)
                    x, y = i_iter.next()

                    selection0 = [0, slice(x+minus_ps[0], x+plus_ps[0]), slice(y+minus_ps[1], y+plus_ps[1])]

                    # Return grayscale patch and edges patch
                    unspread_edgepatch = all_unspread_edges[selection0]
                    edgepatch = all_edges[selection0]
                    #inv_edgepatch = inv_edges[selection]

                    #amppatch = amps[selection]
                    #edgepatch2 = edges2[selection]
                    #inv_edgepatch2 = inv_edges2[selection]
                    #edgepatch_nospread = edges_nospread[selection]

                    # The inv edges could be incorproated here, but it shouldn't be that different.
                    if fr == 0:
                        tot = unspread_edgepatch.sum()
                    else:
                        tot = unspread_edgepatch[fr:-fr,fr:-fr].sum()


                    #if self.settings['threshold'] <= avg <= self.settings.get('max_threshold', np.inf): 
                    if th <= tot:
                        XY = np.matrix([x, y, 1]).T
                        # Now, let's explore all orientations

                        patch = np.zeros((ORI * POL,) + ps + (E,))
                        vispatch = np.zeros((ORI * POL,) + ps)

                        for ori in xrange(ORI * POL):
                            p = matrices[ori] * XY
                            ip = [int(round(p[i])) for i in xrange(2)]

                            selection = [ori, slice(ip[0]+minus_ps[0], ip[0]+plus_ps[0]), slice(ip[1]+minus_ps[1], ip[1]+plus_ps[1])]

                            patch[ori] = all_edges[selection]


                            orig = all_img[selection]
                            span = orig.min(), orig.max() 
                            if span[1] - span[0] > 0:
                                orig = (orig-span[0])/(span[1]-span[0])

                            vispatch[ori] = orig 

                        # Randomly rotate this patch, so that we don't bias 
                        # the unrotated (and possibly unblurred) image

                        shift = rs.randint(ORI)

                        patch[:ORI] = np.roll(patch[:ORI], shift, axis=0)
                        vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)

                        if POL == 2:
                            patch[ORI:] = np.roll(patch[ORI:], shift, axis=0)
                            vispatch[ORI:] = np.roll(vispatch[ORI:], shift, axis=0)

                        the_patches.append(patch)
                        the_originals.append(vispatch)

                        if len(the_patches) >= max_samples:
                            return np.asarray(the_patches), np.asarray(the_originals)

                        break

                    if tries == 99:
                            print("100 tries!")


        return np.asarray(the_patches), np.asarray(the_originals)


    def infoplot(self, vz):
        from pylab import cm

        vz.log(self._train_info['counts'])

        side = int(np.ceil(np.sqrt(self._num_true_parts)))
        grid0 = pnet.plot.ImageGrid(side, side, self._visparts.shape[1:])
        for i in xrange(self._visparts.shape[0]):
            grid0.set_image(self._visparts[i], i//side, i%side, vmin=0, vmax=1, cmap=cm.gray)
    
        grid0.save(vz.generate_filename(), scale=5)

        D = self._parts.shape[-1]
        N = self._num_parts
        # Plot all the parts
        grid = pnet.plot.ImageGrid(N, D, self._part_shape, border_color=(0.6, 0.2, 0.2))

        cdict1 = {'red':  ((0.0, 0.0, 0.0),
                           (0.5, 0.5, 0.5),
                           (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.4, 0.4),
                           (0.5, 0.5, 0.5),
                           (1.0, 1.0, 1.0)),

                 'blue':  ((0.0, 1.0, 1.0),
                           (0.5, 0.5, 0.5),
                           (1.0, 0.4, 0.4))
                }

        from matplotlib.colors import LinearSegmentedColormap
        C = LinearSegmentedColormap('BlueRed1', cdict1)

        for i in xrange(N):
            for j in xrange(D):
                grid.set_image(self._parts[i,...,j], i, j, cmap=C, vmin=0, vmax=1)#cm.BrBG)

        grid.save(vz.generate_filename(), scale=5)

    def save_to_dict(self):
        d = {}
        d['num_true_parts'] = self._num_true_parts
        d['num_orientations'] = self._num_orientations
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        #d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        # This code is just temporary {
        num_true_parts = d.get('num_true_parts')
        if num_true_parts is None:
            num_true_parts = int(d['num_parts'] // d['num_orientations'])
        # }
        obj = cls(num_true_parts, d['num_orientations'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        #obj._weights = d['weights']
        return obj

    def __repr__(self):
        return 'OrientedPartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
