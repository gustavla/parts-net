from __future__ import division, print_function, absolute_import 

# TODO: Temp
import matplotlib as mpl
mpl.use('Agg')

from scipy.special import logit
import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
import pnet
from pnet.bernoulli_mm import BernoulliMM

# TODO: Use later
if 0:
    def _threshold_in_counts(self, threshold, num_edges):
        size = self.settings['part_size']
        frame = self.settings['patch_frame']
        return max(1, int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges))

@Layer.register('hierarchical-parts-layer')
class HierarchicalPartsLayer(Layer):
    def __init__(self, num_parts_per_layer, depth, part_shape, settings={}):
        #, outer_frame=1, threshold=1):
        self._num_parts_per_layer = num_parts_per_layer
        self._depth = depth
        self._num_parts = num_parts_per_layer ** depth
        self._part_shape = part_shape
        #self._outer_frame = outer_frame
        #self._threshold = threshold
        self._settings = settings
        self._train_info = {}
        self._keypoints = None


        #TODO Temporary
        self._min_llhs = None

        self._parts = None
        self._weights = None

    def extract(self, X):
        assert self._parts is not None, "Must be trained before calling extract"

        th = self._settings['threshold']
        n_coded = self._settings.get('n_coded', 1)
        
        support_mask = self._settings.get('support_mask')
        if support_mask is None:
            support_mask = np.ones(self._part_shape, dtype=np.bool)


        if self._num_parts_per_layer == 2:
            from pnet.cyfuncs import code_index_map_binary_tree
            feature_map = code_index_map_binary_tree(X, 
                                                 self._w, 
                                                 self._constant_terms,
                                                 self._depth,
                                                 th,
                                                 outer_frame=self._settings['outer_frame'], 
                                                 min_percentile=self._settings.get('min_percentile', 0.0))
        else:
            from pnet.cyfuncs import code_index_map_hierarchy
            feature_map = code_index_map_hierarchy(X, 
                                                 self._flathier, 
                                                 self._depth,
                                                 th,
                                                 outer_frame=self._settings['outer_frame'], 
                                                 min_percentile=self._settings.get('min_percentile', 0.0))

        feature_map = feature_map[...,[-1]]

        return (feature_map, self._num_parts)
    @property
    def trained(self):
        return self._parts is not None 

    def train(self, X, Y=None):
        assert Y is None
        ag.info('Extracting patches')
        patches = self._get_patches(X)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    #def train_layer(self, flatpatches, depth):

    def _train_inner(self, flatpatches, shape, hier, depth=0):
        min_prob = self._settings.get('min_prob', 0.01)

        #if len(flatpatches) < 5:
            ## Put all of them in both
            #flatpatches.mean()

        mm = BernoulliMM(n_components=self._num_parts_per_layer, 
                         n_iter=20, 
                         tol=1e-15,
                         n_init=5, 
                         random_state=self._settings.get('em_seed', 0), 
                         #params='m',
                         min_prob=min_prob)
        mm.fit(flatpatches)
        logprob, resp = mm.score_samples(flatpatches)
        comps = resp.argmax(-1)

        counts = np.bincount(comps, minlength=self._num_parts_per_layer)

        if 0:
            from scipy.stats import scoreatpercentile
            lower0 = scoreatpercentile(resp[...,0].ravel(), 50)
            lower1 = scoreatpercentile(resp[...,1].ravel(), 50)
            lows = [lower0, lower1]
            hier_flatpatches = [flatpatches[resp[...,m] >= lows[m]] for m in range(self._num_parts_per_layer)]

            # Reset the means
            for m in range(self._num_parts_per_layer):
                mm.means_[m] = np.clip(hier_flatpatches[m].mean(0), min_prob, 1 - min_prob)
        else:
            hier_flatpatches = [flatpatches[comps == m] for m in range(self._num_parts_per_layer)]
        
        #if depth == 0:
            #print('counts', counts) 
        #print(depth, 'hier_flatpatches', map(np.shape, hier_flatpatches))

        all_parts = []
        all_weights = []
        all_counts = []
        all_hierarchies = []

        pp = mm.means_.reshape((self._num_parts_per_layer,) + shape)

        hier[depth].append(pp)

        if depth+1 < self._depth:
            # Iterate
            for m in range(self._num_parts_per_layer):
                parts, weights, counts0 = self._train_inner(hier_flatpatches[m], shape, hier, depth=depth+1)

                all_parts.append(parts)
                all_weights.append(weights)
                all_counts.append(counts0)
                #all_hierarchies.append(hierarchy_parts)

            flat_parts = np.concatenate(all_parts, axis=0)
            all_weights = np.concatenate(all_weights, axis=0)
            all_counts = np.concatenate(all_counts, axis=0)
            return flat_parts, all_weights, all_counts#, [pp] + all_hierarchies
        else:
            parts = mm.means_
            weights = mm.weights_
            #all_parts.reshape((self._num_parts,)+patches.shape[1:])
            return parts, weights, counts


    def train_from_samples(self, patches):
        #from pnet.latent_bernoulli_mm import LatentBernoulliMM
        min_prob = self._settings.get('min_prob', 0.01)

        kp_patches = patches.reshape((patches.shape[0], -1, patches.shape[-1]))
        flatpatches = kp_patches.reshape((kp_patches.shape[0], -1))

        hier = [[] for _ in range(self._depth)]

        all_parts, all_weights, counts  = self._train_inner(flatpatches, patches.shape[1:], hier)

        if 0:
            # Post-training

            # Warm-up the EM with the leaves
            F = self._num_parts_per_layer ** self._depth

            mm = BernoulliMM(n_components=F,
                             n_iter=20, 
                             tol=1e-15,
                             n_init=1, 
                             random_state=100+self._settings.get('em_seed', 0), 
                             init_params='',
                             #params='m',
                             min_prob=min_prob)

            mm.means_ = all_parts
            mm.weights_ = counts / counts.sum()
            mm.fit(flatpatches)
            logprob, resp = mm.score_samples(flatpatches)
            comps = resp.argmax(-1)

            #for d in range(self._depth):
            #import pdb; pdb.set_trace()

            for d in reversed(range(len(hier))):
                for j in range(len(hier[d])):
                    hier_dj = hier[d][j]
                    for k in range(len(hier_dj)):
                        if d == len(hier) - 1:
                            hier[d][j][k] = mm.means_[j*self._num_parts_per_layer + k].reshape(patches.shape[1:])
                        else:
                            hier[d][j][k] = hier[d+1][j*self._num_parts_per_layer + k].mean(0)

                        #hier[d][j][k] = [flatpatches[comps == f].mean(0).reshape((-1,) + patches.shape[1:])
                        #else:
                            #hier[d][j][k] += 


            # Now, update the entire tree with these new results

        if 0:
            def pprint(x):
                if isinstance(x, list):
                    return "[" + ", ".join(map(pprint, x)) + "]"
                elif isinstance(x, np.ndarray):
                    #return "array{}".format(x.shape)
                    return 'A'

            #print(pprint(parts_hierarchy))
            print(pprint(hier))

            #import pdb; pdb.set_trace()

            print(all_parts.shape)

        self._parts = all_parts.reshape((self._num_parts,)+patches.shape[1:])
        self._weights = all_weights 
        self._counts = counts
        #self._parts_hierarchy = parts_hierarchy
        self._hier = hier
        self._flathier = np.asarray(sum(hier, [])) 

        from scipy.special import logit

        if self._num_parts_per_layer == 2:
            self._w = logit(self._flathier[:,1]) - logit(self._flathier[:,0]) 
            self._constant_terms = np.apply_over_axes(np.sum, np.log(1 - self._flathier[:,1]) - np.log(1 - self._flathier[:,0]), [1, 2, 3])[:,0,0,0]

        if 0:
            # Train it again by initializing with this
            mm = BernoulliMM(n_components=self._num_parts, 
                             n_iter=20, 
                             tol=1e-15,
                             n_init=1, 
                             init_params='w',
                             random_state=self._settings.get('em_seed', 0), 
                             min_prob=min_prob)
            mm.means_ = self._parts.reshape((self._num_parts, -1))
            mm.fit(flatpatches)

            self._parts = mm.means_.reshape((self._num_parts,)+patches.shape[1:])
            self._weights = mm.weights_

        Hall = (self._parts * np.log(self._parts) + (1 - self._parts) * np.log(1 - self._parts))
        H = -np.apply_over_axes(np.mean, Hall, [1, 2, 3]).ravel()
        self._train_info['entropy'] = H

    def _get_patches(self, X):
        assert X.ndim == 4

        samples_per_image = self._settings.get('samples_per_image', 20) 
        fr = self._settings.get('outer_frame', 0)
        patches = []

        rs = np.random.RandomState(self._settings.get('patch_extraction_seed', 0))

        th = self._settings['threshold']
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
                    selection = [slice(x, x+self._part_shape[0]), slice(y, y+self._part_shape[1])]

                    patch = Xi[selection]
                    #edgepatch_nospread = edges_nospread[selection]
                    if support_mask is not None:
                        tot = patch[support_mask].sum()
                    elif fr == 0:
                        tot = patch.sum()
                    else:
                        tot = patch[fr:-fr,fr:-fr].sum()

                    if th <= tot: 
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
        D = self._parts.shape[-1]
        N = self._num_parts
        # Plot all the parts

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

        grid = pnet.plot.ImageGrid(N, D, self._part_shape, border_color=(0.6, 0.2, 0.2))
        for i in range(N):
            for j in range(D):
                grid.set_image(self._parts[i,...,j], i, j, cmap=C, vmin=0, vmax=1)#cm.BrBG)

        grid.save(vz.generate_filename(), scale=5)

        vz.title('HIERARCHY')
        base_offset = 0
        for depth in range(self._depth):
            vz.section('Depth', depth)
            for x in range(self._num_parts_per_layer ** depth):
                #grid = pnet.plot.ImageGrid(self._num_parts_per_layer, D, self._part_shape, border_color=(0.6, 0.2, 0.2))
                grid = pnet.plot.ImageGrid(1, D, self._part_shape, border_color=(0.6, 0.2, 0.2))
                #for i in range(self._num_parts_per_layer):
                for j in range(D):
                    grid.set_image(self._w[base_offset+x,...,j], 0, j, cmap=cm.RdBu_r, vmin=-5, vmax=5)#cm.BrBG)

                grid.save(vz.generate_filename(), scale=5)

            base_offset += self._num_parts_per_layer ** depth

        #vz.log('weights:', self._weights)
        #vz.log('entropy', self._train_info['entropy'])

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._counts, label='counts')
        plt.title('Counts')
        plt.savefig(vz.generate_filename(ext='svg'))

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._weights, label='weight')
        plt.title('Weights')
        plt.savefig(vz.generate_filename(ext='svg'))

        vz.log('weights span', self._weights.min(), self._weights.max()) 

        import pylab as plt
        plt.figure(figsize=(6, 3))
        plt.plot(self._train_info['entropy'])
        plt.savefig(vz.generate_filename(ext='svg'))

        vz.log('median entropy', np.median(self._train_info['entropy']))

        llhs = self._train_info.get('llhs')
        if llhs is not None:
            vz.log('llhs', llhs.shape)

            plt.figure(figsize=(6, 3))
            #plt.hist(llhs.mean(0))
            plt.errorbar(np.arange(self._num_parts), llhs.mean(0), yerr=llhs.std(0), fmt='--o')
            plt.title('log likelihood means')
            plt.savefig(vz.generate_filename(ext='svg'))

            parts = np.argmax(llhs, 1) 

            means = np.zeros(self._num_parts)
            sigmas = np.zeros(self._num_parts)
            for f in range(self._num_parts):
                llh0 = llhs[parts == f,f] 
                means[f] = llh0.mean()
                sigmas[f] = llh0.std()
                vz.log('llh', f, ':', means[f], sigmas[f])

            plt.figure(figsize=(6, 3))
            #plt.hist(llhs.mean(0))
            plt.errorbar(np.arange(self._num_parts), means, yerr=sigmas, fmt='--o')

            from scipy.special import logit

            anal_means = -np.prod(self._parts.shape[1:]) * self._train_info['entropy']
            anal_sigmas = np.sqrt(np.apply_over_axes(np.sum, logit(self._parts)**2 * self._parts * (1 - self._parts), [1, 2, 3]).ravel())

            plt.errorbar(np.arange(self._num_parts), anal_means, yerr=anal_sigmas, fmt='--o')
            #plt.plot(np.arange(self._num_parts), anal_means)
            plt.title('coded log likelihood means')
            plt.savefig(vz.generate_filename(ext='svg'))


            plt.figure(figsize=(6, 3))
            plt.hist(llhs.ravel(), 50)
            plt.title('Log likelihood distribution')
            plt.savefig(vz.generate_filename(ext='svg'))


    def save_to_dict(self):
        d = {}
        d['num_parts_per_layer'] = self._num_parts_per_layer
        d['depth'] = self._depth
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['parts'] = self._parts
        d['flathier'] = self._flathier
        d['keypoints'] = self._keypoints
        d['weights'] = self._weights
        d['w'] = self._w
        d['constant_terms'] = self._constant_terms
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts_per_layer'], d['depth'], d['part_shape'], settings=d['settings'])
        obj._parts = d['parts']
        obj._flathier = d['flathier']
        obj._keypoints = d.get('keypoints')
        obj._weights = d['weights']
        obj._w = d['w']
        obj._constant_terms = d['constant_terms']
        return obj

    def __repr__(self):
        return 'PartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
