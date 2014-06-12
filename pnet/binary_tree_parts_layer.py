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

@Layer.register('binary-tree-parts-layer')
class BinaryTreePartsLayer(Layer):
    def __init__(self, max_depth, part_shape, settings={}):
        #, outer_frame=1, threshold=1):
        self._num_parts_per_layer = 2 
        self._part_shape = part_shape
        self._max_depth = max_depth
        #self._outer_frame = outer_frame
        #self._threshold = threshold
        self._settings = settings
        self._train_info = {}
        self._keypoints = None


        #TODO Temporary
        self._min_llhs = None

        self._w = None
        self._tree = None
        self._constant_terms = None

    @property
    def num_parts(self):
        return self._num_parts

    def extract(self, X):
        assert self.trained 

        th = self._settings['threshold']
        
        if self._keypoints is not None:
            from pnet.cyfuncs import code_index_map_binary_tree_keypoints
            feature_map = code_index_map_binary_tree_keypoints(X, 
                                                         self._tree,
                                                         self._w, 
                                                         self._keypoint_constant_terms,
                                                         self._keypoints,
                                                         self._num_keypoints,
                                                         self._num_parts,
                                                         th,
                                                         outer_frame=self._settings['outer_frame'])
        else:
            from pnet.cyfuncs import code_index_map_binary_tree_new

            feature_map = code_index_map_binary_tree_new(X, 
                                                         self._tree,
                                                         self._w, 
                                                         self._constant_terms,
                                                         self._num_parts,
                                                         th,
                                                         outer_frame=self._settings['outer_frame'])

        #feature_map = feature_map[...,[-1]]

        return (feature_map, self._num_parts)
    @property
    def trained(self):
        return self._w is not None

    def train(self, X, Y=None):
        assert Y is None
        ag.info('Extracting patches')
        patches = self._get_patches(X)
        ag.info('Done extracting patches')
        ag.info('Training patches', patches.shape)
        return self.train_from_samples(patches)

    #def train_layer(self, flatpatches, depth):


    def train_from_samples(self, patches):
        min_prob = self._settings.get('min_prob', 0.01)

        kp_patches = patches.reshape((patches.shape[0], -1, patches.shape[-1]))
        flatpatches = kp_patches.reshape((kp_patches.shape[0], -1))

        q = []

        models = []
        constant_terms = []
        constant_terms_unsummed = []

        tree = []# -123*np.ones((1000, 2), dtype=np.int64)
        cur_pos = 0

        s = 0
        cur_part_id = 0
        
        q.insert(0, (cur_pos, 0, flatpatches))
        s += 1
        #cur_pos += 1
        while q:
            p, depth, x = q.pop()

            model = np.clip(np.mean(x, 0), min_prob, 1 - min_prob)
            def entropy(x):
                return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
            H = np.mean(entropy(model))

            sc = self._settings.get('split_criterion', 'H')

            if len(x) < self._settings.get('min_samples_per_part', 20) or depth >= self._max_depth or \
               (sc == 'H' and H < self._settings.get('split_entropy', 0.30)):
                #tree[p,0] = -1
                #tree[p,1] = cur_part_id
                tree.append((-1, cur_part_id))
                cur_part_id += 1

            else:
                mm = BernoulliMM(n_components=self._num_parts_per_layer, 
                                 n_iter=self._settings.get('n_iter', 8), 
                                 tol=1e-15,
                                 n_init=self._settings.get('n_init', 1), # May improve a bit to increase this
                                 random_state=self._settings.get('em_seed', 0), 
                                 #params='m',
                                 min_prob=min_prob)
                mm.fit(x[:self._settings.get('traing_limit')])
                logprob, resp = mm.eval(x)
                comps = resp.argmax(-1)

                w = logit(mm.means_[1]) - logit(mm.means_[0])


                Hafter = np.mean(entropy(mm.means_[0])) * mm.weights_[0] + np.mean(entropy(mm.means_[1])) * mm.weights_[1]
                IG = H - Hafter

                if sc == 'IG' and IG < self._settings.get('min_information_gain', 0.05):
                    tree.append((-1, cur_part_id))
                    cur_part_id += 1
                else:
                    tree.append((len(models), s))
                    K_unsummed = np.log((1 - mm.means_[1]) / (1 - mm.means_[0]))
                    K = np.sum(K_unsummed)

                    models.append(w)
                    constant_terms.append(K)
                    constant_terms_unsummed.append(K_unsummed)
                    #tree[p,1] = s


                    q.insert(0, (s, depth+1, x[comps == 0]))
                    #cur_pos += 1
                    q.insert(0, (s+1, depth+1, x[comps == 1]))
                    #cur_pos += 1
                    s += 2

        shape = (len(models),) + patches.shape[1:]
        weights = np.asarray(models).reshape(shape)
        constant_terms = np.asarray(constant_terms)
        constant_terms_unsummed = np.asarray(constant_terms_unsummed).reshape(shape)
        tree = np.asarray(tree, dtype=np.int64)

        self._tree = tree
        self._num_parts = cur_part_id
        #print('num_parts', self._num_parts)
        self._w = weights 
        self._constant_terms = constant_terms

        supp_radius = self._settings.get('keypoint_suppress_radius', 0)
        if supp_radius > 0:

            NW = self._w.shape[0]
            max_indices = self._settings.get('keypoint_max', 1000)
            keypoints = np.zeros((NW, max_indices, 3), dtype=np.int64)
            kp_constant_terms = np.zeros(NW)
            num_keypoints = np.zeros(NW, dtype=np.int64)

            from gv.keypoints import get_key_points
            for k in range(NW):
                kps = get_key_points(self._w[k], suppress_radius=supp_radius, max_indices=max_indices)

                NK = len(kps)
                num_keypoints[k] = NK
                keypoints[k,:NK] = kps

                for kp in kps:
                    kp_constant_terms[k] += constant_terms_unsummed[k,kp[0],kp[1],kp[2]]
 
            self._keypoints = keypoints
            self._num_keypoints = num_keypoints
            self._keypoint_constant_terms = kp_constant_terms



        #Hall = (self._parts * np.log(self._parts) + (1 - self._parts) * np.log(1 - self._parts))
        #H = -np.apply_over_axes(np.mean, Hall, [1, 2, 3]).ravel()
        #self._train_info['entropy'] = H

    def _get_patches(self, X):
        #assert X.ndim == 4

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
                    x, y = i_iter.next()
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
        D = self._w.shape[-1]
        N = self._w.shape[0]
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


        try:
            import pydot
        except ImportError:
            pydot = None


        me = np.apply_over_axes(np.mean, np.fabs(self._w), [1, 2, 3]).ravel()

        if pydot is not None:
            graph = pydot.Dot(graph_type='graph')
            graph.set_node_defaults(label='""', shape='circle', fixedsize=True, width=0.1, nodesep=0.1, ranksep=0.1, fillcolor='yellow', style='filled')
            graph.set_edge_defaults(weight=10.0)


            def push(q, v):
                q.insert(0, v)
            q = []
            push(q, 0)

            while q:
                v = q.pop()
                nextup, ref = self._tree[v]
                if nextup == -1:
                    graph.add_node(pydot.Node(v, fillcolor='red'))
                    pass 
                else:
                    graph.add_node(pydot.Node(v))
                    for c in range(2):
                        graph.add_edge(pydot.Edge(v, ref+c))
                        push(q, ref+c)

            graph.write_svg(vz.generate_filename(ext='svg'))

        from matplotlib.colors import LinearSegmentedColormap
        C = LinearSegmentedColormap('BlueRed1', cdict1)

        grid = pnet.plot.ImageGrid(N, D, self._part_shape, border_color=(0.6, 0.2, 0.2))
        for i in range(N):
            for j in range(D):
                grid.set_image(self._w[i,...,j], i, j, cmap=C, vmin=-4, vmax=4)#cm.BrBG)

        grid.save(vz.generate_filename(), scale=5)

        #vz.log('weights:', self._weights)
        #vz.log('entropy', self._train_info['entropy'])

        vz.log('tree', self._num_parts)
        vz.log(self._tree)

        vz.log('absolute mean')
        vz.log(me)

        if self._keypoints is not None:
            vz.log(self._num_keypoints)

        if 0:
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
        d['max_depth'] = self._max_depth
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings

        d['tree'] = self._tree
        d['w'] = self._w
        d['constant_terms'] = self._constant_terms
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['num_parts_per_layer'], d['depth'], d['part_shape'], settings=d['settings'])
        obj._tree = d['tree']
        obj._w = d['w']
        obj._constant_terms = d['constant_terms']
        return obj

    def __repr__(self):
        return 'PartsLayer(num_parts={num_parts}, part_shape={part_shape}, settings={settings})'.format(
                    num_parts=self._num_parts,
                    part_shape=self._part_shape,
                    settings=self._settings)
