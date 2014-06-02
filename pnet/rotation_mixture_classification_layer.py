from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import amitgroup as ag
from scipy.special import logit
from pnet.permutation_mm import PermutationMM
import itertools as itr

@Layer.register('rotation-mixture-classification-layer')
class RotationMixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, n_orientations=16, min_prob=0.0001, pooling_settings={}, settings={}):
        self._n_components = n_components
        self._n_orientations = n_orientations
        self._min_prob = min_prob
        self._models = None
        self._pooling_settings = pooling_settings
        self._extra = {}
        self._settings = settings

    @property
    def trained(self):
        return self._models is not None

    def reset(self):
        self._models = None

    @property
    def classifier(self):
        return True

    def extract(self, Z_):
        Z = Z_[0]
        F = Z_[1]
        from pnet.cyfuncs import index_map_pooling_multi as poolf
        X = poolf(Z, F, self._pooling_settings['shape'], self._pooling_settings['strides']) 

        XX =  X[:,np.newaxis,np.newaxis]
        theta = self._models[np.newaxis]
        #print('Z', Z.shape)
        #print('mm', mm.shape)

        llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
        bb = np.apply_over_axes(np.sum, llh, [-3, -2, -1])[...,0,0,0]
        Yhat = np.argmax(bb.max(-1), axis=1)
        return Yhat 
    
    def train(self, X_n, Y):
        X = X_n[0]
        num_parts = X_n[1]
        num_orientations = X_n[2]
        num_true_parts = num_parts // num_orientations

        self._extra['training_comp'] = []

        K = Y.max() + 1

        mm_models = []

        for k in xrange(K):
            Xk = X[Y == k]

            from pnet.cyfuncs import index_map_pooling_multi, orientation_pooling

            # Rotate all the Xk samples

            XB = index_map_pooling_multi(Xk, num_parts, (1, 1), (1, 1))
            XB = XB.reshape(XB.shape[:-1] + (num_true_parts, num_orientations))

            from pnet.more_danny import rotate_patch_map

            blocks = [] 

            for ori in xrange(0, self._n_orientations):
                angle = ori / self._n_orientations * 360
                # Rotate all images, apply rotational spreading, then do pooling

                rots = np.asarray([rotate_patch_map(XB[i], angle) for i in xrange(XB.shape[0])])


                yy = orientation_pooling(rots, 
                                         self._pooling_settings['shape'],
                                         self._pooling_settings['strides'],
                                         self._pooling_settings.get('rotation_spreading_radius', 0))

                blocks.append(yy)#.reshape(yy.shape[:-2] + (-1,)))

            blocks = np.asarray(blocks).transpose((1, 0, 2, 3, 4, 5))

            from pnet.vzlog import default as vz

            import gv

            for i in xrange(self._n_orientations):
                gv.img.save_image(vz.generate_filename(), blocks[0,i,:,:,0].sum(-1))

            vz.finalize()

            shape = blocks.shape[2:4] + (np.prod(blocks.shape[4:]),)

            # Flatten
            blocks = blocks.reshape(blocks.shape[:2] + (-1,))

            n_init = self._settings.get('n_init', 1)
            n_iter = self._settings.get('n_iter', 10)
            seed = self._settings.get('em_seed', 0)

            ORI = self._n_orientations
            POL = 1

            P = ORI * POL


            def cycles(X):
                return np.asarray([np.concatenate([X[i:], X[:i]]) for i in xrange(len(X))])

            RR = np.arange(ORI)
            PP = np.arange(POL)
            II = [list(itr.product(PPi, RRi)) for PPi in cycles(PP) for RRi in cycles(RR)]
            lookup = dict(zip(itr.product(PP, RR), itr.count()))
            permutations = np.asarray([[lookup[ii] for ii in rows] for rows in II])

            print(permutations)

            if 0:
                mm = PermutationMM(n_components=self._n_components, 
                                   permutations=permutations,
                                   n_iter=n_iter, 
                                   n_init=n_init, 
                                   random_state=seed, 
                                   min_probability=self._min_prob)
                mm.fit(blocks)
                mu = mm.means_.reshape((self._n_components,)+shape)

            else:       
                num_angle = self._n_orientations
                d = np.prod(shape)
                permutation = np.empty((num_angle, num_angle * d), dtype=np.int_)
                for a in range(num_angle):
                    if a == 0:
                        permutation[a] = np.arange(num_angle * d)
                    else:
                        permutation[a] = np.roll(permutation[a-1], -d)

                from pnet.bernoulli import em      

                XX = blocks.reshape((blocks.shape[0], -1))
                ret = em(XX, self._n_components, n_iter,
                         permutation=permutation, numpy_rng=seed,
                         verbose=True)

                self._extra['training_comp'].append(ret[3])

                mu = ret[1].reshape((self._n_components * self._n_orientations,) + shape)

            
            mm_models.append(mu)


        self._models = np.asarray(mm_models)


    def save_to_dict(self):
        d = {}
        d['n_components'] = self._n_components
        d['n_orientations'] = self._n_orientations
        d['min_prob'] = self._min_prob
        d['models'] = self._models 
        d['settings'] = self._settings
        d['extra'] = self._extra
        d['pooling_settings'] = self._pooling_settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(n_components=d['n_components'], n_orientations=d['n_orientations'], min_prob=d['min_prob'], pooling_settings=d['pooling_settings'], settings=d['settings'])
        obj._extra = d['extra']
        obj._models = d['models']
        return obj


    def __repr__(self):
        return 'RotationMixtureClassificationLayer(n_components={}, n_orientations={})'.format(self._n_components, self._n_orientations)
