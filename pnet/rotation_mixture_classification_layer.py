from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
from pnet.permutation_mm import PermutationMM
import itertools as itr


@Layer.register('rotation-mixture-classification-layer')
class RotationMixtureClassificationLayer(SupervisedLayer):
    def __init__(self, n_components=1, n_orientations=16, min_prob=0.0001,
                 pooling_settings={}, settings={}):
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

    def _extract(self, phi, data):
        Z, F = phi(data)[:2]
        from pnet.cyfuncs import index_map_pooling_multi as poolf
        X = poolf(Z, F, self._pooling_settings['shape'],
                  self._pooling_settings['strides'])

        XX = X[:, np.newaxis, np.newaxis]
        theta = self._models[np.newaxis]

        llh = XX * np.log(theta) + (1 - XX) * np.log(1 - theta)
        bb = np.apply_over_axes(np.sum, llh, [-3, -2, -1])[..., 0, 0, 0]
        if self._settings.get('return_latent_rotation'):
            yhat = np.vstack([bb.max(-1).argmax(1), bb.max(1).argmax(-1)]).T
        else:
            yhat = np.argmax(bb.max(-1), axis=1)
        return yhat

    def _train(self, phi, data, y):
        X_n = phi(data)
        X = X_n[0]
        num_parts = X_n[1]
        num_orientations = X_n[2]
        num_true_parts = num_parts // num_orientations

        self._extra['training_comp'] = []

        K = int(y.max()) + 1

        mm_models = []

        for k in range(K):
            Xk = X[y == k]
            assert Xk.shape[-1] == 1

            from pnet.cyfuncs import index_map_pooling_multi

            # Rotate all the Xk samples

            print('A')
            XB = index_map_pooling_multi(Xk, num_parts, (1, 1), (1, 1))
            print('B')
            XB = XB.reshape(XB.shape[:-1] + (num_true_parts, num_orientations))

            blocks = []

            print('C')
            for ori in range(0, self._n_orientations):
                angle = ori / self._n_orientations * 360
                # Rotate all images, apply rotational spreading, then do
                # pooling

                from pnet.cyfuncs import rotate_index_map_pooling

                sr = self._pooling_settings.get('rotation_spreading_radius', 0)
                yy1 = rotate_index_map_pooling(Xk[..., 0], angle, sr,
                                               num_orientations,
                                               num_parts,
                                               self._pooling_settings['shape'])

                sh = yy1.shape[:3] + (num_orientations, num_true_parts)
                yy = yy1.reshape(sh)

                blocks.append(yy)

            blocks = np.asarray(blocks).transpose((1, 0, 2, 3, 4, 5))
            print('D')

            """
            from pnet.vzlog import default as vz

            import gv

            for i in range(self._n_orientations):
                gv.img.save_image(vz.impath(), blocks[0, i, :, :, 0].sum(-1))

            vz.finalize()
            """

            shape = blocks.shape[2:4] + (np.prod(blocks.shape[4:]),)

            # Flatten
            blocks = blocks.reshape(blocks.shape[:2] + (-1,))

            n_init = self._settings.get('n_init', 1)
            n_iter = self._settings.get('n_iter', 10)
            seed = self._settings.get('seed', 0)

            ORI = self._n_orientations
            POL = 1

            def cycles(X):
                return np.asarray([np.concatenate([X[i:], X[:i]])
                                   for i in range(len(X))])

            RR = np.arange(ORI)
            PP = np.arange(POL)
            II = [list(itr.product(PPi, RRi))
                  for PPi in cycles(PP)
                  for RRi in cycles(RR)]
            lookup = dict(zip(itr.product(PP, RR), itr.count()))
            permutations = [[lookup[ii] for ii in rows] for rows in II]
            permutations = np.asarray(permutations)

            print('E')

            if 1:
                mm = PermutationMM(n_components=self._n_components,
                                   permutations=permutations,
                                   n_iter=n_iter,
                                   n_init=n_init,
                                   random_state=seed,
                                   min_probability=self._min_prob)
                mm.fit(blocks)
                comps = mm.predict(blocks)
                mu_shape = (self._n_components * self._n_orientations,) + shape
                mu = mm.means_.reshape(mu_shape)

            else:
                num_angle = self._n_orientations
                d = np.prod(shape)
                sh = (num_angle, num_angle * d)
                permutation = np.empty(sh, dtype=np.int_)
                for a in range(num_angle):
                    if a == 0:
                        permutation[a] = np.arange(num_angle * d)
                    else:
                        permutation[a] = np.roll(permutation[a-1], -d)

                from pnet.bernoulli import em


                XX = blocks.reshape((blocks.shape[0], -1))
                print('F Digit:', d)
                ret = em(XX, self._n_components, n_iter,
                         permutation=permutation, numpy_rng=seed,
                         verbose=True)
                print('G')

                comps = ret[3]

                self._extra['training_comp'].append(ret[3])

                mu = ret[1].reshape((self._n_components * self._n_orientations,) + shape)

            if 0: # Build visualizations of all rotations

                ims10k = self._data
                label10k = y

                ims10k_d = ims10k[label10k == d]

                rot_ims10k = np.asarray([[rotate(im, -rot, resize=False) for rot in np.arange(n_orientations) * 360 / n_orientations] for im in ims10k_d])

                vispart_blocks = []

                for phase in range(n_orientations):
                    visparts = np.asarray([
                        rot_ims10k[comps[:, 0]==k, comps[comps[:, 0]==k][:, 1]].mean(0) for k in range(n_comp)
                    ])

                    M = 50

                    grid0 = pnet.plot.ImageGrid(n_comp, min(M, np.max(map(len, XX))), ims10k.shape[1:])
                    for k in range(n_comp):
                        for i in range(min(M, len(XX[k]))):
                            grid0.set_image(XX[k][i], k, i, vmin=0, vmax=1, cmap=cm.gray)
                    grid0.save(vz.impath(), scale=3)

                    for k in range(n_comp):
                        grid.set_image(visparts[k], d, k, vmin=0, vmax=1, cmap=cm.gray)




            mm_models.append(mu)
            print('H')


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
