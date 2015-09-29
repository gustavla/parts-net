from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import Layer
from pnet.layer import SupervisedLayer
import pnet
import pnet.matrix
from scipy import linalg
from sklearn.utils.extmath import pinvh
from amitgroup.plot import ImageGrid, ColorImageGrid
#from itertools import repeat
import multiprocessing as mp

def pre_extract((X, settings, covar, means, lmeans, lstds, part_shape, min_log_prob, weights, gmm_cov_type)):
        print('pre_extracting',X.shape[0])
        ps = part_shape
        dim = (X.shape[1]-ps[0]+1, X.shape[2]-ps[1]+1, 1)

        feature_map = np.empty((X.shape[0],) + dim + (means.shape[0],), dtype=np.float32)
        feature_map[:] = -1
        for i in range(dim[0]):
            for j in range(dim[1]):
                Xij_patch=X[:,i:i+ps[0],j:j+ps[1]]
                #out= self.__extract_helper(Xij_patch, covar,img_stds)

                flatXij_patch = Xij_patch.reshape((Xij_patch.shape[0], -1))
                not_ok = (flatXij_patch.std(-1) <= settings['std_thresh'])
                K = means.shape[0]
                logprob, log_resp, lpr = score_samples(flatXij_patch,
                                          means.reshape((K, -1)),
                                          weights.ravel(),
                                          covar,
                                          gmm_cov_type,
                                          )


                coding = settings['coding']
                if coding == 'hard':
                    C = lpr.argmax(-1)
                    feature_map[np.arange(C.shape[0]),i,j,0,np.array(C)]=1
                else:
                    f=(lpr-lmeans)/lstds+settings['lbias']
                    f[not_ok,:]=-1
                    f=f.clip(min=0)
                    f=f[:,np.newaxis]
                    feature_map[:, i, j]=f
        feature_map=feature_map.clip(min=0)
        return(feature_map)

def logsumexp(a,axis=-1):
    a_max = np.amax(a, axis=axis, keepdims=True)

    c=(a-a_max)
    c[c<-500]=-500
    tmp = np.exp(c)


    out = np.log(np.sum(tmp, axis=axis))
    a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    return out

def lognormalized(a,axis=-1):
    a_max = np.amax(a, axis=axis, keepdims=True)

    c=(a-a_max)

    c[c<-500]=-500
    tmp = np.exp(c)
    outs=np.sum(tmp,axis=axis)
    ii=np.where(outs>0)[0]
    anorm=np.zeros(a.shape)
    lanorm=np.zeros(a.shape)
    out=np.zeros(outs.shape)
    anorm[ii]=tmp[ii]/outs[ii,np.newaxis]
    lanorm[ii]=np.log(anorm[ii])
    out[ii] = np.log(outs[ii])
    a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    return (lanorm, out)

def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""

    # print("X")
    #print('in diag',X.shape)
    # print(X)
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars)))
    cost2= np.sum((means*means) / covars) - 2 * np.dot(X, (means / covars).T) + np.dot(X*X, (1.0 / covars).T)
    lpr-=.5*cost2
    #print('diag',lpr.shape)
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
                  + np.sum(X * np.dot(X, icv), 1)
                  - 2 * np.dot(np.dot(X, icv), means.T)
                  + np.sum(means * np.dot(means, icv)))
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

    lpr=np.zeros((X.shape[0],means.shape[0]))
    for k in range(means.shape[0]):
        lpr[:,k] = (log_multivariate_normal_density(X, means[k,:], covars[k,:],
                                           covariance_type))
    #     + np.log(weights[k]))

    # logprob = logsumexp(lpr, axis=1)
    # #responsibilities = np.exp(lpr - logprob[:, np.newaxis])
    # log_resp = lpr - logprob[:, np.newaxis]
    log_resp, logprob = lognormalized(lpr,-1)
    #log_resp=log_resp[...,np.newaxis]
    return logprob, log_resp, lpr


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
        'ones': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)


@Layer.register('oriented-gaussian-parts-layer')
class OrientedGaussianPartsLayer(SupervisedLayer):
    def __init__(self,settings={},pool=None):

        self._settings = dict(polarities=1,
                              n_iter=10,
                              n_init=1,
                              seed=0,
                              num_orientations=1,
                              part_shape=(3,3),
                              num_parts=1,
                              outer_frame=0,
                              standardize=False,
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
                              min_count=10,
                              lbias=0,
                              coding='hard',
                              num_pool=1,
                              allocate=False,
                              by_class=False
                              )

        self._min_log_prob = -100000
        self._extra = {}

        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self.random_state = np.random.RandomState(self._settings['seed'])
        self._num_true_parts = self._settings['num_parts']
        self._num_orientations = self._settings['num_orientations']
        self._part_shape = self._settings['part_shape']

        self._train_info = None
        self._keypoints = None
        self.pool=pool
        self.num_pool=self._settings['num_pool']
        self._means = None
        self._covar = None
        self._weights = None
        self.pad=True
        self._visparts = None
        self._whitening_matrix = None

        self.w_epsilon = self._settings['whitening_epsilon']

    @property
    def num_parts(self):
        return self._num_true_parts * self._num_orientations

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
        elif cov_type == 'diag' or cov_type=='ones':
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
        logprob, log_resp, lpr = score_samples(flatXij_patch,
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
            f=(lpr-self._lmeans)/self._lstds+self._settings['lbias']
            f[not_ok,:]=self._min_log_prob
            f=f.clip(min=0)
            f=f[:,np.newaxis]
            return f

    def _extract(self, phi, data):
        assert self.trained, "Must be trained before calling extract"
        X = phi(data)
        if (self.pad):
            new_size = [X.shape[1] + self._part_shape[0],
                        X.shape[2] + self._part_shape[1]]
            X_padded = ag.util.pad_to_size(X,
                        (-1, new_size[0], new_size[1],) + X.shape[3:])
            X=X_padded
        covar = self._prepare_covariance()
        sett = (self._settings,
                covar,
                self._means,
                self._lmeans,
                self._lstds,
                self._part_shape,
                self._min_log_prob,
                self._weights,
                self.gmm_cov_type)

        empirical_factor = 2.0
        bytesize = np.prod(X.shape)*8*empirical_factor


        #memory = self._settings['mem_gb'] * 1024**3
        memory = 10 * 1024**3
        n_batches = np.int16(np.ceil(bytesize/memory))
        print(n_batches)
        n_batches=3
        if (self.pool is None and self.num_pool>1):
            self.pool=mp.Pool(self.num_pool)
        print(self.pool)
        X_batches = np.array_split(X, n_batches)
        featl=[]
        for X_b in X_batches:
            if (X_b.shape[0]>2*self.num_pool):
                XX=np.array_split(X_b,self.num_pool, axis=0)
                args = ((X_x,) + sett for X_x in XX)
                if (self.num_pool>1):
                    output=self.pool.map(pre_extract,args)
                    print("Done Parallel")
                else:
                    output=map(pre_extract,args)
                feats=output[0]
                for l in range(1,len(output)):
                    feats=np.concatenate((feats,output[l]))
                featl.append(feats)
            else:
                feats=pre_extract((X_b,)+sett)
                featl.append(feats)

        print("Done batches", len(X_batches))
        feature_map=featl[0]
        for l in range(1,len(featl)):
            feature_map=np.concatenate((feature_map,featl[l]))

        #self.__TEMP_ex_log_probs = ex_log_probs
        #self.__TEMP_ex_log_probs2 = np.concatenate(ex_log_probs2)
        feature_map=np.float64(feature_map)
        print('mean above zero',np.mean(feature_map>0))
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

    def _train(self, phi, data, y):

        raw_patches, raw_originals, patch_indices = self._get_patches(phi, data)
        patch_ys=y[np.int32(patch_indices)]
        self._train_info = {}
        self._train_info['example_patches2'] = raw_originals[:10]


        self.train_from_samples(raw_patches, raw_originals, patch_ys)
        from Draw_means import show_visparts
        show_visparts(self._visparts,self._num_orientations,10, True,100)
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

    def train_from_samples(self, raw_patches, raw_originals, y):
        ORI = self._num_orientations
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
        d = np.prod(raw_patches.shape[2:])
        permutation = np.empty((num_angle, num_angle * d), dtype=np.int_)
        for a in range(num_angle):
            if a == 2:
                permutation[a] = np.arange(num_angle * d)
            else:
                permutation[a] = np.roll(permutation[a-1], d)

        permutations = np.asarray([[lookup[ii] for ii in rows] for rows in II])
        Xflat = raw_patches.reshape(raw_patches.shape[:2] + (-1,))
        from pnet.permutation_gmm import PermutationGMM
        num_class=1
        if self._settings['by_class']:
            num_class=np.max(y)+1
        if (num_class==1):
            y=np.zeros(len(y))
        mm = PermutationGMM(n_components=self._num_true_parts,
                            permutations=permutations,
                            n_iter=n_iter,
                            n_init=n_init,
                            random_state=self.random_state,
                            allocate=self._settings['allocate'],
                            thresh=1e-5,
                            covariance_type=self._settings['covariance_type'],
                            covar_limit=covar_limit,
                            min_covar=self._settings['min_covariance'],
                            params='wmc',
                            )

        ntp=self._num_true_parts
        if (self._num_orientations==1):
            ntp=np.int32(self._num_true_parts/num_class)
        for c in range(num_class):
            mmc=PermutationGMM(n_components=ntp,
                       permutations=permutations,
                       n_iter=n_iter,
                       n_init=n_init,
                       random_state=self.random_state,
                       allocate=self._settings['allocate'],
                       thresh=1e-5,
                       covariance_type=self._settings['covariance_type'],
                       covar_limit=covar_limit,
                       min_covar=self._settings['min_covariance'],
                       params='wmc',
                       )
            mmc.fit(Xflat[y==c])
            mmc=self.organize_parts(Xflat[y==c,:],mmc,raw_originals[y==c,:],ntp)

            if (c==0):
                mm.means_=mmc.means_
                mm.weights_=mmc.weights_*np.float32(np.sum(y==c))/Xflat.shape[0]
                mm.visparts=mmc.visparts
                mm.counts=mmc.counts
                mm._num_true_parts=mmc._num_true_parts
                mm.covars_=mmc.covars_
            else:
                mm.means_=np.concatenate((mm.means_,mmc.means_), axis=0)
                mmc.weights_*=np.float32(np.sum(y==c))/Xflat.shape[0]
                mm.weights_=np.concatenate((mm.weights_,mmc.weights_))
                mm.visparts=np.concatenate((mm.visparts,mmc.visparts))
                mm.counts=np.concatenate((mm.counts,mmc.counts))
                mm.covars_=np.concatenate((mm.covars_,mmc.covars_))
                mm._num_true_parts+=mmc._num_true_parts


        lmeans, lstds=self.get_loglike_stats(mm.means_,mm.weights_,mm.covars_,Xflat)
        IS=(lstds>0)
        print("Length of IS", np.sum(IS))
        # Example patches - initialize to NaN if component doesn't fill it up
        # Rotate the parts into the canonical rotation
        #if POL == 1 and False:
            #self.rotate_indices(mm, ORI)

        means_shape = (mm._num_true_parts * P,) + raw_patches.shape[2:]
        self._num_true_parts=np.sum(IS)
        self._num_parts=self._num_true_parts*P
        ag.info('means_shape', means_shape)
        self._visparts=mm.visparts
        self._means = mm.means_.reshape(means_shape)
        self._means = self._means[IS,:]
        mm.covars_=mm.covars_.reshape((np.prod(mm.covars_.shape[0:2]),)+mm.covars_.shape[2:])
        self._covar = mm.covars_[IS,:]
        self._lmeans=lmeans[IS]
        self._lstds=lstds[IS]
        self._weights = mm.weights_
        if self._settings['uniform_weights']:
            # Set weights to uniform
            self._weights = np.ones(mm.weights_.shape)
            self._weights /= np.sum(self._weights)

        self._train_info['counts'] = mm.counts

    def organize_parts(self,Xflat,mm,raw_originals,ml):
        comps = mm.predict(Xflat)
        counts = np.bincount(comps[:, 0], minlength=ml)
        # Average all images beloinging to true part k at selected rotation given by second coordinate of comps
        visparts = np.asarray([
            raw_originals[comps[:, 0] == k,
                          comps[comps[:, 0] == k][:, 1]].mean(0)
            for k in range(ml)
        ])
        ww = counts / counts.sum()
        ok = counts >= self._settings['min_count']
        # Reject some parts
        #ok = counts >= self._settings['min_count']
        II = np.argsort(counts)[::-1]
        print('Counts',np.sort(counts))
        #II = range(len(counts))
        II = np.asarray([ii for ii in II if ok[ii]])
        ag.info('Keeping', len(II), 'out of', ok.size, 'parts')
        mm._num_true_parts = len(II)
        mm.means_ = mm.means_[II]
        mm.weights_ = mm.weights_[II]
        mm.counts = counts[II]
        mm.visparts = visparts[II]
        covtypes = ['diag', 'diag-perm', 'full', 'full-full','ones']
        if self._settings['covariance_type'] in covtypes:
            mm.covars = mm.covars_[II]
        else:
            mm.covars = mm.covars_
        return(mm)


    def get_loglike_stats(self,means,weights,covars,Xflat):
            XX=Xflat[:,0,:]

            lp, lr, lprs=score_samples(XX,means.reshape(means.shape[0]*means.shape[1],means.shape[2]),weights.ravel(),covars.reshape((covars.shape[0]*covars.shape[1],)+covars.shape[2:]),self._settings['covariance_type'])
            alprs=np.argmax(lprs,axis=1)
            lmeans=np.zeros(means.shape[0]*means.shape[1])
            lstds=-np.ones(lmeans.size)
            for k in range(lmeans.size):
                ii=np.where(alprs==k)
                if (len(ii[0])>10):
                    lmeans[k]=lprs[ii][:,k].mean()
                    lstds[k]=(ii[0].shape[0]*lprs[ii][:,k].std()+.1)/(ii[0].shape[0]+1)
            return lmeans, lstds

    def _get_patches(self, phi, data):
        samples_per_image = self._settings['samples_per_image']
        fr = self._settings.get('outer_frame', 0)
        # invA: This matrix has recursively update over the layers to get the affine transformation
        # that applied on the corners of a window in the current layer will provide the corners of the window in
        # original image layer
        A = np.asarray(pnet.matrix.identity())
        if (hasattr(phi,'pos_matrix') and phi.pos_matrix is not None):
            A = np.asarray(phi.pos_matrix)
        invA = np.linalg.inv(A)
        the_patches = None
        the_originals = None
        ag.info("Extracting patches from")
        ps = self._part_shape
        visps=tuple(np.int64(np.ceil(np.dot(invA,np.append(ps,1))[0:2])))
        minus_ps=np.zeros(2)
        plus_ps=np.zeros(2)
        minus_visps=np.zeros(2)
        plus_visps=np.zeros(2)
        for i in range(2):
            minus_ps[i] = -(ps[i]//2)
            plus_ps[i] = minus_ps[i] + ps[i]
            minus_visps[i] = -(visps[i]//2)
            plus_visps[i] = minus_visps[i] + visps[i]

        ORI = self._num_orientations
        POL = self._settings.get('polarities', 1)
        assert POL in (1, 2), "Polarities must be 1 or 2"

        from skimage import transform
        num_data=data.shape[0]
        top_indices = range(num_data)
        rs = self.random_state
        rs.shuffle(top_indices)
        data=data[top_indices,:]
        size = data.shape[1:3]

        # Make it square, to accommodate all types of rotations
        new_side = np.max(size)

        new_size = [new_side + (new_side - data.shape[1]) % 2,
                    new_side + (new_side - data.shape[2]) % 2]

        pad = [(new_size[i]-size[i])//2 for i in range(2)]

        angles = np.arange(0, 360, 360/ORI)
        radians = angles*np.pi/180

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
        phi.pad=False
        EE = phi(data[:1]).shape
        phi.pad=True
        E=EE[-1]
        C=data.shape[-1]
        print('Original receptive field size',visps)
        print('Input dimensions to this level',EE[1:3])
        c = 0

        batch_size = 20

        max_samples = self._settings.get('max_samples', 10000)
        avoid_edge=np.int32(np.floor(np.max(ps)/2))

        center_adjusts = [ps[0] % 2,
                              ps[1] % 2]
        patches_sh = ((max_samples, self._num_orientations) +
                      ps + (E,))
        the_patches = np.zeros(patches_sh)
        the_indices = -np.ones(max_samples)
        orig_sh = (max_samples, self._num_orientations) + visps+(C,)
        the_originals = np.zeros(orig_sh)
        #the_originals = []
        consecutive_failures = 0


        # This must mean one iteration...
        for b in itr.count(0):
            data_subset = data[b*batch_size:(b+1)*batch_size]
            if data_subset.shape[0] == 0:
                break
            data_padded = ag.util.pad_to_size(data_subset,
                    (-1, new_size[0], new_size[1],) + data.shape[3:])

            ims = data_padded.transpose(1, 2, 0, 3)
            ims = ims.reshape(data_padded.shape[1:3] + (-1,))
            all_data = np.asarray([
                transform.rotate(ims,
                                 angle,
                                 resize=False,
                                 mode='nearest')
                for angle in angles])

            sh = all_data.shape[:3] + (-1, data.shape[3])
            all_data = all_data.reshape(sh)
            all_data = all_data.transpose(3, 0, 1, 2, 4)

            # Add inverted polarity too
            if POL == 2:
                all_data = np.concatenate([all_data, 1 - all_data], axis=1)

            # This avoids hitting the outside of patches, even after rotating.
            # The 15 here is fairly arbitrary


            sh_samples = all_data.shape[:2]
            sh_image = all_data.shape[2:]
            phi.pad=False
            all_X = phi(all_data.reshape((-1,) + sh_image))
            phi.pad=True
            all_X = all_X.reshape(sh_samples + all_X.shape[1:4])

            X_shape = all_X.shape[2:4]

            # These indices represent the center of patches


            offset = (np.asarray(X_shape) - center_adjusts) / 2
            mmatrices = [pnet.matrix.translation(offset[0], offset[1]) *
                         pnet.matrix.rotation(a) *
                         pnet.matrix.translation(-offset[0], -offset[1])
                         for a in radians]
            msk=np.minimum(2,ps[0]//2)
            for n in range(len(all_data)):
                #print('Image',indices[b*batch_size+n])
                # all_img = all_data[n]
                X = all_X[n]
                alld=all_data[n]
                range_x = range(pad[0]+avoid_edge, pad[0]+X_shape[0]-avoid_edge)
                range_y = range(pad[1]+avoid_edge, pad[1]+X_shape[1]-avoid_edge)
                lindices = list(itr.product(range_x, range_y))
                mask=np.ones(X.shape[1:3])
                rs.shuffle(lindices)
                i_iter = itr.cycle(iter(lindices))

                E = X.shape[-1]
                min_std = self._settings['std_thresh']
                # std_thresh = self._settings['std_thresh']

                cl=0
                for ll in lindices:
                    x = ll[0]
                    y = ll[1]
                    if (mask[x,y] and cl < samples_per_image):
                        selection0 = [0,
                                      slice(x+minus_ps[0], x+plus_ps[0]),
                                      slice(y+minus_ps[1], y+plus_ps[1])]

                        # Return grayscale patch and edges patch
                        patch0 = X[selection0]


                        if fr == 0:
                            std = patch0.std()
                        else:
                            std = patch0[fr:-fr, fr:-fr].std()

                        if min_std <= std:
                            XY = np.array([x, y, 1])[:, np.newaxis]
                            # Now, let's explore all orientations

                            patch = np.zeros((ORI * POL,) + ps + (E,))
                            vispatch = np.zeros((ORI*POL,) + visps + (C,))

                            br = False
                            sels = []
                            for ori in range(ORI * POL):
                                p = np.dot(mmatrices[ori], XY)
                                p2 = p
                                H = np.matrix([ps[0] / 2, ps[1] / 2, 0]).T
                                lower = np.int64(np.round(np.asarray(np.dot(invA, p2 - H))))[0:2].reshape(2,)
                                upper = lower+np.array(visps).reshape(2,)
                                ip2 = [int(round(float(p2[i]))) for i in range(2)]
                                slic=list()
                                visslic=list()
                                for s in range(2):
                                    slic.append(slice(ip2[s] + minus_ps[s],ip2[s] + plus_ps[s]))
                                    visslic.append(slice(lower[s], upper[s]))
                                selection = [ori, slic[0], slic[1]]
                                visselection=[ori, visslic[0], visslic[1]]
                                try:
                                    patch[ori] = X[selection]
                                except:
                                    #print('Skipping', selection)
                                    br = True
                                    break
                                try:
                                    vispatch[ori] = alld[visselection]
                                except:
                                    print(visselection)
                                    br=True
                                    break
                                sels.append(selection)
                            if br:
                                continue
                            def assert_equal(t, x, y):
                                assert x == y, '{}: {} != {}'.format(t, x, y)
                            # Randomly rotate this patch, so that we don't bias
                            # the unrotated (and possibly unblurred) image


                            shift = rs.randint(ORI)

                            patch[:ORI] = np.roll(patch[:ORI], shift, axis=0)
                            vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)

                            if POL == 2:
                                patch[ORI:] = np.roll(patch[ORI:], shift, axis=0)
                                vispatch[ORI:] = np.roll(vispatch[ORI:], shift,
                                                         axis=0)

                            # the_patches.append(patch)
                            #the_originals.append(vispatch)
                            the_patches[c] = patch
                            the_indices[c] = top_indices[b*batch_size+n]
                            the_originals[c] = vispatch
                            c += 1
                            cl+= 1
                            mask[x-msk:x+msk,y-msk:y+msk]=0
                            if c % 500 == 0:
                                ag.info('Fetching patches {}/{}'.format(c, max_samples))

                            if c >= max_samples:
                                return (np.asarray(the_patches),
                                        np.asarray(the_originals), the_indices)



        return np.asarray(the_patches[:c]), np.asarray(the_originals[:c]), the_indices

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
        d['num_orientations'] = self._num_orientations
        d['part_shape'] = self._part_shape
        d['settings'] = self._settings
        d['train_info'] = self._train_info
        d['extra'] = self._extra
        d['whitening_matrix'] = self._whitening_matrix
        d['means'] = self._means
        d['covar'] = self._covar
        d['lmeans'] = self._lmeans
        d['lstds'] = self._lstds
        d['weights'] = self._weights
        d['visparts'] = self._visparts
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        obj._means = d['means']
        obj._covar = d['covar']
        obj._weights = d.get('weights')
        obj._lmeans=d.get('lmeans')
        obj._lstds=d.get('lstds')
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
                        n_orientations=self._num_orientations,
                        part_shape=self._part_shape,
                        settings=self._settings)
