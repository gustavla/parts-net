from __future__ import division, print_function, absolute_import

import numpy as np
import itertools as itr
import amitgroup as ag
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import pnet
import pnet.matrix
from amitgroup.plot import ImageGrid
import multiprocessing as mp

def _extract_batch((im, settings, logits, logsum, lmeans, lstds, lpthreshs, num_rot, vecs, means)):
    # TODO: Change the whole framework to this format
    #print("In extract batch",im.shape[0])

    from pnet.cyfuncs import code_index_map as cdmp
    logits=logits.transpose(1,2,3,0)

    if (settings['raw']):
        res=cdmp(im,logits,logsum,settings['threshold'],settings['shape'],settings['strides'], lmeans, lstds, lpthreshs, True, num_rot,settings['lbias'],settings['outer_frame'])
        #res[res<settings['lbias']]=0
    else:
        res=cdmp(im,logits,logsum,settings['threshold'],settings['shape'],settings['strides'], np.float32(np.zeros(0)), np.float32(np.zeros(0)), np.float32(np.zeros(0)),False,num_rot,0,settings['outer_frame'])
        res=np.uint8(res)

    #print("Fraction of non zeros's",np.mean(res>0))
    if (vecs is not None):
            print('Projecting')
            XX=res.reshape((np.prod(res.shape[0:3]),res.shape[3]))
            XX=XX-means
            nn=XX.shape[0]//100
            XXs=np.array_split(XX,nn)
            zzs=[]
            for Xs in XXs:
                zzs.append(np.dot(Xs,vecs))
            zz=np.concatenate((zzs),axis=0)
            res=zz.reshape(res.shape[0:3]+(vecs.shape[1],))
            #print("Done")
    return res[..., np.newaxis]


@Layer.register('oriented-parts-layer')
class OrientedPartsLayer(SupervisedLayer):
    def __init__(self, settings={}, pool=None):
        self._settings = dict(outer_frame=0,
                              n_init=1,
                              n_iter=10,
                              threshold=0.0,
                              samples_per_image=40,
                              max_samples=10000,
                              seed=None,
                              min_prob=0.005,
                              min_count=20,
                              std_thresh=0.05,
                              circular=True,
                              raw=False,
                              mem_gb=4,
                              num_parts=1,
                              num_orientations=1,
                              rot_spread=False,
                              part_shape=(3,3),
                              lbias=0,
                              lperc=95.,
                              shape=(1,1),
                              strides=(1,1),
                              n_coded=1,
                              num_pool=1,
                              var_explained=None,
                              num_labeled=0,
                              by_class=False,
                              allocate=False,
                              grad=False,
                              standardize=False
                              )

        for k, v in settings.items():
            if k not in self._settings:
                raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._num_true_parts = self._settings['num_parts']
        self._num_orientations = self._settings['num_orientations']
        self._part_shape = self._settings['part_shape']
        self._train_info = {}
        self._keypoints = None
        self.pool=pool
        self.pad=True
        self.num_pool=self._settings['num_pool']
        self._parts = None
        self._weights = None
        self._principal_vectors=None
        self._principal_means=None

    @property
    def num_parts(self):
        return self._num_true_parts * self._num_orientations

    @property
    def part_shape(self):
        return self._part_shape

    @property
    def pos_matrix(self):
         S = pnet.matrix.scale(1. / self._settings['strides'][0], 1. / self._settings['strides'][1])
         T = pnet.matrix.translation(-(self._settings['shape'][0] - 1) / 2,
                                     -(self._settings['shape'][1] - 1) / 2)
         return np.dot(S, T)
        #return self.conv_pos_matrix(self._part_shape)

    @property
    def visualized_parts(self):
        return self._visparts

    @property
    def parts(self):
        return self._parts



    def _extract(self, phi, data):
        A=phi.pos_matrix
        print(self.name)
        print(np.linalg.inv(A))
        #with ag.Timer('extract inside parts for layer '+self.name):
        im = phi(data)

        if (self.pad):
            new_size = [im.shape[1] + self._part_shape[0],
                        im.shape[2] + self._part_shape[1]]
            im_padded = ag.util.pad_to_size(im,
                        (-1, new_size[0], new_size[1],) + im.shape[3:])
            im=im_padded
        # Guess an appropriate batch size.
        output_shape = (im.shape[1] - self._part_shape[0] + 1,
                        im.shape[2] - self._part_shape[1] + 1)

        # The empirical factor adjusts the projected size with an empirically
        # measured size.
        empirical_factor = 2.0
        bytesize = im.shape[0]*(self.num_parts * np.prod(output_shape) *
                    np.dtype(self._dtype).itemsize * empirical_factor)/(self._settings['strides'][0]**2)

        #memory = self._settings['mem_gb'] * 1024**3
        memory = 10 * 1024**3
        # print('data shape', data.shape)
        # print('output shape', output_shape)
        # print('num parts', self.num_parts)

        n_batches = np.int16(np.ceil(bytesize/memory))
        print('n_batches',n_batches)




        num_rot=1
        if (self._settings['rot_spread']):
            num_rot=self._settings['num_orientations']
        #with ag.Timer('extract final level for layer '+self.name):
        if (1):
            sett = (self._settings,
                    self._logits,
                    self._logsum,
                    self._lmeans,
                    self._lstds,
                    self._lpthreshs,
                    num_rot,
                    self._principal_vectors,
                    self._principal_means)


            im_batches = np.array_split(im, n_batches)
            if (self.pool is None and self.num_pool>1):
                print('Creating pool in oriented parts layer')
                self.pool=mp.Pool(self.num_pool)

            print('Number of processes',self.num_pool)
            featl=[]
            for im_bb in im_batches:
                if (im_bb.shape[0]>2*self.num_pool):
                    im_subbatches=np.array_split(im_bb,self.num_pool)
                    args = ((im_b,) + sett for im_b in im_subbatches)
                    if (self.num_pool>1):
                        #output=ag.apply_function_at_pool(_extract_batch,args) #
                        output=self.pool.map(_extract_batch,args)
                        #print("Done parallel")
                    else:
                        output=map(_extract_batch,args)
                    feats=output[0]
                    for i in range(1,len(output)):
                        feats=np.concatenate((feats,output[i]))
                        # Reduce to original size:
                    #feats=feats[:,self._part_shape[0]/2:self._part_shape[0]/2+old_size[0],self._part_shape[1]/2:self._part_shape[1]/2+old_size[1],:]
                    featl.append(feats)
                else:
                    feats=_extract_batch((im_bb,)+sett)
                    #feats=feats[:,self._part_shape[0]/2:self._part_shape[0]/2+old_size[0],self._part_shape[1]/2:self._part_shape[1]/2+old_size[1],:]
                    featl.append(feats)
            print("Done batches", len(im_batches))
            feat=featl[0]
            for l in range(1,len(featl)):
                feat=np.concatenate((feat,featl[l]))
            #self.pool.terminate()
            #print(pool)
        # This is a raw output
        #if len(feat.shape)==5:
        print('Fraction of non-zeros',np.mean(feat))
        if (feat.ndim==5 and feat.shape[4]==1):
            feat=feat.reshape(feat.shape[0:4])
        if not self._settings['raw'] and self._principal_vectors is None:
           feat=np.uint8(feat)
        print(feat.size, feat.shape)
        return feat #(feat, self._num_parts, self._num_orientations)

    @property
    def trained(self):
        return self._parts is not None

    def _train(self, phi, data, y):
        global AA
        print('Extracting patches for layer',self.name)
        raw_patches, raw_originals, patch_indices = self._get_patches(phi, data)
        patch_ys=y[np.int32(patch_indices)]
        assert len(raw_patches), "No patches found"
        print('Training layer',self.name,'Patch dims',raw_patches.shape[2:])
        aa=self.train_from_samples(raw_patches, raw_originals, patch_ys)
        from Draw_means import show_visparts
        merge=False
        if (raw_originals.shape[-1]==3):
            merge=True
        show_visparts(self._visparts,self._num_orientations,10, merge, 100)
        if (self._settings['var_explained'] is not None):
            ii=range(data.shape[0])
            rs = np.random.RandomState()
            rs.shuffle(ii)
            X=self._extract(phi,data[ii[0:50],:])

            X=X.reshape((np.prod(X.shape[0:3]),X.shape[3]))
            CC=np.cov(X.T)
            eg,vv=np.linalg.eig(CC)
            cumeg=np.cumsum(eg)/np.sum(eg)
            ii=np.where(cumeg>self._settings['var_explained'])
            self._principal_vectors=vv[:,0:ii[0][0]]
            print('Number of principal vectors:',ii[0][0])
            self._principal_means=X.mean(axis=0)
        return(aa)

    def train_from_samples(self, raw_patches, raw_originals, y):
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
        seed = self._settings['seed']

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
        Xflat = raw_patches.reshape(raw_patches.shape[:2] + (-1,))
        from pnet.permutation_mm import PermutationMM

        num_class=1
        if self._settings['by_class']:
            num_class=np.max(y)+1
        if (num_class==1):
            y=np.zeros(len(y))
        num_features=np.float32(raw_patches.shape[-1])
        quant=np.maximum(np.float64(num_features),10)
        min_prob=1/(2*quant)
        print('QUANT',quant)
        mm = PermutationMM(n_components=self._num_true_parts,
                           permutations=permutations,
                           n_iter=n_iter,
                           n_init=n_init,
                           random_state=seed,
                           min_probability=min_prob)
        ntp=self._num_true_parts
        if (self._num_orientations==1):
            ntp=np.int32(self._num_true_parts/num_class)
        for c in range(num_class):
            mmc=PermutationMM(n_components=ntp,
                       permutations=permutations,
                       n_iter=n_iter,
                       n_init=n_init,
                       random_state=seed,
                       allocate=self._settings['allocate'],
                       min_probability=min_prob)
            if (self._settings['grad']):
                mmc.fit_grad(Xflat[y==c],y==c)
            else:
                mmc.fit(Xflat[y==c],y==c)
            mmc=self.organize_parts(Xflat[y==c,:],mmc,raw_originals[y==c,:],ntp)

            if (c==0):
                mm.means_=mmc.means_
                mm.weights_=mmc.weights_*np.float32(np.sum(y==c))/Xflat.shape[0]
                mm.visparts=mmc.visparts
                mm.counts=mmc.counts
                mm._num_true_parts=mmc._num_true_parts
            else:
                mm.means_=np.concatenate((mm.means_,mmc.means_), axis=0)
                mmc.weights_*=np.float32(np.sum(y==c))/Xflat.shape[0]
                mm.weights_=np.concatenate((mm.weights_,mmc.weights_))
                mm.visparts=np.concatenate((mm.visparts,mmc.visparts))
                mm.counts=np.concatenate((mm.counts,mmc.counts))
                mm._num_true_parts+=mmc._num_true_parts



        mm.means_=np.floor(mm.means_*quant)/quant+1./(2*quant)
        logits, logsum, lmeans, lstds, lpthreshs=self.get_loglike_stats(mm.means_,Xflat)
        IS=(lstds>0)
        print("Length of IS", np.sum(IS))
        print(lpthreshs)


        self._num_true_parts=mm._num_true_parts
        # Visualize parts : we iterate only over 'ok' ones


        ag.info('Training counts:', mm.counts)

        # Store info
        sh = (self._num_true_parts * P,) + raw_patches.shape[2:]
        self._num_true_parts=np.sum(IS)
        self._num_parts=self._num_true_parts*P

        #for n in range(self._num_parts):

        self._visparts=mm.visparts
        self._parts = mm.means_.reshape(sh)
        self._parts = self._parts[IS,:]
        self._parts_vis = self._parts.copy()
        #if (self._settings['raw']):
        if (1):
            self._lmeans=np.float32(lmeans[IS])
            self._lstds=np.float32(lstds[IS])
            self._lpthreshs=np.float32(lpthreshs[IS])
            self._logits=np.float32(logits[IS].reshape(sh))
            self._logsum=np.float32(logsum[IS])
        else:
            self._lmeans=None
            self._lstds=None
            self._lpthreshs=None
        self._train_info['counts_initial'] = mm.counts
        self._train_info['counts'] = mm.counts

        self._preprocess()

    def organize_parts(self,Xflat,mm,raw_originals,ml):
        # For each data point returns part index and angle index.
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
        return(mm)


    def get_loglike_stats(self,mm,Xflat):
        parts=mm
        parts=parts.reshape((parts.shape[0]*parts.shape[1],parts.shape[2]))
        logits = np.float32(np.log(parts / (1 - parts)))
        logsum = np.float32(np.log(1 - parts).sum(1))
        if self._settings['standardize']:
            logits=logits.T
            llm=np.mean(logits,axis=0)
            lls=np.std(logits,axis=0)
            logits=(logits-llm)/lls
            logits=logits.T
            logsum[:]=0
        lprs=np.dot(Xflat[:,0,:],logits.T)+logsum
        lmeans=lprs.mean(axis=0)
        lstds=lprs.std(axis=0)
        lpthresh=np.percentile((lprs-lmeans)/lstds,self._settings['lperc'],axis=0)
        return logits, logsum, lmeans, lstds, lpthresh

    def _get_patches(self, phi, data):
        samples_per_image = self._settings['samples_per_image']
        fr = self._settings.get('outer_frame', 0)
        # invA: This matrix has recursively update over the layers to get the affine transformation
        # that applied on the corners of a window in the current layer will provide the corners of the window in
        # original image layer
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
        rs = np.random.RandomState(self._settings['seed'])
        rs.shuffle(top_indices)
        data=data[top_indices,:]
        size = data.shape[1:3]
        # Make it square, to accommodate all types of rotations
        vis_avoid_edge=0 #np.int32(np.floor(np.max(visps)/2)) #
        avoid_edge = np.int32(np.floor(np.max(ps)/2)) #int(2 + np.max(ps)/2)
        #pad = [(new_size[i]-size[i])//2 for i in range(2)]
        new_side = np.max(size)
        new_size = [new_side + 2*vis_avoid_edge+(new_side - data.shape[1]) % 2,
                    new_side + 2*vis_avoid_edge+(new_side - data.shape[2]) % 2]



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
        E = phi(data[:1]).shape[-1]
        phi.pad=True
        C=data.shape[-1]
        c = 0

        batch_size = 1000

        max_samples = self._settings.get('max_samples', 10000)

        center_adjusts = [ps[0] % 2,
                              ps[1] % 2]
        patches_sh = ((max_samples, self._num_orientations) +
                      ps + (E,))
        the_patches = np.zeros(patches_sh)
        the_indices = -np.ones(max_samples)
        orig_sh = (max_samples, self._num_orientations) + visps+(C,)
        the_originals = np.zeros(orig_sh)


        # This must mean one iteration...
        for b in itr.count(0):
            data_subset = data[b*batch_size:(b+1)*batch_size]
            if data_subset.shape[0] == 0:
                break
            data_padded = ag.util.pad_to_size(data_subset, (-1, new_size[0], new_size[1],) + data.shape[3:])

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
            phi.layer.pad=False
            all_X = phi(all_data.reshape((-1,) + sh_image))
            phi.layer.pad=True
            all_X = all_X.reshape(sh_samples + all_X.shape[1:])

            X_shape = all_X.shape[2:4]

            # These indices represent the center of patches


            offset = (np.asarray(X_shape) - center_adjusts) / 2
            mmatrices = [pnet.matrix.translation(offset[0], offset[1]) *
                         pnet.matrix.rotation(a) *
                         pnet.matrix.translation(-offset[0], -offset[1])
                         for a in radians]
            range_x = range(avoid_edge, X_shape[0]-avoid_edge)
            range_y = range(avoid_edge, X_shape[1]-avoid_edge)
            msk=np.minimum(2,ps[0]//2)
            for n in range(len(all_data)):

                #print("Extracting from image",top_indices[b*batch_size+n])
                # all_img = all_data[n]
                X = all_X[n]
                alld=all_data[n]

                lindices = list(itr.product(range_x, range_y))
                rs.shuffle(lindices)
                C=alld.shape[-1]
                E = X.shape[-1]
                th = self._settings['threshold']
                # std_thresh = self._settings['std_thresh']
                mask=np.ones(X.shape[1:3])
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
                            tot = patch0.sum()
                            # std = patch0.std()
                        else:
                            tot = patch0[fr:-fr, fr:-fr].sum()
                            # std = patch0[fr:-fr, fr:-fr].std()
                        # if std_thresh <= std:
                        if th <= tot:
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
                                if (slic[0].start>=0 and slic[0].stop<=X.shape[1] and slic[1].start>=0 and slic[1].stop<=X.shape[2]):
                                #try:
                                    patch[ori] = X[selection]
                                #except:
                                    #print('Skipping', selection)
                                #    br = True
                                #    break
                                #try:
                                if (visslic[0].start>=0 and visslic[0].stop<=alld.shape[1] and visslic[1].start>=0 and visslic[1].stop<=alld.shape[2]):
                                    vispatch[ori] = alld[visselection]
                                # except:
                                #     #print(visselection)
                                #     br=True
                                #     break
                                sels.append(selection)
                            # if br:
                            #     continue
                            shift = 0 #rs.randint(ORI)
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

    def _create_extract_func(self):

        import theano.tensor as T
        import theano
        from theano.tensor.nnet import conv

        # Create Theano convolution function
        s_input = T.tensor4(name='input', dtype='float32')

        gpu_parts = self._parts.transpose((0, 3, 1, 2)).astype(s_input.dtype, copy=False)
        gpu_parts = (gpu_parts)[:, :, ::-1, ::-1]

        gpu_logits = np.log(gpu_parts / (1 - gpu_parts))
        gpu_rest = np.log(1 - gpu_parts).sum(1).sum(1).sum(1)

        s_logits = theano.shared(gpu_logits, name='logits')
        s_rest = theano.shared(gpu_rest, name='rest')

        s_conv = (conv.conv2d(s_input, s_logits) +
                  s_rest.dimshuffle('x', 0, 'x', 'x'))
        if (not self._settings['raw']):
            s_output = s_conv.argmax(1)
        else:
            s_output = s_conv
        self._dtype = s_input.dtype
        return theano.function([s_input], s_output)

    def _preprocess(self):
        self._dtype='float32'
        return
        #self._extract_func = self._create_extract_func()

    def _vzlog_output_(self, vz):
        from pylab import cm

        if self._train_info:
            vz.log('Initial counts:', self._train_info['counts_initial'])
            vz.log('Final counts:', self._train_info['counts'])

        grid1 = ImageGrid(self._visparts, vmin=0, vmax=1)
        grid1.save(vz.impath(), scale=4)

        # Plot all the parts
        if hasattr(self, '_parts_vis'):
            vz.log('parts', self._parts.shape)
            grid2 = ImageGrid(self._parts_vis.transpose((0, 3, 1, 2)),
                              vmin=0, vmax=1, cmap=cm.jet,
                              border_color=1)
            grid2.save(vz.impath(), scale=3)

        grid2 = ImageGrid(self._parts.transpose((0, 3, 1, 2)),
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
        d['lmeans'] = self._lmeans
        d['lstds'] = self._lstds
        d['lpthreshs']=self._lpthreshs
        d['logits']=self._logits
        d['logsum']=self._logsum
        d['principal_vectors']=self._principal_vectors
        d['principal_means']=self._principal_means
        d['visparts'] = self._visparts
        d['weights'] = self._weights
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(settings=d['settings'])
        obj._parts = d['parts']
        obj._lmeans = d['lmeans']
        obj._lstds = d['lstds']
        obj._lpthreshs=d['lpthreshs']
        obj._logits=d['logits']
        obj._logsum=d['logsum']
        obj._principal_vectors=d['principal_vectors']
        obj._principal_means=d['principal_means']
        obj._num_true_parts=d['num_true_parts']
        obj._num_parts=obj._num_true_parts*obj._num_orientations
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
