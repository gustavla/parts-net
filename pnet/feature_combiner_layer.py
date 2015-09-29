from __future__ import division, print_function, absolute_import

import numpy as np
import pnet
from pnet.layer import Layer
from pnet.layer import SupervisedLayer
import multiprocessing as mp
import itertools
import copy
import sys

def do_trains((layers,X,data,y)):
    pass

def get_pcl_cls(layer,grain,):
    pass

def do_train((layers,X,data,y,grain)):

        for c, la in enumerate(layers):
            pcl=la.pcl
            cls=la.cls
            if (grain>0):
                Ys=np.in1d(y,cls)
                print('Class 1',cls)
                la._train(None,X, Ys,None,data)
            elif (grain<0):
                mask=np.in1d(y,cls)
                Xs=X[ mask ]
                Ys=y[ mask ]
                Ys=np.int32(np.array([Ys==pcl])).squeeze()
                num_class=None
                la._train(None,Xs,Ys,num_class,data)
            else:
                la._train(None,X,y)

        return(layers)

def comp_pool(fe,pool,strides):

    #print('In pool')
    pool2=pool*pool
    dim=fe.shape[1:3]
    num_feat=fe.shape[3]
    xlist=np.arange(0,dim[0],strides)
    ylist=np.arange(0,dim[1],strides)
    newdim=(len(xlist),len(ylist))
    outfe=np.zeros((fe.shape[0],)+newdim+(num_feat,),dtype=np.float16)
    for ox, x in enumerate(xlist):
        for oy, y in enumerate(ylist):
                upx=np.minimum(x+pool,dim[0])
                upy=np.minimum(y+pool,dim[1])
                pool2=(upx-x)*(upy-y)
                Z=np.reshape(fe[:,x:upx,y:upy,:],(fe.shape[0],pool2,num_feat))
                outfe[:,ox,oy,:]=np.max(Z,axis=1)
    return outfe

def _do_extract((X,layers,num_class,pool,strides)):
    feat=[]
    #print('In do extract - number of layers',len(layers), X.shape)
    #for r in range(len(layers)):
    #    fe=layers[r]._extract(None,X,num_class)
    #    feat.append(fe)
    #print('In do extract')
    featout=layers._extract(None,X,num_class)

    #featout=np.concatenate(feat,axis=3)
    if (pool>1):
        featout=comp_pool(featout,pool,strides)

    return featout

@Layer.register('feature-combiner-layer')
class FeatureCombinerLayer(SupervisedLayer):
    def __init__(self, settings={},layers=[],pool=None):

        self._settings = dict(
                              C=.001,
                              sub_region=None,
                              strides=None,
                              num_pool=8,
                              grain=1,
                              smooth_score_output=1.,
                              num_copies_per_layer=1,
                              num_keep_per_layer=1,
                              sgd_batch_size=None,
                              num_sgd_rounds=1,
                              num_components=1,
                              num_bfgs_iters=100,
                              num_sub_region_samples=20,
                              num_sgd_iters=1,
                              num_train=None,
                              num_class=None,
                              rectify=False,
                              pool=1,
                              boost=None,
                              pool_stride=1,
                              seed=None,
                              test_prop=None,
                              soft_thresh=None,
                              max_pool=None,
                              initial_delta=.0001,
                              penalty='L2',
                              prune='False',
                              mine=True
                               )
        for k, v in settings.items():
            if k not in self._settings:
                    raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._layers = layers
        self.num_pool=self._settings['num_pool']
        self.pool=pool
        self._trained = False
        if (self.layers != []):
            self._trained=True
        self.random_state=np.random.RandomState(self._settings['seed'])
    @property
    def layers(self):
        return self._layers

    @property
    def trained(self):
        return self._trained

    def reset(self, num_train, num_pool):
        self._settings['num_train']=num_train
        self._settings['num_pool']=num_pool
        self.num_pool=num_pool


    def _train(self, phi, data, y):


        if (self._settings['num_train'] is not None):
            data=data[0:self._settings['num_train'],:]
            y=y[0:self._settings['num_train']]


        num_class=np.max(y)+1

        X = phi(data)
        #Train a bunch of SVMs on different regions and on different subsets of classes
        if (self._settings['sub_region'] is not None):
            self._layers=[]
            ps=self._settings['sub_region']
            str=self._settings['strides']
            dims=X.shape[1:3]
            grain=self._settings['grain']
            num_copies=self._settings['num_copies_per_layer']
            newdims=(dims[0]-ps[0]+1, dims[1]-ps[1]+1)
            num_regions=newdims[0]*newdims[1]
            #Pick all subsets of -grain classes and then a random class of these to be the positive class.
            if (grain<-1):
                s = set(range(num_class))
                subset_list=list(itertools.combinations(s, -grain))
                num_copies=len(subset_list)
                if (self._settings['num_copies_per_layer']==np.abs(grain)):
                    num_copies*=(-grain)
            elif (grain==-1): # Use one against the rest.
                subset_list=[]
                for i in range(num_class):
                    subset_list.append(range(num_class))
                num_copies=len(subset_list)
            elif (grain>1):
                s = set(range(num_class))
                subset_list=list(itertools.combinations(s, grain))
                num_copies=len(subset_list)
            # Append all the SVMs as sublayers of this layer.
            for ix in range(0,newdims[0],str):
                for iy in range(0,newdims[1],str):
                    for cop in range(num_copies):
                        self._layers.append(pnet.SVMClassificationLayer(settings=dict(C=self._settings['C'],standardize=False,
                                                                                  num_sgd_iters=self._settings['num_sgd_iters'],
                                                                                  smooth_score_output=self._settings['smooth_score_output'],
                                                                                  sgd_batch_size=self._settings['sgd_batch_size'],
                                                                                  num_sgd_rounds=self._settings['num_sgd_rounds'],
                                                                                  num_bfgs_iters=self._settings['num_bfgs_iters'],
                                                                                  num_components=self._settings['num_components'],
                                                                                  allocate=True,
                                                                                  boost=self._settings['boost'],
                                                                                  sub_region=(ix,ix+ps[0],iy,iy+ps[1]),
                                                                                  sub_region_range=self._settings['strides'],
                                                                                  num_sub_region_samples=self._settings['num_sub_region_samples'],
                                                                                  seed=None,
                                                                                  test_prop=self._settings['test_prop'],
                                                                                  max_pool=self._settings['max_pool'],
                                                                                  initial_delta=self._settings['initial_delta'],
                                                                                  penalty=self._settings['penalty'],
                                                                                  mine=self._settings['mine']
                                                                                  #seed=self.random_state
                                                                                  )))
        for j, la in enumerate(self._layers):
            la.round=0
            self.get_pcl_cls(la,j,grain,subset_list)

        self.train_all_layers_once(num_class, X, y, num_copies, subset_list, grain)
        # You've got lots of data so just use first batch

        temp_layer=self.combine_all_layers()
        if (self._settings['prune']):
            aa=temp_layer._svm.coef_.reshape(temp_layer._svm.coef_.size/X.shape[-1],X.shape[-1])
            bb=np.sum(np.abs(aa),axis=0)
            cc=np.sort(bb)
            thr=cc[-64]
            ii=np.where(bb>=thr)
            self.III=ii
            print('III', len(ii[0]))
            X=X[...,ii]
            for la in self._layers:
                la._svm=None
            self.train_all_layers_once(num_class, X, y, num_copies, subset_list, grain)

        # if (grain < 0 ):
        #     num_final=self._settings['num_keep_per_layer']
        #     self.prune_layers(newdims,str,len(self._layers),num_final,num_class)

        self._trained = True

    def get_pcl_cls(self,layer,ind,grain,subset_list):
        all_singles=False
        if (self._settings['num_copies_per_layer']==-grain):
            all_singles=True
        lli = np.int16(np.floor(ind/self._settings['num_copies_per_layer']))
        cls=subset_list[lli]
        pcl=cls[0]
        if (grain<0):
            if (len(cls)>2):
                if (grain < -1):
                    if all_singles:
                        pcl=np.mod(ind,-grain)
                    else:
                        pcl=self.random_state.randint(0,len(cls))
                    pcl=cls[pcl]
            else:
                pcl=cls[0]
        #print('Layer',ind,'cls',cls,'pcl',pcl)
        layer.pcl=pcl
        layer.cls=cls




    def train_all_layers_once(self, num_class, X,y, num_copies, subset_list, grain):
            rs = self.random_state #np.random.RandomState()
            self.round=0

            print('Number of layers', len(self._layers))
            print('DIMS',X.shape[1:3],X.shape,'SIZE',X.nbytes)
            ngig=X.nbytes*5*8/1000000000
            nx=np.ceil(ngig/32)

            print('ngig',ngig,'NNNNXXXX',nx)
            ii=range(X.shape[0])
            num_pool=self.num_pool
            num_r=self._settings['num_sgd_rounds']
            per_pool=np.int16(len(self._layers)/num_pool)
            if (self._settings['mine']):
                num_r=1
            for n in range(num_r):
                rs.shuffle(ii)
                X=X[ii]
                y=y[ii]
                XX=np.array_split(X,nx)
                yy=np.array_split(y,nx)
                print('ROUND',n)
                for ix, XXX in enumerate(XX):
                    #XXX=XX[ix]
                    print('IIIXXX',ix,'shape',XXX.shape)
                    curr_range=np.arange(0, len(self._layers), per_pool*num_pool)
                    for r in curr_range:
                        layersS=[]
                        for p in range(num_pool):
                            rr=np.arange(r+p*per_pool, np.minimum(r+(p+1)*per_pool,len(self._layers)))
                            layers=[]
                            for j in rr:
                                layers.append(self._layers[j])
                            layersS.append(layers)
                        args=((layersS[k],XXX,None,yy[ix],grain) for k in range(num_pool))

                        if (self.pool is not None and self.num_pool>1):
                             layersS=self.pool.map(do_train,args)
                        else:
                            layersS=map(do_train,args)
                        c=0
                        for p in range(num_pool):
                            for la in layersS[p]:
                                self._layers[r+c]=la
                                if (ix==nx-1):
                                    self._layers[r+c].round=n+1
                                c=c+1

            # Remove data from layers.
            if (not self._settings['mine']):
                for c,la in enumerate(self._layers):
                    mask=np.in1d(y,la.cls)
                    Xs=X[ mask ]
                    Ys=y[ mask ]
                    Ys=np.int32(np.array([Ys==la.pcl])).squeeze()
                    Xflat_tr, ytr, Xflat_te, yte=la.get_subregion_samples(Xs,Ys,Xs.shape[0], 0)
                    yhat=la._svm.predict(Xflat_tr)
                    error=np.float32(np.sum(yhat!=ytr))/Xflat_tr.shape[0]

                    print('Round',la.round, 'La',c,'cls',la.cls,'pcl',la.pcl,'loc',la._settings['train_region'],'error',error)
                    ns=np.std(la._svm.coef_[0,:])
                    print('Number of coeficients non-zero', ns, np.size(la._svm.coef_), np.sum(np.abs(la._svm.coef_) <.2*ns), np.sum(np.abs(la._svm.coef_)>.001))



    def prune_layers(self,ii,num_final,num_class):

        t=0
        new_layers=[]
        for i in enumerate(ii):
            new_layers.append(self.layers[i])
        #ldel=np.zeros(len(self._layers))
        # errors=[]
        # index=[]
        # for c in range(num_class):
        #     errors.append([])
        #     index.append([])
        # for i, lay in enumerate(self._layers):
        #     errors[lay.pcl].append(lay._svm.error)
        #     index[lay.pcl].append(i)
        # for ix in range(0,newdims[0],str[0]):
        #         for iy in range(0,newdims[1],str[1]):
        #             tbase=t
        #             #errors=np.zeros(num_copies)
        #             for cop in range(num_copies):
        #                 errors[cop]=self._layers[t]._svm.error
        #                 t=t+1
        #             ii=np.argsort(errors)
        #             for j in range(num_f):
        #                 ldel[tbase+ii[j]]=1
        # new_layers=[]
        # for n, ee in enumerate(errors):
        #     ii=np.argsort(ee)
        #     num_f=np.minimum(num_final,len(ee))
        #     for i in range(num_f):
        #         new_layers.append(self.layers[index[n][ii[i]]])
        # for t in range(len(self._layers)):
        #     if ldel[t]==1:
        #         new_layers.append(self._layers[t])

        self._layers=new_layers

    def combine_all_layers(self):
            temp_layer=copy.deepcopy(self._layers[0])

            coef=[]
            cdim=self._layers[0]._svm.coef_.shape[0]
            doubl=False
            # Only two classes just need first vector of weights, the other is just the negative.
            if (cdim==2 and np.max(np.abs(self._layers[0]._svm.coef_[0,:]+self._layers[0]._svm.coef_[1,:]))<.0001):
                intercept=np.zeros(len(self._layers),dtype=np.float16)
                temp_layer._svm.classes_=range(len(self._layers))
                doubl=True
        # More classes keep all weights.
            else:
                intercept=np.zeros(cdim*len(self._layers),dtype=np.float16)
                temp_layer._svm.classes_=range(len(self._layers)*cdim)
            for l, layer in enumerate(self._layers):
                if (doubl):
                    cc=np.float16(layer._svm.coef_[0,:][np.newaxis,:])
                    intercept[l]=np.float16(layer._svm.intercept_[0])
                else:
                    cc=np.float16(layer._svm.coef_)
                    intercept[l*cdim:(l+1)*cdim]=np.float16(layer._svm.intercept_)
                coef.append(cc)

            temp_layer._svm.coef_=np.concatenate(coef,axis=0)
        #print('coef corr',np.corrcoef(temp_layer._svm.coef_))
            temp_layer._svm.intercept_=intercept
            return temp_layer

    def _extract(self, phi, data):
        X = phi(data)
        X=np.float16(X)
        feats = []
        if (self.pool is None and self.num_pool>1):
               print('Creating pool in combiner layer')
               self.pool=mp.Pool(self.num_pool)
        # Createa temporary layer that will be the union of all the trained layers (LGM) in this feature combiner layer
        temp_layer=self.combine_all_layers()
        num_super_subs=np.maximum(X.shape[0]//5000,1)
        num_subs=np.maximum(8,self.num_pool)
        XS=np.array_split(X,num_super_subs)
        num_class=None
        # The classes sent to the SVMs are real not aggregates.
        # if (self._settings['grain']<0):
        #     num_class=self._settings['num_class']
        for i, Xs in enumerate(XS):
            features=[]
            X_subs=np.array_split(Xs,num_subs)
            print('Extract',self.num_pool, self.pool)
            #args=((X_s,)+(self._layers,num_class,self._settings['pool'],self._settings['pool']) for X_s in X_subs)
            args=((X_s,)+(temp_layer,num_class,self._settings['pool'],self._settings['pool_stride']) for X_s in X_subs)
            if (self.pool is not None):
                 pfeatures=self.pool.map(_do_extract,args)
            else:
                 pfeatures=map(_do_extract,args)
            # for pf in pfeatures:
            #     features.append(pf)
            fe=np.concatenate(pfeatures,axis=0)
            if (self._settings['rectify']):
                if (self._settings['soft_thresh'] is None):
                    fe=fe.clip(min=0)
                else:
                    thr=self._settgings['soft_thresh']
                    fe[np.abs(fe)<thr]=0

            if i==0:
                featout=np.zeros((num_super_subs*fe.shape[0],)+fe.shape[1:], dtype=np.float16)
                print('Final size',featout.nbytes/1000000000)
            featout[i*fe.shape[0]:(i+1)*fe.shape[0],:]=fe
            #feats.append(fe)
            print('Batch',i,'Min',np.min(fe), 'Size', fe.nbytes/1000000)
        print('finished parallel processing going to concatenate')
        #featout=np.concatenate(feats,axis=0)
        print('Final size',featout.nbytes/1000000000)
        print('Output of layer',featout.shape[1:])
        return featout

    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        d['layers'] = []
        for layer in self._layers:
            if ('seed' in layer._settings):
                layer._settings['seed']=None
            layer_dict = layer.save_to_dict()
            layer_dict['name'] = layer.name
            d['layers'].append(layer_dict)
        return d

    @classmethod
    def load_from_dict(cls, d):
        layers = [
            Layer.getclass(layer_dict['name']).load_from_dict(layer_dict)
            for layer_dict in d['layers']
        ]
        obj = cls(d['settings'],layers)
        return obj

    def __repr__(self):
        return 'FeatureCombinerLayer({})'.format(num_parts=self._layers)
