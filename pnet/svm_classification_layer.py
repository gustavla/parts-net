from __future__ import division, print_function, absolute_import
from pnet.layer import SupervisedLayer
from pnet.layer import Layer
import numpy as np
import itertools as itr
import amitgroup as ag
from sklearn.utils import check_random_state
from sklearn.svm import LinearSVC as SVM
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _multinomial_loss_grad
from sklearn.linear_model.logistic import _multinomial_loss_grad_hess
from sklearn.linear_model.logistic import _multinomial_loss
from scipy import optimize, sparse
from sklearn.utils.extmath import (safe_sparse_dot, logsumexp, squared_norm)
import copy
from pnet.permutation_mm import PermutationMM
from Draw_means import show_visparts
import matplotlib.pyplot as py

def decision_function(svm,X):

    scores=np.dot(X,svm.coef_.T)+svm.intercept_
    return(scores)

def get_probs(svm,Xflat, beta):
        scores=svm.decision_function(Xflat)
        scores*=beta
        lss=logsumexp(scores,axis=1)
        lss=lss[:,np.newaxis]
        lscores=scores-lss
        scores=np.exp(lscores,lscores)
        #np.exp(scores, scores)
        #scores /= scores.sum(axis=1).reshape((scores.shape[0], -1))
        return scores

def multinomial_loss(w, X, Y, alpha, sample_weight, max_pool=None, penalty='L2'):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    per_sample=X.shape[0]/Y.shape[0]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    max_list=None
    XX=X
    if (per_sample>1 and max_pool is not None and n_classes==2):
        if max_pool:
            pt=p.T
            # Get maximum output in region for first class and clip at 0.
            pp=pt[0,:].reshape((-1,per_sample)).clip(0)
            ppp=np.max(pp,axis=1)
            max_listp=np.argmax(pp,axis=-1)
            # Get maximum output in region for second class and clip at 0.
            qq=pt[1,:].reshape((-1,per_sample)).clip(0)
            qqq=np.max(qq,axis=1)
            max_listq=np.argmax(qq,axis=-1)
            # First class maxes are positive.
            zz=(ppp>0)
            # Create list of sub-sample points that achieved the maximum for each sample
            max_list=np.zeros(len(ppp))
            max_list[zz]=max_listp[zz]
            max_list[~zz]=max_listq[~zz]
            ii=np.array(range(len(max_list)))
            ml=np.int64((ii*per_sample)+max_list)
            # Data at sub-sample points that achieved maximum
            XX=X[ml,:]
            po=np.zeros((len(zz),2))
            po[zz,:]=np.concatenate((ppp[zz,np.newaxis],-ppp[zz,np.newaxis]),axis=1)
            po[~zz,:]=np.concatenate((-qqq[~zz,np.newaxis],qqq[~zz,np.newaxis]),axis=1)
            p=po
        else:
            p=p.T
            pp=np.sum(p.reshape((p.shape[0],-1,per_sample)),axis=2)
            p=pp.T

    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    if (penalty=='L2'):
        loss += 0.5 * alpha * squared_norm(w)
    else:
        loss += alpha*np.sum(np.abs(w))
    p = np.exp(p, p)
    return loss, p, w, XX


def multinomial_loss_grad(w, X, Y, alpha, sample_weight, max_pool=None, penalty='L2'):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : ndarray, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    """


    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)))
    loss, p, w, XX = multinomial_loss(w, X, Y, alpha, sample_weight, max_pool, penalty=penalty)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)

    grad[:, :n_features] = safe_sparse_dot(diff.T, XX)
    if (penalty=='L2'):
        grad[:, :n_features] += alpha * w
    else:
        grad[:, :n_features] += alpha*np.sign(w)
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def logistic_reg(X, y, classes, C=1., Delta=1.,fit_intercept=True, coef=None, intercept=None,
                             max_iters=100, sample_weight=None, penalty='L2', tol=1e-4, verbose=0,max_pool=None):


    #print('C',C,'num data in step',X.shape[0])


    # sample_weight = np.ones(X.shape[0])#/X.shape[0]
    # if (classes.size==2):
    #     prop=np.int16(np.sum(y[:,0])/np.sum(y[:,1]))
    #     sample_weight[y[:,1]==1]=prop
    #sample_weight=sample_weight/np.sum(sample_weight)
    Y_bin = y
    w0 = np.zeros((Y_bin.shape[1], X.shape[1] + int(fit_intercept)))
    if coef is not None:
        w0[:, :coef.shape[1]] = coef
    if intercept is not None:
        w0[:,coef.shape[1]]=intercept
    # Simple gradient descent move with a given Delta
    wold=w0
    #print('loss before',_multinomial_loss(w0,X,Y_bin,C,sample_weight)[0])
    if (max_iters==1):

        ll,ww,p = multinomial_loss_grad(w0,X,Y_bin,C,sample_weight,max_pool, penalty=penalty)
        w0=w0-Delta*ww.reshape(w0.shape)
        #print('loss ',_multinomial_loss(w0,X,Y_bin,C,sample_weight)[0])
    # Solve logistic minimization.
    else:
        #w0 = w0.ravel()
        target = Y_bin
        func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        w0, loss, info = optimize.fmin_l_bfgs_b(
            func, w0, fprime=None,
            args=(X, target, C, sample_weight),
            iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iters
            )
        #
        w0 = np.reshape(w0, (classes.size, -1))
        # if classes.size == 2:
        #     multi_w0 = multi_w0[1][np.newaxis, :]
    if (0):
        print('loss after',_multinomial_loss(w0,X,Y_bin,C,sample_weight)[0])
    newcoef=w0[:,:X.shape[1]]
    newintercept=w0[:,X.shape[1]]

    return newcoef, newintercept


@Layer.register('svm-classification-layer')\




class SVMClassificationLayer(SupervisedLayer):
    def __init__(self, settings={}):

        self._settings = dict(standardize=False,
                              C=100,
                              epsilon=.01,
                              sub_region=None,
                              num_sub_region_samples=20,
                              sub_region_range=None,
                              num_train=None,
                              num_sgd_iters=0,
                              num_bfgs_iters=None,
                              num_sgd_rounds=1,
                              sgd_batch_size=None,
                              smooth_score_output=None,
                              boost=None,
                              seed=None,
                              simple_aggregate=False,
                              num_components=1,
                              allocate=True,
                              min_count=2,
                              test_prop=None,
                              max_pool=None,
                              initial_delta=.0001,
                              penalty='L2',
                              prune=False,
                              mine=True,
                              train_region=None
                               )


        for k, v in settings.items():
            if k not in self._settings:
                    raise ValueError("Unknown settings: {}".format(k))
            else:
                self._settings[k] = v
        self._penalty = self._settings['C']
        self._extra = {}
        self._svm = None
        self.epsilon = self._settings['epsilon']
        self.sub_region=self._settings['sub_region']
        if (type(self._settings['seed']) is not np.random.RandomState):
            self.random_state= np.random.RandomState(self._settings['seed'])
        else:
            self.random_state=self._settings['seed']
        self._trained=False
        if (self._svm is not None):
            self._trained=True

    @property
    def trained(self):
        return self._trained

    @property
    def classifier(self):
        return True

    def reset(self, num_train):
        self._svm=None
        self._settings['num_train']=num_train

    def _extract(self, phi, data, numcl=None):
        if (phi is not None):
            X = phi(data)
        else:
            X=data
# From mixture classification layer:
        num_comp=self._settings['num_components']
        if (isinstance(X,tuple)):
            ysub=X[1]
            num_comp=np.max(ysub)+1
            X=X[0]
        if (self.sub_region is not None):

            # Don't expand margin if subregion is exactly the size of the data.
            ps=(self.sub_region[1]-self.sub_region[0],self.sub_region[3]-self.sub_region[2])
            correction=np.mod(ps,2)
            halfps=np.array(ps)//2
            dim = (X.shape[1], X.shape[2])
            reg=self._settings['train_region']
            lowx=0 #reg[0]
            lowy=0 #reg[2]
            highx=dim[0] #reg[1]
            highy=dim[1] #reg[3]


            # if (ps[0]==dim[0] and ps[1]==dim[1]):
            #     correction=ps
            #     halfps=(0,0)
            #     dim=(1,1)
            # Extend domain to end up with the same dimension as the input after convolving with the filters.
            XX=np.zeros((X.shape[0],X.shape[1]+ps[0]-correction[0],X.shape[2]+ps[1]-correction[1], X.shape[3]),dtype=np.float16)
            XX[:,halfps[0]:XX.shape[1]-halfps[0],halfps[1]:XX.shape[1]-halfps[1],:]=X
            X=XX

            if (numcl is None):
                numclass=len(self._svm.classes_)
            else:
                numclass=numcl
            if (numclass>2):
                feature_map = np.zeros((X.shape[0],) + dim + (numclass,),dtype=np.float16)
            else:
                # Only one vector for a two class problem
                feature_map = np.zeros((X.shape[0],) + dim + (1,),dtype=np.float16)
            xrange=np.arange(lowx,highx,1)
            yrange=np.arange(lowy,highy,1)
            for x in xrange:
                for y in yrange:
                    Xr=X[:,x:x+ps[0],y:y+ps[1],:]
                    Xflat = Xr.reshape((X.shape[0], -1)).astype(np.float64)
                    scores=self._svm.decision_function(Xflat)
                    if (numcl is not None):
                        if (len(self._svm.classes_)==2):
                            scores=np.hstack((scores[:,np.newaxis],-scores[:,np.newaxis]))
                        if (len(self._svm.classes_)<numclass):
                            newscores=np.zeros((X.shape[0],numclass),dtype=np.float16)
                            newscores[:,self._svm.classes_]=np.float16(scores)
                    else:
                        # Take care of different output formats for the linear classifier
                        if (len(self._svm.classes_)==2):
                            # Make output from SVM into an nx1 arrray
                            if (scores.shape[1]==1):
                                newscores=np.float16(scores[:,np.newaxis])
                            # Output from logistic regression is 2 dim +/- use only first dim
                            else:
                                newscores=np.float16(scores[:,0])
                                newscores=newscores[:,np.newaxis]
                        else:
                            newscores=np.float16(scores)
                    if (self._settings['smooth_score_output'] is not None and numcl is not None):
                        scores*=self._settings['smooth_score_output']
                        np.exp(newscores, newscores)
                        newscores /= newscores.sum(axis=1).reshape((newscores.shape[0], -1))
                    feature_map[:,x,y,:]=newscores
            return feature_map
        else:
            Xflat = X.reshape((X.shape[0], -1))
            if self._settings.get('standardize'):
                means = self._extra['means']
                variances = self._extra['vars']
                Xflat = (Xflat - means) / np.sqrt(variances + self.epsilon)
            if (self._settings['smooth_score_output'] is not None):
                scores=get_probs(self._svm,Xflat, self._settings['smooth_score_output'])
                yhat=np.argmax(scores,axis=1)
            else:
                yhat=self._svm.predict(Xflat)
            if num_comp > 1:
                yhat=np.int32(yhat//num_comp)
            return yhat

    def adapt_component_classes(self, y):
        num_comp=self._settings['num_components']
        if (num_comp>1):
            true_component_factor=2
            #clf=self.ovr_sgd(num_comp,X,y)
            num_class=np.max(y)+1
            YS=np.zeros((y.size,num_class))
            for c in range(num_class):
                ii=np.where(y==c)
                true_class=c//num_comp
                sister_classes=np.arange(true_class*num_comp,(true_class+1)*num_comp)
                ee=np.zeros(num_class)
                ee[sister_classes]=(1-true_component_factor/num_comp)/(num_comp-1)
                ee[c]=true_component_factor/num_comp
                YS[ii,:]=ee
        else:
            y=np.int32(y)
            num_class=np.max(y)+1
            YS=np.zeros((y.size,num_class))
            v=(range(y.size),y)
            YS[v]=1
        return YS


    def run_default_minimization(self,X,YS,y, sample_weight, classes, coef, intercept):
            clf_def=LogisticRegression(multi_class='multinomial', solver='lbfgs')
            clf_def.classes_=classes
            clf_def.coef_=coef
            clf_def.intercept_=intercept
            X=np.float64(X)
            newcoef, newintercept=logistic_reg(X, YS, classes, C=10,Delta=1, max_iters=100, coef=clf_def.coef_, intercept=clf_def.intercept_, sample_weight=sample_weight)
            clf_def.coef_=newcoef
            clf_def.intercept_=intercept
            # yhat=clf_def.predict(X)
            # yerror=(y!=yhat)
            # error_def=np.sum(sample_weight*yerror)/np.sum(sample_weight)
            # print('Default Error on training',error_def)
            return clf_def

    def _train_sgd(self,X,X_te,y,y_te, sample_weight, num_class=None):
        do_mine=self._settings['mine']
        #ag.info('Entering SVM-SGD training with X.shape = ', X.shape)
        if (X.dtype == 'float64'):
            print("IN SVM TRAIN",X.size*8)

# Set parameters of training size, number of rounds, number of iterations.
        if (self._settings['sgd_batch_size'] is not None):
            num_batches=np.int(np.ceil(y.shape[0]/self._settings['sgd_batch_size']))
        max_iters=1
        # This means do the full logistic minimization
        if (self._settings['num_bfgs_iters'] is not None):
            max_iters=self._settings['num_bfgs_iters']
        num_rounds=self._settings['num_sgd_rounds']
        if (do_mine):
            print('Number of training batches for SGD svm',num_batches,'Number of rounds', num_rounds)
        # More compnents per class than 1 use one vs. rest which ignores other components in same class.
        YS=self.adapt_component_classes(y)
        num_class=np.max(y)+1
        rs=self.random_state
        classes=np.unique(y)
        max_pool=None
        per_sample=X.shape[0]/YS.shape[0]
        if (per_sample>1):
            max_pool=True
        # Initialize with weights that are one on correct class and 0 elsewhere. For aggregating outputs of other classifiers.
        if (self._settings['simple_aggregate']):
            coef=np.zeros((num_class,X.shape[1]))
            for c in range(num_class):
                coef[c,np.arange(c,X.shape[1],num_class)]=1.
            intercept=np.zeros(num_class)
        else:
            if (self._svm is not None and self._svm.coef_ is not None):
                coef=self._svm.coef_
                intercept=self._svm.intercept_
            else:
                if (num_class>2 or do_mine):
                    if (num_class>2):
                        coef=(rs.rand(num_class,X.shape[1])-.5)*.005
                        intercept=np.zeros(num_class) # For SGD
                    else:
                        coef=np.zeros((num_class,X.shape[1]))
                        coef[0,:]=(rs.rand(X.shape[1])-.5)*.005
                        coef[1,:]=-coef[0,:]
                        intercept=np.zeros(num_class)
                else:
                    coef=(rs.rand(X.shape[1])-.5)*.005
                    intercept=(0,)
        if (0):
            n=0
            coef=np.zeros((num_class,X.shape[1]))
            clf=self.run_default_minimization(X, YS, y, sample_weight, classes, coef, intercept)
        else:
            if (self._svm is None):
                if (num_class>2 or do_mine):
                    clf=LogisticRegression(multi_class='multinomial', solver='lbfgs')
                else:
                    clf = SGDClassifier(alpha=self._penalty, class_weight=None, epsilon=0.1,
                            eta0=0.0, fit_intercept=True, l1_ratio=1.,
                            learning_rate='optimal', loss='log', n_iter=1, n_jobs=1,
                            penalty=self._settings['penalty'], power_t=0.5, random_state=None, shuffle=True,
                            verbose=0, warm_start=True)
                clf.num_rounds=0
            else:
                clf=self._svm

            if (num_class==2 and not do_mine):
                YS=y # For SGD
            clf.classes_=classes
            clf.coef_=coef
            clf.intercept_=intercept

            yyy=np.repeat(y,per_sample)
            ii=range(YS.shape[0])
            StartDelta=.0001/np.maximum(self._settings['C'],1)
            StepDelta=self._settings['initial_delta'] #.0001

            # Rounds through all data
            XT=X.reshape(len(ii),-1)
            if (not do_mine):
                clf.partial_fit(XT,YS)
            else:
                for n in range(num_rounds):
                    rs.shuffle(ii)
                    iii=ii
                    #print('Iteration',n)
                    Xbs=np.array_split(XT[iii],num_batches)
                    Ys=np.array_split(YS[iii],num_batches)
                    Smws=np.array_split(sample_weight[iii],num_batches)
                    #print('Delta',Delta)
                    # Loop through batches

                    coef=clf.coef_
                    intercept=clf.intercept_
                    for j in (range(len(Xbs))):
                        Delta=StartDelta/((n*len(Xbs)+j)*StepDelta+1.)
                        #print(n*len(Xbs)+j,Delta)
                        Xbsr=Xbs[j].reshape(Xbs[j].shape[0]*per_sample,-1)
                        if (not do_mine):
                            #print('Standard SGD partial fit')
                            clf.partial_fit(Xbsr, Ys[j],classes)
                        else:
                            newcoef, newintercept=logistic_reg(Xbsr, Ys[j], classes, C=self._settings['C'],Delta=Delta, max_iters=max_iters, coef=coef, intercept=intercept,
                                                      sample_weight=Smws[j],max_pool=max_pool, penalty=self._settings['penalty'])
                            coef=newcoef
                            intercept=newintercept
                    if (do_mine):
                        clf.coef_=coef
                        clf.intercept_=intercept
                    clf.num_rounds+=1
                if (0):
                    yhat=clf.predict(X)
                    print('round',n,np.mean((yhat!=yyy)))
        # if (num_class==2 and ~do_mine):
        #     ss=np.std(clf.coef_)
        #     clf.coef_=(clf.coef_)/ss
        #     clf.intercept_=(clf.intercept_)/ss
        if (X_te is not None):
            XX=X_te
            yy=y_te
        else:
            XX=X
            yy=y
        if (do_mine):
            yhat=clf.predict(XX)
            yyy=np.repeat(yy,per_sample)
            conf=np.zeros((num_class,num_class))
            for c in range(num_class):
                for d in range(num_class):
                    conf[c,d]=np.sum(yhat[yyy==c]==d)
            np.set_printoptions(precision=3, suppress=True)
            print(conf)
            ns=np.std(clf.coef_[0,:])
            print('Number of coeficients non-zero', ns, np.size(clf.coef_), np.sum(np.abs(clf.coef_) <.2*ns), np.sum(np.abs(clf.coef_)>.001))
            yerror=(yyy!=yhat)
            error=np.float32(np.sum(yerror))/len(yerror)
            print('Error on training',error)
            clf.error=error
        self._svm = clf

        #if (num_class==2):
        #    self.reweight(sample_weight,yerror,error)


    def reweight(self,sample_weight,yerror,error):
            tot=sample_weight.sum()
            new_sample_weight=np.ones(sample_weight.size)
            fac=1
            if (error>0):
                fac=((1-error)/error)
            new_sample_weight[yerror]=sample_weight[yerror]*fac
            sample_weight[yerror]=new_sample_weight[yerror]
            self._svm.sample_weight=sample_weight*tot/np.sum(sample_weight)






    def cluster(self,X,data,y,num_components, num_features, orig):
        ldata=data
        if (orig is not None):
            ldata=orig
        K = y.max() + 1
        mm_models = []
        mm_comps = []
        mm_weights = []
        mm_visparts=[]
        quant=np.minimum(50,np.maximum(np.float64(num_features),10))
        min_prob=1/(2*quant)
        comps=[]
        permutations=np.zeros((1,1), dtype=np.int64)
        for k in range(K):
            Xk = X[y == k]
            dim=np.prod(Xk.shape[1:])
            Xk = Xk.reshape((Xk.shape[0],1,dim))
            mm = PermutationMM(n_components=num_components,
                           permutations=permutations,
                           n_iter=10,
                           n_init=1,
                           random_state=self.random_state,
                           allocate=self._settings['allocate'],
                           min_probability=min_prob)

            mm.fit(Xk,y[y==k])

            #print('After PMM',mm.random_state.get_state(),self.random_state.get_state())

            mm,compsk=self.organize_parts(Xk,mm,ldata[y==k,:],num_components, quant)
            comps.append(compsk)

            merge= False
            if (ldata.shape[-1]==3):
                merge=True
            show_visparts(mm.visparts,1,10, merge,k)


            mm_models.append(mm.means_)
            #mm_.append(mm.weights_)
            # mm_visparts.append(mm.visparts)
            # np.set_printoptions(precision=2, suppress=True)
            # print('Weights:', mm.weights_)

        # self._extra['weights'] = mm_weights
        # self._extra['vis_parts'] = mm_visparts
        self._extra['models'] = mm_models
        return comps


    def organize_parts(self,Xflat,mm,raw_originals,ml,quant):
            # For each data point returns part index and angle index.

        mm.means_=np.floor(mm.means_*quant)/quant+1./(2*quant)
        comps = mm.predict(Xflat)
        comps=comps[:,0]
        counts = np.bincount(comps, minlength=ml)
        # Average all images beloinging to true part k at selected rotation given by second coordinate of comps
        visparts = np.asarray([
            raw_originals[comps == k,:].mean(0)
            for k in range(ml)
        ])

        ok = counts >= self._settings['min_count']

        II = np.argsort(counts)[::-1]
        print('Counts',np.sort(counts))
        #II = range(len(counts))
        II = np.asarray([ii for ii in II if ok[ii]])
        # ag.info('Keeping', len(II), 'out of', ok.size, 'parts')
        # mm._num_true_parts = len(II)
        mm.means_ = mm.means_[II]
        mm.weights_ = mm.weights_[II]
        mm.n_components=len(II)
        comps=mm.predict(Xflat)
        comps=comps[:,0]
        #mm.weights_ = mm.weights_[II]
        # mm.counts = counts[II]

        mm.visparts = visparts[II]
        # mm.mu=mu[II,:]
        return(mm, comps)

    def get_subregion_samples(self, X, y, train_num, test_num, cx=None, cy=None):
        Xflat_te=None
        yte=None
        ps=self.sub_region
        if (self._settings['mine']):
            print('Original Dims', X.shape[1], X.shape[2], 'New Dims', ps)
        dim=(ps[1]-ps[0],ps[3]-ps[2])
        num_samp_per_region=self._settings['num_sub_region_samples']
        num_sq=np.ceil(np.sqrt(num_samp_per_region))
        # Figure out the range of data on which you are sampling regions.
        highx=np.minimum(ps[0]+self._settings['sub_region_range'],X.shape[1]-dim[0]+1)
        highy=np.minimum(ps[2]+self._settings['sub_region_range'],X.shape[2]-dim[1]+1)
        qr1x=np.minimum(ps[0],X.shape[1]-dim[0]+1)
        qr1y=np.minimum(ps[2],X.shape[2]-dim[1]+1)
        qr2x=np.minimum(ps[0]+3*self._settings['sub_region_range']/4,X.shape[1]-dim[0]-num_sq+1)
        qr2y=np.minimum(ps[2]+3*self._settings['sub_region_range']/4,X.shape[2]-dim[1]-num_sq+1)

        # If too many samples select randomly.
        aa=list(itr.product(np.arange(ps[0],highx), np.arange(ps[2],highy)))
        if len(aa)>num_samp_per_region:
            if (cx is None or cy is None):
                lowx=self.random_state.random_integers(qr1x,qr2x,1)
                lowy=self.random_state.random_integers(qr1y,qr2y,1)
            else:
                lowx=cx
                lowy=cy
            if (self._settings['mine']):
                print('lowx, lowy',lowx, lowy)
            if (self._settings['train_region'] is None):
                self._settings['train_region']=(lowx,lowx+num_sq,lowy,lowy+num_sq)
            else:
                lowx=self._settings['train_region'][0]
                highx=self._settings['train_region'][1]
                lowy=self._settings['train_region'][2]
                highy=self._settings['train_region'][3]

            aa=list(itr.product(np.arange(lowx,lowx+num_sq),np.arange(lowy,lowy+num_sq)))
            #self.random_state.shuffle(aa)
        XXtr=[]
        XXte=[]
        YYtr=[]
        YYte=[]
        max_pool=self._settings['max_pool']
        if (max_pool):
            YYtr.append(y[:train_num, np.newaxis])
            if (test_num):
                YYte.append(y[train_num:,np.newaxis])
        for i,a in enumerate(aa):
                # Coordinates of random samples
                xx=a[0]
                yy=a[1]
                XXtr.append(X[:train_num,xx:xx+dim[0],yy:yy+dim[1],:].reshape((train_num,-1)))
                if (not max_pool):
                    YYtr.append(y[:train_num, np.newaxis])
                    if (test_num):
                        XXte.append(X[train_num:,xx:xx+dim[0],yy:yy+dim[1],:])
                        YYte.append(y[train_num:, np.newaxis])
        Xtr=np.concatenate(XXtr,axis=1)
        Xtr=Xtr.reshape(Xtr.shape[0]*len(aa),-1)
        ytr=np.concatenate(YYtr,axis=1)
        ytr=ytr.ravel()
        Xflat_tr = Xtr   #.reshape((Xtr.shape[0], -1))#.astype(np.float64)
        if (test_num):
            Xte=np.concatenate(XXte,axis=1)
            Xte=Xtr.reshape(Xte.shape[0]*len(aa),-1)
            yte=np.concatenate(YYte,axis=0)
            Xflat_te = Xte #.reshape((Xte.shape[0], -1)).astype(np.float64)
        #X=X[:,ps[0]:ps[1],ps[2]:ps[3],:]
        return Xflat_tr, ytr, Xflat_te, yte

    def flatten_and_adjust_for_components(self,X,y, data, train_num, test_num):
        print("training on full domain one classifier")
        Xflat_te=None
        yte=None
        num_comp=1
        Xflat_tr = X[:train_num,:].reshape((train_num, -1))#.astype(np.float64)\
        if self._settings['num_components']>1:
            num_class=np.max(y)+1
            num_comp=self._settings['num_components']
            comps=self.cluster(Xflat_tr,data,y,num_comp,X.shape[-1],orig)
            ya=np.copy(y)
            for c in range(num_class):
                y[ya==c]=comps[c]+num_comp*c
        ytr=y[:train_num]
        if (test_num):
            Xflat_te = X[train_num:,:].reshape((test_num, -1)).astype(np.float64)
            yte=y[train_num:]
        return Xflat_tr, ytr, Xflat_te, yte, num_comp


    def _train(self, phi, data, y, num_class=None, orig=None):
        if self._settings['num_train'] is not None:
            data=data[0:self._settings['num_train'],:]
            y=y[0:self._settings['num_train']]
        #print('numtrain',data.shape[0])

        if phi is not None:
            X = phi(data)
        else:
            X=data
        train_num = X.shape[0]
        test_num=0
        if (self._settings['test_prop'] is not None):
            train_num=np.int32(np.ceil(X.shape[0]*(1-self._settings['test_prop'])))
            test_num=X.shape[0]-train_num


        num_comp=1

        num_feat=X.shape[-1]
        if (self._settings['sub_region'] is not None):
            #if (not hasattr(self,'Xflat_tr')):
            Xflat_tr, ytr, Xflat_te, yte = self.get_subregion_samples(X,y,train_num, test_num)
            #     self.Xflat_tr=Xflat_tr
            #     self.ytr=ytr
            #     self.Xflat_te=Xflat_te
            #     self.yte=yte
            # else:
            #     Xflat_tr=self.Xflat_tr
            #     ytr=self.ytr
            #     Xflat_te=self.Xflat_te
            #     yte=self.Xflat_te
        else:
            Xflat_tr, ytr, Xflat_te, yte, num_comp = self.flatten_and_adjust_for_components(X,y,data, train_num, test_num)

        if (self._settings['num_sgd_iters'] != 0):

            sample_weight=np.ones(Xflat_tr.shape[0])
            if (np.max(ytr)==1):
                prop=np.sum(ytr==0)/np.sum(ytr==1)
                prop=1
                sample_weight[ytr==1]=prop
                boost=self._settings['boost']
            if (np.max(ytr)==1 and boost is not None):
                    svm=[]
                    if (boost>1):
                        print('boosting',boost)
                    for b in range(boost):
                        print('boosting step',b)
                        self._train_sgd(Xflat_tr,Xflat_te,ytr,yte,sample_weight,num_class)

                        sample_weight=svm[b].sample_weight
                    self.merge_boosted_svms(svm,boost)
                    return;
            else:
                num_tries=1

                self._train_sgd(Xflat_tr,Xflat_te,ytr,yte,sample_weight,num_class)
                if (self._settings['prune']):

                    len=self._svm.coef_.size/num_feat
                    aa=self._svm.coef_.reshape(len,num_feat)
                    bb=np.sum(np.abs(np.abs(aa)),axis=0)
                    cc=np.sort(bb)
                    thr=cc[-64]
                    ii=np.where(bb>thr)
                    self.III=ii

                    self.III=ii
                    XX=Xflat_tr.reshape(Xflat_tr.size/num_feat,num_feat)
                    XX=XX[...,ii]
                    Xflat_tr=XX.reshape(train_num,-1)
                    if (test_num):
                        XXt=Xflat_te.reshape(Xflat_te.size/num_feat,num_feat)
                        XXte=XXte[...,ii]
                        Xflat_te=XXte.reshape(test_num,-1)
                    self._svm=None
                    self._train_sgd(Xflat_tr,Xflat_te,ytr,yte,sample_weight,num_class)
                svm_old=copy.deepcopy(self._svm)

                if (self._settings['sub_region'] is not None and num_tries>1):

                    for t in np.arange(1,num_tries):
                            print('try')
                            self._svm=None
                            Xflat_tr, ytr, Xflat_te, yte = self.get_subregion_samples(X,y,train_num, test_num)
                            sample_weight=np.ones(Xflat_tr.shape[0])
                            self._train_sgd(Xflat_tr,Xflat_te,ytr,yte,sample_weight,num_class)
                            if (svm_old.error >  self._svm.error):
                                svm_old=copy.deepcopy(self._svm)
                self._svm=svm_old
                self._trained=True
                return;

        self.train_regular_svm(Xflat_tr,ytr,Xflat_te, yte, num_comp, test_num)

    def train_regular_svm(self,Xflat_tr,ytr, Xflat_te, yte, num_comp, test_num):
        #ag.info('Entering SVM training with X.shape = ', Xflat_tr.shape)


        clf = SVM(C=self._penalty, random_state=self.random_state)
        num_class=np.max(ytr)+1
        if (self._settings['simple_aggregate']):
            clf.coef_=np.zeros((num_class,Xflat_tr.shape[1]))
            for c in range(num_class):
                clf.coef_[c,np.arange(c,Xflat_tr.shape[1],num_class)]=1.
                clf.intercept_=np.zeros(num_class)

        # Just to set up the multiclass SVM - and then run ovr explicitly
        clf.fit(Xflat_tr[0:num_class,:],np.array(range(num_class)))
        # Test
        CLF=[]
        for c in range(num_class):
            true_c=c//num_comp
            print('sub class index', c, 'true class', true_c)
            ii_class=np.where(ytr==c)[0]
            if (len(ii_class)):
                ii_not=np.where(ytr//num_comp != true_c)[0]
                self.random_state.shuffle(ii_not)
                ll=len(ii_not) #/num_comp
                ii=np.concatenate((ii_class,ii_not[0:ll]))
                ytemp=ytr[ii]
                Xtemp=Xflat_tr[ii,:]
                ytemp=np.int64((ytemp==c))
                clft=SVM(C=self._penalty, random_state=self.random_state)
                clft.fit(Xtemp,ytemp)

                yt=clft.predict(Xtemp)
                print('error',1-(yt==ytemp).mean())
            else:
                clft.coef_=np.zeros((1,Xflat_tr.shape[1]))
                clft.intercept_=-1000000*np.ones((1,))
            if (c==0):
                clf.coef_=clft.coef_
                clf.intercept_=clft.intercept_
            else:
                clf.coef_=np.concatenate((clf.coef_,clft.coef_))
                clf.intercept_=np.concatenate((clf.intercept_,clft.intercept_))
            #print("Done training")
        # Check training error rate
        #zz=np.dot(Xflat,clf.coef_.T)+clf.intercept_
        Xflat=Xflat_tr
        yy=ytr
        if (test_num>0):
            Xflat=Xflat_te
            yy=yte
        yhat = clf.predict(Xflat)
        conf=np.zeros((num_class,num_class))
        for c in range(num_class):
            for d in range(num_class):
                conf[c,d]=np.sum(yhat[yy==c]==d)
        np.set_printoptions(precision=3, suppress=True)
        print(conf)
        training_error_rate = 1 - (yy//num_comp == yhat//num_comp).mean()
        ag.info('Training error rate: {}%'.format(100 * training_error_rate))

        self._svm = clf
        self._trained=True

    def merge_boosted_svms(self,svm,boost):
        svm_new=svm[0]
        coef=[]
        rr=np.arange(0,boost,1)
        intercept=np.zeros(rr.size)
        for i,b in enumerate(rr):
            cc=svm[b].coef_[0,:][np.newaxis,:]
            intercept[i]=svm[b].intercept_[0]
            coef.append(cc)
        svm_new.intercept=intercept
        svm_new.intercept_=svm_new.intercept
        svm_new.coef=np.concatenate(coef,axis=0)
        #print('Corr between filters',np.corrcoef(svm_new.coef))
        svm_new.coef_=svm_new.coef
        svm_new.classes=range(boost)
        self._svm=svm_new
        self._trained=True
        return;


    def _vzlog_output_(self, vz):
        import pylab as plt
        if self._settings.get('standardize'):
            vz.log('means', self._extra['means'])
            vz.log('vars', self._extra['vars'])
            vz.log('min var', np.min(self._extra['vars']))

            plt.figure()
            plt.hist(self._extra['means'])
            plt.title('Means')
            plt.savefig(vz.impath())

            plt.figure()
            plt.hist(self._extra['vars'])
            plt.title('Variances')
            plt.savefig(vz.impath())

    def ovr_sgd(self,num_comp,X,y):
        num_class=np.max(y)+1
        num_cl=num_class
        if (num_class==2):
            num_cl=1
        rs=self.random_state
        classes=np.unique(y)

        # Just initializing the final clf that will combine all the ovr's
        clf = SGDClassifier(alpha=self._penalty, class_weight=None, epsilon=0.1,
            eta0=0.0001, fit_intercept=True, l1_ratio=0.15,
            learning_rate='constant', loss='log', n_iter=1, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=True,
            verbose=0, warm_start=True)
        clf.partial_fit(X[0:num_class*5,:],y[0:num_class*5],classes)
        num_batches=1
        if (self._settings['sgd_batch_size'] is not None):
            num_batches=np.int(np.ceil(X.shape[0]/self._settings['sgd_batch_size']))
        num_rounds=self._settings['num_sgd_rounds']

        print('Number of training batches for SGD svm',num_batches,'Number of rounds', num_rounds)
        yerror_rounds=np.zeros(num_rounds)

        for c in range(num_cl):
            true_c=c//num_comp
            print('sub class index', c, 'true class', true_c)
            slic=np.logical_or(y==c,y//num_comp!=true_c)
            ytemp=y[slic]
            print('YY',len(ytemp))
            Xtemp=X[slic,:]
            sample_weight=np.ones(np.size(ytemp))
            if num_cl>1:
                ytemp=np.int64(ytemp==c)
            else:
                prop=np.int64(np.sum(ytemp==0)/np.sum(ytemp==1))*2
                sample_weight=np.ones(np.size(ytemp))
                sample_weight[y==1]=prop
            print('Length of current data',Xtemp.shape[0])
            clft = SGDClassifier(alpha=self._penalty, class_weight=None, epsilon=0.1,
                eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                learning_rate='optimal', loss='log', n_iter=1, n_jobs=1,
                penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                verbose=0, warm_start=True)
            classest=np.unique(ytemp)
            ii=range(Xtemp.shape[0])
            for n in range(num_rounds):
                rs.shuffle(ii)
                print('Iteration',n)
                Xbs=np.array_split(Xtemp[ii],num_batches)
                Ys=np.array_split(ytemp[ii],num_batches)
                SSs=np.array_split(sample_weight[ii],num_batches)

                for j in range(len(Xbs)):
                    #print('Fitting batch',j)
                    #print('YSS',np.sum(Ys[j]))
                    clft.partial_fit(Xbs[j],Ys[j],classest, sample_weight=SSs[j])
                    yhat=clft.predict(Xtemp)
                    yerror_rounds[n]= 1 - (ytemp == yhat).mean()
                print('Error on round',n,yerror_rounds[n])
                #if (n>0 and np.abs(yerror_rounds[n]-yerror_rounds[n-1])/yerror_rounds[n-1]<.01):
                #    break
            if (c==0):
                clf.coef_=clft.coef_
                clf.intercept_=clft.intercept_
            else:
                clf.coef_=np.concatenate((clf.coef_,clft.coef_))
                clf.intercept_=np.concatenate((clf.intercept_,clft.intercept_))
        yhat = clf.predict(X)
        conf=np.zeros((num_class,num_class))
        for c in range(num_class):
            for d in range(num_class):
                conf[c,d]=np.sum(yhat[y==c]==d)
        np.set_printoptions(precision=3, suppress=True)
        print(conf)
        training_error_rate = 1 - (y//num_comp == yhat//num_comp).mean()
        ag.info('Training error rate: {}%'.format(100 * training_error_rate))
        return(clf)


    def save_to_dict(self):
        d = {}
        d['settings'] = self._settings
        d['extra'] = self._extra
        d['penalty'] = self._penalty
        aa=None
        if (hasattr(self._svm,'loss')):
            d['loss']=self._svm.loss
            aa=None
        d['svm'] = dict(coef=self._svm.coef_,
                        intercept=self._svm.intercept_,
                        classes=self._svm.classes_,
                        loss=aa,
                        SGD=(self._settings['num_sgd_iters']>0))
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['settings'])
        obj._extra = d['extra']

        # Reconstruct the SVM object. This is a bit clonky and I should
        # probably just preform the prediction manually from coef and
        # intercept.
        svm = d['svm']

        if (svm.get('SGD')):
            aa=None
            if (svm.get('loss')):
                aa=svm['loss']
            obj._svm=LogisticRegression(multi_class='multinomial', solver='lbfgs')
        else:
            obj._svm = SVM(C=d['penalty'], random_state=obj._settings.get('seed', 0))

        from sklearn.preprocessing.label import LabelEncoder
        obj._svm._enc = LabelEncoder()
        obj._svm._enc.classes_ = svm.get('classes')
        if (svm.get('SGD')):
            obj._svm.classes_=svm.get('classes')
        #obj._svm.class_weight_ = svm['coef']
        obj._svm.coef_ = svm['coef']
        #obj._svm.raw_coef_ = svm['raw_coef']
        obj._svm.intercept_ = svm['intercept']
        return obj


