__author__ = 'amit'
import pnet
import test_edge
import deepdish as dd
import numpy as np
pdir='_Part_images'

def draw_means(Dname, ll=None, num_cols=1, merge=True):

    if (ll is None):
        ll=test_edge.find_next_net_index(Dname)
        ll-=1

    import os
    three=3
    net=pnet.Layer.load(Dname+'/net'+str(ll)+'.h5')
    if (Dname=='Cnets'):
        for l in range(len(net._layers)):
            if ("oriented" in net._layers[l].name):
                visparts = net._layers[l]._visparts
                # SHow the model means as color images
                if (merge):
                    if (visparts.shape[3]==1):
                        grid = dd.plot.ImageGrid(visparts[...,0], cols=net._layers[l]._settings['num_orientations']*num_cols)
                    else:
                        grid = dd.plot.ColorImageGrid(visparts, cols=net._layers[l]._settings['num_orientations']*num_cols)
                else:
                    # Show different channels separately
                    if (visparts.ndim==4):
                        bigvisparts=np.zeros((visparts.shape[0]*3,visparts.shape[1],visparts.shape[2],3))
                        for ii in range(3):
                            bigvisparts[np.arange(ii,bigvisparts.shape[0],3),:,:,ii]=visparts[:,:,:,ii]
                    grid = dd.plot.ColorImageGrid(bigvisparts, cols=net._layers[l]._settings['num_orientations']*num_cols)
                grid.save(pdir+'/parts_col'+str(l)+'.png', scale=5)
            elif ("gmm-classification" in net._layers[l].name):
                layer=net._layers[l]
                numclass=len(layer._models)
                num_models=layer._n_components
                visparts=[]
                for K in range(numclass):
                    visparts.append(layer._models[K]["visparts"])
                visparts=np.array(visparts)
                visparts=visparts.reshape((visparts.shape[0]*visparts.shape[1],visparts.shape[2])+(visparts.shape[3:]))
                if (visparts.ndim==4):
                    grid=dd.plot.ColorImageGrid(visparts,cols=num_models)
                else:
                    grid=dd.plot.ImageGrid(vispart,cols=num_models)
                grid.save(pdir+'/parts_col'+str(l)+'.png',scale=5)
                print("Here")
    else:
        for l in range(len(net._layers)):
            if ("oriented" in net._layers[l].name):
                visparts = net._layers[l]._visparts
                num_cols=np.minimum(net._layers[l]._settings['num_orientations']*num_cols,20)
                if merge:
                    grid = dd.plot.ColorImageGrid(visparts, cols=num_cols)
                else:
                    if (visparts.ndim==4):
                        visparts=visparts.transpose(0,3,1,2)
                        visparts=visparts.reshape((visparts.shape[0]*visparts.shape[1],visparts.shape[2],visparts.shape[3]))
                    grid = dd.plot.ImageGrid(visparts, cols=num_cols)
                grid.save(pdir+'/parts'+str(l)+'.png', scale=5)


def show_visparts(visparts, num_orientations, num_cols, merge=False, index=0):

                if merge:
                    grid = dd.plot.ColorImageGrid(visparts, cols=num_cols)
                else:
                    if (visparts.ndim==4):
                        visparts=visparts.transpose(0,3,1,2)
                        visparts=visparts.reshape((visparts.shape[0]*visparts.shape[1],visparts.shape[2],visparts.shape[3]))
                    grid = dd.plot.ImageGrid(visparts, cols=num_cols)
                grid.save(pdir+'/parts'+str(index)+'.png', scale=5)