#!python:
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, abs, fabs, fmax, fmin, log, pow, sqrt
from libc.stdlib cimport rand, srand 
from cpython cimport bool

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

def subsample_offset_shape(shape, size):
    return [int(shape[i]%size[i]/2 + size[i]/2)  for i in xrange(2)]

def code_index_map(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                   np.ndarray[ndim=4,dtype=np.float64_t] part_logits, 
                   np.ndarray[ndim=1,dtype=np.float64_t] constant_terms, 
                   int threshold, outer_frame=0, 
                   #float min_llh=-np.inf,
                   np.ndarray[ndim=1,dtype=np.float64_t] min_llh=np.tile(-np.inf, 0),
                   return_llhs=False):
    cdef unsigned int part_x_dim = part_logits.shape[0]
    cdef unsigned int part_y_dim = part_logits.shape[1]
    cdef unsigned int part_z_dim = part_logits.shape[2]
    cdef unsigned int num_parts = part_logits.shape[3]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.ndarray[dtype=np.float64_t, ndim=4] llhs
    if return_llhs:
        llhs = np.zeros((n_samples, new_x_dim, new_y_dim, num_parts))
    else:
        llhs = np.zeros((0, new_x_dim, new_y_dim, num_parts))

    cdef np.float64_t[:,:,:,:] llhs_mv = llhs

    
    cdef np.ndarray[dtype=np.int64_t, ndim=3] out_map = -np.ones((n_samples,
                                                                  new_x_dim,
                                                                  new_y_dim), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    #cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(np.float64), 0, 4).copy()

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()

    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits
    cdef np.float64_t[:] constant_terms_mv = constant_terms

    cdef np.int64_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # TODO: Remove
    cdef np.float64_t[:] min_llh_mv = min_llh

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in xrange(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count:
                    vs[:] = constant_terms
                    #vs_alt[:] = constant_terms_alt
                    with nogil:
                        for i in range(part_x_dim):
                            for j in range(part_y_dim):
                                for z in range(X_z_dim):
                                    if X_mv[n,i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            vs_mv[k] += part_logits_mv[i,j,z,k]

                    max_index = vs.argmax()
                    if vs_mv[max_index] >= min_llh_mv[max_index]:
                        out_map_mv[n,i_start,j_start] = max_index

                    if return_llhs:
                        llhs[n,i_start,j_start] = vs
                

    if return_llhs:
        return out_map, llhs
    else:
        return out_map


def code_index_map_with_keypoints(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                                  np.ndarray[ndim=4,dtype=np.float64_t] log_parts, 
                                  np.ndarray[ndim=4,dtype=np.float64_t] log_invparts, 
                                  np.ndarray[ndim=3,dtype=np.int64_t] keypoints, 
                                  int threshold, outer_frame=0, float min_llh=-np.inf):
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int num_keypoints = keypoints.shape[1]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,i,j,z,k, cx0, cx1, cy0, cy1, kp 
    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int64_t, ndim=3] out_map = -np.ones((n_samples,
                                                                  new_x_dim,
                                                                  new_y_dim), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits = log_parts - log_invparts

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()
    cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.zeros(num_parts)
    cdef np.float64_t[:] constant_terms_mv = constant_terms
    cdef np.float64_t[:,:,:,:] log_invparts_mv = log_invparts 
    cdef np.int64_t[:,:,:] keypoints_mv = keypoints 


    for k in xrange(num_parts):
        for kp in xrange(num_keypoints):
            i = keypoints_mv[k,kp,0]
            j = keypoints_mv[k,kp,1]
            z = keypoints_mv[k,kp,2]

            constant_terms_mv[k] += log_invparts_mv[k,i,j,z] 

    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits

    cdef np.int64_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in xrange(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count:
                    vs[:] = constant_terms
                    #vs_alt[:] = constant_terms_alt
                    with nogil:
                        for k in range(num_parts):
                            for kp in range(num_keypoints):
                                i = keypoints_mv[k,kp,0]
                                j = keypoints_mv[k,kp,1]
                                z = keypoints_mv[k,kp,2]

                                if X_mv[n,i_start+i,j_start+j,z]:
                                    vs_mv[k] += part_logits_mv[k,i,j,z]

                    max_index = vs.argmax()
                    if vs[max_index] >= min_llh:
                        out_map_mv[n,i_start,j_start] = max_index
                

    return out_map

def code_index_map_with_same_keypoints(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                                       np.ndarray[ndim=4,dtype=np.float64_t] log_parts, 
                                       np.ndarray[ndim=4,dtype=np.float64_t] log_invparts, 
                                       np.ndarray[ndim=2,dtype=np.int64_t] keypoints, 
                                       int threshold, outer_frame=0, float min_llh=-np.inf):
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int num_keypoints = keypoints.shape[0]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,i,j,z,k, cx0, cx1, cy0, cy1, kp 
    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int64_t, ndim=3] out_map = -np.ones((n_samples,
                                                                  new_x_dim,
                                                                  new_y_dim), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits = log_parts - log_invparts

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()
    cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.zeros(num_parts)
    cdef np.float64_t[:] constant_terms_mv = constant_terms
    cdef np.float64_t[:,:,:,:] log_invparts_mv = log_invparts 
    cdef np.int64_t[:,:] keypoints_mv = keypoints 


    for kp in xrange(num_keypoints):
        i = keypoints_mv[kp,0]
        j = keypoints_mv[kp,1]
        z = keypoints_mv[kp,2]

        for k in xrange(num_parts):
            constant_terms_mv[k] += log_invparts_mv[k,i,j,z] 

    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits

    cdef np.int64_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in xrange(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count:
                    vs[:] = constant_terms
                    #vs_alt[:] = constant_terms_alt
                    with nogil:
                        for kp in range(num_keypoints):
                            i = keypoints_mv[kp,0]
                            j = keypoints_mv[kp,1]
                            z = keypoints_mv[kp,2]
                            if X_mv[n,i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    vs_mv[k] += part_logits_mv[k,i,j,z]

                    max_index = vs.argmax()
                    if vs[max_index] >= min_llh:
                        out_map_mv[n,i_start,j_start] = max_index
                

    return out_map

def index_map_pooling(np.ndarray[ndim=3,dtype=np.int64_t] part_index_map, 
            int num_parts,
            pooling_shape,
            strides):

    offset = subsample_offset_shape((part_index_map.shape[1], part_index_map.shape[2]), strides)
    cdef:
        int sample_size = part_index_map.shape[0]
        int part_index_dim0 = part_index_map.shape[1]
        int part_index_dim1 = part_index_map.shape[2]
        int stride0 = strides[0]
        int stride1 = strides[1]
        int pooling0 = pooling_shape[0]
        int pooling1 = pooling_shape[1]
        int half_pooling0 = pooling0 // 2
        int half_pooling1 = pooling1 // 2
        int offset0 = offset[0]
        int offset1 = offset[1]


        int feat_dim0 = part_index_dim0//stride0
        int feat_dim1 = part_index_dim1//stride1

        np.ndarray[np.uint8_t,ndim=4] feature_map = np.zeros((sample_size,
                                                              feat_dim0,
                                                              feat_dim1,
                                                              num_parts),
                                                              dtype=np.uint8)
        np.int64_t[:,:,:] part_index_mv = part_index_map 
        np.uint8_t[:,:,:,:] feat_mv = feature_map 


        int p, x, y, i, j, n,i0, j0


    with nogil:
         for n in range(sample_size): 
            for i in range(feat_dim0):
                for j in range(feat_dim1):
                    x = offset0 + i*stride0 - half_pooling0
                    y = offset1 + j*stride1 - half_pooling1
                    for i0 in range(int_max(x, 0), int_min(x + pooling0, part_index_dim0)):
                        for j0 in range(int_max(y, 0), int_min(y + pooling1, part_index_dim1)):
                            p = part_index_mv[n,i0,j0]
                            if p != -1:
                                feat_mv[n,i,j,p] = 1
     
     
      
    return feature_map 

def convert_to_index_map(np.ndarray[ndim=3,dtype=np.int64_t] part_index_map, 
                         int num_parts,
                         pooling_shape,
                         strides):

    offset = subsample_offset_shape((part_index_map.shape[1], part_index_map.shape[2]), strides)
    cdef:
        int sample_size = part_index_map.shape[0]
        int part_index_dim0 = part_index_map.shape[1]
        int part_index_dim1 = part_index_map.shape[2]
        int stride0 = strides[0]
        int stride1 = strides[1]
        int pooling0 = pooling_shape[0]
        int pooling1 = pooling_shape[1]
        int half_pooling0 = pooling0 // 2
        int half_pooling1 = pooling1 // 2
        int offset0 = offset[0]
        int offset1 = offset[1]


        int feat_dim0 = part_index_dim0//stride0
        int feat_dim1 = part_index_dim1//stride1

        np.ndarray[np.uint8_t,ndim=4] feature_map = np.zeros((sample_size,
                                                              feat_dim0,
                                                              feat_dim1,
                                                              num_parts),
                                                              dtype=np.uint8)
        np.int64_t[:,:,:] part_index_mv = part_index_map 
        np.uint8_t[:,:,:,:] feat_mv = feature_map 


        int p, x, y, i, j, n,i0, j0


    with nogil:
         for n in range(sample_size): 
            for i in range(feat_dim0):
                for j in range(feat_dim1):
                    x = offset0 + i*stride0 - half_pooling0
                    y = offset1 + j*stride1 - half_pooling1
                    for i0 in range(int_max(x, 0), int_min(x + pooling0, part_index_dim0)):
                        for j0 in range(int_max(y, 0), int_min(y + pooling1, part_index_dim1)):
                            p = part_index_mv[n,i0,j0]
                            if p != -1:
                                feat_mv[n,i,j,p] = 1
     
     
      
    return feature_map 

def resample_and_arrange_image(np.ndarray[dtype=np.uint8_t,ndim=2] image, target_size, np.ndarray[dtype=np.float64_t,ndim=2] lut):
    cdef:
        int dim0 = image.shape[0]
        int dim1 = image.shape[1]
        int output_dim0 = target_size[0]
        int output_dim1 = target_size[1]
        np.ndarray[np.float64_t,ndim=3] output = np.empty(target_size + (3,), dtype=np.float64)
        np.uint8_t[:,:] image_mv = image
        np.float64_t[:,:,:] output_mv = output
        np.float64_t[:,:] lut_mv = lut 
        double mn = image.min()
        int i, j, c

    with nogil:
        for i in range(output_dim0):
            for j in range(output_dim1):
                for c in range(3):
                    output_mv[i,j,c] = lut_mv[image[dim0*i/output_dim0, dim1*j/output_dim1],c]

    return output

# EXPERIMENTAL STUFF ##########################################################

def code_index_map_multi(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                         np.ndarray[ndim=4,dtype=np.float64_t] part_logits, 
                         np.ndarray[ndim=1,dtype=np.float64_t] constant_terms, 
                         int threshold, outer_frame=0, int n_coded=1, float min_llh=-np.inf):
    cdef unsigned int part_x_dim = part_logits.shape[0]
    cdef unsigned int part_y_dim = part_logits.shape[1]
    cdef unsigned int part_z_dim = part_logits.shape[2]
    cdef unsigned int num_parts = part_logits.shape[3]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,i,j,z,k, cx0, cx1, cy0, cy1, m
    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int64_t, ndim=4] out_map = -np.ones((n_samples,
                                                                  new_x_dim,
                                                                  new_y_dim,
                                                                  n_coded), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    #cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(np.float64), 0, 4).copy()

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()

    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits
    cdef np.float64_t[:] constant_terms_mv = constant_terms

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in xrange(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count:
                    vs[:] = constant_terms
                    #vs_alt[:] = constant_terms_alt
                    with nogil:
                        for i in range(part_x_dim):
                            for j in range(part_y_dim):
                                for z in range(X_z_dim):
                                    if X_mv[n,i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            vs_mv[k] += part_logits_mv[i,j,z,k]

            
                    for m in xrange(n_coded):
                        max_index = vs.argmax()
                        if vs[max_index] >= min_llh:
                            out_map_mv[n,i_start,j_start,m] = max_index
                        vs[max_index] = -np.inf

    return out_map

def orientation_pooling(np.ndarray[ndim=5,dtype=np.uint8_t] X,
                        pooling_shape,
                        strides,
                        int rotational_spreading=0):
    offset = subsample_offset_shape((X.shape[1], X.shape[2]), strides)
    cdef:
        int sample_size = X.shape[0]
        int part_index_dim0 = X.shape[1]
        int part_index_dim1 = X.shape[2]
        #int n_coded = part_index_map.shape[3]
        int num_true_parts = X.shape[3]
        int num_orientations = X.shape[4]
        int stride0 = strides[0]
        int stride1 = strides[1]
        int pooling0 = pooling_shape[0]
        int pooling1 = pooling_shape[1]
        int half_pooling0 = pooling0 // 2
        int half_pooling1 = pooling1 // 2
        int offset0 = offset[0]
        int offset1 = offset[1]


        int feat_dim0 = part_index_dim0//stride0
        int feat_dim1 = part_index_dim1//stride1

        np.ndarray[np.uint8_t,ndim=5] feature_map = np.zeros((sample_size,
                                                              feat_dim0,
                                                              feat_dim1,
                                                              num_true_parts,
                                                              num_orientations),
                                                              dtype=np.uint8)
        np.uint8_t[:,:,:,:,:] X_mv = X 
        np.uint8_t[:,:,:,:,:] feat_mv = feature_map 


        int x, y, i, j, n, i0, j0, m, r, z, rs


    with nogil:
         for n in range(sample_size): 
            for i in range(feat_dim0):
                for j in range(feat_dim1):
                    x = offset0 + i*stride0 - half_pooling0
                    y = offset1 + j*stride1 - half_pooling1
                    for i0 in range(int_max(x, 0), int_min(x + pooling0, part_index_dim0)):
                        for j0 in range(int_max(y, 0), int_min(y + pooling1, part_index_dim1)):
                            for z in xrange(num_true_parts):
                                for r in xrange(num_orientations):
                                    if X[n,i0,j0,z,r]:
                                        feat_mv[n,i,j,z,r] = 1
                                        for rs in xrange(1, rotational_spreading+1):
                                            feat_mv[n,i,j,z,(r+rs)%num_orientations] = 1
                                            feat_mv[n,i,j,z,(r-rs)%num_orientations] = 1
     
     
      
    return feature_map 

def index_map_pooling_multi(np.ndarray[ndim=4,dtype=np.int64_t] part_index_map, 
            int num_parts,
            pooling_shape,
            strides):

    offset = subsample_offset_shape((part_index_map.shape[1], part_index_map.shape[2]), strides)
    cdef:
        int sample_size = part_index_map.shape[0]
        int part_index_dim0 = part_index_map.shape[1]
        int part_index_dim1 = part_index_map.shape[2]
        int n_coded = part_index_map.shape[3]
        int stride0 = strides[0]
        int stride1 = strides[1]
        int pooling0 = pooling_shape[0]
        int pooling1 = pooling_shape[1]
        int half_pooling0 = pooling0 // 2
        int half_pooling1 = pooling1 // 2
        int offset0 = offset[0]
        int offset1 = offset[1]


        int feat_dim0 = part_index_dim0//stride0
        int feat_dim1 = part_index_dim1//stride1

        np.ndarray[np.uint8_t,ndim=4] feature_map = np.zeros((sample_size,
                                                              feat_dim0,
                                                              feat_dim1,
                                                              num_parts),
                                                              dtype=np.uint8)
        np.int64_t[:,:,:,:] part_index_mv = part_index_map 
        np.uint8_t[:,:,:,:] feat_mv = feature_map 


        int p, x, y, i, j, n,i0, j0, m


    with nogil:
         for n in range(sample_size): 
            for i in range(feat_dim0):
                for j in range(feat_dim1):
                    x = offset0 + i*stride0 - half_pooling0
                    y = offset1 + j*stride1 - half_pooling1
                    for i0 in range(int_max(x, 0), int_min(x + pooling0, part_index_dim0)):
                        for j0 in range(int_max(y, 0), int_min(y + pooling1, part_index_dim1)):
                            for m in xrange(n_coded):
                                p = part_index_mv[n,i0,j0,m]
                                if p != -1:
                                    feat_mv[n,i,j,p] = 1
     
     
      
    return feature_map 

def index_map_pooling_multi_support(np.ndarray[ndim=4,dtype=np.int64_t] part_index_map, 
                                    np.ndarray[ndim=2,dtype=np.uint8_t] support, 
                                    int num_parts,
                                    pooling_shape,
                                    strides):

    offset = subsample_offset_shape((part_index_map.shape[1], part_index_map.shape[2]), strides)
    cdef:
        int sample_size = part_index_map.shape[0]
        int part_index_dim0 = part_index_map.shape[1]
        int part_index_dim1 = part_index_map.shape[2]
        int n_coded = part_index_map.shape[3]
        int stride0 = strides[0]
        int stride1 = strides[1]
        int pooling0 = pooling_shape[0]
        int pooling1 = pooling_shape[1]
        int half_pooling0 = pooling0 // 2
        int half_pooling1 = pooling1 // 2
        int offset0 = offset[0]
        int offset1 = offset[1]

        np.uint8_t[:,:] support_mv = support

        int feat_dim0 = part_index_dim0//stride0
        int feat_dim1 = part_index_dim1//stride1

        np.ndarray[np.uint8_t,ndim=4] feature_map = np.zeros((sample_size,
                                                              feat_dim0,
                                                              feat_dim1,
                                                              num_parts),
                                                              dtype=np.uint8)
        np.int64_t[:,:,:,:] part_index_mv = part_index_map 
        np.uint8_t[:,:,:,:] feat_mv = feature_map 


        int p, x, y, i, j, n,i0, j0, m


    with nogil:
         for n in range(sample_size): 
            for i in range(feat_dim0):
                for j in range(feat_dim1):
                    x = offset0 + i*stride0 - half_pooling0
                    y = offset1 + j*stride1 - half_pooling1
                    for i0 in range(int_max(x, 0), int_min(x + pooling0, part_index_dim0)):
                        for j0 in range(int_max(y, 0), int_min(y + pooling1, part_index_dim1)):
                            if support_mv[i0-x,j0-y]:
                                for m in xrange(n_coded):
                                    p = part_index_mv[n,i0,j0,m]
                                    if p != -1:
                                        feat_mv[n,i,j,p] = 1
     
     
      
    return feature_map 

def code_index_map_general(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                           np.ndarray[ndim=4,dtype=np.float64_t] parts, 
                           np.ndarray[ndim=2,dtype=np.uint8_t] support, 
                           int threshold, 
                           outer_frame=0, 
                           int n_coded=1, 
                           int standardize=0,
                           int max_threshold=10000,
                           min_percentile=None):
    cdef unsigned int part_x_dim = parts.shape[1]
    cdef unsigned int part_y_dim = parts.shape[2]
    cdef unsigned int part_z_dim = parts.shape[3]
    cdef unsigned int num_parts = parts.shape[0]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start, j_start, i_end, j_end, i, j, z, k, cx0, cx1, cy0, cy1, m

    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.ndarray[dtype=np.float64_t,ndim=1] means = np.zeros(num_parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] sigmas = np.ones(num_parts)

    cdef np.ndarray[dtype=np.float64_t,ndim=1] postmax_means = np.zeros(num_parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] postmax_sigmas = np.ones(num_parts)

    cdef np.uint8_t[:,:] support_mv = support

    cdef np.float64_t[:] means_mv = means
    cdef np.float64_t[:] sigmas_mv = sigmas 

    cdef np.float64_t[:] postmax_means_mv = postmax_means
    cdef np.float64_t[:] postmax_sigmas_mv = postmax_sigmas 

    cdef float min_standardized_llh = -np.inf
    if standardize and min_percentile is not None:
        from scipy.stats import norm
        min_standardized_llh = norm.ppf(min_percentile/100.0)
    
    cdef np.ndarray[dtype=np.int64_t,ndim=4] out_map = -np.ones((n_samples,
                                                                  new_x_dim,
                                                                  new_y_dim,
                                                                  n_coded), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    cdef np.ndarray[dtype=np.float64_t,ndim=4] log_parts = np.log(parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=4] log_invparts = np.log(1 - parts)

    cdef np.ndarray[dtype=np.float64_t,ndim=4] part_logits = log_parts - log_invparts

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()
    cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.zeros(num_parts)
    cdef np.float64_t[:] constant_terms_mv = constant_terms
    cdef np.float64_t[:,:,:,:] log_invparts_mv = log_invparts 

    # Construct an array with only the keypoint parts

    cdef np.ndarray[dtype=np.float64_t,ndim=3] kp_parts = parts[:,support.astype(np.bool)]

    from scipy.special import logit
    cdef int do_premax_standardization = 0

    if standardize == 1:
        postmax_means[:] = np.sum(kp_parts * np.log(kp_parts) + (1 - kp_parts) * np.log(1 - kp_parts))
        postmax_sigmas[:] = np.sqrt(np.sum(logit(kp_parts)**2 * kp_parts * (1 - kp_parts)))
    elif standardize == 2:
        postmax_means[:] = np.apply_over_axes(np.sum, kp_parts * np.log(kp_parts) + (1 - kp_parts) * np.log(1 - kp_parts), [1, 2]).ravel()
        postmax_sigmas[:] = np.sqrt(np.apply_over_axes(np.sum, logit(kp_parts)**2 * kp_parts * (1 - kp_parts), [1, 2]).ravel())
    elif standardize == 3:
        do_premax_standardization = 1
        means[:] = np.apply_over_axes(np.sum, kp_parts * np.log(kp_parts) + (1 - kp_parts) * np.log(1 - kp_parts), [1, 2]).ravel()
        sigmas[:] = np.sqrt(np.apply_over_axes(np.sum, logit(kp_parts)**2 * kp_parts * (1 - kp_parts), [1, 2]).ravel())

    constant_terms[:] = np.apply_over_axes(np.sum, np.log(1 - kp_parts), [1, 2]).ravel()

    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in range(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count <= max_threshold:
                    vs[:] = constant_terms
                    with nogil:
                        for i in range(part_x_dim):
                            for j in range(part_y_dim):
                                if support_mv[i,j]:
                                    for z in range(X_z_dim):
                                        if X_mv[n,i_start+i,j_start+j,z]:
                                            for k in range(num_parts):
                                                vs_mv[k] += part_logits_mv[k,i,j,z]

                    if do_premax_standardization:
                        vs[:] = (vs - means) / sigmas

                    for m in range(n_coded):
                        max_index = vs.argmax()
                        if (vs[max_index] - postmax_means_mv[max_index]) / postmax_sigmas_mv[max_index] >= min_standardized_llh:
                            out_map_mv[n,i_start,j_start,m] = max_index

                        vs[max_index] = -np.inf
                

    return out_map

def code_index_map_hierarchy(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                             np.ndarray[ndim=5,dtype=np.float64_t] parts, 
                             int depth,
                             int threshold, 
                             outer_frame=0, 
                             int standardize=0,
                             min_percentile=None):
    cdef unsigned int part_x_dim = parts.shape[2]
    cdef unsigned int part_y_dim = parts.shape[3]
    cdef unsigned int part_z_dim = parts.shape[4]
    cdef unsigned int num_parts = parts.shape[0]
    cdef unsigned int num_parts_per_layer = parts.shape[1]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start, j_start, i_end, j_end, i, j, z, k, cx0, cx1, cy0, cy1, m, level, offset, base_offset, pos

    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.ndarray[dtype=np.float64_t,ndim=1] means = np.zeros(num_parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] sigmas = np.ones(num_parts)

    cdef np.ndarray[dtype=np.float64_t,ndim=1] postmax_means = np.zeros(num_parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] postmax_sigmas = np.ones(num_parts)

    cdef np.float64_t[:] means_mv = means
    cdef np.float64_t[:] sigmas_mv = sigmas 

    cdef np.float64_t[:] postmax_means_mv = postmax_means
    cdef np.float64_t[:] postmax_sigmas_mv = postmax_sigmas 

    cdef float min_standardized_llh = -np.inf
    if standardize and min_percentile is not None:
        from scipy.stats import norm
        min_standardized_llh = norm.ppf(min_percentile/100.0)
    
    cdef np.ndarray[dtype=np.int64_t,ndim=4] out_map = -np.ones((n_samples,
                                                                 new_x_dim,
                                                                 new_y_dim,
                                                                 1), dtype=np.int64)
    cdef np.ndarray[dtype=np.float64_t,ndim=1] vs = np.ones(num_parts_per_layer, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    cdef np.uint8_t[:,:,:,:] X_mv = X

    cdef np.ndarray[dtype=np.float64_t,ndim=5] log_parts = np.log(parts)
    cdef np.ndarray[dtype=np.float64_t,ndim=5] log_invparts = np.log(1 - parts)

    cdef np.ndarray[dtype=np.float64_t,ndim=5] part_logits = log_parts - log_invparts

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()
    cdef np.ndarray[dtype=np.float64_t, ndim=2] constant_terms = np.zeros((num_parts, num_parts_per_layer))
    cdef np.float64_t[:,:] constant_terms_mv = constant_terms
    cdef np.float64_t[:,:,:,:,:] log_invparts_mv = log_invparts 

    # Construct an array with only the keypoint parts

    from scipy.special import logit
    cdef int do_premax_standardization = 0

    constant_terms[:] = np.apply_over_axes(np.sum, np.log(1 - parts), [2, 3, 4])[:,:,0,0,0]

    cdef np.float64_t[:,:,:,:,:] part_logits_mv = part_logits

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for n in range(n_samples):
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[n,i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]

        base_offset = 0

        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count:
                    base_offset = 0
                    offset = 0
                    #print "Starting..."
                    pos = 0
                    for level in range(depth):
                        offset = base_offset + pos 
                        base_offset += num_parts_per_layer ** level

                        #print 'offset', offset, 'base offset', base_offset

                        vs[:] = constant_terms[offset]
                        with nogil:
                            for i in range(part_x_dim):
                                for j in range(part_y_dim):
                                    for z in range(X_z_dim):
                                        if X_mv[n,i_start+i,j_start+j,z]:
                                            for k in range(num_parts_per_layer):
                                                vs_mv[k] += part_logits_mv[offset,k,i,j,z]


                        max_index = vs.argmax()
                        pos = pos * num_parts_per_layer + max_index

                    out_map_mv[n,i_start,j_start,0] = pos 
    return out_map

def code_index_map_binary_tree(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                               np.ndarray[ndim=4,dtype=np.float64_t] weights, 
                               np.ndarray[ndim=1,dtype=np.float64_t] constant_terms,
                               int depth,
                               int threshold, 
                               outer_frame=0, 
                               min_percentile=None):
    cdef unsigned int part_x_dim = weights.shape[1]
    cdef unsigned int part_y_dim = weights.shape[2]
    cdef unsigned int part_z_dim = weights.shape[3]
    cdef unsigned int num_parts = weights.shape[0]
    cdef unsigned int num_parts_per_layer = 2 
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start, j_start, i_end, j_end, i, j, z, k, cx0, cx1, cy0, cy1, m, level, offset, base_offset, pos, mult

    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.ndarray[dtype=np.int64_t,ndim=4] out_map = -np.ones((n_samples,
                                                                 new_x_dim,
                                                                 new_y_dim,
                                                                 1), dtype=np.int64)
    cdef np.uint8_t[:,:,:,:] X_mv = X

    from scipy.special import logit
    #cdef np.ndarray[dtype=np.float64_t,ndim=4] weights = logit(parts[:,1]) - logit(parts[:,0]) 
    #cdef np.ndarray[dtype=np.float64_t,ndim=1] constant_terms = np.apply_over_axes(np.sum, np.log(1 - parts[:,1]) - np.log(1 - parts[:,0]), [1, 2, 3])[:,0,0,0]
    cdef np.float64_t[:,:,:,:] weights_mv = weights 
    cdef np.float64_t[:] constant_terms_mv = constant_terms 

    # Construct an array with only the keypoint parts

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    with nogil:
        for n in range(n_samples):
            for i in range(X_x_dim):
                for j in range(X_y_dim):
                    count = 0
                    for z in range(X_z_dim):
                        count += X_mv[n,i,j,z]
                    integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
            # Now accumulate the other axis
            for j in range(X_y_dim):
                for i in range(X_x_dim):
                    integral_counts[1+i,1+j] += integral_counts[i,1+j]

            base_offset = 0

            # Code parts
            for i_start in range(X_x_dim-part_x_dim+1):
                i_end = i_start + part_x_dim
                for j_start in range(X_y_dim-part_y_dim+1):
                    j_end = j_start + part_y_dim

                    # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                    cx0 = i_start+i_frame
                    cx1 = i_end-i_frame
                    cy0 = j_start+i_frame
                    cy1 = j_end-i_frame
                    count = integral_counts[cx1, cy1] - \
                            integral_counts[cx0, cy1] - \
                            integral_counts[cx1, cy0] + \
                            integral_counts[cx0, cy0]

                    if threshold <= count:
                        base_offset = 0
                        offset = 0
                        #print "Starting..."
                        pos = 0
                        mult = 1
                        for level in range(depth):
                            offset = base_offset + pos 
                            base_offset += mult 
                            mult *= num_parts_per_layer

                            #print 'offset', offset, 'base offset', base_offset

                            v = constant_terms_mv[offset] 
                            for i in range(part_x_dim):
                                for j in range(part_y_dim):
                                    for z in range(X_z_dim):
                                        if X_mv[n,i_start+i,j_start+j,z]:
                                            v += weights_mv[offset,i,j,z]

                            max_index = <int>(v > 0)
                            pos = pos * num_parts_per_layer + max_index

                        out_map_mv[n,i_start,j_start,0] = pos 
    return out_map

def code_index_map_binary_tree_new(np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                                   np.ndarray[ndim=2,dtype=np.int64_t] tree,
                                   np.ndarray[ndim=4,dtype=np.float64_t] weights, 
                                   np.ndarray[ndim=1,dtype=np.float64_t] constant_terms,
                                   int num_parts,
                                   int threshold, 
                                   outer_frame=0, 
                                   min_percentile=None):
    cdef unsigned int part_x_dim = weights.shape[1]
    cdef unsigned int part_y_dim = weights.shape[2]
    cdef unsigned int part_z_dim = weights.shape[3]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start, j_start, i_end, j_end, i, j, z, k, cx0, cx1, cy0, cy1, m, level, offset, base_offset, pos, mult, nextup, ref

    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.int64_t[:,:] tree_mv = tree

    cdef np.ndarray[dtype=np.int64_t,ndim=4] out_map = -np.ones((n_samples,
                                                                 new_x_dim,
                                                                 new_y_dim,
                                                                 1), dtype=np.int64)
    cdef np.uint8_t[:,:,:,:] X_mv = X

    from scipy.special import logit
    #cdef np.ndarray[dtype=np.float64_t,ndim=4] weights = logit(parts[:,1]) - logit(parts[:,0]) 
    #cdef np.ndarray[dtype=np.float64_t,ndim=1] constant_terms = np.apply_over_axes(np.sum, np.log(1 - parts[:,1]) - np.log(1 - parts[:,0]), [1, 2, 3])[:,0,0,0]
    cdef np.float64_t[:,:,:,:] weights_mv = weights 
    cdef np.float64_t[:] constant_terms_mv = constant_terms 

    # Construct an array with only the keypoint parts

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    with nogil:
        for n in range(n_samples):
            for i in range(X_x_dim):
                for j in range(X_y_dim):
                    count = 0
                    for z in range(X_z_dim):
                        count += X_mv[n,i,j,z]
                    integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
            # Now accumulate the other axis
            for j in range(X_y_dim):
                for i in range(X_x_dim):
                    integral_counts[1+i,1+j] += integral_counts[i,1+j]

            base_offset = 0

            # Code parts
            for i_start in range(X_x_dim-part_x_dim+1):
                i_end = i_start + part_x_dim
                for j_start in range(X_y_dim-part_y_dim+1):
                    j_end = j_start + part_y_dim

                    # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                    cx0 = i_start+i_frame
                    cx1 = i_end-i_frame
                    cy0 = j_start+i_frame
                    cy1 = j_end-i_frame
                    count = integral_counts[cx1, cy1] - \
                            integral_counts[cx0, cy1] - \
                            integral_counts[cx1, cy0] + \
                            integral_counts[cx0, cy0]

                    if threshold <= count:
                        base_offset = 0
                        offset = 0
                        #print "Starting..."
                        #pos = 0
                        mult = 1
                        while True:
                            nextup = tree_mv[offset,0]
                            ref = tree_mv[offset,1]

                            if nextup == -1:
                                out_map_mv[n,i_start,j_start,0] = ref
                                break
                            else:

                                #offset = base_offset + pos 
                                #base_offset += mult 
                                #mult *= num_parts_per_layer

                                #print 'offset', offset, 'base offset', base_offset

                                v = constant_terms_mv[nextup] 
                                for i in range(part_x_dim):
                                    for j in range(part_y_dim):
                                        for z in range(X_z_dim):
                                            if X_mv[n,i_start+i,j_start+j,z]:
                                                v += weights_mv[nextup,i,j,z]


                                offset = ref + <int>(v > 0)

    return out_map

def code_index_map_binary_tree_keypoints(
                                   np.ndarray[ndim=4,dtype=np.uint8_t] X, 
                                   np.ndarray[ndim=2,dtype=np.int64_t] tree,
                                   np.ndarray[ndim=4,dtype=np.float64_t] weights, 
                                   np.ndarray[ndim=1,dtype=np.float64_t] constant_terms,
                                   np.ndarray[ndim=3,dtype=np.int64_t] keypoints,
                                   np.ndarray[ndim=1,dtype=np.int64_t] num_keypoints,
                                   int num_parts,
                                   int threshold, 
                                   outer_frame=0, 
                                   min_percentile=None):
    cdef unsigned int part_x_dim = weights.shape[1]
    cdef unsigned int part_y_dim = weights.shape[2]
    cdef unsigned int part_z_dim = weights.shape[3]
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start, j_start, i_end, j_end, i, j, z, k, cx0, cx1, cy0, cy1, m, level, offset, base_offset, pos, mult, nextup, ref, kp

    cdef int count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges

    cdef np.int64_t[:,:] tree_mv = tree

    cdef np.ndarray[dtype=np.int64_t,ndim=4] out_map = -np.ones((n_samples,
                                                                 new_x_dim,
                                                                 new_y_dim,
                                                                 1), dtype=np.int64)
    cdef np.uint8_t[:,:,:,:] X_mv = X

    from scipy.special import logit
    #cdef np.ndarray[dtype=np.float64_t,ndim=4] weights = logit(parts[:,1]) - logit(parts[:,0]) 
    #cdef np.ndarray[dtype=np.float64_t,ndim=1] constant_terms = np.apply_over_axes(np.sum, np.log(1 - parts[:,1]) - np.log(1 - parts[:,0]), [1, 2, 3])[:,0,0,0]
    cdef np.float64_t[:,:,:,:] weights_mv = weights 
    cdef np.float64_t[:] constant_terms_mv = constant_terms 

    # Construct an array with only the keypoint parts

    cdef np.int64_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.int64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.int64)
    cdef np.int64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t v
    cdef int max_index 

    cdef np.int64_t[:,:,:] keypoints_mv = keypoints 
    cdef np.int64_t[:] num_keypoints_mv = num_keypoints 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    with nogil:
        for n in range(n_samples):
            for i in range(X_x_dim):
                for j in range(X_y_dim):
                    count = 0
                    for z in range(X_z_dim):
                        count += X_mv[n,i,j,z]
                    integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
            # Now accumulate the other axis
            for j in range(X_y_dim):
                for i in range(X_x_dim):
                    integral_counts[1+i,1+j] += integral_counts[i,1+j]

            base_offset = 0

            # Code parts
            for i_start in range(X_x_dim-part_x_dim+1):
                i_end = i_start + part_x_dim
                for j_start in range(X_y_dim-part_y_dim+1):
                    j_end = j_start + part_y_dim

                    # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                    cx0 = i_start+i_frame
                    cx1 = i_end-i_frame
                    cy0 = j_start+i_frame
                    cy1 = j_end-i_frame
                    count = integral_counts[cx1, cy1] - \
                            integral_counts[cx0, cy1] - \
                            integral_counts[cx1, cy0] + \
                            integral_counts[cx0, cy0]

                    if threshold <= count:
                        base_offset = 0
                        offset = 0
                        mult = 1
                        while True:
                            nextup = tree_mv[offset,0]
                            ref = tree_mv[offset,1]

                            if nextup == -1:
                                out_map_mv[n,i_start,j_start,0] = ref
                                break
                            else:
                                v = constant_terms_mv[nextup] 
                                for kp in range(num_keypoints_mv[nextup]):
                                    i = keypoints_mv[nextup,kp,0]
                                    j = keypoints_mv[nextup,kp,1]
                                    z = keypoints_mv[nextup,kp,2]

                                    if X_mv[n,i_start+i,j_start+j,z]:
                                        v += weights_mv[nextup,i,j,z]

                                offset = ref + <int>(v > 0)

    return out_map



