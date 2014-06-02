from __future__ import division, print_function, absolute_import
import numpy as np
import math

def rotate_location(location, org_shape, angle, rotated_shape):
    """
    When a rectangular region of the given original shape is rotated around its
    center counter-clockwise by the given angle and being embedded into another
    rectangular region with the same center, where is the rotated location
    relative to this embedding rectangular region?

    Parameters
    ----------
    location : (num_location,2)-shape NumPy array
        The original locations relative to the top left corner of the original
        grid.
    org_shape : 2-tuple
        It is (n_org,m_org) of the original grid.
    angle : number
        The rotation angle in degrees in counter-clockwise.
    rotated_shape : 2-tuple
        It is (n_rotated,m_rotated) of the embedding grid.

    Returns
    -------
    location : (num_location,2)-shape np.float64 NumPy array
        The rotated locations relative to the top left corner of the embedding
        grid.
    """
    y = 0.5 * (org_shape[0] - 1) - location[:,0]
    x = location[:,1] - 0.5 * (org_shape[1] - 1)
    cos_angle = math.cos(angle * math.pi / 180)
    sin_angle = math.sin(angle * math.pi / 180)

    location = np.empty_like(location, dtype=np.float64)
    location[:,0] = 0.5 * (rotated_shape[0] - 1) - (sin_angle * x + cos_angle * y)
    location[:,1] = (cos_angle * x - sin_angle * y) + 0.5 * (rotated_shape[1] - 1)
    return location


def rotate_patch_map(patch_map, angle):
    """
    Rotate the patch map by the given angle.

    Parameters
    ----------
    patch_map : (n,m,num_patch_model,num_angle)-shape np.bool_ NumPy array
        The patch map.  Assume the underlying angles are equally spaced in
        [0, 360).
    angle : number
        The rotation angle in degrees in counter-clockwise.

    Returns
    -------
    patch_map : (n,m,num_patch_model,num_angle)-shape np.bool_ NumPy array
        The rotated patch map.
    """
    n, m, _, num_angle = patch_map.shape
    # Identify patch locations.
    location_i, location_j, which_m, which_a = np.nonzero(patch_map)
    if location_i.size == 0:
        return np.copy(patch_map)
    location = np.vstack((location_i, location_j)).transpose()
    # Rotate the locations.
    location = rotate_location(location, (n,m), angle, (n,m))
    # Keep only inbound locations.
    is_inside = np.logical_and(np.all(location >= 0, axis=1),
                               np.all(location < np.array([n,m]), axis=1))
    if not np.all(is_inside):
        location = location[is_inside]
        which_m = which_m[is_inside]
        which_a = which_a[is_inside]
    location = location.astype(np.int_) # Floor np.float64 to np.int_.
    # Rotate the angle indices.
    a = np.mod(which_a * (360 / num_angle) + angle, 360)
    which_a = np.round(a / (360 / num_angle)).astype(np.int_)
    which_a[which_a == num_angle] = 0
    # Put together.
    patch_map = np.zeros_like(patch_map)
    patch_map[location[:,0], location[:,1], which_m, which_a] = True
    return patch_map

