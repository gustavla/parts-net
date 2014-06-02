import os
import struct
#from array import array as pyarray
import numpy as np
#import amit.hdf5

def parse_background_random(path=None):
    """
    Parse the 2 files
        mnist_background_random_train.amat
        mnist_background_random_test.amat
    for MNIST with random background from a folder, store them as a Python
    dictionary of C-contiguous np.uint8 NumPy arrays, and save to disk using
    amit.hdf5.save().
    
    References
    ----------
        http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007#Downloadable_datasets
    """
    if path is None:
        path = os.environ['RAND_MNIST']

    # Parse training images.
    training = np.loadtxt(os.path.join(path, 'mnist_background_random_train.amat'))
    training_image = np.ascontiguousarray((training[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    training_label = (training[:,-1]).astype(np.uint8)
    
    # Parse test images.
    test = np.loadtxt(os.path.join(path, 'mnist_background_random_test.amat'))
    test_image = np.ascontiguousarray((test[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    test_label = (test[:,-1]).astype(np.uint8)

    # Save together.
    data = {'training_image':training_image.astype(np.float64)/255.0, 'training_label':training_label,
            'test_image':test_image.astype(np.float64)/255.0, 'test_label':test_label}
    #amit.hdf5.save(os.path.join(path, filename), data)

    return data

def parse_background_images(path=None):
    if path is None:
        path = os.environ['BKG_MNIST']

    # Parse training images.
    training = np.loadtxt(os.path.join(path, 'mnist_background_images_train.amat'))
    training_image = np.ascontiguousarray((training[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    training_label = (training[:,-1]).astype(np.uint8)
    
    # Parse test images.
    test = np.loadtxt(os.path.join(path, 'mnist_background_images_test.amat'))
    test_image = np.ascontiguousarray((test[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    test_label = (test[:,-1]).astype(np.uint8)

    # Save together.
    data = {'training_image':training_image.astype(np.float64)/255.0, 'training_label':training_label,
            'test_image':test_image.astype(np.float64)/255.0, 'test_label':test_label}
    #amit.hdf5.save(os.path.join(path, filename), data)

    return data

<<<<<<< HEAD



def parse_basic_mnist(path=None):
=======
def parse_basic(path=None):
>>>>>>> 019b90c462ba3a85d703a16ecab0417f63585f86
    if path is None:
        path = os.environ['BASIC_MNIST']

    # Parse training images.
    training = np.loadtxt(os.path.join(path, 'mnist_train.amat'))
    training_image = np.ascontiguousarray((training[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)))
    training_label = (training[:,-1]).astype(np.uint8)
    
    # Parse test images.
    test = np.loadtxt(os.path.join(path, 'mnist_test.amat'))
    test_image = np.ascontiguousarray((test[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)))
    test_label = (test[:,-1]).astype(np.uint8)

    # Save together.
    data = {'training_image':training_image.astype(np.float64)/255.0, 'training_label':training_label,
            'test_image':test_image.astype(np.float64)/255.0, 'test_label':test_label}
    #amit.hdf5.save(os.path.join(path, filename), data)

    return data

<<<<<<< HEAD



def parse_rotated_mnist(path=None):
=======
def parse_rotation(path=None):
>>>>>>> 019b90c462ba3a85d703a16ecab0417f63585f86
    if path is None:
        path = os.environ['ROT_MNIST']

    # Parse training images.
    training = np.loadtxt(os.path.join(path, 'mnist_all_rotation_normalized_float_train_valid.amat'))
    training_image = np.ascontiguousarray((training[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    training_label = (training[:,-1]).astype(np.uint8)
    
    # Parse test images.
    test = np.loadtxt(os.path.join(path, 'mnist_all_rotation_normalized_float_test.amat'))
    test_image = np.ascontiguousarray((test[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    test_label = (test[:,-1]).astype(np.uint8)

    # Save together.
    data = {'training_image':training_image.astype(np.float64)/255.0, 'training_label':training_label,
            'test_image':test_image.astype(np.float64)/255.0, 'test_label':test_label}
    #amit.hdf5.save(os.path.join(path, filename), data)

    return data

<<<<<<< HEAD




def parse_rotated_img_mnist(path=None):
    if path is None:
        path = os.environ['ROT_IMG_MNIST']
=======
def parse_rotation_images(path=None):
    if path is None:
        path = os.environ['ROTBKG_MNIST']
>>>>>>> 019b90c462ba3a85d703a16ecab0417f63585f86

    # Parse training images.
    training = np.loadtxt(os.path.join(path, 'mnist_all_background_images_rotation_normalized_train_valid.amat'))
    training_image = np.ascontiguousarray((training[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    training_label = (training[:,-1]).astype(np.uint8)
    
    # Parse test images.
    test = np.loadtxt(os.path.join(path, 'mnist_all_background_images_rotation_normalized_test.amat'))
    test_image = np.ascontiguousarray((test[:,:-1] * 255).astype(np.uint8).reshape((-1,28,28)).transpose((0,2,1)))
    test_label = (test[:,-1]).astype(np.uint8)

    # Save together.
    data = {'training_image':training_image.astype(np.float64)/255.0, 'training_label':training_label,
            'test_image':test_image.astype(np.float64)/255.0, 'test_label':test_label}
    #amit.hdf5.save(os.path.join(path, filename), data)

    return data
