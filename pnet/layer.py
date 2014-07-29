from __future__ import division, print_function, absolute_import

from .saveable import SaveableRegistry
import pnet

@SaveableRegistry.root
class Layer(SaveableRegistry):
    def train(self, phi, data, y=None):
        pass

    def extract(self, phi, data):
        raise NotImplemented("Subclass and override to use")

    @property
    def trained(self):
        return True

    @property
    def supervised(self):
        return False

    @property
    def classifier(self):
        return False

    def conv_pos_matrix(self, kernel_shape):
        """
        Helper function that can be used to set pos_matrix for
        convolution operations.
        """
        return pnet.matrix.translation(-(kernel_shape[0] - 1) / 2,
                                       -(kernel_shape[1] - 1) / 2)

    @property
    def pos_matrix(self):
        """
        Returns an affine matrix that translates a position in the feature
        space in the incoming layer to a position in the feature space in
        this layer.
        """
        return pnet.matrix.identity()


class UnsupervisedLayer(Layer):
    @property
    def supervised(self):
        return False


class SupervisedLayer(Layer):
    @property
    def supervised(self):
        return True
