from __future__ import division, print_function, absolute_import 

from .saveable import SaveableRegistry

@SaveableRegistry.root
class Layer(SaveableRegistry):
    def train(self, X, Y=None):
        pass 

    def extract(self, X):
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

    def infoplot(self, vz):
        pass
    
    #@property(self):
    #def output_shape(self):
        #raise NotImplemented("Subclass and override to use")
    

class UnsupervisedLayer(Layer):
    pass

class SupervisedLayer(Layer):
    @property
    def supervised(self):
        return True
