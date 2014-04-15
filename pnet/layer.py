from __future__ import division, print_function, absolute_import 

from .saveable import SaveableRegistry

@SaveableRegistry.root
class Layer(SaveableRegistry):

    def train(self, X):
        pass 

    def extract(self, X):
        raise NotImplemented("Subclass and override to use")

    @property
    def trained(self):
        return True

    def infoplot(self, vz):
        pass
    
    #@property(self):
    #def output_shape(self):
        #raise NotImplemented("Subclass and override to use")

