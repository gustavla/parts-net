from __future__ import division, print_function, absolute_import 

from .saveable import SaveableRegistry

@SaveableRegistry.root
class Layer(SaveableRegistry):

    def train(self, X):
        pass 

    def extract(self, X):
        raise NotImplemented("Subclass and override to use")

