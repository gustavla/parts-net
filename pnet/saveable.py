from __future__ import absolute_import
from .named_registry import NamedRegistry

# Which backend (it needs to have load and save)
#import numpy as backend 
import amitgroup.io as backend

class Saveable(object):
    @classmethod
    def load(cls, path):
        if path is None:
            return cls.load_from_dict({})
        else:
            d = backend.load(path)#.flat[0]
            return cls.load_from_dict(d)
        
    def save(self, path):
        backend.save(path, self.save_to_dict())

    @classmethod
    def load_from_dict(cls, d):
        raise NotImplementedError("Must override load_from_dict for Saveable interface")

    def save_to_dict(self):
        raise NotImplementedError("Must override save_to_dict for Saveable interface")


class SaveableRegistry(Saveable, NamedRegistry):
    @classmethod
    def load(cls, path):
        if path is None:
            return cls.load_from_dict({})
        else:
            d = backend.load(path)#.flat[0]
            # Check class type
            class_name = d.get('name')
            if class_name is not None:
                return cls.getclass(class_name).load_from_dict(d)
            else:
                return cls.load_from_dict(d)

    def save(self, path):
        d = self.save_to_dict()
        d['name'] = self.name 
        import pdb; pdb.set_trace()
        backend.save(path, d)
     
