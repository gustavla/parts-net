import sys
import itertools as itr
import numpy as np

if sys.version_info >= (3,):
    imap_unordered = map
    imap = map
else:
    imap_unordered = itr.imap
    imap = itr.imap
starmap_unordered = itr.starmap
starmap = itr.starmap

def main(name=None):
    return name == '__main__'
