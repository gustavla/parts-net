from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import extension
import numpy as np

setup(
    name = "pnet",
    ext_modules = cythonize('pnet/*.pyx'), # accepts a glob pattern
    include_dirs = [np.get_include()],

    # Uncomment for debugging
    #extra_compile_args=["-g"],
    #extra_link_args=["-g"],
)

