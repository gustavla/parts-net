from __future__ import division, print_function, absolute_import

import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import os
from copy import copy

__all__ = ['VzLog', 'default']

DEFAULT_NAME = os.environ.get('VZ_DEFAULT_NAME', 'vzlog')
_g_initialized_vz_names = set()

_HEADER = """
<!DOCTYPE html>
<html>
<head>
    <title>Log</title>
    <style>
    html {
        background: white;
        font-family: Verdana;
        font-size: 12px
    }

    div.section {
        border-bottom: 2px solid gray;
    }

    p.title {
        text-weight: bold;        
    }
    </style>
</head>
<body>
"""

_FOOTER = """
</body>
</html>
"""

class VzLog(object):
    def __init__(self, name):
        self.name = name
        self._file_rights = os.environ.get('VZ_FILE_RIGHTS')
        #self._filenames = set()
        self._filename_stack = set() 
        if self._file_rights is not None:
            self._file_rights = int(self._file_rights, 8)

        self.initialize()

        self._counter = 0

    #def __del__(self):
        #if self._open:
        #    self.finalize()
        #pass

    def register_filename(self, fn):
        self._filename_stack.add(fn)

    def set_rights(self, fn):
        if self._file_rights is not None:
            return os.chmod(fn, self._file_rights)

    def update_rights(self):
        for fn in copy(self._filename_stack):
            if os.path.exists(fn):
                self.set_rights(fn)
                self._filename_stack.remove(fn)

    def initialize(self):
        """
        Initialize folder for logging.
        """
        global _g_initialized_vz_names

        # Construct path. Note that if 
        root = self._get_root()
        dot_vz_fn = os.path.join(root, '.vz')
        
        # First, remove previous folder. Only do this if it looks like
        # it was previously created with vz. Otherwise, through an error
        if os.path.isdir(root):
            # Check if it has a '.vz' file
            if os.path.exists(dot_vz_fn):
                # Delete the whole directory
                import shutil
                shutil.rmtree(root)
            else:
                raise Exception("Folder does not seem to be a vz folder. Aborting.") 

        self._open = True

        # Create folder
        os.mkdir(root)

        self._output_html(_HEADER)

        with open(dot_vz_fn, 'w') as f:
            print('ok', file=f)

    def _get_root(self):
        return os.path.join(os.path.expandvars(os.path.expanduser(os.environ.get('VZ_DIR', ''))), self.name)

    def _output_surrounding_html(self, prefix, suffix, *args):
        self._output_html(*((prefix,) + args + (suffix,)))

    def _output_html(self, *args):
        with open(os.path.join(self._get_root(), 'index.html'), 'a') as f:
            print(*args, file=f)

    def finalize(self):
        self._output_html(_FOOTER) 
        self.register_filename(os.path.join(self._get_root(), 'index.html'))
        self.update_rights()
        self.set_rights(self._get_root())

        if self._filename_stack:
            self.log("WARNING: Could not finalize these files: {}".format(self._filename_stack))

        self._open = False

    def generate_filename(self, ext='png'):
        #fn = '-'.join(title.lower().split()) + '.png'
        fn = 'plot-{:04}.'.format(self._counter) + ext
        self._counter += 1
        
        #self._output_html('<p class="title">' + title + '</p>')
        self._output_surrounding_html('<div>', '</div>', '<img src="{}" />'.format(fn))

        self.register_filename(os.path.join(self._get_root(), fn))
        # The file won't exist yet, but this can still update older files
        self.update_rights()
        return os.path.join(self._get_root(), fn)

    def log(self, *args):
        self._output_surrounding_html('<pre>', '</pre>', *args)

    def title(self, *args):
        self._output_surrounding_html('<h1>', '</h1>', *args)

    def section(self, *args):
        self._output_surrounding_html('<h2>', '</h2>', *args)

    def text(self, *args):
        self._output_surrounding_html('<p>', '</p>', *args)




default = VzLog(DEFAULT_NAME)

#def savefig(name
