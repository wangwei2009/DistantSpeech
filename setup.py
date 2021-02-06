#! /usr/bin/env python

# from distutils.core import setup, Extension
# import numpy
# import os

# os.environ['CC'] = 'g++';
# setup(name='matt_simple_test', version='1.0', ext_modules =[Extension('_mcra',
#  ['mcra.c', 'mcra.i'], include_dirs = [numpy.get_include(),'.'])])



# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# ezrange extension module
_mcra = Extension("_mcra",
                   ["mcra.i","mcra.c"],
                   include_dirs = [numpy_include],
                   )

# ezrange setup
setup(  name        = "range function",
        description = "range takes an integer and returns an n element int array where each element is equal to its index",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_mcra]
        )