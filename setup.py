import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

ext_modules = [Extension("convolve", ["my_cython/convolve.pyx"], include_dirs = [numpy.get_include(),'.','/opt/local/include'],
                extra_compile_args = ['-O3'])]

setup(
  name = 'convolve',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)