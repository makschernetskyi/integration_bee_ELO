from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define modules
extensions = [
    Extension("player", ["player.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_run_tournament", ["cython_run_tournament.pyx"], include_dirs=[numpy.get_include()]),
]

# Setup
setup(
    ext_modules=cythonize(extensions, language_level="3"),
)