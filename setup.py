from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "test_cython", 
        ["test_cython.pyx"], 
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],)
