from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "fourier_grid_ham.fgh_fast",
        ["fourier_grid_ham/fgh_fast.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
    )
]

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name="fourier_grid_ham",
    version="0.0.1",
    author="Ryan Pederson",
    author_email="pedersor@uci.edu",
    description="Minimal Fourier Grid Hamiltonian (FGH) method implementation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=('>=3.8.0'),
    install_requires=requirements,
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
