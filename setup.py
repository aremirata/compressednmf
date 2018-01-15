from distutils.core import setup
from distutils.extension import Extension
import numpy
import scipy

setup(
    name='nmf_compressed',
    version='0.1.0',
    description='Factorization methods in python and cython',
    author='Alger Remirata',
    author_email='abremirata21@gmail.com',
    packages=[
        'nmf_compressed',
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],
)

