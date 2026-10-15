# builds the official FiRE extension: copy python/FiRE from the checkout next to this file, as setup.py
import os, sys
from setuptools import setup, Extension
from Cython.Build import cythonize
inc = os.environ.get("BOOST_INC", "/usr/include")
setup(name="FiRE", ext_modules=cythonize(Extension("FiRE", sources=["FiRE.pyx", "cppFiRE.cpp"], language="c++", include_dirs=[inc],
      extra_compile_args=["-std=c++14", "-O2"]), language_level=3))
