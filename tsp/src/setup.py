"""
Setup script for building the faco_cpp Python extension.

Build with:
    pip install -e ./tsp/faco
    
Or manually:
    cd tsp/faco && python setup.py build_ext --inplace
"""

import sys
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Try to import pybind11 - will fail if not installed
try:
    import pybind11
except ImportError:
    print("pybind11 is required. Install with: pip install pybind11")
    sys.exit(1)


class BuildExt(build_ext):
    """Custom build extension to add compiler-specific flags."""
    
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/openmp'],
        'unix': ['-O3', '-march=native', '-ffast-math', '-fopenmp'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append('-std=c++17')
            opts.append('-fvisibility=hidden')
            # Check for Apple clang which needs different OpenMP flags
            if sys.platform == 'darwin':
                # macOS with Homebrew LLVM
                opts = [o for o in opts if o != '-fopenmp']
                link_opts = [o for o in link_opts if o != '-fopenmp']
                # Try Homebrew libomp
                opts.append('-Xpreprocessor')
                opts.append('-fopenmp')
                link_opts.append('-lomp')
        elif ct == 'msvc':
            opts.append('/std:c++17')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
        build_ext.build_extensions(self)


# Extension module
ext_modules = [
    Extension(
        'faco_cpp',
        sources=[
            'mfaco_train.cpp',
            'binding.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '.',  # Current directory for headers
        ],
        define_macros=[
            ('FACO_BUILD_PYEXT', '1'),
        ],
        language='c++',
    ),
]

setup(
    name='faco_cpp',
    version='0.1.0',
    author='NGFACO',
    description='C++ MFACO Training Module for fast neural-guided ACO training',
    long_description='Fast C++ implementation of MFACO for TSP with pybind11 bindings',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'pybind11>=2.6',
        'numpy>=1.19',
    ],
    python_requires='>=3.8',
    zip_safe=False,
)
