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


# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(os.path.dirname(here), 'src')  # ../src relative to faco/

# Extension module
ext_modules = [
    Extension(
        'faco_cpp',
        sources=[
            os.path.join(src_dir, 'mfaco_train.cpp'),
            os.path.join(src_dir, 'binding.cpp'),
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            src_dir,  # Directory containing headers
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
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    # Explicitly disable package discovery
    packages=[],
    py_modules=[],
    zip_safe=False,
)
