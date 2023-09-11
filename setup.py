import os
import site
import sys

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

# pip bug workaround https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

import numpy
from extension_helpers import add_openmp_flags_if_available

# Available at setup time due to pyproject.toml (use pip install .)
from pybind11.setup_helpers import Pybind11Extension, build_ext

# universal stable ABI is not supported for PyBind11 :-(
cpp_module = Pybind11Extension("optimap._cpp", ["optimap/_cpp/lib.cpp"], cxx_std=17)
cpp_module.include_dirs.append(os.path.join(PROJECT_ROOT, "optimap/_cpp/include"))
cpp_module.include_dirs.append(numpy.get_include())

if add_openmp_flags_if_available(cpp_module):
    cpp_module.extra_compile_args.append("-DUSE_OMP")

setup(
    name="opticalmapping",
    url="https://github.com/cardiacvision/optimap",
    description="A toolbox for analyzing optical mapping and fluorescence imaging data.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics"
        ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-image",
        "scikit-video",
        "tqdm",
        "pooch",
        "seasonal",
        ],
    tests_require=["pytest"],
    ext_modules=[cpp_module],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    extras_require={
        "all": [
            "opencv-contrib-python",
            "PySide6",
            "pycairo",
        ],
        "test": [
            "pytest"
        ],
        "docs": [
            "sphinx",
            "sphinxcontrib-napoleon",
            "sphinx-autobuild",
            "sphinx-copybutton",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-video",
            "furo",
            "myst_nb @ git+https://github.com/executablebooks/MyST-NB.git",  # TODO: use pypi once released
            "jupytext",
        ]
    }
)
