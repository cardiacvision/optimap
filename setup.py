# ruff: noqa: I001,D100
import os
from pathlib import Path

from setuptools import find_packages, setup
# Available at setup time due to pyproject.toml (use pip install .)
import numpy
from extension_helpers import add_openmp_flags_if_available
from pybind11.setup_helpers import Pybind11Extension, build_ext

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

# universal stable ABI is not supported for PyBind11 :-(
cpp_module = Pybind11Extension("optimap._cpp", ["optimap/_cpp/lib.cpp"], cxx_std=17)
cpp_module.include_dirs.append(os.path.join(PROJECT_ROOT, "optimap/_cpp/include"))
cpp_module.include_dirs.append(numpy.get_include())

if add_openmp_flags_if_available(cpp_module):
    cpp_module.extra_compile_args.append("-DUSE_OMP")

core_requirements = [
    "matplotlib",
    "mpl-pan-zoom",
    "numpy",
    "packaging",
    "Pillow>=10.0.1",  # not a strict requirement, but it makes importing 16-bit PNGs more consistent
    "pooch",
    "pymatreader",
    "scikit-image",
    "scikit-video",
    "scipy",
    "seasonal",
    "static_ffmpeg",
    "tqdm",
]

recommended_requirements = [
    "monochrome",
    "opencv-contrib-python",
    "PySide6",
]

# Update requirements-core.txt if needed
core_reqs_fn = Path(__file__).parent / "requirements-core.txt"
core_reqs_fn.write_text("\n".join(core_requirements))
if (core_reqs_fn.read_text().strip().splitlines() != core_requirements):
    core_reqs_fn.write_text("\n".join(core_requirements))

setup(
    name="optimap",
    url="https://github.com/cardiacvision/optimap",
    author="Jan Lebert",
    author_email="jan.lebert@ucsf.edu",
    license="MIT",
    description="A toolbox for analyzing optical mapping and fluorescence imaging data.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={"optimap": ["assets/*.png"]},
    install_requires=core_requirements + recommended_requirements,
    tests_require=["pytest"],
    ext_modules=[cpp_module],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    extras_require={
        "test": ["pytest"],
        "docs": [
            "sphinx>=7.4.6",
            "sphinxcontrib-napoleon",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-video",
            "sphinx-autobuild",
            "sphinx-copybutton",
            "sphinx-codeautolink",
            "furo",
            "myst_nb>=1.0.0",
            "jupytext",
            "jupyter-cache",
            "sphinx-remove-toctrees",
            "sphinx-design",
            "sphinx-togglebutton",
        ],
    },
)
