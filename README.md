# optimap
[![docs](https://readthedocs.org/projects/optimap/badge/?version=latest&style=)](https://optimap.readthedocs.org)
[![tests](https://github.com/cardiacvision/optimap/actions/workflows/main.yml/badge.svg)](https://github.com/cardiacvision/optimap/actions/workflows/main.yml)
[![PyPI](https://img.shields.io/pypi/v/opticalmapping.svg)](https://pypi.org/project/opticalmapping/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/opticalmapping.svg)](https://python.org)
[![DOI](https://zenodo.org/badge/677528623.svg)](https://zenodo.org/badge/latestdoi/677528623)


### optimap: An open-source library for the processing of fluorescence video data

``optimap`` is an open-source Python toolbox for exploring, visualizing, and analyzing high-speed fluorescence imaging data with a focus on cardiac optical mapping data. It includes modules for loading, processing and exporting videos, extracting and measuring optical traces, action potential or calcium waves, performing motion compensation, spatio-temporal smoothing, measuring contractility and further post-processing, analyzing and visualizing the results.

> ⚠️ optimap is currently in early development, expect breaking changes and bugs.

## Installation
### Installing pre-built binaries (Mac OSX, Windows, Linux)

optimap is available for Mac OSX, Windows and Linux. See [Installing Optimap](https://optimap.readthedocs.io/en/latest/chapters/getting_started/#installing-optimap) for more detailed information regarding the installation of optimap.

```bash
pip install opticalmapping[all]
```

will install optimap and all recommended dependencies (including OpenCV and PySide2). If you wish to install your own version of OpenCV (e.g. for CUDA support) or Qt implementation use

```bash
pip install opticalmapping
```
optimap is a script-based software package, which means that you run Python-based analysis scripts rather than working with a graphical user interface.

## Getting Started
We provide several examples which explain the usage of optimap, see [Tutorials](https://optimap.readthedocs.io/en/latest/tutorials/basics/). See the [Getting Started](https://optimap.readthedocs.io/en/latest/chapters/getting_started/) guide for installation instructions and a quick introduction to optimap.

## Links

* [Documentation](https://optimap.readthedocs.io)
* [Issue tracker](https://github.com/cardiacvision/optimap/issues)
* [Source code](https://github.com/cardiacvision/optimap)

## Contributing

We welcome bug reports, questions, ideas for new features and pull-requests to fix issues or add new features to optimap. See [Contributing](https://optimap.readthedocs.io/en/latest/chapters/contributing/) for more information.

## License

optimap is licensed under the [MIT License](https://github.com/cardiacvision/optimap/blob/main/LICENSE.md).