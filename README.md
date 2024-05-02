# optimap

[![docs](https://readthedocs.org/projects/optimap/badge/?version=latest&style=)](https://optimap.readthedocs.org)
[![tests](https://github.com/cardiacvision/optimap/actions/workflows/main.yml/badge.svg)](https://github.com/cardiacvision/optimap/actions/workflows/main.yml)
[![PyPI](https://img.shields.io/pypi/v/opticalmapping.svg)](https://pypi.org/project/opticalmapping/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/opticalmapping.svg)](https://python.org)
[![DOI](https://zenodo.org/badge/677528623.svg)](https://zenodo.org/badge/latestdoi/677528623)

### optimap: An open-source library for the processing of fluorescence video data

`optimap` is an open-source Python toolbox for exploring, visualizing, and analyzing high-speed fluorescence imaging data with a focus on cardiac optical mapping data. It includes modules for loading, processing and exporting videos, extracting and measuring optical traces, visualizing action potential or calcium waves, tracking motion and compensating motion artifacts, computing activation maps, conduction velocities, action potential durations, as well as measuring contractility and further analyzing and visualizing the results. Refer to the [Tutorials](https://optimap.readthedocs.io/en/latest/tutorials/) and the [Documentation](https://optimap.readthedocs.io/en/latest/) for more information about optimap's features.

> ⚠️ optimap is currently in early development, expect breaking changes and bugs.

## Installation

`optimap` is available for macOS, Windows and Linux, see the [Getting Started](https://optimap.readthedocs.io/en/latest/chapters/getting_started/) guide for more information.

### Installing pre-built binaries (macOS, Windows, Linux)

Pre-built binaries can be installed using pip:

```bash
pip install opticalmapping[all]
```

The above command will install optimap and all recommended dependencies including OpenCV and PySide2. If you wish to install your own version of OpenCV (e.g. for CUDA support) or Qt implementation use:

```bash
pip install opticalmapping
```

To update optimap to the latest version run

```bash
pip install --upgrade opticalmapping[all]
```

## About optimap

`optimap` is an interactive, script or notebook-based software library created for cardiovascular scientists in particular, but might also be useful for scientists in other fields. For instance, when performing calcium imaging or physiological research with moving cells or tissues. It is designed to be a flexible and customizable analysis workflow toolkit, which allows for a wide range of analyses and visualizations. See the [Tutorials](https://optimap.readthedocs.io/en/latest/tutorials/) for examples and more information about the usage of `optimap`. The tutorials can be downloaded by clicking on the link in the green box at the top of each tutorial page.

`optimap` is developed by members of the [Cardiac Vision Laboratory](https://cardiacvision.ucsf.edu) at the [University of California, San Franicsco](https://www.ucsf.edu). It is open-source, freely available, and relies on open-source packages such as NumPy, SciPy, Matplotlib and OpenCV.

## Links

- [Documentation](https://optimap.readthedocs.io)
- [Issue tracker](https://github.com/cardiacvision/optimap/issues)
- [Source code](https://github.com/cardiacvision/optimap)

## Contributing

We welcome bug reports, questions, ideas for new features and pull-requests to fix issues or add new features to optimap. See [Contributing](https://optimap.readthedocs.io/en/latest/chapters/contributing/) for more information.

## License

optimap is licensed under the [MIT License](https://github.com/cardiacvision/optimap/blob/main/LICENSE.md).
