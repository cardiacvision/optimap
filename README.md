# optimap

### Python toolbox for analyzing optical mapping data

``optimap`` is an open-source Python toolbox for exploring, visualizing, and analyzing high-speed fluorescence imaging data with a focus on cardiac optical mapping data. It includes modules for data input/output, processing scientific video recordings, visualization, motion compensation, trace extraction, and analysis.

> ⚠️ optimap is currently in early development, expect breaking changes and bugs.

## Installation
### Installing pre-built binaries (Mac OSX, Windows, Linux)

```bash
pip install opticalmapping[all]
```

will install optimap and all recommended dependencies (including OpenCV and PySide2). If you wish to install your own version of OpenCV (e.g. for CUDA support) or Qt implementation use

```bash
pip install opticalmapping
```

instead. See [Installing Optimap](https://optimap.readthedocs.io/en/latest/chapters/getting_started/#installing-optimap) for more information.

## Getting Started
See the [Getting Started](https://optimap.readthedocs.io/en/latest/chapters/getting_started/) guide for installation instructions and a quick introduction to optimap. See the [Tutorials](https://optimap.readthedocs.io/en/latest/tutorials/basics/) for more detailed examples.

## Links

* [Documentation](https://optimap.readthedocs.io)
* [Issue tracker](https://github.com/cardiacvision/optimap/issues)
* [Source code](https://github.com/cardiacvision/optimap)

## License

optimap is licensed under the [MIT License](https://github.com/cardiacvision/optimap/blob/main/LICENSE.md).