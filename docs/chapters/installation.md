(installing)=
# Installing optimap

``optimap`` is available for Windows, Mac OSX, and Linux.

``optimap`` requires Python 3.8 or later. To install [Python](https://en.wikipedia.org/wiki/Python_programming_language) we recommend installing the [Anaconda distribution](https://www.anaconda.com/distribution/), which includes Python and many useful packages for scientific computing, or by installing [Python directly](https://code.visualstudio.com/docs/python/python-tutorial#_install-a-python-interpreter).

```{tip}
optimap relies heavily on [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org). We recommend the [Scientific Python Lectures](https://lectures.scientific-python.org) and the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for an introduction to Python and these libraries.
```

## Installing optimap on Windows

Installing `optimap` on Windows requires Python to be installed on the system.

## Installing optimap on Mac OSX

Installing `optimap` on Mac OSX requires [Homebrew](https://brew.sh/). `Homebrew` is a popular and very useful package manager which is freely available for Mac OSX. Paste and execute the following command in the Mac OSX terminal to install `Homebrew`: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`. Note that Mac OSX has a native Python version installed and it is necessary to install a separate Python 3.8 version or later. Next, you need to install `pip`, which is a package installer specifically for Python packages. You can install `pip` using Homebrew: `brew install pip`. `pip` is a 

## Installing optimap on Linux

With `pip` and `python` already installed it is very straight-forward to install `optimap` on Linux systems:

```{code-block} bash
pip install opticalmapping[all]
```

This will install


The easiest way to install optimap is using `pip` in the command line:

```{code-block} bash
pip install opticalmapping[all]
```

If this command fails, please try the following:
```{code-block} bash
python -m pip install opticalmapping\[all\]
```

```{note}
`pip install opticalmapping` will install the core functionality of optimap. However, the [OpenCV](https://opencv.org/) and dependencies are not installed by default which are required for the {mod}`optimap.motion` and {mod}`optimap.video` modules. To install these dependencies use `pip install opticalmapping[all]` or install OpenCV manually.

To use GPU-accelerated motion tracking algorithms a CUDA-enabled version of OpenCV is required, which is currently not available on PyPI. See [](#opencv) for more information.
```
To update optimap to the latest version run `pip install --upgrade opticalmapping[all]`. See [](#contributing) for instructions on how to install optimap from source.




## GPU Acceleration

By default, `optimap` will be installed without GPU-acceleration (graphics processing unit). 





# How to cite

A publication about optimap is currently in preparation. In the meantime, please cite the following papers if you use optimap in your research: {footcite:t}`Lebert2022,Christoph2018a`.

```{footbibliography}
```