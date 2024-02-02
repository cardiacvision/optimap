(installing)=
# Installing optimap

`optimap` is available for Windows, Mac OSX, and Linux. Pre-built binaries can be installed using `pip`, see below. You can find more detailed installation instructions in our [Installation Guide](https://optimap.readthedocs.io/en/latest/chapters/installation/). You can find the latest version of our sourcode here:

[https://github.com/cardiacvision/optimap](https://github.com/cardiacvision/optimap).

`optimap` requires Python 3.8 or later. To install [Python](https://en.wikipedia.org/wiki/Python_programming_language) we recommend installing the [Anaconda distribution](https://www.anaconda.com/download), which includes Python and many useful packages for scientific computing, or by installing [Python directly](https://code.visualstudio.com/docs/python/python-tutorial#_install-a-python-interpreter).

```{tip}
optimap relies heavily on [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org). We recommend the [Scientific Python Lectures](https://lectures.scientific-python.org) and the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for an introduction to Python and these libraries.
```

### Installing pre-built binaries (Mac OSX, Windows, Linux)

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

If you do not have pip or Python installed you will first have to install these packages.

To use GPU-accelerated motion tracking algorithms a CUDA-enabled version of OpenCV is required, which is currently not available on PyPI. See [](#opencv) for more information.
```

To update optimap to the latest version run `pip install --upgrade opticalmapping[all]`. See [](#contributing) for instructions on how to install optimap from source.

# Overview

optimap consists of the following modules:

```{eval-rst}
.. autosummary::
   optimap.video
   optimap.image
   optimap.trace
   optimap.motion
   optimap.phase
   optimap.utils
```

Some of the most important functions are imported into the top-level namespace:

```{eval-rst}
.. autosummary::
   optimap
```

for convenience.

(vscode)=
# Using optimap

We highly recommend using [Visual Studio Code](https://code.visualstudio.com) for working with optimap. Visual Studio Code is a free and open-source editor with excellent support for Python and Jupyter notebooks.

- Download and install [Visual Studio Code](https://code.visualstudio.com)
- Install the Microsoft [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- Install the Microsoft [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

In particular, we recommend using regular `.py` files with automatic code cells by typing `# %%`. This allows you to run code cells directly from the editor, as shown below:

![code cells](/_static/vscode-code-cells.png)

See the [Visual Studio Code documentation](https://code.visualstudio.com/docs/python/jupyter-support-py) for more information. Code cells are also supported in other editors such as [PyCharm](https://www.jetbrains.com/pycharm/) and [Spyder](https://www.spyder-ide.org/).

Purists who would like to run their `.py` scripts in the command line
You could run the Python script in a terminal with `python3 basics.py` from the folder where the file is located. Alternatively, one can start Python in the command line (by typing in `python3.9` and pressing 'Enter') and type in the above command and press 'Enter'.

![plot viewer](/_static/vscode-plot-viewer.gif)

(monochrome)=
# Monochrome

Monochrome is a separate project ... TODO

# Interactive Plots

```{admonition} Working with remote Jupyter notebooks
:class: warning
optimap is currently not designed to be used with Jupyter Notebook running on a remote server. The interactive plotting functions listed below might not work as expected.
```

optimap uses [Matplotlib](https://matplotlib.org/) for plotting. The following functions require an interactive Matplotlib backend:

```{eval-rst}
.. autosummary::
   optimap.video.play
   optimap.video.play2
   optimap.trace.select_positions
   optimap.trace.select_traces
   optimap.trace.compare_traces
   optimap.motion.play_displacements
   optimap.motion.play_displacements_points
```

We strongly recommend using the Qt matplotlib backend for interactive plotting. optimap will attempt to automatically switch to the Qt backend for these functions and switch back to the inline backend afterwards (if applicable).

```{note}
{meth}`optimap.utils.disable_interactive_backend_switching` can be used to disable the interactive backend switching if it causes problems. Please report any issues you encounter.
```

Alternatively, you can manually switch to the Qt backend by running the following code before calling any of the functions listed above:

```python
import matplotlib.pyplot as plt
plt.switch_backend('QtAgg')  # or 'QtCairo'
```

# How to cite

A publication about optimap is currently in preparation. In the meantime, please cite the following DOI and paper if you use optimap in your research: {footcite:t}`optimap_zenodo,Lebert2022`.

```{footbibliography}
```
