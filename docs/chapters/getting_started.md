(installing)=
# Installing optimap

`optimap` is available for Windows, Mac OSX, and Linux. Pre-built binaries can be installed using `pip`, see below. You can find more detailed installation instructions in our [Installation Guide](#installation). You can find the latest version of our source code at [https://github.com/cardiacvision/optimap](github.com/cardiacvision/optimap).

`optimap` requires Python 3.8 or later. To install [Python](https://en.wikipedia.org/wiki/Python_programming_language) we recommend installing the [Anaconda distribution](https://www.anaconda.com/download), which includes Python and many useful packages for scientific computing, or by installing [Python directly](https://code.visualstudio.com/docs/python/python-tutorial#_install-a-python-interpreter).

```{tip}
optimap relies heavily on [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org). We recommend the [Scientific Python Lectures](https://lectures.scientific-python.org) and the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) for an introduction to Python and these libraries.
```

#### Installing pre-built binaries (macOS, Windows, Linux)

The easiest way to install optimap is using `pip` in the command line:

```bash
pip install "opticalmapping[all]"
```

```{note}
`pip install opticalmapping` will install the core functionality of optimap. However, the [OpenCV](https://opencv.org/) and dependencies are not installed by default which are required for the {mod}`optimap.motion` and {mod}`optimap.video` modules. To install these dependencies use `pip install "opticalmapping[all]"` or install OpenCV manually.

If you do not have pip or Python installed you will first have to install these packages.

To use GPU-accelerated motion tracking algorithms a CUDA-enabled version of OpenCV is required, which is currently not available on PyPI. See [](#opencv) for more information.
```

To update optimap to the latest version run

```bash
pip install --upgrade "opticalmapping[all]"`
```

#### Installing from source

To install optimap from source, clone the GitHub repository and run `pip install .` in the root directory:

```bash
git clone https://github.com/cardiacvision/optimap.git
cd optimap
pip install .
```

See [](#contributing) for more details.

# Overview of optimap

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

Some of the most important functions are also imported into the top-level namespace:

```{eval-rst}
.. autosummary::
   optimap
```

for convenience.

See the tutorials listed below for an introduction to the main features of optimap and the API reference for a complete list of functions and classes.

```{toctree}
:maxdepth: 2
:titlesonly:

/tutorials/index.md
```

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

[Monochrome](https://github.com/sitic/monochrome/) is a separate project to view high-dynamic range monochromatic videos, such as those produced by optical mapping. It can be used as a standalone application or as a Python library. Monochrome is not required to use optimap, but it can be useful for viewing optical mapping data in addition to the functions provided by optimap.

# Interactive Plots

```{admonition} Working with remote Jupyter notebooks
:class: warning
optimap is currently not designed to be used with Jupyter Notebook running on a remote server. The interactive plotting functions listed below might not work as expected.
```

optimap uses [Matplotlib](https://matplotlib.org/) for plotting. The following functions require an interactive Matplotlib backend:

```{eval-rst}
.. autosummary::
   optimap.video.show_video
   optimap.video.show_videos
   optimap.video.show_video_pair
   optimap.video.show_video_overlay
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
