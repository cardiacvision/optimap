(installing)=
# Installing optimap

`optimap` is a [Python](https://en.wikipedia.org/wiki/Python_programming_language) library for optical mapping analysis that supports Windows, Mac OSX, and Linux. This guide will help you get started with installation and basic usage.

`optimap` requires Python 3.8 or later. To install Python we recommend installing the [Anaconda distribution](https://www.anaconda.com/download), which includes Python and many useful packages for scientific computing, or by installing [Python directly](https://code.visualstudio.com/docs/python/python-tutorial#_install-a-python-interpreter).

```{tip}
optimap relies heavily on [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org). 
For an introduction to Python and these libraries, we recommend:
- [Scientific Python Lectures](https://lectures.scientific-python.org)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
```

#### Installation using pip (macOS, Windows, Linux)

The easiest way to install optimap is using [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/) in the command line:

```bash
pip install optimap
```

To update optimap to the latest version run

```bash
pip install --upgrade optimap
```

````{admonition} Installing without Recommended Dependencies
:class: note, dropdown

`pip install optimap` will install optimap with all recommended dependencies including [OpenCV](https://opencv.org/) (`opencv-contrib-python`) and PySide6, which may not be desired in some advanced use cases. If you need a custom version of OpenCV (e.g., with CUDA support for GPU-accelerated motion tracking) or a different Qt implementation use the following command to install optimap with minimal dependencies:

```bash
pip install --no-deps optimap
pip install -r https://raw.githubusercontent.com/cardiacvision/optimap/refs/heads/main/requirements-core.txt
```

or using [pip-mark-installed](https://pypi.org/project/pip-mark-installed/):

```bash
pip install pip-mark-installed
pip-mark-installed opencv-contrib-python PySide6
pip install optimap
```
````

#### Installing from source

To install optimap from source, clone the GitHub repository and run `pip install .` in the root directory:

```bash
git clone https://github.com/cardiacvision/optimap.git
cd optimap
pip install .
```

See [](#contributing) for more details.

# Overview of optimap

optimap is organized into the following modules:

```{eval-rst}
.. autosummary::
   optimap
   optimap.video
   optimap.image
   optimap.trace
   optimap.motion
   optimap.activation
   optimap.phase
   optimap.utils
```

The most important functions are also imported into the top-level namespace `optimap`. Click on the module name to see the functions and classes it contains.

See the tutorials listed below for a comprehensive introduction to optimap's main features. For a complete reference of all available functions and classes, consult the [📚 API documentation](/api).

```{toctree}
:maxdepth: 2
:titlesonly:

/tutorials/index.md
```

(vscode)=
# Using optimap

We highly recommend using [Visual Studio Code](https://code.visualstudio.com) for working with optimap. Visual Studio Code is a free and open-source editor with excellent support for Python and Jupyter notebooks.

To get started:
1. Download and install [Visual Studio Code](https://code.visualstudio.com)
2. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) by Microsoft
3. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) by Microsoft

## Code Cells in VS Code

VS Code allows you to create interactive code cells directly in Python files by typing `# %%`. This lets you run code segments individually, similar to Jupyter notebooks or MATLAB Sections:

![code cells](/_static/vscode-code-cells.png)

For more details, see the [VS Code documentation on Jupyter support](https://code.visualstudio.com/docs/python/jupyter-support-py). Similar code cell functionality is also available in other editors like [PyCharm](https://www.jetbrains.com/pycharm/) and [Spyder](https://www.spyder-ide.org/).

![plot viewer](/_static/vscode-plot-viewer.gif)

## Command Line Usage

If you prefer using the command line, you can:
- Run scripts directly with `python3 your_script.py` from the script's directory
- Start an interactive Python session by running `python3` and enter commands directly

(monochrome)=
# Monochrome Viewer

[Monochrome](https://github.com/sitic/monochrome/) is a companion tool for viewing high-dynamic range monochromatic videos, commonly used in optical mapping. While not required for optimap, it provides additional visualization capabilities. To view a video in Monochrome use the following Python code:

```python
import monochrome as mc
mc.show(video)
```

[<center><img src="https://cardiacvision.github.io/optimap/main/_static/Monochrome-screenshot1.webp"></center>](https://github.com/sitic/monochrome/)

See the [Monochrome documentation](https://monochrome.readthedocs.io/) for details.

# Interactive Plots

```{admonition} Working with **remote** Jupyter notebooks
:class: warning
The interactive plotting functions in optimap may not work as expected when using Jupyter Notebook on a **remote** server (i.e., not on your local computer). This is because the interactive plots require a local display to work properly.
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
plt.switch_backend("QtAgg")  # or "QtCairo"
```

# How to cite

A publication about optimap is currently in preparation. In the meantime, please cite the following DOI and paper if you use optimap in your research: {footcite:t}`optimap_zenodo,Lebert2022`.

```{footbibliography}
```
