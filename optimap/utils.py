"""General utility functions."""

__all__ = [
    "set_verbose",
    "is_verbose",
    "print_bar",
    "print_properties",
    "enable_interactive_backend_switching",
    "disable_interactive_backend_switching",
    "retrieve_example_data",
]

import functools
import urllib.parse
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pooch

from ._version import __version__

VERBOSE = False
INTERACTIVE_BACKEND = "QtAgg"
INTERACTIVE_BACKEND_SWITCHING = True
FILE_HASHES = {
    "Example_01_Sinus_Rabbit_Basler.npy":
        "sha256:5c692cca0459c931b7f767c27162a42b31e3df90a6aeac53bb182affa2135678",
    "Example_02_VF_Rabbit_Di-4-ANEPPS_Basler_acA720-520um.npy":
        "sha256:6252da91db434cad95758830fdf7c13a9db6793da30dd8e85e6878736e20787e",
    "Example_03_Pacing_Rabbit_Di-4-ANEPPS_Basler_acA720-520um.npy":
        "sha256:50113334e6955f5abb3658b5027447f507fd9eef6bfef766a628c2365ff848be",
    "Example_04_Pacing_Rabbit_Di-4-ANEPPS_Basler_acA720-520um.npy":
        "sha256:674603f64ccf754f73a264986e0bf1ee93d03ce3a9ea88f248620632046e3c40",
    "Example_05_Ratiometry.npy":
        "sha256:10a59863ee23abc689d8ee4cd27542ef1b7b8b8eb5668a7f2dc49572f18319f2",
    # used in tests
    "optimap-test-download-file.npy":
        "sha256:0d3cfca36d8e3ad935de4d0681ddd510c1590212a99dccb196353c8ce85b7491",
    # warped version of Example_02, used to speed up documentation build
    "Example_02_VF_Rabbit_Di-4-ANEPPS_Basler_acA720-520um_warped.npy":
        "sha256:a1781582b669a69a9753b1c43d23e0acf26fb372426eeb6d880d2e66420b2107",
    "Example_02_VF_Rabbit_Di-4-ANEPPS_Basler_acA720-520um_warped_mask.npy":
        "sha256:3f5d8402c8251f3cb8e8d235b459264ff7e7e2cf2b81f08129f0897baa262db6"
}


def set_verbose(state=True):
    """Set verbosity of optimap.

    Parameters
    ----------
    state : bool
        If True, optimap will print more information.
    """
    global VERBOSE
    VERBOSE = state
    if VERBOSE:
        print(f"optimap - v{__version__}")
        print("A free python-based framework for processing optical mapping and fluorescence imaging data.")
        print_bar(force=True)


def is_verbose():
    """Returns whether optimap is verbose.

    Returns
    -------
    bool
        Whether optimap is verbose.
    """
    return VERBOSE


def _print(string):
    if VERBOSE:
        print(string)


def print_bar(force=False):
    """Print a bar to separate sections of output."""
    if VERBOSE or force:
        print(
            "------------------------------------------------------------------------------------------"
        )


def print_properties(array: np.ndarray):
    """Print properties of an array."""
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy array")
    print_bar(force=True)
    print(f"array with dimensions: {array.shape}")
    print(f"datatype of array: {array.dtype}")
    print(f"minimum value in entire array: {np.nanmin(array)}")
    print(f"maximum value in entire array: {np.nanmax(array)}")
    if nans := np.sum(np.isnan(array)) > 0:
        print(f"number of NaNs in array: {nans}")
    print_bar(force=True)


def enable_interactive_backend_switching():
    """Enable automatic switching of the matplotlib backend to Qt when necessary.

    See also :py:func:`disable_interactive_backend_switching`.
    """
    global INTERACTIVE_BACKEND_SWITCHING
    INTERACTIVE_BACKEND_SWITCHING = True

def disable_interactive_backend_switching():
    """Disable automatic switching of the matplotlib backend to Qt when necessary.

    See also :py:func:`enable_interactive_backend_switching`.
    """
    global INTERACTIVE_BACKEND_SWITCHING
    INTERACTIVE_BACKEND_SWITCHING = False


def interactive_backend(func):
    """Function decorator to change backend temporarily in Ipython session.

    Switches to QtAgg backend if in Ipython session and back to inline afterwards.
    """

    def get_ipython():
        try:
            from IPython import get_ipython
        except ImportError:
            return None
        return get_ipython()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not INTERACTIVE_BACKEND_SWITCHING:
            return func(*args, **kwargs)

        ipython = get_ipython()
        current_backend = mpl.get_backend()

        try:
            _print(f"Switching matplotlib backend from {current_backend} to backend to {INTERACTIVE_BACKEND}")
            plt.switch_backend(INTERACTIVE_BACKEND)
        except Exception as e:
            print(f"ERROR: Failed to switch matplotlib backend to '{INTERACTIVE_BACKEND}': {e}. "
                  "Automatic backend switching has been disabled.")
            disable_interactive_backend_switching()

        result = func(*args, **kwargs)

        # Switch back
        if ipython is not None and "inline" in current_backend:
            _print("Switching matplotlib backend to inline")
            plt.switch_backend(current_backend)
            ipython.run_line_magic("matplotlib", "inline")
        return result

    return wrapper

def retrieve_example_data(name, directory="./example_data", silent=False):
    """Download example data if not already present.

    Parameters
    ----------
    name : str
        Name of the file to download.
    directory : str
        Directory to download the file to.
    silent : bool
        If True, set logging level to WARNING.

    Returns
    -------
    str
        Path to the file.
    """
    known_hash = FILE_HASHES.get(name, None)
    if known_hash is None:
        warnings.warn(f"WARNING: Example file '{name}' is not known. Attempting to download it anyway.", UserWarning)
    if silent:
        silent = pooch.get_logger().level
        pooch.get_logger().setLevel("WARNING")

    # The CMS server only allows files with a certain extensions to be uploaded.
    # We use .webm as dummy extension to upload the files, and rename them after download.
    remote_path = urllib.parse.quote(f"{name}", safe="")
    url = f"https://cardiacvision.ucsf.edu/sites/g/files/tkssra6821/f/{remote_path}_.webm"

    path = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        fname=name,
        path=directory,
    )

    if silent:
        pooch.get_logger().setLevel(silent)
    return path
