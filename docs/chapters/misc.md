(opencv)=
# OpenCV with CUDA support

Some of motion estimation & compensation functions of {mod}`optimap.motion` support GPU-accelerated computing using supported NVIDIA GPUs. They rely on a OpenCV version which was built with CUDA support, which is unfortunately not the default on most prebuilt OpenCV versions.

To check if your OpenCV version supports CUDA, run the following code:

```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())
```

which prints `0` if CUDA is not supported or no NVIDIA GPU was found.

## Building OpenCV with CUDA support

### Windows

Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and a C++ compiler such as the [Microsoft Visual C++ (MSVC) compiler toolset](https://visualstudio.microsoft.com/downloads/#remote-tools-for-visual-studio-2022).

Open a PowerShell window and run the following commands:

```powershell
git clone --recurse-submodules https://github.com/opencv/opencv-python.git
cd opencv-python
$env:CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_CUDNN=OFF -DWITH_NVCUVID=OFF -DOPENCV_DNN_CUDA=OFF -DOPENCV_ENABLE_NONFREE=OFF -DBUILD_opencv_cudacodec=OFF -DCUDA_GENERATION=Auto -DCUDA_TOOLKIT_ROOT_DIR='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6'"
$env:ENABLE_CONTRIB=1
py -m pip wheel .
```

Please note that the CUDA Toolkit version in the `CUDA_TOOLKIT_ROOT_DIR` variable must match the version you installed.

Install the generated wheel file (name will be different):

```powershell
py -m pip install .\opencv_contrib_python-4.6.0+4638ce5-cp311-cp311-win_amd64.wh
```

Test if CUDA is supported:

```bash
py -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

If you get an error similar to:

```
File "C:\Users\USERNAME\AppData\Local\Programs\Python\Python311\Lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```

then you might need to load the CUDA DLLs first before importing OpenCV:

```python
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import cv2
```

To avoid having to do this every time you import OpenCV, you can edit the OpenCV Python package to do this automatically.
Add the `os.add_dll_directory` line to the top of `C:\Users\USERNAME\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\__init__.py` (see error message for the exact path).

### Linux

If you are using Arch/Manjaro Linux prebuilt CUDA versions of OpenCV are available in the AUR: [opencv-cuda](https://archlinux.org/packages/extra/x86_64/opencv-cuda/).

Otherwise you can build OpenCV with CUDA support yourself. The following instructions are for Arch/Manjaro Linux, but should be easily adaptable to other distributions.

```bash
git clone --recurse-submodules https://github.com/opencv/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_CUDNN=OFF -DWITH_NVCUVID=OFF -DOPENCV_DNN_CUDA=OFF -DOPENCV_ENABLE_NONFREE=OFF -DBUILD_opencv_cudacodec=OFF -DCUDA_GENERATION=Auto -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda -DCUDA_HOST_COMPILER=/usr/bin/gcc-10"
export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=1
python -m pip wheel .
```

Now install the generated wheel file (name will be different):

```bash
python -m pip install ./opencv_contrib_python_headless-4.5.5+209d32e-cp310-cp310-linux_x86.whl
```

and test if everything works:

```bash
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```
