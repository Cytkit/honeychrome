[Cytkit](https://cytkit.com) | [Honeychrome](https://honeychrome.cytkit.com/) 
---


# <img src="https://honeychrome.cytkit.com/src/honeychrome/view_components/assets/cytkit_web_logo.png" width="60"> Honeychrome

Open-Source Cross-Platform GUI App for Cytometry Acquisition and Analysis. Built in Python. [Source code on GitHub.](https://github.com/salmansamson/honeychrome) Part of the [Cytkit](https://cytkit.com) project to provide a complete open-source spectral cytometer.

<div>
  <a href="https://youtu.be/RQ4-RQkDCm4"><img src="https://honeychrome.cytkit.com/other/honeychrome screenshot animation.webp" alt="Animated Demo" width="800"></a>
</div>

Our aim is to make Honeychrome so intuitive that you don't need instructions. But just in case, we have prepared this instructional video on YouTube: [Introduction to Honeychrome in 30 Minutes](https://youtu.be/RQ4-RQkDCm4). 

We also have a set of guides to using Honeychrome: 

* Installation (below)
* [Acquire Data](https://honeychrome.cytkit.com/docs/acquisition)
* [Spectral Analysis](https://honeychrome.cytkit.com/docs/spectral_analysis)
* [AutoSpectral in Honeychrome](https://honeychrome.cytkit.com/docs/autospectral_in_honeychrome)
* [Conventional Analysis](https://honeychrome.cytkit.com/docs/conventional_analysis)
* [Manipulate Plots and Gates](https://honeychrome.cytkit.com/docs/cytometry_plots_and_gates)
* [Reports, Exports & Sample Comparison](https://honeychrome.cytkit.com/docs/reports)
* [Programming and Plugins](https://honeychrome.cytkit.com/docs/programming_plugins)

## Mission statement

For too long, our field lacked a general purpose, 
free and open-source cytometry software package. 
We are plugging that gap to provide a software package that is useful for everybody: 
power cytometrists, biologists, students, innovators. 
The app is lightweight but provides all the features one expects: 

- acquire data (starting with Cytkit, perhaps with drivers for other instruments in the future)
- analyse data from any instrument's FCS files 
- provide both conventional compensation and spectral unmixing
- give simple statistical comparisons between samples, publication-quality graphs and exports
- provide intuitive usability with minimal instructions
- support the major operating systems (Windows, MacOS, Linux...)
- no collection or processing of personal information

Honeychrome is fully open source to provide a platform that innovators can exploit, 
for developing both new instrumentation and new methods in data analysis/visualisation.

See more about Honeychrome and the Cytkit project here: [cytkit.com](https://cytkit.com/)

## Installation
You have the following options, depending on whether you want to use the Python source 
code or just download an executable. For most users, it is sufficient to download
an executable for your platform (Windows/Linux/MacOS).

This project uses the SignPath Foundation for code signing.

Note that the plugin system is only available if Honeychrome is run from the Python source code.

### For non-programmers: download and run executable
A set of compressed binary installers are available below.

#### Windows
1. Download [Honeychrome for Windows x64](https://apps.microsoft.com/detail/9nxl24t3t6lv?ocid=webpdpshare) (from the Microsoft Store)
2. Double click to run the installer

#### Linux (Ubuntu with Wayland) > 24.04)
1. Download [Honeychrome for Linux x64 Ubuntu > 24.04](https://github.com/salmansamson/honeychrome/releases/download/v0.8.3/honeychrome-v0.8.3-linux-x64.tar.gz)
2. Extract the honeychrome folder and put it somewhere appropriate for the single user or system
3. Change to the honeychrome folder and run ./install_linux.sh

(For other Linux systems, please use the Python instructions below.)

#### MacOs
1. Download [Honeychrome for MacOS (arm64)](https://github.com/salmansamson/honeychrome/releases/download/v0.8.3/honeychrome-v0.8.3-macos.dmg)
2. Open the disk image and drag honeychrome.app to your Applications folder.
3. Double click the app in your Applications folder to run 

#### Other systems
For all other systems, please follow the Python instructions below. 

> **Note:** If you want to use the plugin functionality, please clone and run the full Python package according to the instructions below.

### Run Honeychrome as a Python package
This should work on any platform for which Python is available. The simple way is to install Honeychrome from PyPi (pip). You can alternatively clone and run the source code. Either way works if you want to run plugins.

#### Install Honeychrome package from PyPi
*Requires Python 3.12 and a C/C++ compiler (some dependencies are compiled during installation):
the Xcode command-line tools on macOS (`xcode-select --install`), the Visual Studio 2022 Build Tools
with "Desktop development with C++" on Windows, `build-essential` on Linux.*

1.  **Create a virtual environment:**
    ```bash
    python3.12 -m venv .venv
    ```

2.  **Activate the virtual environment:**
    * **Linux/macOS:** `source .venv/bin/activate`
    * **Windows:** `.venv\Scripts\activate`

3.  **Update the installation tools:**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    ```

4.  **macOS and Linux only: install Annoy without fast-math** (PaCMAP needs this):
    ```bash
    ANNOY_COMPILER_ARGS=-D_CRT_SECURE_NO_WARNINGS,-fpermissive,-O3,-fno-associative-math \
      pip install --no-binary annoy --no-cache-dir annoy==1.17.3
    ```

5.  **Install Honeychrome with pip:**
    ```bash
    pip install honeychrome
    ```
   (Optional: use pip to install any further packages on which your plugins may depend.)

6.  **Build the compiled kernels (recommended):**
    Honeychrome runs without them, but FlowSOM clustering in Honeycluster and Spectral Variants
    are then unavailable, autofluorescence extraction uses k-means in place of the SOM, and the
    noise model and Spectral Correction run more slowly. First install the build tools
    for your platform (see [Build tools](#build-tools) below), then go to the kernels' folder
    inside the installed package:
    * **Linux/macOS:**
      ```bash
      cd "$(python -c "import honeychrome, os; print(os.path.join(os.path.dirname(honeychrome.__file__), 'bundled_plugins'))")"
      ```
    * **Windows (PowerShell):**
      ```powershell
      cd (python -c "import honeychrome, os; print(os.path.join(os.path.dirname(honeychrome.__file__), 'bundled_plugins'))")
      ```

    and build them:
    ```bash
    python build_som_kernel.py
    python build_autospectral_opt_kernel.py
    python build_fix_my_unmix_kernel.py
    ```
    Build them again after each upgrade of Honeychrome.

7.  **Run Honeychrome:**
    ```bash
    honeychrome
    ```

#### Run Honeychrome from Python source
Follow these steps to clone the repository and run the application from source:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/salmansamson/honeychrome.git
    cd honeychrome
    ```

2.  **Create a virtual environment:**
    *Requires Python 3.12.*
    ```bash
    python3.12 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    * **Linux/macOS:** `source .venv/bin/activate`
    * **Windows:** `.venv\Scripts\activate`

4.  **Upgrade pip and install dependencies:**
    ```bash
    python -m pip install --upgrade pip setuptools wheel
    python install_annoy.py
    pip install -r requirements.txt
    python install_opentsne.py   # macOS only (needs `brew install libomp`); elsewhere it does nothing
    ```

5.  **Install the package in editable mode:**
    *This ensures all internal modules are correctly mapped to your path, and that the kernels
    built in the next step are found.*
    ```bash
    pip install -e .
    ```

6.  **Build the compiled kernels** (install the [Build tools](#build-tools) for your platform first):
    ```bash
    cd src/honeychrome/bundled_plugins
    python build_som_kernel.py
    python build_autospectral_opt_kernel.py
    python build_fix_my_unmix_kernel.py
    cd ../../..
    ```
    Check that they load:
    ```bash
    python -c "import sys; sys.path.insert(0, 'src/honeychrome/bundled_plugins'); [__import__(n) for n in ('_som_kernel', '_autospectral_opt_kernel', '_gls_m2p_kernel', '_fmu_kernel')]; print('kernels OK')"
    ```
    After `git pull`, build them again if any `.c`, `.cpp`, `.hpp`, `CMakeLists.txt` or
    `build_*kernel*.py` file in `src/honeychrome/bundled_plugins` has changed. Quit Honeychrome
    before rebuilding.

7.  **Run the application:**
    ```bash
    python3 -m honeychrome.main
    ```

#### Build tools
The compiled kernels need a C/C++ compiler, CMake, Armadillo with BLAS/LAPACK and OpenMP:

* **macOS:**
  ```bash
  xcode-select --install
  brew install libomp armadillo cmake
  ```
* **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt install build-essential cmake libarmadillo-dev libopenblas-dev
  ```
* **Windows:** install the Visual Studio 2022 Build Tools with "Desktop development with C++",
  CMake and Git, then Armadillo with OpenBLAS through vcpkg (PowerShell; the vcpkg step takes
  about 15 minutes the first time):
  ```powershell
  git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
  C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
  C:\vcpkg\vcpkg.exe install armadillo:x64-windows-static-md
  Set-Content -Path "$env:USERPROFILE\vcpkg_static_md.cmake" -Encoding ascii -Value @(
    'set(VCPKG_TARGET_TRIPLET x64-windows-static-md CACHE STRING "")',
    'include("C:/vcpkg/scripts/buildsystems/vcpkg.cmake")'
  )
  $env:CMAKE_TOOLCHAIN_FILE = "$env:USERPROFILE\vcpkg_static_md.cmake"
  ```
  Build the kernels in the same PowerShell window, so that `CMAKE_TOOLCHAIN_FILE` is set.

## Contributions
We have many ideas for Honeychrome and welcome new ones. 
We also welcome anyone wishing to contribute to and improve the software!

We will shortly release fuller instructions and documentation for programmers wishing to build on Honeychrome in
new open source cytometry projects. In the meantime, please contact us if you require assistance.

Honeychrome builds on several great open source Python packages:
- [FlowKit](https://github.com/whitews/flowkit) - for definition of gates, gating hierarchies, transforms, and FCS input/output
- [PySide6](https://pypi.org/project/PySide6/) - the GUI system, Qt for Python
- [pyqtgraph](https://www.pyqtgraph.org/) - extremely fast plotting and manipulation of plots in Qt
- Several others for processing of data, including numpy, scikit, sklearn
- Seaborn and Matplotlib for publication-quality graphics

Samson Rogers and Oliver Burton have been the main developers of Honeychrome so far. Oliver contributed the AutoSpectral Cleaning and AF functions and many aspects of the spec and usability. 
Thanks to support and advice from many people including C. Vant, Lotte Carr, Scott Tasker, Philip Jones, Robyn Pritchard.

## Roadmap
By January 2027, we hope to get to a stable release labeled v1.0.0, including the following:
- all bugs fixed
- documentation for users
- documentation for developers
