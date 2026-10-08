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
1.  **Create a virtual environment:**
    *Requires Python 3.12 or higher.*
    ```bash
    python3 -m venv .venv
    ```
    
2.  **Activate the virtual environment:**
    * **Linux/macOS:** `source .venv/bin/activate`
    * **Windows:** `.venv\Scripts\activate`
    

3.  **Install Honeychrome with pip:**
    ```bash
    pip install honeychrome
    ```  
   (Optional: use pip to install any further packages on which your plugins may depend, e.g. the example plugins require umap-learn and hdbscan.)
   

4.  **Run Honeychrome:**
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
    *Requires Python 3.12 or higher.*
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    * **Linux/macOS:** `source .venv/bin/activate`
    * **Windows:** `.venv\Scripts\activate`

4.  **Upgrade pip and install dependencies:**
    ```bash
    pip install --upgrade pip
    python install_annoy.py
    pip install -r requirements.txt
    ```

5.  **Install the package in editable mode:**
    *This ensures all internal modules are correctly mapped to your path.*
    ```bash
    pip install -e .
    ```

6.  **Run the application:**
    ```bash
    python3 -m honeychrome.main
    ```

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
