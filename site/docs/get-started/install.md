---
title: Install
lead: Download the app for your computer, or run Honeychrome as a Python package if you want to use plugins.
redirect_from:
  - /readme
  - /readme.html
---
{% assign r = site.data.release %}
Most people should download the app for their platform. It needs no Python, R or other software, and the macOS and Windows builds are signed. Run Honeychrome with Python instead if you want to use [plugins](/docs/developers/plugins/), or if there is no app for your system.

| | Get it | Plugins |
|---|---|---|
| **macOS** (Apple Silicon) | [{{ r.platforms.mac.file }}]({{ r.platforms.mac.url }}) | No |
| **Windows** (x64) | [Microsoft Store]({{ r.platforms.windows.url }}) | No |
| **Linux** (Ubuntu 24.04 or later, x64) | [{{ r.platforms.linux.file }}]({{ r.platforms.linux.url }}) | No |
| **Any system with Python {{ r.python }} or later** | `pip install honeychrome` | Yes |

## Install the app

### macOS
1. Download [Honeychrome for macOS]({{ r.platforms.mac.url }}) (Apple Silicon).
2. Open the disk image and drag **honeychrome.app** to your Applications folder.
3. Double-click the app in Applications to run it.

### Windows
1. Open [Honeychrome in the Microsoft Store]({{ r.platforms.windows.url }}).
2. Click **Get** or **Install**. The Store keeps Honeychrome up to date.
3. Start Honeychrome from the Start menu.

### Linux
1. Download [Honeychrome for Linux]({{ r.platforms.linux.url }}) (x64, built for Ubuntu 24.04 or later with Wayland).
2. Extract the `honeychrome` folder and put it somewhere suitable for a single user or for the whole system.
3. In a terminal, change to the `honeychrome` folder and run `./install_linux.sh`.

For other Linux distributions, run Honeychrome with Python as below.

Code signing for Windows is provided by the [SignPath Foundation](https://signpath.org). The macOS app is signed and notarised by Apple.

## Run Honeychrome with Python

This works on any platform with Python {{ r.python }} or later, and is the way to use plugins. You can install the package from PyPI or run the source code; both run plugins.

### Install the package from PyPI
1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
2. Activate it:
   * Linux or macOS: `source .venv/bin/activate`
   * Windows: `.venv\Scripts\activate`
3. Install Honeychrome:
   ```bash
   pip install honeychrome
   ```
   Use pip to install anything else your plugins need as well.
4. Run Honeychrome:
   ```bash
   honeychrome
   ```

### Run from the source code
1. Clone the repository:
   ```bash
   git clone https://github.com/Cytkit/honeychrome.git
   cd honeychrome
   ```
2. Create and activate a virtual environment (Python {{ r.python }} or later):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   On Windows, activate with `.venv\Scripts\activate`.
3. Upgrade pip and install the dependencies:
   ```bash
   pip install --upgrade pip
   python install_annoy.py
   pip install -r requirements.txt
   python install_opentsne.py
   ```
   `install_opentsne.py` builds openTSNE with OpenMP on macOS (it needs `brew install libomp`); on other systems it does nothing.
4. Install Honeychrome in editable mode, so that all its modules are found:
   ```bash
   pip install -e .
   ```
5. Run it:
   ```bash
   python3 -m honeychrome.main
   ```

## Next steps
- [Spectral unmixing](/docs/unmixing/spectral-unmixing/): your first experiment, from FCS files to unmixed data.
- [Watch the 30-minute introduction]({{ r.video_url }}) on YouTube (recorded with version 0.8).
