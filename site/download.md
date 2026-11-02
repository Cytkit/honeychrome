---
# Platform names, file names and download links are in _data/release.yml.
# The install steps for each platform are listed below. The page text after
# the front matter is shown in two columns, separated by a line of three
# asterisks.
layout: download
title: Download Honeychrome
eyebrow: Download
lead: Free for everyone. Choose your platform, or install the Python package to use plugins.
permalink: /download/
wide: true
search_include: true
platforms:
  - key: mac
    icon: apple
    steps:
      - Download the disk image.
      - Open it and drag **honeychrome.app** to Applications.
      - Double-click Honeychrome in Applications.
  - key: windows
    icon: windows
    steps:
      - Open Honeychrome in the Microsoft Store.
      - Click **Get** or **Install**.
      - Start Honeychrome from the Start menu. The Store keeps it up to date.
  - key: linux
    icon: linux
    steps:
      - Download and extract the archive.
      - Move the `honeychrome` folder where you want it.
      - In that folder, run `./install_linux.sh`.
---
## {% include icon.html name="python" %} Python package and plugins {#python}

Honeychrome runs on any system with Python {{ site.data.release.python }} or later. Install it this way if you want to use [plugins](/docs/developers/plugins/), which the downloaded apps do not load, or if there is no app for your system.

```
python3 -m venv .venv
source .venv/bin/activate
pip install honeychrome
honeychrome
```

On Windows, activate the environment with `.venv\Scripts\activate`. The [Install guide](/docs/get-started/install/) also covers running from the source code.

***

## Release {{ site.data.release.version }} {#details}

- [What's new in {{ site.data.release.version }}](/releases/)
- [All releases and previous versions]({{ site.data.release.releases_url }}) on GitHub
- [Source code](https://github.com/Cytkit/honeychrome) (GNU LGPL v3.0)
- [honeychrome on PyPI]({{ site.data.release.pypi_url }})

The macOS app is signed and notarised by Apple. The Windows app is checked and hosted by the MS Store.

Honeychrome collects no personal data. Crash reports are optional and can be switched off; see [Privacy and debug data](/docs/reference/privacy/).
