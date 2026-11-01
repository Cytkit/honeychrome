---
# The home page. The text at the top of the page is set here. The feature
# blocks further down are the Markdown files in _home/features, and the
# sections after them are the files in _home/sections.
# The version number is added automatically wherever it appears.
layout: home
title: Honeychrome
description: Free, open-source desktop software for spectral flow cytometry, with AutoSpectral unmixing, automated gating, clustering and statistics on macOS, Windows and Linux.
body_class: home

hero:
  eyebrow: Spectral flow cytometry software
  headline: Open-source spectral cytometry with AutoSpectral built in.
  # This part of the headline is drawn in the spectrum colours.
  highlight: AutoSpectral
  lead: Honeychrome takes you from raw data (instrument acquisition or FCS import) to unmixed data, gates, clusters and statistics in one free desktop app for macOS, Windows and Linux.
  screenshot: hero/honeychrome-tour.webp
  alt: Honeychrome's main window, cycling through the raw data, spectral process, unmixed data and statistics views
  width: 1536
  height: 960

cards:
  - title: Free and open source
    text: No licences or seats. The code is on GitHub under the LGPL, so anyone can check how the numbers are made.
  - title: Any instrument
    text: Reads standard FCS files from spectral and conventional cytometers, and runs the Cytkit cytometer directly.
  - title: Your data stays with you
    text: Everything runs on your own computer. Honeychrome collects no personal data and never uploads your files.
  - title: macOS, Windows, Linux
    text: A signed app for each platform, or a Python package if you want to write your own plugins.

workflow:
  title: One workflow, from raw files to results
  intro: Each step uses the unmixing you chose, so plots, gates, clusters, statistics and exports always agree.
  steps:
    - title: Import or Acquire
      link: /docs/unmixing/spectral-unmixing/#import-your-fcs-files
      text: Import FCS files from any instrument, or interface with compatible instruments for direct acquisition
    - title: Unmix
      link: /docs/unmixing/autospectral/
      text: AutoSpectral cleaning and per-cell autofluorescence
    - title: Gate
      link: /docs/plots-and-gating/plots-and-gates/
      text: By hand, per sample, or with Automated Gating
    - title: Cluster
      link: "#honeycluster"
      text: UMAP, FlowSOM, Leiden and automatic annotation
    - title: Compare
      link: /docs/statistics/simple-stats/
      text: Group statistics, heatmaps and volcano plots
    - title: Report
      link: /docs/outputs/reports-and-exports/
      text: Figures, reports, CSV and FCS exports

features:
  eyebrow: What's in Honeychrome
  title: Everything for a spectral experiment, in one app

cta:
  title: Try Honeychrome
  text: Free for everyone. No account or licence needed.
---
