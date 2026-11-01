---
title: Release notes
eyebrow: What's new
lead: Changes in each version of Honeychrome. Downloads for every release are on GitHub.
permalink: /releases/
search_include: true
---
{% assign r = site.data.release %}
[Download the current version](/download/){: .btn .btn--honey} [All releases on GitHub]({{ r.releases_url }}){: .btn .btn--ghost}

## Version 1.0.0

Version 1.0.0 is finally here!

Honeychrome 1.0 is our first full release. Alongside the spectral unmixing tools, it now includes a complete analysis workflow from raw files to statistics: automated gating, dimensionality reduction and clustering, annotation and differential testing, all inside the app.

### Major new features

- **Improved and faster AutoSpectral workflow.** AutoSpectral now lives in a single AutoSpectral tab with four pages: Autofluorescence, Noise, Spectral Variants and Spectral Correction.
  - Autofluorescence profiles can be assigned per sample or per sample group, or auto-assigned from scatter, with a summary to review before accepting.
  - Per-cell autofluorescence unmixing is faster and no longer needs a compiled C kernel. AF and AF Index are written out as channels you can gate on.
  - Noise model: an optional weighted unmixing method fitted to your controls. The unmixing method you choose is used everywhere in the app (plots, statistics, exports and plugins).
  - Spectral variants are de-duplicated, making optimised unmixing faster and more accurate.
  - Spectral Correction (beta): detects and corrects residual unmixing errors in fully stained samples, including tandem dye breakdown.
- **Automated Gating plugin:** build a gating model once and apply it to every sample.
  - Start from your existing Honeychrome gates or import a FlowJo workspace (.wsp).
  - Gates are converted into data-driven gating algorithms, trained on samples you choose, then run across the experiment. Each gate can stay fixed or be recalculated per sample, with a "nudge" to make thresholds stricter or looser.
  - Results include population frequencies, group comparison statistics with heatmaps and volcano plots, sample PCA, predictive analysis and Flow Set Enrichment Analysis (human and mouse cell-type libraries).
  - Long runs can be paused and partial runs saved; results persist with the experiment and export to CSV.
- **Honeycluster plugin:** the modern spectral workflow, including dimensionality reduction, clustering, annotation and statistics.
  - Dimensionality reduction: UMAP, t-SNE (openTSNE), PaCMAP and PCA.
  - Clustering: FlowSOM (fast, batched), HDBSCAN and Leiden, with cluster trees for every method.
  - Annotation: automatic cluster labelling (Cluster ID and MEM labels) from a curated cell-type and marker database; clusters can be renamed and recoloured.
  - Statistics: differential abundance (count GLMs with an edgeR-style F test for small studies) and differential expression, multi-group comparisons with FDR control, paired designs and numeric covariates, volcano plots, heatmaps that can be split by factor, sample PCA and predictive analysis.
  - Layout: arrange plots into publication-ready figures and export them; a report generator summarises each run.
- **Per-sample gating:** right click on a gate to customise it for the current sample. Every sample shares one gating template, and only the gates you change are stored for that sample. Per-sample gates are used throughout, including Statistics, the Spectral Process and the analysis plugins.
- **Spectral library:** look up profiles of matching fluorophores for your instrument. The new Reference Library window (View > Reference Library) lets you browse shipped and saved reference spectra for your detector configuration, compare them, choose which spectrum to use as the reference or QC target, and import or export them. Controls can be loaded straight from the library in the Spectral Process.

### Other new features and improvements

- Spectral Process
  - Bead controls are detected automatically (including bead vs cell particle type from scatter) and matched to the right unstained beads when several bead types are present; you can mark or unmark beads with a right click.
  - "Auto generate spectral controls" now continues straight into control cleaning (on by default; can be turned off in App Settings or per experiment).
  - Warnings for controls that contain no real signal, for a poorly conditioned unmixing matrix, and for files acquired with inconsistent voltages.
  - Switching a control's sample updates its particle type, negative, gates and cleaning automatically.
  - Unmixed gating is kept when the spectral model is regenerated.
  - All warnings from a run are collected into a single message at the end.
- One heavy job at a time: long calculations (unmixing, clustering, gating runs, exports) are queued so they do not compete for memory and CPU. A running job can be released by right-clicking it in the status bar.
- The Statistics tab is now called Simple Stats and adds a sample x marker heatmap.
- Unstained, single-stained and bead samples are excluded by default when choosing samples for analysis.
- Bead samples are shown in green in the sample browser.
- Detector labels on laser-coloured plots are angled for legibility.

### Instrument support

- Added CytoStellar support.
- Improved FACSDiscover compatibility.
- Better automatic transforms for Bigfoot and Xenith data.

### Performance

- Large FCS files load 5-6x faster (thanks to Joan Escriva Font).
- Large files are exported in chunks to limit memory use.
- Unmixed data are cached, so switching between samples and tabs is faster.
- Reduced memory use and faster plotting throughout.

### Bug fixes

- Corrected the spillover matrix written on export.
- Fixed a crash when unticking "use cleaned" in the Spectral Process.
- Fixed a crash when switching tabs during a refresh.
- Fixed text visibility in dark mode.
- Numerous crash fixes from error reports.

### Acknowledgements

The AutoSpectral noise model builds on work by Paul Heisig (Amsterdam UMC) and David Novo. Thanks to everyone who tested pre-release builds and sent feedback.

## Version 0.8.3

We have switched Honeychrome to the LGPL (from GPL) license. As more people try out Honeychrome and tell us of their interests in contributing to the code or using it for various purposes, we think it's a good time to make this change. For example, LGPL will allow people to develop their own separate commercial plugins for Honeychrome, while requiring that any developments of the existing code have to be released as open source under the same license.

- Improved and faster AutoSpectral workflow.
- Improved determination of major (peak) channels in the spectral process.
- Improved compatibility with spectral and conventional cytometers, in particular the FACSDiscover.

## Version 0.8.2

Added support for FCS files from the following spectral cytometers:

- Cytek Aurora and Northern Lights
- Sony ID7000
- BD FACSDiscover A8 and S8
- BD FACSymphony A5SE
- Beckman Coulter CytoFLEX mosaic
- Thermo Fisher Attune Xenith
- NovoCyte Opteon

## Version 0.8.0

### Major new features

- The two major components of AutoSpectral have been added: Control Cleaning and Autofluorescence. These (optional) enhancements of the unmixing workflow are rapidly becoming the standard tool for fixing unmixing artifacts caused by variable autofluorescence.
- Tabbed plugin system: Honeychrome now automatically loads plugins from a plugins folder; two example plugins are included. Plugins are only available if Honeychrome is run in the Python environment (see the [installation instructions](/docs/get-started/install/)), not as a precompiled executable.
- 2D plots can now be shown as dot plots coloured by gate.
- Debug data: Honeychrome can send error and crash data if allowed by the user. No personally identifiable information is collected or sent.

### PyPI release

You can now use Honeychrome in a Python environment: run `pip install honeychrome`, then run `honeychrome` from the command line.

### Improvements

- Automatic recognition of antigens and fluorophore names in the spectral model editor.
- Better colourmap ranges to avoid saturation.
- Imported FCS files can now live anywhere: if they are not in the experiment folder, a symbolic link is set up from the experiment folder, so there are no more broken relative paths.
- The build system now produces executable installers for Linux (Ubuntu 24.04 or later), Windows (10, 11) and macOS (Apple Silicon).
- Numerous bug fixes.

## Version 0.6.1

### Features and improvements

- Cytometry plots make better use of the colourmap range: there is less blue and more of the higher colours, and the inside and edges of the 2D histograms are scaled independently.
- A hotspot matrix viewer has been added to the Spectral Process tab, following Mage et al.
- A "density cutoff" setting has been added to App Configuration: the lowest number of events per bin to show a colour in the 2D histograms. Set it to 0 to ignore the cutoff.
- New statistical comparison: samples can be compared by histogram overlays.

### Usability

- You must now click to select a cell of the spillover matrix or NxN plots before rolling the scroll wheel, which stops accidental changes to spillover.
- A status message on window reload invites you to load a sample.
- Improved scrolling: cytometry plots now ignore the mouse wheel.
- Improved import: either place FCS files in the experiment folder or link to an existing folder.
- Click to open file dialogs in experiment settings.
- Tooltips on heatmap horizontal headers avoid clipped names.
- Scrolling the spectral model editor no longer changes the selection.

### Bug fixes

- Fixed import of Bigfoot and FCMPASS files.
- Fixed a log ticks bug.
- Fixed a raw transforms bug.
- Fixed a colormap loading bug in App Configuration.
- Numerous further bug fixes.
