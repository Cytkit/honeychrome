---
layout: default
title: AutoSpectral in Honeychrome
---
[Cytkit](https://cytkit.com) | [Honeychrome](https://honeychrome.cytkit.com/) 
---

# <img src="/src/honeychrome/view_components/assets/cytkit_web_logo.png" width="60"> Honeychrome

{% include menu.md %}

# AutoSpectral in Honeychrome
AutoSpectral is an optimised unmixing workflow.

Originally written in R, AutoSpectral is a set of tools to enhance spectral analysis by providing you with the best possible spectral signatures of your spectral controls, optimising the removal of autofluorescence, and providing reproducible mixing. See the Colibri Cytometry blog and the AutoSpectral Github repository for more information on AutoSpectral.
- [AutoSpectral on the Colibri Cytometry blog](https://www.colibri-cytometry.com/post/introducing-autospectral-an-optimized-unmixing-workflow)
- [AutoSpectral package on Github](https://github.com/DrCytometer/AutoSpectral)

Honeychrome now provides the following two major components of AutoSpectral natively as steps in the spectral process:
- AutoSpectral Cleaning (optimisation of control profiles) 
- AutoSpectral AF (per-cell autofluorescence extraction)

AutoSpectral Cleaning clears up artifacts due to poor single stain control profiles, for example hypernegative events, skewed data and diagonal correlations, by carrying out saturation exclusion, brightest-event selection, scatter matching and noise removal on the control profiles.

AutoSpectral AF clears up artifacts due to variable autofluorescence, for example negative spreading, by pairing every event with its best matched autofluorescence spectrum. 

See related articles on Colibri Cytometry: [AutoSpectral Cleaning](https://www.colibri-cytometry.com/post/autospectral-cleaning) and [AutoSpectral AF](https://www.colibri-cytometry.com/post/autospectral-single-cell-autofluorescence)

This guide assumes you have already installed Honeychrome using the [executable installers (Windows/Mac/Linux) or Python (cross-platform)](/readme), that you have a set of FCS files to import, and that you are already familiar with the basic spectral unmixing process. If not, please go to  [Spectral Analysis](/docs/spectral_analysis) first. 

We will be continuing use of AutoSpectral Full Workflow Example (a 9-colour panel from the 5-laser Cytek Aurora from the AutoSpectral Full Workflow Example as set up in the tutorial [Spectral Analysis](/docs/spectral_analysis).

## How to perform AutoSpectral Cleaning (optimisation of control profiles)

### Show the AutoSpectral Cleaning process and click "Clean Controls"

First make sure that you have at least one unstained control (i.e. with "unstained" in the name) in the single stain controls folder. Preferably, you should have additional unstained controls if you are dealing with cell types that have variable autofluorescence. You can mark any additional samples as unstained in the Sample Browser (left pane) by right clicking them and checking "Mark as Unstained". Unstained samples are marked in the sample browser with a blue droplet icon.

![check_mark_unstained.png](/assets/check_mark_unstained.png)

Now in the Spectral Process tab, generate your spectral controls (e.g. with the Autogenerate Spectral Controls button, see [Spectral Analysis](/docs/spectral_analysis)). Make sure you select "Using unstained negative" instead of "Using internal negatives". 

Click "Show AutoSpectral Cleaning", which brings up the AutoSpectral control cleaning box and a workflow help guide. 

Beside it, "Automatically Clean" makes "Auto generate spectral controls" continue straight into Clean Controls, so the cleaning below runs without a second click and the spectral model is only recalculated once. It is ticked by default in new experiments (the default can be changed in App Settings) and is saved with each experiment. Untick it to generate and clean separately.

Two new columns are shown in the spectral controls table: "Use cleaned" and "Exclude Noise". The cleaned profiles are applied when you run Clean Controls. The noise removal is not checked by default, as it is not necessary for controls with low autofluorescence: run it first and see the results. Then consider applying noise removal to controls with high autofluorescence and re-running.

You may now click "Clean Controls". The AutoSpectal cleaning pipeline will automatically proceed. Each control will be processed for:
- Saturation exclusion
- Brightest-event selection
- Scatter matching (selection of the best-matching negative cells)
- Noise removal (subtraction of matching autofluorescence per control event)

![show_autospectral_cleaning2.png](/assets/show_autospectral_cleaning2.png)

### Show peak channel event histograms
There is now a checkbox below the spectral model table, "Show peak channel event histograms". Enable this and click on a line of the spectral model table to see the events selected for both positive and negative controls after control cleaning, against all events in that sample.

You can adjust the gating here before re-running control cleaning.

![show_peak_channel_event_histograms.png](/assets/show_peak_channel_event_histograms.png)

### Scatter-matching Diagnostics

A further checkbox "Show scatter-matching plots" is shown. These show the unstained cells that best match the positive cells on the morphology plot, and are therefore the best choice to use for negatives of that profile.

![show_scatter_matching_diagnostics2.png](/assets/show_scatter_matching_diagnostics2.png)

### Spectral QC

A further checkbox "Spectral QC" is shown, which . 

![spectral_qc.png](/assets/spectral_qc.png)

## How to perform AutoSpectral AF (per-cell autofluorescence extraction)

### Show the AutoSpectral AF process
Go to the AutoSpectral tab and check the box "Show AutoSpectral AF process"
![show_autospectral_af.png](/assets/show_autospectral_af.png)

### Extract AF profile from an unstained sample
Under Step 1, select your (first) unstained sample, and press the button Extract AF Profile. (Default number of clusters: 200.) You should see a set of spectra of clusters below. This step should be repeated for each unstained control that you have available, generating multiple AF profiles. After extraction, the clusters can be inspected by clicking on the Stored AF Profiles list.

![extract_af_step1.png](/assets/extract_af_step1.png)

### Assign the AF profiles to samples
Under Step 2, select the AF profiles to apply to each sample. Multiple controls (columns) can be selected for each sample (row). Each cell in the sample will be automatically assigned its nearest cluster from of the selected controls for a more accurate AF subtraction.

To assign many samples at once, select a profile in the Stored AF Profiles list and use **Assign Selected Profile to All Samples**, or choose a folder and use **Assign Selected Profile to Folder** (every sample shown in the grid in that folder or its subfolders).

**Auto-assign from Scatter** fills in the samples shown in the grid that have no AF profile yet (bead samples are skipped; existing assignments are never changed). The forward and side scatter of each unstained sample with an AF profile is modelled as a Gaussian mixture, and each sample is given the profile(s) of the unstained sample(s) whose scatter best explains its own, chosen by BIC: a second or third profile is added only when it improves the fit and accounts for at least 10% of the events. A summary opens at the end, listing samples whose scatter lies partly outside every unstained sample assigned to them. Review the assignments before analysing: similar scatter does not guarantee similar autofluorescence.

![assign_af_step2.png](/assets/assign_af_step2.png)

### Inspect the AutoSpectral AF results against standard unmixing 
Under Step 3, select a sample (from the sample browser, left pane), and select channels and a source gate on the side-by-side plots.

Negative spread and other artifacts caused by inhomogeneous autofluorescence should be much improved.
![inspect_af_step3.png](/assets/inspect_af_step3.png)

This workflow recalculates all unmixed data for analysis in the Unmixed Data / Statistics tabs, and is applied whenever an assigned sample is loaded in the sample browser.