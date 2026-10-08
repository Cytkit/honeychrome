---
layout: default
title: Spectral Flow Cytometry Analysis
---
[Cytkit](https://cytkit.com) | [Honeychrome](https://honeychrome.cytkit.com/) 
---

# <img src="/src/honeychrome/view_components/assets/cytkit_web_logo.png" width="60"> Honeychrome

{% include menu.md %}

# How to analyse spectral cytometry data in Honeychrome
This guide assumes you have already installed Honeychrome using the [executable installers (Windows/Mac/Linux) or Python (cross-platform)](/readme), and that you have a set of FCS files to import. We will be using a 9-colour panel run on a 5-laser Cytek Aurora from the [AutoSpectral Full Workflow Example](https://www.colibri-cytometry.com/post/autospectral-full-workflow-example). If you wish to follow this example, download the data from [Mendeley Data](https://www.colibri-cytometry.com/post/autospectral-full-workflow-example) and unzip.

{:toc}

## Create a new Experiment file
Run Honeychrome and select New Experiment. Type "AutoSpectral Full Workflow Example". This creates a .kit file with the experiment metadata and a folder of the same name to organise the FCS files and exports.
![Splash screen](/assets/splash.png)

## Import your FCS files
Go to File menu > Import FCS Files, which brings up the following dialog window. You can either copy/move your FCS files into the experiment's Raw subfolder, or create a link to an existing data folder.
![Import data dialog](/assets/import_data_dialog.png)

After dragging the data into the Raw subfolder, you should see the files as follows. Note that Honeychrome will respect any folder grouping of the files on the filesystem, and present the same hierarchy in the sample browser:
![raw_subfolder_contents.png](/assets/raw_subfolder_contents.png)

You must also tell Honeychrome where the single stained controls are located. Open Experiment Settings (from the dialog or from menu Edit > Experiment Settings) and click on single stained controls to choose the correct folder. Here it is for this example:
![expt_settings.png](/assets/expt_settings.png)

Click Update Experiment Configuration at the bottom of the dialog, which will update your experiment metadata and check that the channels in the FCS files thatt you provided are consistent.
![update_expt_config.png](/assets/update_expt_config.png)

## Check raw cytometry
You can now browse your raw data. Click on any sample in the Sample Browser (left pane) to load the data. A full set of morphology plots, a ribbon plot and histograms are provided.

> **Tip:** You can add and manipulate gates, but don't attempt to do your analysis yet! You must first set up a spectral model so that you can work with the unmixed data. 
> 
![browse_raw_data.png](/assets/browse_raw_data.png)


## Build spectral model, unmix and fine tune

Select the Spectral Process tab. In this example, we will just press "Auto generate spectral controls", which takes all the FCS samples in the Single Stain Controls folder and calculates its profile. The default profile is the average intensity of the brightest events within the sample, minus an internal negative control (the dimmer events in the same sample). 

> **Tip:** You can alternatively use an unstained negative control: make sure first that one sample has a name containing the word "unstained" (case insensitive) and that you have defined gates Neg Unstained and Pos Unstained.

![auto_spectral_controls.png](/assets/auto_spectral_controls.png)

In this example, there are duplicate cell and bead controls. Select the bead controls and press the button "Delete Selected". You should now have the following controls in the spectral model editor. You can rename the labels at this point, or go back to the raw and adjust the gates that were set up automatically to define these controls.

![correct_spectral_controls.png](/assets/correct_spectral_controls.png)

If a control has failed, you can use a profile from the Reference Library. To add a control to the library, right-click it in the spectral model editor and choose Save to Reference Library. To use a saved (or shipped) profile, set Control Type to Single Stained Spectral Control from Library, enter the fluorophore as the label, and pick a match under Sample Name. Only profiles recorded for the same instrument and detector configuration are offered. Browse and manage the library from View > Reference Library.

> **Tip:** Selecting one or more controls shows only these in the spectral viewer (and lines of the matrices below), which makes it easier to work with large panels.

![spectral_control_from_library.png](/assets/spectral_control_from_library.png)

Now press Select None (or Select All) so we can see all profiles and the full matrices below:
- Similarity Matrix
- Hotspot Matrix
- Unmixing Matrix

![similarity_matrix.png](/assets/similarity_matrix.png)
![hotspot_matrix.png](/assets/hotspot_matrix.png)
![unmixing_matrix.png](/assets/unmixing_matrix.png)

These look reasonable.

You can do fine tuning with the spillover matrix and/or the NxN plots:

![spillover_matrix.png](/assets/spillover_matrix.png)
![nxn_plots.png](/assets/nxn_plots.png)

> **Tip:** If you have a large panel, don't waste time doing this on the full set - it is slow and difficult to find the right row. Click one or more labels in the spectral model editor first, so that you can see the relevant rows and hide the others. Click on the relevant cell before rolling the mouse wheel to adjust fine tuning:

![filtered_fine_tuning.png](/assets/filtered_fine_tuning.png)

## Analyse unmixed cytometry
You can now analyse your unmixed data. Manipulating plots and gates should hopefully be intuitive, but here are [some instructions](/docs/cytometry_plots_and_gates.md).

![unmixed_data_start.png](/assets/unmixed_data_start.png)


## Further instructions: 
- [Manipulate Plots and Gates](/docs/cytometry_plots_and_gates)
- [AutoSpectral in Honeychrome](/docs/autospectral_in_honeychrome) 
- [Reports, Exports & Sample Comparison](/docs/reports)
