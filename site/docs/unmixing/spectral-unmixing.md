---
title: Spectral unmixing
lead: Create an experiment, import FCS files, build a spectral model from single-stained controls, then unmix and check the result.
redirect_from:
  - /docs/spectral_analysis
  - /docs/spectral_analysis.html
---

This guide takes you from FCS files to unmixed data. It uses the **AutoSpectral Full Workflow Example**, a 9-colour panel run on a 5-laser Cytek Aurora. To follow along, download the data from the [AutoSpectral Full Workflow Example](https://www.colibri-cytometry.com/post/autospectral-full-workflow-example) post (it links to the files on Mendeley Data) and unzip them. If you have not installed Honeychrome yet, see [Install](/docs/get-started/install/).

## Create an experiment

Start Honeychrome and choose **New Experiment**. Type a name, for example "AutoSpectral Full Workflow Example". This creates a `.kit` file, which holds the experiment's settings, and a folder of the same name for the FCS files, exports and reports.

{% include figure.html src="docs/splash.webp" alt="Honeychrome start-up screen with New Experiment and Open Experiment" width="1600" height="1031" %}

## Import your FCS files

Choose **File > Import FCS Files**. You can either copy or move your FCS files into the experiment's `Raw` folder, or link to an existing data folder.

{% include figure.html src="docs/import_data_dialog.webp" alt="Import FCS Files dialog offering to copy, move or link FCS files into the Raw folder" width="922" height="760" %}

Honeychrome keeps the folder structure of your files and shows the same folders in the sample browser on the left. Folders are also used as sample groups later, for example in Simple Stats and when assigning autofluorescence profiles.

{% include figure.html src="docs/raw_subfolder_contents.webp" alt="Sample browser showing the imported FCS files grouped by their folders" width="1600" height="1258" %}

Tell Honeychrome where the single-stained controls are. Open **Edit > Experiment Settings** (or use the button in the import dialog) and click **single stained controls** to choose the folder.

{% include figure.html src="docs/expt_settings.webp" alt="Experiment Settings with the single stained controls folder selected" width="1600" height="1319" %}

Then click **Update Experiment Configuration** at the bottom of the dialog. Honeychrome reads the channels of every file and checks that they are consistent. If the detector voltages or gains differ between files, it warns you, because Honeychrome does not adjust for them.

{% include figure.html src="docs/update_expt_config.webp" alt="Update Experiment Configuration button at the bottom of the Experiment Settings dialog" width="1044" height="1176" %}

## Check the raw data

Click any sample in the sample browser to load it. The **Raw Data** tab shows morphology plots, a ribbon plot of every detector, and histograms.

Set broad scatter gates here that keep every cell of interest, large and small, but not doublets. The spectral controls use them.

> You can draw gates on the raw data, but do your analysis in the **Unmixed Data** tab, once the spectral model is built.
{: .tip}

{% include figure.html src="docs/browse_raw_data.webp" alt="Raw Data tab with morphology plots, a ribbon plot and histograms for the selected sample" width="1600" height="1000" %}

### Unstained and bead samples

The sample browser marks two kinds of sample, and you can change either with a right click:

- **Unstained** samples have a blue droplet icon. Files with "unstained" in the name are recognised automatically. Use **Mark as Unstained** or **Unmark as Unstained** for others.
- **Bead** samples are shown in green. Unstained samples and single-stained controls whose scatter looks like beads are marked automatically. Use **Mark as Beads** or **Unmark as Beads** to correct it; your choice is final.

Unstained, single-stained and bead samples are left out by default when you choose samples for analysis.

## Build the spectral model

Open the **Spectral Process** tab and press **Auto generate spectral controls**. Honeychrome makes one control for each file in the single-stained controls folder:

- The **label** comes from the start of the file name, and antigen and fluorophore names are recognised.
- The **particle type** comes from the name, for example "CD4 BUV395 (Cells)" or "(Beads)". When the name does not say, Honeychrome decides from the scatter, and bead controls are matched to the right unstained bead sample when there are several bead types.
- The **major channel** is the detector where the fluorophore varies most. A positive gate is placed on the brightest events and a negative gate on the dimmest (5% and 25% by default, set in App Configuration). The gates are added to the raw data plots, where you can adjust them.

{% include figure.html src="docs/auto_spectral_controls.webp" alt="Spectral Process tab after Auto generate spectral controls, listing a control for each FCS file" width="1600" height="1000" %}

### Choose the negative

Each spectrum is the mean of the positive events minus a negative:

- **Using internal negatives** (the default) takes the dimmest events of the same control.
- **Using unstained negative** takes an unstained sample instead, which is usually better. Choose the unstained sample for each control in the **Unstained Negative** column, or for all at once with **Set negative for all**.

With an unstained negative and **Automatically Clean** ticked (the default), **AutoSpectral control cleaning** runs straight after the controls are generated. See [Control cleaning](/docs/unmixing/control-cleaning/).

### Tidy the controls

This example has both cell and bead controls for some fluorophores. Select the bead controls and press **Delete Selected**. You can also rename labels here, or go back to the Raw Data tab and adjust the gates of a control.

{% include figure.html src="docs/correct_spectral_controls.webp" alt="Spectral model editor after deleting the duplicate bead controls" width="1600" height="976" %}

If you change the sample of a control, for example from cells to beads, its particle type and negative follow the new sample. Its gates are placed again, and it is cleaned again if Automatically Clean is on.

Unmixed gates are kept when the spectral model is regenerated.

### Use a spectrum from the Reference Library

If a control has failed, use a spectrum from the Reference Library instead:

1. Set the control's **Control Type** to **Single Stained Spectral Control from Library**.
2. Enter the fluorophore as the label.
3. Choose a match under **Sample Name**. Only spectra recorded on the same instrument and detector configuration are offered.

To add one of your own controls to the library, right-click it in the spectral model editor and choose **Save to Reference Library**. **View > Reference Library** opens the library itself, where you can browse, compare, import and export spectra for your instrument.

{% include figure.html src="docs/spectral_control_from_library.webp" alt="A control set to Single Stained Spectral Control from Library, with a matching library profile chosen" width="1600" height="545" %}

## Check the spectral model

> Select one or more controls in the table to show only their spectra, and only their rows of the matrices and NxN plots. This makes large panels much easier to work with.
{: .tip}

Press **Select None** (or **Select All**) to see every spectrum and the full matrices:

- the **similarity matrix**: how alike each pair of spectra is;
- the **hotspot matrix**: where unmixing will spread signal between fluorophores;
- the **unmixing matrix**.

The condition number of the spectral model is shown above the table. Higher values mean less stable unmixing, and Honeychrome warns about a poorly conditioned matrix.

{% include figure.html src="docs/similarity_matrix.webp" alt="Similarity matrix of the fluorophore spectra" width="982" height="816" %}

{% include figure.html src="docs/hotspot_matrix.webp" alt="Hotspot matrix of the fluorophore spectra" width="982" height="816" %}

{% include figure.html src="docs/unmixing_matrix.webp" alt="Unmixing matrix" width="1600" height="481" %}

## Fine tune

The **spillover matrix** and the **NxN plots** let you fine tune the unmixing by hand. The NxN plots show every pair of unmixed labels for the sample selected in the sample browser.

- Click a cell of the spillover matrix or a tile of the NxN plots, then roll the mouse wheel to adjust it. You must click first, so scrolling the page cannot change the spillover by accident.
- Double-click a cell of the spillover matrix to type a value.
- Hover over a tile to see its spillover.

{% include figure.html src="docs/spillover_matrix.webp" alt="Spillover matrix used for fine tuning" width="1108" height="820" %}

{% include figure.html src="docs/nxn_plots.webp" alt="NxN plots of every pair of unmixed labels" width="1476" height="1393" %}

> With a large panel, select the labels you are working on first. With nothing selected, every pair is drawn: a 60-colour panel means 1,800 plots at once.
{: .tip}

{% include figure.html src="docs/filtered_fine_tuning.webp" alt="Fine tuning with only the selected labels shown in the spillover matrix and NxN plots" width="1600" height="565" %}

## Analyse the unmixed data

Open the **Unmixed Data** tab to analyse your unmixed data; see [Plots and gates](/docs/plots-and-gating/plots-and-gates/).

{% include figure.html src="docs/unmixed_data_start.webp" alt="Unmixed Data tab with plots of the unmixed labels" width="1600" height="489" %}

## Next steps
- [AutoSpectral in Honeychrome](/docs/unmixing/autospectral/): clean the controls and remove autofluorescence cell by cell.
- [Plots and gates](/docs/plots-and-gating/plots-and-gates/)
- [Simple Stats](/docs/statistics/simple-stats/) and [Reports and exports](/docs/outputs/reports-and-exports/)

## Further reading
- [What's an Unmixing Matrix?](https://www.colibri-cytometry.com/post/what-s-an-unmixing-matrix) on the Colibri Cytometry blog
- Novo D, Grégori G, Rajwa B. Generalized unmixing model for multispectral flow cytometry utilizing nonsquare compensation matrices. *Cytometry A* 2013;83(5):508–520. [doi:10.1002/cyto.a.22272](https://doi.org/10.1002/cyto.a.22272)
- Mage PL, Konecny AJ, Mair F. Measurement and prediction of unmixing-dependent spreading due to collinearity in spectral flow cytometry. *Cytometry A* 2026;109(6):436–455. [doi:10.1002/cyto.a.70044](https://doi.org/10.1002/cyto.a.70044). The similarity and hotspot matrices follow this paper.
