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

Honeychrome provides these parts of AutoSpectral natively:
- **AutoSpectral Cleaning** (optimisation of control profiles), in the Spectral Process tab.
- **The AutoSpectral tab**, with four pages, each building on the one before:
  - **Autofluorescence** (AutoSpectral AF): per-cell autofluorescence extraction.
  - **Noise**: noise-weighted unmixing with a positivity prior. This is new.
  - **Spectral Variants**: per-cell fluorophore spectra.
  - **Spectral Correction** (Fix My Unmix): attempts to corrects the fluorophore spectra from residual spillover in fully stained samples. This is new. Use it
  with caution.

AutoSpectral Cleaning clears up artifacts due to poor single stain control profiles, for example hypernegative events, skewed data and diagonal correlations, by carrying out saturation exclusion, brightest-event selection, scatter matching and noise removal on the control profiles.

AutoSpectral AF clears up artifacts due to variable autofluorescence, for example negative spreading, by pairing every event with its best matched autofluorescence spectrum. The noise model, the spectral variants and the spectral corrections then reduce the remaining spread and spillover further.

Whatever you choose on the AutoSpectral tab is used for the whole experiment: the Unmixed Data tab, unmixed exports, Honeycluster, the gating plugins and Statistics all unmix each sample the same way.

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

Below the scatter-matching plots, tick "Show spectral QC plot" and choose a control to compare its calculated spectrum (purple) with the reference library spectrum for that fluorophore (green). The cosine similarity, the number of events kept and the expected and observed peak channels are shown with it. Clean Controls warns about a control whose cosine similarity is low.

![spectral_qc.png](/assets/spectral_qc.png)

## The AutoSpectral tab

Open the **AutoSpectral** tab. The pages are listed down the left side: Autofluorescence, Noise, Spectral Variants and Spectral Correction. Each page opens with a **Show Help** button that explains every setting, with references, and ends with a side-by-side comparison of the current sample (the sample selected in the sample browser) unmixed two ways. The comparison plots open on the Singlets gate, else Cells, else the root gate, and you can change the gate, the channels and the scaling on either plot.

Work through the pages in order. Autofluorescence is the basis of everything else. The noise model or the spectral variants are optional, and you can use one or the other, not both. Spectral correction is optional and works with whichever method you have chosen.

## Autofluorescence (AutoSpectral AF)

### Extract AF profiles from unstained samples
Under Step 1, select your (first) unstained sample, and press **Extract AF Profile**. Only samples marked as unstained are listed (see above). The AF events are clustered on a self-organising map: **SOM grid size** 10 gives at most 100 AF spectra, and near-duplicates are removed, so you usually see fewer. Repeat this for each unstained sample you have, for example one per tissue or cell preparation, generating one AF profile per unstained sample.

Click a profile in the Stored AF Profiles list to see its spectra; the profile mean is drawn in black. To remove one spectrum that you do not want (for example one that is clearly a fluorophore rather than autofluorescence), right-click it in the plot and choose **Delete spectrum…**. The mean cannot be deleted. Deleting a spectrum renumbers the experiment's AF Index, so gates drawn on AF Index may select different spectra afterwards; re-extract the profile to restore it. **Delete Selected** removes a whole profile and its assignments. Profiles can also be saved to and loaded from CSV.

![extract_af_step1.png](/assets/extract_af_step1.png)

### Assign the AF profiles to samples
Under Step 2, select the AF profiles to apply to each sample. Multiple profiles (columns) can be selected for each sample (row). Each cell in the sample is then given the best-fitting AF spectrum from all of its selected profiles.

To assign many samples at once, select a profile in the Stored AF Profiles list and use **Assign Selected Profile to All Samples**, or choose a folder and use **Assign Selected Profile to Folder** (every sample shown in the grid in that folder or its subfolders). Tick **Include unstained controls** or **Include single-stained controls** to show those samples in the grid too.

**Auto-assign from Scatter** fills in the samples shown in the grid that have no AF profile yet (bead samples are skipped; existing assignments are never changed). The forward and side scatter of each unstained sample with an AF profile is modelled as a Gaussian mixture, and each sample is given the profile(s) of the unstained sample whose scatter best explains its own. With **One profile per sample** ticked (the default) that is a single unstained sample; unticked, up to three are chosen by BIC, a second or third being added only when it improves the fit and accounts for at least 30% of the events. A summary opens at the end; Show Details lists first the samples whose scatter lies partly outside every unstained sample assigned to them. Review the assignments before analysing: similar scatter does not guarantee similar autofluorescence.

![assign_af_step2.png](/assets/assign_af_step2.png)

### Compare with standard unmixing
Under Step 3, the current sample is shown unmixed by ordinary least squares without AF (left) and with per-cell AF (right). "Assigned to sample" uses the Step 2 assignment; choosing a single profile previews it without changing the assignment. Negative spread and other artifacts caused by inhomogeneous autofluorescence should be much improved.

![inspect_af_step3.png](/assets/inspect_af_step3.png)

Every sample with an assigned profile is now unmixed with per-cell AF wherever it is used. Two channels are added to the unmixed data: **AF Abundance** (how much autofluorescence was removed from each cell) and **AF Index** (which AF spectrum was chosen for each cell).

## Noise (noise-weighted unmixing)

Every detector reads with noise: a constant electronic floor plus photon noise that grows with the signal. Ordinary least squares trusts every detector equally. The noise model weights each detector by how noisy it is, and a positivity prior pulls fluorophores that a cell is clearly negative for towards zero, so their spread no longer lands in the other channels. The result is narrower negatives in crowded parts of the panel.

1. Choose the **Unstained cells** and the **Bead negative** (both default to Auto: the unstained cell sample with an AF profile and the lowest mean signal, and the unstained bead samples or the bead controls' negatives). The noise model also uses the single-stained controls of the Spectral Process. It selects its own events (the main scatter population and the singlets) and does not use your gates.
2. Press **Model Noise**. Each control is split: the model is fitted on 70% of the events and checked on the other 30%. The summary says how well the model predicts the held-out events; tick **Show calibration and detector tables** for the details.
3. **Use the noise model for the experiment's unmixing** is ticked when the fit has finished. Untick it to go back to per-cell AF.
4. **Positivity prior (lambda)**, default 1, sets how strongly likely-negative fluorophores are pulled to zero. Higher values narrow the negatives further, at the cost of over-estimating very dim cells positive for two very similar dyes. This
is based on the modelled dispersion of the spillover spread in the controls.

Some cytometers write data clamped at zero (no negative values on any detector; seen on the Attune Xenith and Bigfoot). The noise model does not describe that data, so Model Noise leaves it off and offers **Use the weights for clamped data** instead.

The comparison shows the current sample with per-cell AF, and with per-cell AF plus the noise model.

Unmixing with the noise model takes longer than per-cell AF. When you open a sample, the Unmixed Data tab shows per-cell AF at once and swaps in the noise-model result when it is ready (the status bar says so). The result is kept on disk in the experiment's `cache/unmixed` folder, so a sample you open again appears at once. The space this cache may take is set in App Configuration (**Unmixing cache size**, default 8 GB per experiment); the least recently used files are removed first.

The noise model was shaped by the work of, and discussions with, Paul Heisig (UMC Amsterdam) and David Novo.

## Spectral Variants (per-cell fluorophore spectra)

A fluorophore's spectrum is not quite the same in every cell: tandem dyes break down to different degrees, and the local environment shifts a dye's spectrum a little. This page finds that variation in each single-stained control and lets unmixing choose, for every cell, the variant of each fluorophore that fits it best, together with the cell's autofluorescence. It needs AF profiles with at least two spectra assigned to the samples.

1. Press **Compute Spectral Variants**. The variants are found from each fluorophore's single-stained control, and a necessity score decides which fluorophores are worth optimising per cell. Fluorophores with too little clean signal keep their reference spectrum.
2. **Plot Spectral Variants** shows the variants of a chosen fluorophore against its reference spectrum.
3. Tick **Use spectral variants for the experiment's unmixing**. This is not available while the noise model is in use.

Tick **Show Advanced Settings** for the settings of the variant search, the necessity score, the unmixing and the comparison; the Unmixing settings are stored with the experiment. As with the noise model, the Unmixed Data tab shows per-cell AF first and swaps in the result when it is ready, and the results are cached. Unmixed exports with spectral variants start with a time estimate.

The comparison shows the current sample with per-cell AF, and with per-cell AF plus spectral variants.

## Spectral Correction (Fix My Unmix)

Single-stained controls give each fluorophore's spectrum, but a reference spectrum can still be slightly wrong for your fully stained cells: a tandem that has degraded differently, a control stained on beads, a different buffer. This shows as residual spillover: cells negative for a marker reading above or below zero in proportion to another marker. Fix My Unmix measures that spillover on the fully stained samples, where a marker-negative population must read zero, and corrects the spectra.

1. The **Setup** box lists what the earlier pages provide: the AF profiles, the spillover spread, the spectral variants and the experiment's unmixing method.
2. Choose the **fully stained samples**. By default every sample that is not a bead sample, an unstained sample or a single-stain control is used, concatenated (**Default Samples** restores this). Tick **Concatenate per group** to correct each raw subfolder on its own; each sample is then unmixed with its own group's corrected spectra.
3. Each group needs an unstained sample: **Auto** picks the unstained cell sample whose scatter fits the group best (**Match Unstained** shows the choice), or choose one yourself. Set the gate and the number of events per group (400,000 by default; half of each run's corrections on a 40–50 colour panel varied between halves of 200,000 events, so fewer makes the corrections noisier).
4. Optionally tick fluorophores under **Fluorophores kept fixed**.
5. Press **Fix Unmix**. Every run starts again from the base spectra. With **Smooth updates** (Advanced Settings, on by default) it runs five rounds on fresh halves of the events and averages them; this takes about three times as long as a single pass. **Smooth updates: keep** set to **Only corrections the rounds agree on** runs every round from the base spectra and keeps a fluorophore's correction only when most rounds found it and agree on it; fewer fluorophores change, and what changes reproduces.
6. Review the **Results**: a table of the fluorophores that changed, with the angle of the change and an **Active** tick for each, the before/after spectra, and a heatmap of the change (**Show spectral change heatmap**). Untick **Active** to keep a fluorophore's original spectrum.
7. **Use spectral corrections for the experiment's unmixing** is ticked after the run. Every fully stained sample of the experiment (not an unstained, single-stain or bead sample) is then unmixed with the corrected spectra; after a run per group, each sample takes its own raw subfolder's corrections, and a subfolder Fix Unmix did not run on keeps the base spectra. Untick a raw subfolder under **Apply the corrections to the samples of** to keep its samples on the base spectra.

The comparison shows the current sample, whichever it is, with the base spectra and with the corrected spectra it takes, both with the experiment's unmixing method, on the pair of channels that changes most. For a sample the experiment's unmixing does not correct (a control, an unticked subfolder, or with the corrections unticked) the comparison is marked as a preview.

The corrections are kept as an overlay: your single-stain control spectra in the Spectral Process are not changed. If you change those spectra (for example by re-cleaning the controls), the corrections no longer apply and are cleared, with a message on the page; run Fix Unmix again.

## Long jobs

Only one long job runs at a time in Honeychrome: AF extraction, Model Noise, Compute Spectral Variants, Fix Unmix, Clean Controls, unmixed exports, Honeycluster runs, Run Gating and Statistics, among others. While one runs, the status bar shows **Running: …**, and the buttons that would start another, or change the unmixing, are greyed out; hover over one to see which job is running. The comparisons and the background unmixing of the sample you are viewing wait and continue when the job has finished. You can still browse samples.

If a job has stopped without finishing and the buttons stay greyed, right-click **Running: …** in the status bar and choose **Release**. Only do this when you are sure the job is no longer running.

## Further reading
- Burton OT et al. AutoSpectral improves spectral flow cytometry accuracy through optimised spectral unmixing and autofluorescence-matching at the cellular level. *bioRxiv* 2025. [doi:10.1101/2025.10.27.684855](https://doi.org/10.1101/2025.10.27.684855)
- Novo D, Grégori G, Rajwa B. Generalized unmixing model for multispectral flow cytometry utilizing nonsquare compensation matrices. *Cytometry A* 2013; 83(5): 508–520. [doi:10.1002/cyto.a.22272](https://doi.org/10.1002/cyto.a.22272)
- Novo D. A comparison of spectral unmixing to conventional compensation for the calculation of fluorochrome abundances from flow cytometric data. *Cytometry A* 2022. [doi:10.1002/cyto.a.24669](https://doi.org/10.1002/cyto.a.24669)
- Kmet R, Novo D. Reducing spreading: removing the impact of irrelevant dyes improves unmixed flow cytometry data. *Cytometry A* 2025. [doi:10.1002/cyto.a.24957](https://doi.org/10.1002/cyto.a.24957)
- Nguyen R, Perfetto S, Mahnke YD, Chattopadhyay P, Roederer M. Quantifying spillover spreading for comparing instrument performance and aiding in multicolor panel design. *Cytometry A* 2013; 83(3): 306–315. [doi:10.1002/cyto.a.22251](https://doi.org/10.1002/cyto.a.22251)
- Cai X, Garcia-Garcia S, Kuhnen L, Gianniou M, Garcia Vallejo JJ. Unmixing spread estimation based on residual model in spectral flow cytometry. *bioRxiv* 2026. [doi:10.64898/2026.01.27.701929](https://doi.org/10.64898/2026.01.27.701929)
- [AutoSpectral package on GitHub](https://github.com/DrCytometer/AutoSpectral)
