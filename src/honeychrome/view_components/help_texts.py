process_help_text = '''
<h3>Spectral Process Help</h3>
<p>
Use the Spectral Model Editor to specify the spectral model, for both spectral and conventional cytometry. Single stained controls can be set automatically (see Auto Generate below).
The spectral model should consist of a set of "controls", where each may be of the following <i>control types</i>:
</p>

<ul>
<li>
<b>Single Stained Control:</b> 
this is a sample of beads or cells labelled with one fluorophore only, or autofluorescence only. 
Single stained controls can be set automatically (see Auto Generate below).
To set each single stained control manually, edit the <i>label</i> (e.g. CD4-FITC or Unstained), 
set <i>particle type</i> to cells or beads, and select the control sample's FCS file from the pull-down menu under </i>sample name</i>, 
which lists all FCS files within the Single Stain Controls subfolder. Under <i>positive gate</i>, select the gate that marks the positive events. 
</li>

<li>
<b>Single Stained Control from Library:</b> 
if the required single stained control sample is not present in the current experiment, it may be loaded from the Reference Library. 
To add a control to the Reference Library, right-click it in this table and choose <i>Save to Reference Library</i>. 
The library also ships with reference spectra for supported instruments, and can be browsed from <i>View &gt; Reference Library</i>. 
Enter the <i>label</i> (matching the fluorophore name, ignoring case) and choose a match under <i>sample name</i>. 
Only profiles recorded with the same instrument and detector configuration are listed.
</li>

<li>
<b>Channel Assignment:</b> 
in the absence of single stained spectral controls, you may assign a channel directly to a label. 
Each channel assignment is equivalent to assigning a label name to a channel name, or setting the spectral profile to a delta function 
(i.e. intensity = 1 for the <i>major channel</i>, 0 for all other channels). 
Normally, this is only applicable in conventional cytometry with a low number of fluorophores, where you can either compensate manually or ignore compensation.
</li>
</ul>

<h4>How to use <i>Auto Generate Spectral Controls</i></h4>
<p>
This function parses the names of the FCS files in the single stain controls and builds the spectral model automatically. 
The single stain controls subfolder (default: <tt>Raw/Single stain controls</tt>) should include the full set of controls including one named "Unstained". 
</p>

<ol>
<li>
The single stained controls names should start with "Label (Cells)..." or "Label (Beads)". 
(If the name does not contain the particle type, "Cells" is assumed.)
Each is processed to find the fluorescence channel with the largest variation according to that label ("major channel"). 
A positive gate is assigned to the brightest events and a negative gate to the dimmest (percentages are set in Application Configuration; 
default 5% brightest and 25% dimmest). Positive and negative gates are added to the raw cytometry plots as appropriate. 
Select one or more controls in the spectral model to inspect their spectra in the profiles viewer below, or select all / none to inspect all spectra.
</li>

<li>
The spectral profiles use <i>Internal Negatives</i> by default, i.e. the profile is defined as the mean fluorescence of the positively gated sample, 
minus the fluorescence of the negatively gated sample, normalised to peak intensity = 1.
An alternative (better) negative is the <i>Unstained Negative</i>. I.e. the profile is defined as the mean fluorescence of the positively gated sample, 
minus the fluorescence of the negatively gated <i>unstained</i> sample. "Pos Unstained" and "Neg Unstained" gates are added to the raw gating if you didn't already create them.
</li>

<li>
If any singly stained spectral control is missing, you may load it from the Reference Library. 
Select <i>control type</i> as "Single Stained Control from Library", and enter the label name. 
Matching profiles for this instrument configuration will be listed under <i>sample name</i>.
</li>
</ol>

<h4>Profiles Viewer, Similarity Matrix, Hotspot Matrix, Unmixing Matrix</h4>
<p>
The spectral profiles are displayed in the profiles viewer; the similarity matrix, hotspot matrix and unmixing matrix are also calculated and displayed. 
Selecting one or more labels shows only the relevant profiles and rows of the similarity matrix, unmixing matrix, spillover matrix and NxN plots.
</p>

<h4>Spillover, Compensation, Fine-Tuning and NxN Plots</h4>
<p>
In addition to the non-square unmixing matrix, a square spillover matrix is defined, which can be used for fine tuning in spectral cytometry or manual compensation in conventional cytometry. 
The initial values of spillover matrix are 0 for all non-diagonal elements (1 for diagonal elements). 
</p>

<p>
The spillover matrix can be edited by double clicking on the relevant cell. Alternatively it can be adjusted by rolling the mouse wheel on the spillover matrix or the corresponding tile of the NxN plots.
</p>

<p>
2D histograms of all pairwise combinations of fluorophores (for the current sample, if selected) are displayed in the NxN plots, 
and can be used to inspect the unmixing process and look for issues.
</p>



<h4>Further Reading and Background</h4>

<p>
Honeychrome currently generates an unmixing matrix using only the ordinary least squares (OLS) method, which is the mathematically simplest spectral 
unmixing algorithm [1]. Nevertheless, good results can be obtained with careful selection of clean single stained controls, one or more representative unstained
negatives, and sanity checking of the profiles, similarity matrix, unmixing matrix and NxN plots [2]. 
Spectral unmixing is an area of active development, with much more sophisticated algorithms now available that take proper account of errors in the controls and 
sources of autofluorescence [2]. 
Highly recommended is the Colibri Cytometry blog <a href="https://www.colibri-cytometry.com/post/what-s-an-unmixing-matrix">What's an Unmixing Matrix?</a>.
The similarity and hotspot matrices are defined according to [3].
</p>
<p>
References:<br/>
[1]<a href="https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22272">Novo et al. 2013</a><br/>
[2]<a href="https://www.biorxiv.org/content/10.1101/2025.10.27.684855v1.full">Burton et al. 2025</a><br/>
[3]<a href="https://www.biorxiv.org/content/10.1101/2025.04.17.649396v2.full">Mage et al. 2025</a><br/>
</p>
'''

nxn_help_text = '''
<h3>Select any sample from the sample browser to view it in the NxN plots below.</h3>
<h5>Select from the sample browser on the left, not the spectral model above.</h5>
<ul>
<li>If your panel is large, select one or more rows in the editor above, so that the corresponding rows of the NxN array are plotted below. This makes the plots easier to find, quicker to plot, and quicker to fine-tune.</li>
<li>If no labels are selected above, the full array of NxN plots will be calculated. <b>Warning: this is slow if N is large!</b> If you are using a 60-colour panel, that means calculating 1800 2D histograms at once.</li>
<li>Click to select a cell and roll scroll wheel to adjust spillover (fine tuning) (row label spills into column labels)</li>
<li>Hover to inspect spillover of each cell in the array</li>
</ul>
'''

autospectral_af_help_text = '''
<h5>This is an optional enhancement of the spectral process, which implements the autofluorescence (AF) extraction part of the AutoSpectral package (AutoSpectral AF).</h5>
<p>By following the steps below, you can extract each cell’s individual autofluorescent background in a manner specific to that cell, producing better unmixing with less spread.</p>
<ol>
<li>Extract AF profile from an unstained sample. This can be repeated for each unstained control that you have available, generating multiple AF profiles. </li>
<li>Only unstained cell control samples will appear as options. Mark your unstained samples by right-clicking on the sample and selecting "Mark as Unstained".</li>
<li>Default: 200 clusters (duplicates will be removed, so you will see fewer). After extraction, the clusters can be inspected by clicking on the Stored AF Profiles list.</li>
<li>Assign the AF profiles to Samples. Each cell in the sample is automatically assigned its nearest cluster from of the selected controls for a more accurate AF subtraction.</li>
<li>Inspect the AutoSpectral AF results against standard unmixing. Negative spread and other artifacts caused by inhomogeneous autofluorescence should be much improved.</li>
<li>
</ol>
<p>AutoSpectral AF is intended for use in spectral unmixing, not in conventional flow with compensation.</p>
<p>New profiles also store a <b>negative envelope</b>: the spread of each fluorophore in the unstained sample once the profile's mean AF is removed, as a function of AF brightness, to which the spillover spread of the single-stain controls is added for each cell. A cell's AF is chosen without crediting an AF spectrum for cancelling real fluorophore signal: each channel is judged once, at the profile's mean AF, by its probability of being positive against that distribution of the negative, and counts towards choosing the AF in proportion to its probability of being negative. A negative channel is fitted towards zero; the more clearly positive a channel is, the less it counts, and no channel switches on or off at a boundary. Events with a saturated detector are unmixed from the remaining detectors. The envelope follows the spectral model automatically while the spectra move by less than 2 degrees; larger changes rebuild it from 5,000 stored unstained events, and extracting a profile re-estimates the spread. Envelopes stored by an earlier version are upgraded from the same stored events. Profiles without an envelope (from older experiments or loaded from CSV) use the previous scoring.</p>
<p><b>Note on unstained samples:</b> If you intend to use <b>AutoSpectral AF</b>, do <em>not</em> add unstained samples to the Spectral Process table. Unstained samples should only be used as the negative reference (assigned via the "Unstained Negative" column), not as spectral controls in their own right. Including them as controls will produce manual AF extraction profiles and can be used as an alternative to AutoSpectral AF.</p>
<p>Documentation:</p>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/introducing-autospectral-an-optimized-unmixing-workflow">AutoSpectral on the Colibri Cytometry blog</a></li>
<li><a href="https://www.colibri-cytometry.com/post/autospectral-single-cell-autofluorescence">AutoSpectral AF on the Colibri Cytometry blog</a></li>
<li><a href="https://github.com/DrCytometer/AutoSpectral">AutoSpectral package on Github</a></li>
<li><a href="https://www.biorxiv.org/content/10.64898/2026.01.27.701929v2.full">The Residual Model of Spillover Spread, Cai et al.</a></li>
<li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3678531/">Quantifying Spillover Spread, Nguyen et al.</a></li>
</ul>
'''

autospectral_cleaning_help_text = '''
<h3>AutoSpectral Control Cleaning — Workflow Guide</h3>
<p><b>Important:</b> AutoSpectral Control Cleaning is designed for <b>spectral flow cytometry</b> single-stained cell controls. It will have less impact on bead controls and has not been tested on conventional flow samples.</p>
<p><b>Note on unstained samples:</b> If you intend to use <b>AutoSpectral AF</b>, do <em>not</em> add unstained samples to the Spectral Process table. Unstained samples should only be used as the negative reference (assigned via the "Unstained Negative" column), not as spectral controls in their own right. Including them as controls will produce manual AF extraction profiles and can be used as an alternative to AutoSpectral AF.</p>
<h4>Workflow</h4>
<ol>
<li><b>Set up the basic scatter gates in the Raw Data tab:</b> for ease of use, set broad gates that capture all events of interest, including large and small cells but not doublets. The cleaning pipeline will automatically select the relevant events for each control based on the positive gate and particle type.</li>
<li><b>Run "Auto generate spectral controls":</b> do this in the Spectral Model Editor to set up the single stained controls, if you haven't already. This will add positive and negative gates to the raw gating for each control. With "Automatically Clean" ticked (beside "Show AutoSpectral Cleaning"), the controls are then cleaned straight away, as in the "Clean Controls" step below; use "Clean Controls" to re-run cleaning after changing gates, negatives or particle types.</li>
<li><b>Ensure that "Particle type" is set correctly for each control:</b> this determines the scatter-matching process. If particle type is incorrect, the cleaning will not work well.</li>
<li><b>For best results, use "Unstained Negative" profiles (instead of "Internal Negatives":</b> these provide a more accurate negative reference for the spectral profiles than the default internal negatives.</li>
<li><b>Assign unstained negatives:</b> Ensure each cell control has an "Unstained Negative" assigned in the table. Right-click a sample in the Sample panel and select "Mark as Unstained" to make it available as a negative reference.</li>
<li><b>Optional: Adjust the positive and negative gates:</b> By default, the positive gate is set to the brightest 5% of events and the negative gate to the dimmest 25% of events (percentages are configurable in Application Configuration). If these gates do not capture the positive and negative populations well, adjust them by moving the gate boundaries. The gates will be updated in the Spectral Model Editor and used for cleaning.</li>
<li><b>Run "Clean Controls":</b> Click the "Clean Controls" button. The pipeline will run for all eligible cell controls (those with an unstained negative assigned). Steps performed per control:
  <ul>
    <li>Saturation exclusion — removes detector-saturated events.</li>
    <li>Scatter matching — selects negative events whose scatter profile matches the positive control, reducing background due to cell size/granularity differences.</li>
    <li>Noise exclusion (if ticked) — identifies and removes intrusive autofluorescent events from the positive control using PCA on the matched unstained.</li>
  </ul>
</li>
<li><b>Inspect diagnostic plots:</b> After cleaning, use the scatter-matching and spectral QC diagnostic plots (visible below when cleaning is active and relevant box ticked) to verify that the cleaning has worked as expected.</li>
<li><b>If unsatisfied:</b> adjust the options and re-run or untick the "Use Cleaned" checkbox to revert to the original profiles for that control.</li>
<li><b>Use Cleaned profiles:</b> Once cleaning is complete, each eligible control will show a "Use Cleaned" checkbox. This is ticked by default and the shown profiles will be the cleaned ones. Uncheck to revert to the standard gate-mean method for a specific control. Hit "Recalculate" to update the spectral model.</li>
<li><b>Recalculate:</b> The spectral model is automatically recalculated after cleaning. If you change any cleaning settings, click "Clean controls" again to update. Only controls with updated settings will be updated.</li>
<li><b>Review:</b> fluorophore profiles, similarity matrix and HotSpot Matrices.</li>
<li><b>Inspect the NxN plots:</b> check for any remaining unmixing issues, and fine-tune the spillover matrix if needed.</li>
<li><b>Proceed:</b> to unmixing, AutoSpectral AF extraction if desired, and downstream analysis or batch export as normal, using the cleaned profiles.</li>
</ol>
<p>Documentation: <a href="https://drcytometer.github.io/AutoSpectral/articles/09_Cleaning.html">AutoSpectral package on Github</a></p>
<br/>
'''

autospectral_noise_help_text = '''
<h3>Noise: noise-weighted unmixing</h3>
<p>Every detector reads with noise: a constant floor (σ<sub>0</sub>, electronic noise) plus photon shot noise that grows with the signal. Ordinary least squares (OLS) trusts every detector equally. Weighting each detector by the inverse of its noise variance (generalised least squares, GLS) lets the quiet detectors count for more and the noisy ones for less, and a positivity prior then pulls fluorophores that a cell is clearly negative for towards zero, so their spread no longer lands in the other channels.</p>
<ol>
<li><b>Model Noise</b> fits, for every detector <i>d</i>, Var(<i>y<sub>d</sub></i>) = σ<sub>0,d</sub><sup>2</sup> + <i>μ<sub>d</sub></i> / κ<sub>d</sub> from the single-stained controls, an unstained cell sample and unstained beads (which anchor σ<sub>0</sub>). Each control is split 70/30: the model is fitted on 70% of the events and checked on the other 30%. Each control is described by its spectrum, its own spectral variants (found on the training events) and, for cell controls, the autofluorescence (AF) of its negative, so per-cell spectral variation is not mistaken for noise. A good model predicts the held-out residual variance within 0.8–1.25 for at least 80% of the control cells.</li>
<li><b>Event selection:</b> the noise model picks its own events and ignores the gates. Each sample is read in full; the densest population on FSC-A × SSC-A is found and the 95% ellipse around it kept (saturated scatter events included, as bright beads saturate it on some cytometers), then the singlets on FSC-A / FSC-H. Doublets and aggregates of beads would otherwise read as extra noise at high signal. The tables give the selected and total events per sample.</li>
<li><b>Estimator options</b> (validation; off by default): <i>the negative's own AF rows</i> lets bead negatives and bead controls carry their own bead autofluorescence, which varies from bead to bead; <i>one design row per laser</i> lets each control's signal vary between laser intercepts; the <i>per-control quadratic term</i> absorbs spread specific to one control, so it does not bias κ; <i>σ<sub>0</sub> per particle class</i> fits one floor for beads and one for cells (κ shared), as beads and cells differ in pulse width; <i>σ<sub>0</sub> from the same laser</i> replaces a floor the data could not pin down by the median of the other detectors on that laser, instead of 1% of the median.</li>
<li><b>Clamped data:</b> some exports write values clamped at zero, with no negative values on any detector (seen on the Attune Xenith and Bigfoot). Their noise is cut off at zero, which the noise model does not describe. When the unstained cells and bead negatives have no detector with 1% of values at or below zero, Model Noise leaves the noise model off and unmixes with Poisson weights from the fitted κ with a floor of 10 photoelectrons, scaled to the unstained cells' residual variance, and the positivity prior; the noise model can still be ticked.</li>
<li><b>Unmixing with the noise model</b> starts from the per-cell AF result, with each cell's AF spectrum chosen in the geometry of the weighted fit (one weight per detector for the sample). It then refits every fluorophore and the AF abundance by GLS with the positivity prior: each fluorophore is shrunk in proportion to how likely the cell is to be negative for it, judged against the spread of the negative (from the AF profile's negative envelope and the spillover spread). Each channel is reported with its own prior removed, so a dim positive keeps its scale.</li>
<li><b>Positivity prior (λ)</b> sets how strongly likely-negative fluorophores are shrunk. Higher values narrow the negatives further, at the cost of over-estimating dim cells positive for two near-identical dyes: across four large panels, λ = 0.5 gave negatives about 20% narrower than the AutoSpectral optimisation with about +9% bias on such double positives, and λ = 1 (the default) about 28% narrower with about +14%.</li>
<li><b>Use the noise model for the experiment's unmixing</b> (ticked after Model Noise): every sample is unmixed with the noise model in the main window, unmixed exports, Honeycluster, the gating plugins and Statistics. The main window shows per-cell AF at once and swaps the noise-model result in when it is ready; results are cached, so a sample shown again appears at once. Changing lambda unmixes the displayed sample again. Spectral variants are not used while this is ticked.</li>
<li><b>Comparison:</b> the current sample unmixed with per-cell AF, and with per-cell AF plus the noise model. A sample without an AF profile is shown as OLS against the noise model without AF.</li>
</ol>
<p><b>Credit:</b> Paul Heisig (UMC Amsterdam) and David Novo, whose work and discussions shaped the noise model and noise-weighted unmixing.</p>
<p><b>Further reading</b></p>
<ul>
<li>Novo D, Grégori G, Rajwa B. Generalized unmixing model for multispectral flow cytometry utilizing nonsquare compensation matrices. <i>Cytometry A</i> 2013; 83(5): 508–520. <a href="https://doi.org/10.1002/cyto.a.22272">doi:10.1002/cyto.a.22272</a></li>
<li>Novo D. A comparison of spectral unmixing to conventional compensation for the calculation of fluorochrome abundances from flow cytometric data. <i>Cytometry A</i> 2022. <a href="https://doi.org/10.1002/cyto.a.24669">doi:10.1002/cyto.a.24669</a></li>
<li>Kmet R, Novo D. Reducing spreading: removing the impact of irrelevant dyes improves unmixed flow cytometry data. <i>Cytometry A</i> 2025. <a href="https://doi.org/10.1002/cyto.a.24957">doi:10.1002/cyto.a.24957</a></li>
<li>Rajwa B, Roederer M. Lost in translation: harmonizing terminology and defining mathematical tools for panel optimization. <i>Cytometry A</i> 2025. <a href="https://doi.org/10.1002/cyto.a.24972">doi:10.1002/cyto.a.24972</a></li>
<li>Cai X, Garcia-Garcia S, Kuhnen L, Gianniou M, Garcia Vallejo JJ. Unmixing spread estimation based on residual model in spectral flow cytometry. <i>bioRxiv</i> 2026. <a href="https://doi.org/10.64898/2026.01.27.701929">doi:10.64898/2026.01.27.701929</a></li>
</ul>
'''

autospectral_variants_help_text = '''
<h3>Spectral Variants: per-cell fluorophore spectra</h3>
<p>A fluorophore's emission spectrum is not quite the same in every cell: tandem dyes break down to different degrees, and the local environment of a dye shifts its spectrum a little. This page finds that variation in each single-stained control and lets unmixing choose, for every cell, the variant of each fluorophore that fits it best, together with the cell's autofluorescence (AF). It is the per-cell optimisation of the AutoSpectral package, run by a compiled kernel.</p>
<p><b>Prerequisite:</b> a sample can only be compared here once AF profile(s) with at least two spectra are assigned to it on the <b>Autofluorescence</b> page.</p>
<p>AF profiles that carry a negative envelope (see the <b>Autofluorescence</b> page) are scored with it here too, including the saturated-detector refit: events with a saturated detector are unmixed, variants included, from the remaining detectors. Profiles without an envelope use the previous scoring. The compiled kernel must be rebuilt to use the envelope; an older build scores AF by joint L2.</p>
<ol>
<li><b>Compute Spectral Variants:</b> discovers spectral variation for each fluorophore from its single-stained control, by clustering background-corrected, scatter-matched positive events. Near-identical variants are then merged: a variant within <i>dedup.threshold</i> (default 1% of the peak, in any single detector) of the reference or of a better-populated variant is dropped, and the threshold is raised for a fluorophore until no more than <i>max.variants</i> (default 10) remain; the two variants furthest from the reference are always kept. Fluorophores with too little clean signal fall back to the single reference spectrum (not an error — just nothing to optimise for that one). Positivity thresholds reuse AF profiles already extracted from each unstained sample; AF is extracted only for unstained samples without a stored profile. A necessity score then estimates how much each fluorophore's variation would change the unmixing of the rest of the panel; fluorophores at or above <i>optimize.necessity.threshold</i> (Advanced Settings) are optimised per cell, the others keep their reference spectrum.</li>
<li><b>Plot Spectral Variants:</b> shows the discovered variant spectra (faint) against the reference spectrum (solid) for a selected fluorophore.</li>
<li><b>Use spectral variants for the experiment's unmixing:</b> samples with an AF profile are unmixed with per-cell spectral variants in the main window, unmixed exports (after a time estimate), Honeycluster, the gating plugins and Statistics; samples without one keep per-cell AF. The Unmixing settings in Advanced Settings are stored with the experiment. Not available while the noise model is in use (Noise page).</li>
<li><b>Comparison:</b> the current sample unmixed with per-cell AF alone and with per-cell AF plus spectral variants, subsampled for speed (<i>comparison.max_events</i> in Advanced Settings).</li>
</ol>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/introducing-autospectral-an-optimized-unmixing-workflow">AutoSpectral on the Colibri Cytometry blog</a></li>
<li><a href="https://github.com/DrCytometer/AutoSpectral">AutoSpectral package on Github</a></li>
</ul>
'''
autospectral_correction_help_text = '''
<h3>Spectral Correction: Fix My Unmix</h3>
<p>Single-stained controls give each fluorophore's spectrum, but a reference spectrum can still be slightly wrong for the fully stained cells: a tandem dye that has degraded differently, a control stained on beads, a different buffer. Part of such an error lies inside the span of the other spectra, and no residual check can see it; it shows as residual spillover, cells negative for a marker reading above or below zero in proportion to another marker. Fix My Unmix (from the AutoSpectral package's <tt>fix.my.unmix()</tt>) measures that spillover on fully stained samples, where a marker-negative population must read zero, and corrects the spectra.</p>
<ol>
<li><b>Setup</b> lists what the other pages provide: the AF profiles (every sample is unmixed with its own, by per-cell AF), the spillover spread (it widens the positivity boundaries), the spectral variants (a prior on how large each correction can plausibly be) and the experiment's unmixing method, which the corrected spectra feed.</li>
<li><b>Samples:</b> by default every sample that is not a bead sample, an unstained sample or a single-stain control, run together; the <i>Include</i> tick boxes offer those types too. <i>Concatenate per group</i> runs each raw subfolder on its own, and each sample is then unmixed with its own group's corrected spectra. Each group needs an unstained sample for its positivity thresholds: <i>Auto</i> takes the unstained cell sample whose scatter fits the group best. The events are drawn equally from the group's samples, in the gate.</li>
<li><b>Fluorophores kept fixed:</b> ticked fluorophores keep their spectra; Fix Unmix gathers no evidence with them as the source and does not re-measure them. Tick the tag fluorophores of a FlowCode experiment: their FRET error is removed by the FlowCode plugin, and correcting their spectra here would remove it a second time. The FlowCode plugin ticks them when it opens this page.</li>
<li><b>Fix Unmix</b> runs in two phases. Phase one fits, for every pair of fluorophores, the residual spillover from the clearly positive cells of one into the cells negative for the other, with a trust weight for each coefficient; phase two re-measures the spectrum of each fluorophore phase one changed from the cells it dominates, and adopts it only if it passes every check (fit, offset, angle, collinearity, conditioning, leakage). Every run starts from the base spectra. With <i>Smooth updates</i> (Advanced Settings, on by default) Fix My Unmix runs five rounds, each on a fresh half of the events unmixed with the current spectra, and moves the spectra only part of the way toward each round's result, so the noise of one subsample is averaged rather than adopted; it takes longer than a single pass.</li>
<li><b>Results:</b> the change of every corrected spectrum (heatmap and before/after traces), and a table of the fluorophores phase one changed with the angle adopted, the angle phase one implied and phase two's verdict. Untick <i>Active</i> to keep a fluorophore's base spectrum. What depends on the spectra follows the corrected ones: the AF envelopes are rebuilt, the spillover spread is recomputed from the controls, and spectral variants and positivity thresholds move with each correction (the noise model describes the detectors and is kept).</li>
<li><b>Use spectral corrections for the experiment's unmixing</b> (ticked after Fix Unmix): every corrected sample is unmixed with its group's corrected spectra by the experiment's method (per-cell AF, the noise model or spectral variants), in the main window, exports, Honeycluster, the gating plugins and Statistics. The corrections are cleared when the base spectra change.</li>
<li><b>Comparison:</b> the current sample with the base spectra and with its group's corrected spectra, both by the experiment's method, on the channel pair that changes most.</li>
</ol>
<ul>
<li><a href="https://github.com/DrCytometer/AutoSpectral">AutoSpectral package on Github</a></li>
</ul>
'''
