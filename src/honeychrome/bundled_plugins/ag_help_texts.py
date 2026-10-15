"""
ag_help_texts.py — per-tab help text for the automated gating plugins
======================================================================
Companion module to ``gating_model_builder_tab.py`` and
``automated_gating_tab.py`` (the file name does not end in ``_tab.py``, so
the plugin loader does not treat it as a tab).

One HTML string per tab, shown and hidden with the core app's
HelpToggleWidget (honeychrome.view_components.help_toggle_widget).
"""

BUILDER_HIERARCHY = '''
<h3>Hierarchy — define the gates the model will learn</h3>
<p>A gating model is a hierarchy of gates. Each gate is calculated from its
parent population only, so a child gate adapts to whatever its parent
selects. This is automation for targeted analysis of well-characterised
populations in a defined panel, as opposed to discovery by clustering
[1, 3]; benchmarks of manual gating, clustering and automated gating are
reviewed in [2].</p>

<h4>Getting gates in</h4>
<ul>
<li><b>Import from Hierarchy</b> copies the gates you drew on the main
cytometry plots. Each gate keeps its shapes exactly as drawn (algorithm
<i>imported</i>, applied as drawn) and the names you gave the populations,
and records the trainable gate it converts to:
<ul>
<li>the forward-scatter area against height (or width) gate becomes a
<i>singlets</i> gate and moves to the top of the hierarchy; gates drawn above
it are placed directly below it; a side-scatter area against height gate is a
<i>singlets</i> gate too, left where it is;</li>
<li>ellipses become 10-vertex polygons of the same area; no gate is ever an
ellipse;</li>
<li>sibling gates drawn on the same pair of parameters become one gate with a
population each: <i>2dsep</i>, or <i>1dsep</i> on the fluorescence parameter
when the other is scatter (proposed algorithm <i>mixture</i>);</li>
<li>a histogram gate becomes a <i>1dsep</i> gate with a histogram view:
<i>tail</i>, or <i>mixture</i> when its positive population holds more than
25% of the parent (from the counts in the workspace, or on the Review tab from
the review samples when the workspace has none);</li>
<li>a single gate on a pair becomes a <i>1dsep</i> gate on its fluorescence
parameter when the other is scatter, or a <i>2dsep</i> gate when both are
fluorescence (<i>mixture</i>);</li>
<li>shapes drawn only on scatter parameters (e.g. FSC-A against SSC-A)
become <i>free</i> gates with the <i>cells</i> algorithm.</li>
</ul>
Gates that share a name in the import (e.g. <i>TIGIT+</i> drawn on many T cell
subsets) are named after the population they are drawn on, <i>TIGIT+ (Tem)</i>,
with the populations above it added only when needed, <i>TIGIT+ (CD4+ › Tem)</i>.
They keep their shared name as a link: converting one converts the others
still as drawn the same way (see the Review tab).
Which side of a fluorescence parameter a population is on comes from the drawn
coordinates: entirely above the channel's zero is the positive side, reaching
zero or below is the negative side. Compare each gate as drawn with its
trainable form on the Review tab, then use <b>Convert to trainable</b> (or
<b>Convert all to trainable</b>) so Train calculates it from the training
samples. <b>Split…</b> replaces
a 2dsep gate with sequential 1dsep gates; you choose the order, algorithms and
view parameters. <b>Keep imported boundary</b> in the right-click menu
reverses a conversion.</li>
<li><b>Refresh from Hierarchy</b> adds gates drawn since the last import and
keeps any parameters you have changed. Gates that no longer exist on the
main plots are marked stale rather than deleted.</li>
<li><b>Import FlowJo…</b> opens a FlowJo workspace (.wsp) or template (.wspt).
A section expands under the buttons: choose the group or sample whose gating
to import, check how each FlowJo parameter maps to an experiment channel
(matched by fluorophore name where possible), and assign or drop any
parameter that could not be matched. Populations that use a dropped
parameter, and those below them, are not imported. <b>Import all — add</b>
adds the gates to the model; <b>Import all — replace</b> replaces its gates.
FlowJo gates are converted by the same rules as <b>Import from
Hierarchy</b>. The gate preview is hidden while this section is open. The
experiment's own gating is never changed. Review the imported gates on the
Review tab before training.</li>
<li><b>Add gate manually</b> creates a gate that is not on the main plots.</li>
</ul>

<h4>Gate types and algorithms</h4>
<ul>
<li><b>1dsep</b> — one threshold on one channel, the separation parameter
(marked <i>(sep)</i>), drawn against a view parameter (<i>(view)</i>) or as a
histogram. <b>1dsep separation axis</b> above the buttons sets which plot axis
carries the separation parameter of gates you add or split; each gate's Edit
dialog can change it.</li>
<li><b>2dsep</b> — a threshold on each of two channels (four quadrants).</li>
<li><b>singlets</b> — a band around the area-against-height diagonal.</li>
<li><b>free</b> — a shape on two scatter channels. <b>cells</b> cuts the
debris: the density valley left of the main forward-scatter peak (or where
the density there falls below 5% of the peak), keeping everything above it;
the other sides stay open except just below a pile of saturated events.
<b>all</b> is the convex hull of the parent population.</li>
<li><b>replicate</b> (gate type) — reuses another gate's boundary on a
different parent.</li>
</ul>
<p>The <b>replicate</b> algorithm of a threshold gate axis takes its threshold
from elsewhere, chosen on the Review tab: another gate separating on the same
channel (its threshold is copied), or a <b>reference population</b>. A
reference population is one whose biology is known and whose events are
plentiful: the threshold is calculated there, by default with <i>tail</i>, and
transferred to a population that is rare, noisy or has no clear boundary. For
example, CD3+ T cells, which do not express CD19, set the CD19 threshold, and
Foxp3− CD4 T cells set the CD25 threshold applied to Tregs.</p>
<p>Threshold algorithms: <b>tail</b> (the upper tail of a symmetric negative
population; negative populations are symmetric on a logicle or biexponential
axis [6]), <b>kde_min</b> and <b>bimodal</b> (the density minimum between
two peaks), <b>mixture</b> (where two Gaussian components cross), and
<b>otsu</b> (the split that best separates two classes [7]). After an import,
each threshold gate is given a suggested algorithm: <i>mixture</i> when its
channel looks bimodal (a two-component Gaussian mixture fits clearly better
than one), otherwise <i>tail</i>.</p>
<p>Threshold populations are open-ended: every event beyond the threshold
belongs to the population, however far outside the plotted range. Shapes are
tested where the plots draw the events: an event beyond an axis sits on that
axis edge, so a shape drawn up to the edge includes the events past it.</p>
<p>Thresholds are calculated once, on the pooled training samples, rather than
recalculated for each sample as OpenCyto [4] and flowDensity [5] do; the Setup
tab offers per-sample recalculation as an option.</p>

<h4>Editing</h4>
<p>Double-click a gate, or right-click it, to change its type, channels,
populations or algorithm. <b>Advanced…</b> exposes the algorithm
parameters: tail fraction, number of mixture components, quantile trimming,
the part of the axis used for the calculation and, for singlets gates, the
band width in standard deviations.</p>

<h4>Further Reading and Background</h4>
<p>Documentation:</p>
<ul>
<li><a href="https://github.com/DrCytometer/AutoGating">AutoGating</a> — R package implementing the gating algorithms (GitHub)</li>
<li><a href="https://www.colibri-cytometry.com/references">Useful Publications</a> — curated reference library, Colibri Cytometry</li>
</ul>
<p>References:<br/>
[1]<a href="https://doi.org/10.1002/cyto.a.24320">Cheung et al. 2021, Cytometry A (current trends in automated analysis software)</a><br/>
[2]<a href="https://doi.org/10.1093/bib/bbae633">Liu et al. 2025, Briefings in Bioinformatics (evaluation of gating methods)</a><br/>
[3]<a href="https://doi.org/10.1038/nri.2016.56">Saeys, Van Gassen and Lambrecht 2016, Nature Reviews Immunology (computational flow cytometry)</a><br/>
[4]<a href="https://doi.org/10.1371/journal.pcbi.1003806">Finak et al. 2014, PLOS Computational Biology (OpenCyto)</a><br/>
[5]<a href="https://doi.org/10.1093/bioinformatics/btu677">Malek et al. 2015, Bioinformatics (flowDensity)</a><br/>
[6]<a href="https://doi.org/10.1002/cyto.a.20258">Parks, Roederer and Moore 2006, Cytometry A (Logicle display)</a><br/>
[7]<a href="https://doi.org/10.1109/TSMC.1979.4310076">Otsu 1979, IEEE Trans Syst Man Cybern (threshold selection)</a><br/>
</p>
'''

BUILDER_TRAIN = '''
<h3>Train — calculate every gate from training samples</h3>
<ol>
<li>Tick the <b>training samples</b>. Single-stain controls and unstained
samples are hidden unless <b>Show controls</b> is ticked.</li>
<li>Set <b>Events per sample</b>. Each sample contributes a reproducible
random subsample of this size, so a large file cannot dominate the pooled
thresholds. <i>All</i> uses every event.</li>
<li>Click <b>Calculate Gates</b>. Samples are unmixed off the main thread,
each with its own autofluorescence profile assignment, exactly as the main
window unmixes them; time-QC keep-masks are applied when present. Gates are
then calculated in hierarchy order on the pooled, transformed events.</li>
</ol>

<h4>Checking and adjusting</h4>
<p>Each gate appears as a tile showing its parent population and boundary,
beside the gating hierarchy. Tiles are shown for the gates drawn on the
populations open in the tree: the first three levels to begin with. Expand a
population to show the gates drawn on it, or select it to jump to them.
Click a tile's <b>Detail…</b> to open it full size, move the threshold (1D
gates) or the threshold lines (2D gates), and accept to keep the adjustment.
Adjusted gates are marked in the model.</p>
<p>A gate whose parent population has too few events to calculate a
boundary is reported in the status line, and its children are skipped.</p>
<p>Training results are kept with the experiment and restored when you come
back to it. Boundaries are stored in transformed units together with each
channel's transform, so they are converted if a transform has changed.</p>
'''

BUILDER_EXPORT = '''
<h3>Export — save or load a gating model</h3>
<p><b>Save Model…</b> writes a <b>.agmodel</b> file: the gate hierarchy,
the trained boundaries, the transform of every gated channel, the training
samples with their event counts and autofluorescence assignments, and
time-QC details. The Automated Gating plugin applies it to new samples.</p>
<p><b>Load Model…</b> opens an existing model to inspect it or to continue
training. When the model's transforms differ from this experiment's, its
boundaries are converted to the current transforms; channels without stored
transforms (older model files) are loaded unchanged and reported.</p>
'''

AG_SETUP = '''
<h3>Setup — load a model and match it to this experiment</h3>
<ol>
<li><b>Load Model…</b> opens a <b>.agmodel</b> file from the Gating Model
Builder. The summary shows where and how it was trained, including the
autofluorescence profiles of the training samples.</li>
<li><b>Channel alignment.</b> Every channel a gate uses is matched to this
experiment by antigen (synonyms such as CD197 and CCR7 are recognised), so a
panel that moved an antigen to another detector still lines up. Green rows
matched by antigen. Amber rows need checking: the same detector carries a
different antigen, the match is only by similar name, or the antigen is on
more than one detector. Red rows have no match; choose a channel. Then click
<b>Accept Alignment</b>.</li>
<li>On accepting, the model's boundaries are converted to this experiment's
axis transforms where they differ; the message area lists the channels
converted.</li>
</ol>

<h4>How each gate is applied</h4>
<ul>
<li><b>Fixed</b> (default) — the stored boundary is applied to every sample
as it is.</li>
<li><b>Recalculate</b> — the gate's algorithm is run again on each sample's
own parent population, with the model's parameters (the per-sample approach
of <a href="https://doi.org/10.1371/journal.pcbi.1003806">OpenCyto</a> and
<a href="https://doi.org/10.1093/bioinformatics/btu677">flowDensity</a>). If the result moves
further than the <b>drift limit</b> from the stored boundary (in transformed
axis units, on an axis spanning about 0 to 1) the sample is flagged, and
either keeps its recalculated boundary or uses the stored one, as chosen.
If the calculation fails (too few events), the stored boundary is used and
the sample is flagged.</li>
</ul>
'''

AG_RUN = '''
<h3>Run — gate samples</h3>
<ol>
<li>Tick the samples to gate in the <b>Gate</b> column. Single-stain controls
and unstained samples are hidden unless <b>Show controls</b> is ticked. A
warning lists samples whose autofluorescence profile assignment none of the
training samples had, since their unmixing differs from the training
data.</li>
<li>Tick <b>Validate</b> for samples you want to inspect one at a time.</li>
<li>Click <b>Run Gating</b>. Each sample is unmixed with its own
autofluorescence profiles, restricted to its time-QC keep-mask when present,
transformed, and gated through the whole hierarchy using every event.</li>
</ol>
<p><b>Pause</b> stops at the next gate, keeps every sample gated so far and
saves them with the experiment; the sample being gated is dropped. <b>Run
Gating</b> then asks whether to <b>Resume</b> (gate only the ticked samples
without results) or <b>Restart</b> (discard the results so far). A paused run
can only be restarted once the model, the channel alignment or the gate
settings have changed. <b>Cancel</b> stops the same way but discards the run,
keeping the results from before it.</p>

<h4>Inspecting the result</h4>
<p><b>View</b> shows every gate either for all gated samples pooled (with the
stored boundaries) or for one validation sample (with the boundary that
sample was gated with). Plots use a random subsample of each sample
(<b>Display events per sample</b>); counts and statistics always use every
event.</p>
<p>The table lists each sample's event counts, autofluorescence profiles and
flags: gates that could not be calculated, recalculated gates that drifted,
and events removed by time QC.</p>
<p>Gated results are saved with the experiment. The display events are not,
so plots need gating to be run again after a restart.</p>
'''

AG_RESULTS = '''
<h3>Results — compare groups</h3>
<h4>Groups</h4>
<p>Add groups, give each an optional regular expression matched against the
sample name, and click <b>Assign by pattern</b>, or choose each sample's group
by hand. Covariates hold per-sample values: use one to <b>pair</b> samples
(e.g. donor) or to <b>adjust</b> for (e.g. age, sex, batch). Every group
compared needs at least 3 samples.</p>
<p><b>Import table…</b> reads groups and covariates from a CSV or tab-separated
file with a <i>sample</i> column (the file name, with or without its
extension), an optional <i>group</i> column and one column per covariate.
New groups are added. <b>Export table…</b> writes the same format.</p>

<h4>Statistics</h4>
<p>The same framework has been applied in systems-immunology studies of flow
cytometry cohorts [6, 7].</p>
<ul>
<li><b>Population frequencies</b> — each population's share of its parent (or
its stats parent) as log odds, with half an event added to the population and
to the rest of the parent so 0% and 100% stay finite; moderated linear model
[1, 2]. Effects are log2 odds ratios.</li>
<li><b>Population counts</b> — negative-binomial model of event counts [5] with
the parent count as offset. Effects are log2 fold changes.</li>
<li><b>Marker medians</b> — each population's median of each marker (and of
AF Abundance, the per-cell autofluorescence level), in transformed units. Populations are screened first and markers are tested
only within populations that pass (stage-wise FDR [4]). By default a population
is not tested on the markers of its own and its ancestors' gates, which
differ by construction.</li>
</ul>
<p>With <b>TREAT</b> [3] on, p-values test whether the effect exceeds the
threshold; with it off, they test for any difference and the threshold only
filters. FDR is pooled over all comparisons or taken within each. Changing a
threshold updates significance without refitting.</p>

<h4>Sample PCA and prediction</h4>
<p>Both use the per-sample values of the last statistics run. <b>Sample
PCA</b> places every sample on the first two principal components of the
chosen tables (each column scaled to unit variance), coloured by group, with
the features that load most strongly as arrows. The <b>predictive
analysis</b> asks which features best separate two groups, and how well:
lasso logistic regression and a random forest, assessed by repeated nested
cross-validation, give the cross-validated ROC curve with its AUC (mean ± SD
over repeats) and balanced accuracy, the most informative features, and the
AUC reached with the top 1, 2, 3 … features [6, 7]. It needs at least 10
samples per group; 30 or more is recommended.</p>

<h4>Figures and populations</h4>
<p>Heatmaps, volcano plots and a population × marker summary are drawn for
the comparison chosen at the top. Click a volcano point to open its
population in <b>Populations</b>, which shows that gate for every sample
gated in this session, with the boundary the sample was gated with.</p>

<h4>Export</h4>
<p><b>Generate Report</b> writes a folder under
<i>Automated_Gating_Reports</i> in the experiment folder: the population and
marker tables as CSV, a settings document, a PNG or CSV per ticked item
(sample tables, one plot per gate, and each comparison's figures and results)
and one PDF with all of them. Gate plots need the display events, so they are
offered only for samples gated in this session. The sample PCA and
predictive figures are included once they have been run. <b>Cancel</b> on the
progress window stops the report and removes its half-written folder.</p>

<h4>Further Reading and Background</h4>
<p>References:<br/>
[1]<a href="https://doi.org/10.1093/nar/gkv007">Ritchie et al. 2015, Nucleic Acids Research (limma software)</a><br/>
[2]<a href="https://doi.org/10.2202/1544-6115.1027">Smyth 2004, Stat Appl Genet Mol Biol (empirical Bayes moderation)</a><br/>
[3]<a href="https://doi.org/10.1093/bioinformatics/btp053">McCarthy and Smyth 2009, Bioinformatics (TREAT)</a><br/>
[4]<a href="https://doi.org/10.1111/rssb.12028">Benjamini and Bogomolov 2014, J R Stat Soc B (selective inference on families of hypotheses)</a><br/>
[5]<a href="https://doi.org/10.1093/bioinformatics/btp616">Robinson, McCarthy and Smyth 2009, Bioinformatics (edgeR)</a><br/>
[6]<a href="https://doi.org/10.1002/alz.70952">Humblet-Baron et al. 2025, Alzheimer's &amp; Dementia (systems immunology of Alzheimer's disease)</a><br/>
[7]<a href="https://doi.org/10.1038/s41467-026-70895-1">Veiga et al. 2026, Nature Communications (autoinflammation of unknown origin)</a><br/>
</p>
'''

BUILDER_REVIEW = '''
<h3>Review — compare imported gates with their trainable form</h3>
<p>Imported gates (from the experiment's hierarchy or a FlowJo workspace)
keep their shapes exactly as drawn until they are converted to trainable.
FlowJo coordinates are raw instrument units converted with this experiment's
transforms, and unmixing here can differ from the unmixing the gates were
drawn on, so look at them before training.</p>
<ol>
<li>Choose a <b>population</b> in the gating hierarchy on the left or in the
population list above it; the two stay in step. A population with no gates
below it shows the plot it is gated on.</li>
<li>Choose the <b>review samples</b>. Tick several to pool them, each giving the
same number of events. When the imported counts came from a sample in this
experiment it is chosen for you, and with that one sample each population
label shows its fraction of parent next to the fraction recorded at import,
in red when they differ by 2 percentage points or by 5% of the imported
fraction.</li>
<li>Each plot's gates <b>as drawn</b> are on the left; on the right, level
with it, each gate's <b>trainable form</b>: calculated on the review samples'
parent population with the algorithm chosen above it, or its trained boundary
once trained. A 2dsep gate has an algorithm per axis. Choosing
<b>replicate</b> shows a source: another gate on the same channel, whose
threshold is copied, or a reference population, on which the threshold is
calculated. <b>Update</b> calculates the shown gates again.</li>
<li><b>Convert to trainable</b> makes Train calculate the gate with the chosen
algorithm; <b>Use shapes as drawn</b> reverses it. <b>Convert all to
trainable</b> converts every gate still as drawn to its proposed algorithm.
<b>Split…</b> replaces a 2dsep gate with sequential 1dsep gates. The red
<b>−</b> deletes a gate and the gates below it, after asking.</li>
<li>Gates that share an imported name (e.g. every <i>CD80+</i>) are kept on
one algorithm: converting one, or changing a converted one's algorithm,
converts the others still as drawn (or set this way before) to the same
algorithm and parameters. Once two of them have been set differently by
hand, the others are left alone and a message at the bottom says so; when
you set them alike again, the others follow.</li>
<li>Tick <b>Edit gates</b> to drag the shapes as drawn of gates that still
apply them (left; long polygons are thinned to 32 corners), and the
boundaries of converted gates (right). A boundary moved on the right is kept
as an adjustment until the gate is trained again. <b>Reset to import</b>
restores the shapes shown to their imported state.</li>
<li>Drag an axis to rescale it, as on the main plots: the lower half of a
biexponential axis changes its width basis, the upper half zooms. The gates
move with the axis, so they stay on the same events.</li>
</ol>
<h4>Hierarchy menu</h4>
<p>Right-click the hierarchy (Ctrl- or Shift-click selects several
populations) to <b>Copy gate</b> or <b>Copy gating branch</b> (the gate and
every gate below it) and <b>Paste</b> them onto other populations; copies are
named after the population they are pasted on and follow their same-named
gates. <b>Delete gate…</b> and <b>Delete population…</b> remove the selection
with what is below it. <b>Use as reference population</b> marks a population
as a source for replicated thresholds.</p>
<p>Changing a gate clears the training of that gate and of the gates below
it; train again to apply the change.</p>
'''
