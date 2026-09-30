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
selects.</p>

<h4>Getting gates in</h4>
<ul>
<li><b>Import from Hierarchy</b> copies the gates you drew on the main
cytometry plots. Quadrant gates become <i>2dsep</i> gates with one
population per quadrant; a forward-scatter area against height (or width)
polygon becomes a <i>singlets</i> gate.</li>
<li><b>Refresh from Hierarchy</b> adds gates drawn since the last import and
keeps any parameters you have changed. Gates that no longer exist on the
main plots are marked stale rather than deleted.</li>
<li><b>Add gate manually</b> creates a gate that is not on the main plots.</li>
</ul>

<h4>Gate types and algorithms</h4>
<ul>
<li><b>1dsep</b> — one threshold on one channel (negative / positive).</li>
<li><b>2dsep</b> — a threshold on each of two channels (four quadrants).</li>
<li><b>singlets</b> — a band around the area-against-height diagonal.</li>
<li><b>free</b> — a polygon around the parent population (convex hull or
ellipse).</li>
<li><b>replicate</b> — reuses another gate's boundary on a different
parent.</li>
</ul>
<p>Threshold algorithms: <b>tail</b> (the upper tail of a symmetric negative
population), <b>kde_min</b> and <b>bimodal</b> (the density minimum between
two peaks), <b>mixture</b> (where two Gaussian components cross), and
<b>otsu</b> (the split that best separates two classes). After an import,
each threshold gate is given a suggested algorithm: <i>mixture</i> when its
channel looks bimodal (a two-component Gaussian mixture fits clearly better
than one), otherwise <i>tail</i>.</p>
<p>Threshold populations are open-ended: every event beyond the threshold
belongs to the population, however far outside the plotted range.</p>

<h4>Editing</h4>
<p>Double-click a gate, or right-click it, to change its type, channels,
populations or algorithm. <b>Advanced…</b> exposes the algorithm
parameters: tail fraction, number of mixture components, quantile trimming,
the part of the axis used for the calculation and, for singlets gates, the
band width in standard deviations.</p>
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
<p>Each gate appears as a tile showing its parent population and boundary.
Click a tile to open it full size, move the threshold (1D gates) or the
threshold lines (2D gates), and accept to keep the adjustment. Adjusted
gates are marked in the model.</p>
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
own parent population, with the model's parameters. If the result moves
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

<h4>Statistics</h4>
<ul>
<li><b>Population frequencies</b> — each population's share of its parent (or
its stats parent) as log odds, with half an event added to the population and
to the rest of the parent so 0% and 100% stay finite; moderated linear model.
Effects are log2 odds ratios.</li>
<li><b>Population counts</b> — negative-binomial model of event counts with
the parent count as offset. Effects are log2 fold changes.</li>
<li><b>Marker medians</b> — each population's median of each marker (and of
AF Abundance, the per-cell autofluorescence level), in transformed units. Populations are screened first and markers are tested
only within populations that pass (stage-wise FDR). By default a population
is not tested on the markers of its own and its ancestors' gates, which
differ by construction.</li>
</ul>
<p>With <b>TREAT</b> on, p-values test whether the effect exceeds the
threshold; with it off, they test for any difference and the threshold only
filters. FDR is pooled over all comparisons or taken within each. Changing a
threshold updates significance without refitting.</p>

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
offered only for samples gated in this session.</p>
'''
