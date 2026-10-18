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
<h3>Import/Setup — define the gates the model will learn</h3>
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
25% of the parent (on the Hierarchy tab, in the shapes as drawn on the review
samples; until review samples are loaded, from the counts in the workspace);</li>
<li>the same 25% rule sets each axis of the other threshold gates on the
Hierarchy tab, from the shapes as drawn on the review samples: <i>mixture</i>
for an axis whose positive side holds more than 25% of the parent, <i>tail</i>
otherwise (<i>mixture</i> until review samples are loaded). A template's counts
are not used for this: they describe the group, not the samples. Once review
samples are loaded, the Hierarchy tab refines these proposals (see its help);</li>
<li>a single gate on a pair becomes a <i>1dsep</i> gate on its fluorescence
parameter when the other is scatter, or a <i>2dsep</i> gate when both are
fluorescence (<i>mixture</i>); on the Hierarchy tab, a 2dsep gate whose shape
does not cut one of its parameters converts to 1dsep on the other;</li>
<li>shapes drawn only on scatter parameters (e.g. FSC-A against SSC-A)
become <i>free</i> gates with the <i>cells</i> algorithm.</li>
</ul>
Gates that share a name in the import (e.g. <i>TIGIT+</i> drawn on many T cell
subsets) are named after the population they are drawn on, <i>TIGIT+ (Tem)</i>,
with the populations above it added only when needed, <i>TIGIT+ (CD4+ › Tem)</i>.
They keep their shared name as a link: converting one converts the others
still as drawn the same way (see the Hierarchy tab).
Which side of a fluorescence parameter a population is on comes from the drawn
coordinates: entirely above the channel's zero is the positive side, reaching
zero or below is the negative side. Sibling gates on two fluorescence parameters
take the quadrants that best match their shapes instead: a gate spanning both
sides of a threshold keeps both (e.g. <i>CD14-</i> drawn over the whole height
of the plot is the whole CD14- side; <i>NK</i> drawn around <i>non-NK</i> is
the L of the other three quadrants), and when one parameter separates none of
the gates the gate becomes a <i>1dsep</i> gate on the other. Compare each gate as drawn with its
trainable form on the Hierarchy tab, then use <b>Convert to trainable</b> (or
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
parameter that could not be matched. The search box and <b>Flagged or
unmatched</b> narrow the parameter list to the rows that need a decision.
Populations that use a dropped
parameter, and those below them, are not imported. <b>Import all — add</b>
adds the gates to the model; <b>Import all — replace</b> replaces its gates.
FlowJo gates are converted by the same rules as <b>Import from
Hierarchy</b>. The gate preview is hidden while this section is open. The
experiment's own gating is never changed. Review the imported gates on the
Hierarchy tab before training.</li>
<li><b>Add gate manually</b> creates a gate that is not on the main plots.</li>
</ul>

<h4>Finding gates</h4>
<p>The X and Y columns show each axis by its marker (the fluorophore when the
channel has no marker). Type in the search box to keep the gates whose name,
populations, markers, channels or algorithm match; the populations above a
match stay visible. <b>Show</b> narrows the list to <i>Unconverted</i> gates
(still applying their imported shapes), <i>Conflicting algorithms</i> (gates
sharing a name whose algorithms were set differently by hand) or <i>Flagged
conversions</i> (gates whose trainable form lands far from the imported shape;
see the Hierarchy tab) or <i>Overlapping gates</i> (populations defined twice;
see the Hierarchy tab). Flagged gates are marked ⚠ with the reason in the
tooltip.</p>

<h4>Gate types and algorithms</h4>
<ul>
<li><b>1dsep</b> — one threshold on one channel, the separation parameter
(marked <i>(sep)</i>), drawn against a view parameter (<i>(view)</i>) or as a
histogram. <b>1dsep separation axis</b> above the buttons sets which plot axis
carries the separation parameter of gates you add or split; each gate's Edit
dialog can change it.</li>
<li><b>2dsep</b> — a threshold on each of two channels (four quadrants). A
population is one quadrant or several joined: a whole side, two opposite
corners, or an L of three quadrants around another population.</li>
<li><b>singlets</b> — a band around the area-against-height diagonal.</li>
<li><b>free</b> — a shape on two scatter channels. <b>cells</b> is a
rectangle: its lower forward-scatter side cuts the debris (the density valley
left of the main forward-scatter peak, or where the density there falls below
5% of the peak); its other three sides are where the density of the events
above the cut falls below its <b>fall-off</b> on each axis (0.002 of its peak,
i.e. 0.2%, set per axis on the Hierarchy tab or in Advanced parameters), and
just below a pile of saturated events when there is one. A shape drawn on the
debris (below the cut) converts to a cells gate that keeps the debris side
instead.
<b>all</b> is the convex hull of the parent population.</li>
<li><b>replicate</b> (gate type) — reuses another gate's boundary on a
different parent.</li>
</ul>
<p>The <b>replicate</b> algorithm of a threshold gate axis takes its threshold
from elsewhere, chosen on the Hierarchy tab: another gate separating on the same
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
<p>Changing a gate between <i>1dsep</i> and <i>2dsep</i> keeps its population
names, so the gates below stay attached: going to 1dsep, each population takes
its side of the separation parameter (a population on both sides, or two on
the same side, cannot be kept: use <b>Split…</b>); going to 2dsep, each side
becomes that half of the plot. Renaming a population in the dialog renames it
for the gates below too. A change that would leave gates below without their
parent population is refused, naming them.</p>

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
<p>The hierarchy is coloured by training status: green trained, blue
adjusted by hand, amber not yet trained and red failed; the line below it
counts each. Its search box and <b>Show</b> filter work as on the Import/Setup
tab. Threshold lines on the tiles take the colour of their algorithm.</p>
<p>A gate whose parent population has too few events to calculate a
boundary is reported in the status line, and its children are skipped.</p>

<h4>Pause and Abort</h4>
<p><b>Pause</b> stops after the gate being calculated and keeps every gate
trained so far. <b>Calculate Gates</b> then asks whether to <b>Resume</b>,
keeping those gates and training the rest on the same pooled events, or to
<b>Overwrite</b>, training every gate again. A paused run can be resumed only
while the training samples, events per sample, gates and transforms are as
they were; otherwise it is trained again from the start. <b>Abort</b> stops
and keeps nothing from the run: the boundaries from before stay.</p>
<p>Training results, and a paused run's progress, are kept with the
experiment and restored when you come back to it. Boundaries are stored in
transformed units together with the builder's transforms (see the Transforms
tab).</p>
'''

BUILDER_EXPORT = '''
<h3>Export — save or load a gating model</h3>
<p><b>Save Model…</b> writes a <b>.agmodel</b> file: the gate hierarchy,
the trained boundaries, the builder's transform of every channel (those on
the Transforms tab), the training samples with their event counts and
autofluorescence assignments, and time-QC details. The Automated Gating
plugin applies it to new samples and shows them in its transforms by
default.</p>
<p><b>Load Model…</b> opens an existing model to inspect it or to continue
training. Its transforms become the builder's, so its boundaries are shown
where they were trained; channels without a stored transform (older model
files) take this experiment's.</p>
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
<li>On accepting, the model's boundaries are converted to the transforms on
the Transforms page (the model's own unless you changed them) where they
differ; the message area lists the channels converted.</li>
</ol>
<p>Long lists can be searched, and the alignment table can show only the
rows to check.</p>

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
<p>Double-click a gate to switch it between fixed and recalculated, or
right-click to set every selected gate.</p>
'''

AG_RUN = '''
<h3>Run — gate samples</h3>
<ol>
<li>Tick the samples to gate in the <b>Gate</b> column. Single-stain controls
and unstained samples are hidden unless <b>Show controls</b> is ticked. A
warning lists samples whose autofluorescence profile assignment none of the
training samples had, since their unmixing differs from the training
data.</li>
<li>Tick <b>Validate</b> for samples whose gates you want to see one by one.
Validation samples are gated too, whether or not <b>Gate</b> is ticked.</li>
<li>Click <b>Run Gating</b>. Each sample is unmixed with its own
autofluorescence profiles, restricted to its time-QC keep-mask when present,
transformed with the Transforms page's transforms, and gated through the
whole hierarchy using every event.</li>
</ol>
<p><b>Pause</b> stops at the next gate, keeps every sample gated so far and
saves them with the experiment; the sample being gated is dropped. <b>Run
Gating</b> then asks whether to <b>Resume</b> (gate only the ticked samples
without results) or <b>Restart</b> (discard the results so far). A paused run
can only be restarted once the model, the channel alignment, the gate
settings or the transforms have changed. <b>Cancel</b> stops the same way but
discards the run, keeping the results from before it.</p>

<h4>Inspecting the result</h4>
<p><b>Gated views</b> show the gates drawn on the population picked in the
hierarchy on the left (search it by name): a row for all gated samples pooled,
with the stored boundaries, and a row for each validation sample, with the
boundaries it was gated with. Plots use a random subsample of each sample
(<b>Display events per sample</b>); counts and statistics always use every
event.</p>
<p>The table lists each sample's event counts, autofluorescence profiles and
flags: gates that could not be calculated, recalculated gates that drifted,
and events removed by time QC.</p>
<p>Gated results and their plot events are saved with the experiment and
restored when you open it again. Each run's population counts and marker
medians are also written as CSV files to <i>Automated_Gating_Results</i> in
the experiment folder, named after the model with a timestamp.</p>
'''

AG_RESULTS = '''
<h3>Results — compare groups</h3>
<h4>Groups</h4>
<p>Add groups, give each an optional regular expression matched against the
sample name, and click <b>Assign by pattern</b>, or choose each sample's group
by hand. Covariates hold per-sample values: use one to <b>pair</b> samples
(e.g. donor) or to <b>adjust</b> for (e.g. age, sex, batch). Tick
<b>Numeric</b> under a covariate's title to enter it as a continuous number
(e.g. age, CRP) with a value for every sample; unticked covariates enter as
categories, so leave numeric codes such as batch 1, 2, 3 unticked. Every group
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

<h4>Figures</h4>
<p>Heatmaps, volcano plots and a population × marker summary are drawn for
the comparison chosen at the top. Large heatmaps open at half size (or
smaller); the size slider enlarges them. <b>Pop Out</b> opens a figure in its
own window, which can stay open while you work. In either place,
double-click a heatmap row or a volcano point to show that population on the
<b>Populations</b> page and tick it for the report; right-click it to show
it, open its comparison in a window, or include it in the report.</p>

<h4>Populations</h4>
<p>Lists the significant populations of the chosen comparison (marked ★), or
all of them, and can be searched by name. For the population picked: each
group's events pooled over its samples (up to 100,000 events per group) on
the population's gate, with the stored boundary and the pooled gated
frequency, beside a dot and violin plot of every sample's frequency (% of
parent) with the test's p-value and FDR. Tick <b>Include in report</b> to
add the comparison to the report.</p>

<h4>Export</h4>
<p><b>Generate Report</b> writes a folder under
<i>Automated_Gating_Reports</i> in the experiment folder: the population and
marker tables as CSV, a settings document, a PNG or CSV per ticked item
(sample tables, one plot per gate, each comparison's figures and results, and
the populations ticked for the report) and one PDF with all of them. The
sample PCA and predictive figures are included once they have been run.
<b>Cancel</b> on the progress window stops the report and removes its
half-written folder.</p>
<p>Every statistics run is written automatically to
<i>Automated_Gating_Results</i> as CSV files (each test's results, and the
per-sample values tested), named after the model with a timestamp; the
statistics are also kept with the experiment and restored when you open it
again. <b>Export Gating Results…</b> and <b>Export Statistics…</b> write the
same files to a folder of your choice.</p>

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

AG_ENRICHMENT = '''
<h3>Enrichment — immune states</h3>
<p>Flow Set Enrichment Analysis asks whether the features of a curated
<b>immune state</b> (for example CD8 T-cell exhaustion: PD-1, TIGIT, LAG3 and
CD39 up and TCF1 and CD127 down in CD8 T cells) shift together between two
groups, in the manner of gene set enrichment analysis [1]. It uses the samples,
groups, pairing, covariates and comparisons of the Results page.</p>

<h4>Samples</h4>
<p>A comparison is tested only with at least <b>30 samples</b> in its two
groups and <b>10 in each</b>. The p-values come from permuting the samples'
group labels; with fewer samples there are too few distinct labellings, and
the correlation between features cannot be estimated, so set-level results are
not reliable.</p>

<h4>Library</h4>
<p>The bundled library (<i>ag_fsea_library.csv</i>) holds draft immune states,
each a list of rows: a cell type with either its <b>abundance</b> (Abund) or a
<b>marker</b> read out in it, and the direction (UP or DOWN) the feature moves
when the state is gained. Competing phenotypes are separate states that may
share markers (PD-1 is in exhaustion and in acute activation); the data decide
which one moves. Each row carries a citation. The draft entries need review
before results are published. <b>Choose library…</b> uses your own CSV with the
same columns.</p>

<h4>Features and cell types</h4>
<ul>
<li><b>Positivity</b> — the fraction of a population positive for a marker
(gates such as PD1+ drawn on CD8 TEM), as log odds within the parent.</li>
<li><b>Subsets</b> — each phenotype population within its parent, as log odds.</li>
<li><b>Abundance</b> — each phenotype population within the <b>anchor</b>, so
samples with different numbers of events, debris or dead cells compare on the
same footing. The default is the model's anchor population: the one marked in
the Gating Model Builder, else the viable cells (the deepest gate using a
viability dye, recognised from the fluorophore database), else single cells,
else <i>Cells</i>.</li>
<li><b>Marker medians</b> — medians of phenotype populations only; medians
inside positivity gates repeat their parent.</li>
</ul>
<p>Exact duplicates and the two halves of a two-way split are counted once.
Panels that re-use one fluorochrome for markers of different lineages are
handled by reading the marker from the positivity gates drawn on that channel
nearest to the population; channels whose marker cannot be told apart are
tested but match no library row.</p>
<p><b>Check matching</b> shows how each population maps to a library cell type
(by name, accepted only when its lineage agrees with the population's
ancestors) and how each library row matched. Choose a cell type in the table
to correct a mapping. A library cell type matches its own population only, not
the populations below it. A state needs the minimum number of distinct matched
features (5 by default) to be tested.</p>

<h4>Statistics</h4>
<p>Each feature is ranked by sign(logFC) × −log10 p from the same moderated
linear model as the Results page [2] (no TREAT threshold). A state's enrichment
score (ES) is the weighted running sum over that ranking, with the DOWN members
negated: ES above zero means the state is <b>gained</b> in the second group,
below zero <b>lost</b>. The group labels are permuted and every feature refitted
to give the null distribution, which keeps the correlation between features;
with pairing or covariates the residuals of the model without the group terms
are permuted (within pairs) [3]. NES is the ES divided by the mean null ES of
the same sign; p is never zero [4]; FDR is Benjamini–Hochberg over the states
of a comparison and FWER the chance that the largest null NES of any state
reaches the NES. The FDR threshold is the Results page's. Read FDR: the
FWER estimate was liberal when calibrated on 15 samples (10% of null labellings
had a state below 0.05, against 5% expected).</p>

<h4>Results</h4>
<p>The dot plot shows every tested state in every comparison (colour NES, size
the share of members in the leading edge, ringed when FDR is below the
threshold). Select a state in the table for its running sum and its members;
the leading edge is the set of members that drive the score.</p>

<h4>References</h4>
<ol>
<li>Subramanian A et al. (2005) Gene set enrichment analysis. PNAS 102:15545–15550.</li>
<li>Smyth GK (2004) Stat Appl Genet Mol Biol 3: Article 3.</li>
<li>Freedman D, Lane D (1983) J Bus Econ Stat 1:292–298.</li>
<li>Phipson B, Smyth GK (2010) Permutation p-values should never be zero.
Stat Appl Genet Mol Biol 9: Article 39.</li>
</ol>
'''

BUILDER_REVIEW = '''
<h3>Hierarchy — compare imported gates with their trainable form</h3>
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
once trained. A 2dsep gate has an algorithm per axis. Each algorithm has a
colour, used for its name in the lists and for its threshold on the plots:
<span style="color:#8e44ad">mixture</span>,
<span style="color:#dc6e00">tail</span>,
<span style="color:#1e64dc">replicate</span>,
<span style="color:#008c8c">otsu</span>,
<span style="color:#c82878">bimodal</span>,
<span style="color:#8c5f28">kde_min</span> and
<span style="color:#1ea032">imported</span>. Choosing
<b>replicate</b> shows a source: another gate on the same channel, whose
threshold is copied, or a reference population, on which the threshold is
calculated; until a source gives a threshold, the imported boundary is shown
in its place, captioned. Once a source is chosen, a plot below the pair shows
the population the threshold is calculated on (the reference population, or
the parent population of the gate copied) with the gate drawn on it and each
population's share of it. With <b>tail</b> (also for a reference population),
<b>Tail fraction</b> beside the algorithm sets the share of the mirrored
negative population left above the threshold: smaller is stricter. It starts at
0.0001; the arrows step by factors of ten and any other value can be typed. The tail
algorithm uses the whole axis; the others ignore the lowest 1% of it.
<b>Update</b> calculates the shown gates again.</li>
<li>With review samples loaded, an imported gate's proposal comes from its
shapes as drawn on them (an algorithm you choose for an axis is kept; the
algorithm's tooltip gives the reason):
<ul>
<li><b>1dsep</b>, from its parent population on the separation channel:
<i>tail</i> when at least 40% of the parent lies below zero (centred on zero);
when at least 5% does, whichever of <i>tail</i>, <i>mixture</i> and a
three-component <i>mixture</i> cut between its lowest two or highest two
components reproduces the shapes as drawn best; when less than 5% does and
more than 40% of the parent is drawn positive, the parent's own events cannot
place the threshold, so it is <i>replicate</i>d from the <b>anchor
population</b> (with <i>mixture</i> when more than 25% of the anchor lies above
the drawn threshold, else <i>tail</i>) and a red line asks you to choose a
reference population (the source list offers the anchor and the marked
reference populations; <b>Keep</b> accepts it); otherwise the 25% rule. Gates
drawn on the anchor or above it are not replicated.</li>
<li><b>2dsep</b>: the combination of these candidates on the two axes that
reproduces the shapes as drawn best.</li>
<li>The <b>anchor population</b> is the population marked with <b>Use as
anchor population</b> in the tree's right-click menu; until one is marked it is
the viable cells (the population on the negative side of the deepest gate
using a viability dye, recognised from the fluorophore database), else single
cells, else <i>Cells</i>. It is marked <i>[anchor]</i> in the tree. Automated
Gating's enrichment analysis uses the same population as its default
anchor.</li>
<li>A <b>single shape</b> on two fluorescence parameters whose edges on one
parameter leave out at most 1% of the parent's events within its band on the
other converts to <b>1dsep</b> on the other parameter (e.g. a CD3 rectangle
spanning the whole CD20 range).</li>
<li>A 2dsep gate with one population on a whole side of one parameter and one
population in each quadrant of the other side is <b>split</b> by Convert into
sequential 1dsep gates: that parameter first (its other side named after the
marker, e.g. <i>CD38+</i>), then the other parameter on that side. The new
gates keep the shapes as drawn and are then converted by these rules.</li>
<li>A scatter shape drawn on the debris (most of its events below the debris
cut) converts to a <i>cells</i> gate that keeps the debris side. A
<i>cells</i> gate shows its fall-off per axis beside the algorithm.</li>
<li>A <i>mixture</i> axis shows its number of <b>components</b> and its
<b>cut</b>: which two components, ordered by mean, cross at the threshold
(lowest and highest, the lowest two, or the highest two; with three
components, e.g. a negative, a dim and a bright population).</li>
</ul>
A 2dsep threshold that separates populations on one side only (a joined
region) is drawn on that side only.</li>
<li><b>Convert to trainable</b> makes Train calculate the gate with the chosen
algorithm; <b>Use shapes as drawn</b> reverses it. <b>Convert all to
trainable</b> converts every gate still as drawn to its proposed form, first
checking every gate on the review samples (click it again to stop).
<b>Split…</b> replaces a 2dsep gate with sequential 1dsep gates. The red
<b>−</b> deletes a gate and the gates below it, after asking.</li>
<li>Gates that share an imported name (e.g. every <i>CD80+</i>) are kept on
one algorithm: converting one, or changing a converted one's algorithm,
converts the others still as drawn (or set this way before) to the same
algorithm and parameters. Once two of them have been set differently by
hand, the others are left alone and a message at the bottom says so; when
you set them alike again, the others follow. The <b>⇊</b> button beside a
converted gate applies its algorithm and settings to every gate of the same
name below the population it is drawn on, set by hand or not: set up
<i>CD25+</i> on one T cell subset, for example, then pass it to the subsets
below it. The <b>⇉</b> button beside it does the same for every gate of that
name on the same channels anywhere in the model.</li>
<li>Tick <b>Edit gates</b> to drag the shapes as drawn of gates that still
apply them (left; long polygons are thinned to 32 corners), and the
boundaries of converted gates (right). A boundary moved on the right is kept
as an adjustment until the gate is trained again. <b>Reset to import</b>
restores the shapes shown to their imported state.</li>
<li>Drag an axis to rescale it, as on the main plots: the lower half of a
biexponential axis changes its width basis, the upper half zooms. The gates
move with the axis, so they stay on the same events.</li>
<li>Right-click a population on any plot to <b>open</b> it: the page moves to
that population and shows the gates drawn on it.</li>
</ol>
<h4>Finding and checking gates</h4>
<p>The search box above the hierarchy keeps the populations whose gates match
(by name, marker, channel or algorithm), and <b>Show</b> narrows it to
unconverted gates, gates with conflicting algorithms or flagged conversions,
as on the Import/Setup tab. <b>Flag conversions</b> compares, on the review
samples, each gate's imported shape with its trainable form and flags it when
a threshold moves more than the set distance (in axis units: the axis spans
about 0 to 1) or a population's share of its parent changes by more than the
set number of percentage points. Flagged gates are marked ⚠ in the hierarchy,
with the reason in the tooltip.</p>
<p><b>Flag overlaps</b> looks for populations defined twice: two populations,
neither below the other, that share at least the set share of their events on
the review samples (events in both as a share of events in either). Only
populations with gates drawn on them are compared, and a population that is
nearly the same as one above it (90% shared) counts as that one, so a branch
defined twice is reported once rather than for every marker gate below it.
Flagged gates are marked ⚠ in the hierarchy and listed by <b>Show: Overlapping
gates</b>; the Import/Setup table marks them ⚠ Overlap.</p>
<p>The colour of a population in the hierarchy is the status of the gate that
makes it, as in the Import/Setup table:
<span style="color:#c0392b">red</span> unconverted (still applying its shapes
as drawn), <span style="color:#946100">amber</span> converted but not trained,
<span style="color:#0a7a23">green</span> trained.</p>
<h4>Hierarchy menu</h4>
<p>Right-click a population for <b>Edit gate…</b> (the gate editor of the
Import/Setup tab: name, type, channels, parent and populations) and
<b>Advanced parameters…</b> of the gate that makes it. Right-click the
hierarchy (Ctrl- or Shift-click selects several populations) to <b>Copy gate</b> or <b>Copy gating branch</b> (the gate and
every gate below it) and <b>Paste</b> them onto other populations; copies are
named after the population they are pasted on and follow their same-named
gates. <b>Delete gate…</b> and <b>Delete population…</b> remove the selection
with what is below it. <b>Use as reference population</b> marks a population
as a source for replicated thresholds.</p>
<p>Changing a gate clears the training of that gate and of the gates below
it; train again to apply the change.</p>
'''

AG_TRANSFORMS = '''
<h3>Transforms — the axis scales gating uses</h3>
<p>Threshold gates are calculated on transformed (logicle) values, so where a
gate falls depends on the transform. By default the transforms are the
model's own, the ones it was trained in, so a recalculated gate behaves as it
did in training; channels the model has none for use this experiment's.</p>
<ul>
<li>Pick a <b>population</b> of the experiment's gating hierarchy (the gates of
the Unmixed Data tab; Singlets or Cells to start with). Its events are pooled
from a few of the samples ticked on the Run page, each unmixed and gated as
the main window does: about 30,000 events of the population in all, an equal
share from each sample, read without loading the whole file.</li>
<li>Select a channel on the right to see its histogram and biplots. Drag an
axis to adjust it as on the main plots: the lower half of a logicle axis
changes its width basis W, the upper half the displayed range.</li>
<li><b>Auto-transform</b> sets W for every logicle channel from the events
shown (Parks, Roederer and Moore 2006).</li>
<li><b>Revert to the model's transforms</b> undoes your changes.</li>
</ul>
<p>The model's boundaries move with a transform, so they stay on the same
events. Gating results record the transforms they were made in; changing the
transforms means a paused run is restarted rather than resumed.</p>
'''

BUILDER_TRANSFORMS = '''
<h3>Transforms — the axis scales the model is trained in</h3>
<p>Threshold algorithms work on transformed (logicle) values, so the transform
of a channel changes where its gates are placed. The model is trained in
these transforms and saves them; Automated Gating uses them by default.
They start as this experiment's display transforms and are the builder's
own: changing them here does not change the main window's plots.</p>
<ul>
<li>Pick a <b>population</b> of the experiment's gating hierarchy (the gates of
the Unmixed Data tab; Singlets or Cells to start with). Its events are pooled
from the training samples (or the sample on show), each unmixed and gated as
the main window does: about 30,000 events of the population in all, an equal
share from each sample, read without loading the whole file.</li>
<li>Select a channel on the right to see its histogram and biplots. Drag an
axis to adjust it: the lower half of a logicle axis changes its width basis
W, the upper half the displayed range. Axes dragged on the Import/Setup and
Hierarchy plots change the same transforms.</li>
<li><b>Auto-transform</b> sets W for every logicle channel from the events
shown (Parks, Roederer and Moore 2006).</li>
<li><b>Reset to the experiment's transforms</b> returns to the main window's
transforms.</li>
</ul>
<p>Imported shapes and trained boundaries move with a transform, so they stay
on the same events. Retrain after changing transforms to calculate the gates
on the new scales.</p>
'''
