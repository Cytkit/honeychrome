"""
drc_help_texts.py — Per-tab help text for the DR / Clustering / Statistics plugin
===================================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does NOT
end in ``_tab.py``, so it is not picked up as a separate plugin tab — same
convention as drc_report.py, drc_stats.py, etc.).

One HTML help string per tab, shown/hidden via the core app's
HelpToggleWidget (honeychrome.view_components.help_toggle_widget), the same
mechanism used for the Spectral Process help and the AutoSpectral AF help.
"""

transforms_tab_help_text = '''
<h3>Transforms — read-only preview of Logicle parameters</h3>
<p>This tab lets you preview and locally fine-tune the per-channel transform
(Logicle/biexponential) used to load event data for training and plotting
elsewhere in this plugin. It reads the transforms currently configured in
the main Honeychrome experiment, but any adjustments you make here are
<b>local previews only</b> — they are saved to this plugin's own settings,
not written back to the experiment. To change the transforms used
elsewhere in Honeychrome, use the Transforms panel in the main
application.</p>

<h4>Before you start</h4>
<ol>
<li>Select one or more <b>gates</b> from the gate tree. Their events are
unioned to build the preview pool (and, later, the training pool used for
DR and clustering).</li>
<li>Click <b>Reload from experiment</b> if you've changed transforms in
the main Honeychrome window and want this tab to pick them up.</li>
</ol>

<h4>Working with the preview</h4>
<ul>
<li>Select a channel from the list on the left to see its 1-D histogram
and current transform type.</li>
<li><b>Drag the histogram's x-axis</b> to adjust the transform, exactly
as in the main Honeychrome cytometry plots (lower half of the axis
adjusts Logicle W; upper half zooms the display range).</li>
<li>Use <b>Auto-transform preview (all channels)</b> to run an automated
W estimate across every channel in one step, pooling all selected
training samples for a more stable estimate than a single sample would
give.</li>
<li>Add or remove <b>biplots</b> below the 1-D histogram to inspect two
channels together; drag either axis to adjust that channel's transform.</li>
<li>Use <b>Save CSV</b> / <b>Load CSV</b> to export or re-import the
current set of local transform overrides, e.g. to share a fine-tuned set
of parameters between sessions.</li>
</ul>

<p>These same local transform parameters are what the Configuration tab
uses when loading training data for DR and clustering, so it is worth
checking them here before training a model.</p>
'''

configuration_tab_help_text = '''
<h3>Configuration — data selection, dimensionality reduction, and clustering</h3>
<p>This tab controls what data goes into training, and lets you train
dimensionality-reduction (DR) embeddings and run clustering on the
result. Runs produced here are archived and become available for
annotation, statistics, and plotting in the other tabs.</p>

<h4>1. Data Selection</h4>
<ul>
<li><b>Gate(s):</b> check one or more gates; their events are unioned
before loading. This selection is shared with the Transforms tab.</li>
<li><b>Parameters to use for DR and clustering:</b> check the channels
to include. Scatter channels are pre-unchecked by default, and
<tt>Time</tt>/<tt>event_id</tt> are always excluded.</li>
<li><b>Training events per sample</b> and <b>Training samples:</b>
choose how many events to pool per sample and which samples contribute
to training. Single-stain controls are automatically excluded from the
picker.</li>
</ul>

<h4>2. Dimensionality Reduction</h4>
<p>Four algorithms are available, each trading off preservation of local
structure (relationships within a "blob" of similar cells) against
global structure (distances between blobs):</p>
<ul>
<li><b>UMAP</b> — fast, widely used, strong local-structure
preservation.</li>
<li><b>tSNE</b> (via openTSNE) — excellent local-structure preservation;
distances across white space between islands are not meaningful.</li>
<li><b>PaCMAP</b> — aims to balance local and global structure in a
single embedding.</li>
<li><b>PHATE</b> — designed for developmental/continuous data;
represents transitions between related populations as branches or
connections rather than gaps.</li>
</ul>
<p>None of these algorithms preserves inter-cluster distances in an
absolute sense — they are useful for visual grouping and getting a feel
for the data, not for treating on-plot distances as a quantitative
metric. See the Further Reading links below for a practical comparison
of how these algorithms behave on real flow cytometry data.</p>
<p>Click <b>Train DR Model</b> to fit the selected algorithm on the
training pool, then <b>Apply to All Samples</b> to project every sample
(including ones not used for training) through the trained model.</p>

<h4>3. Clustering</h4>
<ul>
<li><b>FlowSOM</b> — self-organising map followed by metaclustering; set
the SOM grid size and number of metaclusters. Honeychrome uses a batched
FlowSOM training for fast processing of even millions of events.
tab.</li>
<li><b>Leiden</b> — graph-based community detection; reuses UMAP's
neighbour graph if one is available. Leiden was designed specifically to
fix a defect in the Louvain-based algorithms behind classic
Phenograph-style clustering, which can produce badly-connected or even
disconnected "communities" and can be sensitive to how many cells and
which kNN implementation are used — see the Phenograph link below for
examples of this in practice. Leiden guarantees well-connected
communities and is deterministic for a given random seed.</li>
<li><b>HDBSCAN</b> — density-based clustering; automatically flags
sparse events as noise (label −1, shown in grey) rather than forcing
every event into a cluster.</li>
</ul>
<p>Choose whether to <b>cluster on original features</b> or on a
<b>trained DR embedding</b>; whether to downsample training data for
speed; and whether to assign the resulting cluster labels to every
sample or just the training set. Click <b>Run Clustering</b> to fit.</p>

<h4>4. Archived Runs</h4>
<p>Every DR and clustering run is kept here. Double-click a run's label
to rename it, or any other cell to view its full configuration. Renaming
or deleting a run here updates every run selector elsewhere in the
plugin (Cluster Annotation, Stats, Workspace) immediately.</p>

<h4>Further Reading and Background</h4>
<p>Documentation:</p>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/data-analysis-dimensionality-reduction">Data Analysis: Dimensionality Reduction</a> — Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/data-analysis-comparing-dimensionality-reductions">Data Analysis: Comparing Dimensionality Reductions</a> — Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/the-peculiarities-of-phenograph">The peculiarities of Phenograph</a> — Colibri Cytometry blog</li>
</ul>
<p>References:<br/>
[1]<a href="https://arxiv.org/abs/1802.03426">McInnes, Healy and Melville 2018 (UMAP)</a><br/>
[2]<a href="https://doi.org/10.1101/731877">Poličar, Stražar and Zupan 2019 (openTSNE)</a><br/>
[3]<a href="https://arxiv.org/abs/2012.04456">Wang, Huang, Rudin and Shaposhnik 2021, JMLR (PaCMAP)</a><br/>
[4]<a href="https://doi.org/10.1038/s41587-019-0336-3">Moon et al. 2019, Nat Biotechnol (PHATE)</a><br/>
[5]<a href="https://doi.org/10.1002/cyto.a.22625">Van Gassen et al. 2015, Cytometry A (FlowSOM)</a><br/>
[6]<a href="https://doi.org/10.1038/s41598-019-41695-z">Traag, Waltman and van Eck 2019, Sci Rep (Leiden)</a><br/>
[7]<a href="https://doi.org/10.1007/978-3-642-37456-2_14">Campello, Moulavi and Sander 2013, PAKDD (HDBSCAN)</a><br/>
[8]<a href="https://doi.org/10.12688/f1000research.21642.2">Kratochvil, Koladiya and Vondrasek 2020, F100Res (EmbedSOM)</a><br/>
[9]<a href="https://doi.org/10.1016/j.cell.2015.05.047">Levine et al. 2015, Cell (Phenograph)</a><br/>
[10]<a href="https://doi.org/10.1038/s41467-019-13055-y">Belkina et al. 2019, Nature Communications (OptSNE)</a><br/>
</p>
'''

cluster_annotation_tab_help_text = '''
<h3>Cluster Annotation: inspect, name, and QC your clusters</h3>
<p><b>Prerequisite:</b> run (or select an existing) clustering run in the
Configuration tab first, then pick it from the <i>Clustering run</i>
selector at the top of this tab.</p>

<h4>Annotation sub-tab</h4>
<ul>
<li><b>Per-Marker Violin Plots:</b> select one or more channels and click
<b>Recompute Violins</b> to see the per-cluster expression distribution
for each. Use this to sanity-check that clusters actually separate on
the markers you expect.</li>
<li><b>Cluster Map:</b> a scatter plot of the selected DR embedding,
coloured by cluster.</li>
<li><b>Cluster Labels:</b> a table of cluster names, with two
independent ways to get suggestions (see below). You always have final
control to rename any cluster by hand.</li>
</ul>

<h4>Cluster ID Suggestions</h4>
<p>Two independent mechanisms feed the Cluster Labels table, trying to
help you figure out what type(s) of cells are present:</p>
<ol>
<li><b>MEM (Marker Enrichment Modeling)</b>: a descriptive statistic of
the cluster's own transformed marker expression relative to a background
reference, producing labels like "CD4+6 CD8−5". Because it only
describes what the cluster's data actually shows, it is safe to adopt
automatically; use <b>Adopt All MEM Labels</b> to do so. Set the
<i>MEM threshold</i> to control how large a score has to be before a
marker is reported in the label.</li>
<li><b>Cell-type scoring</b>: matches each cluster's MEM profile
against a database of expected marker signatures for named cell types
(<i>species</i> selector, plus <tt>drc_cell_type_database.csv</tt>). This
is a <b>biological claim</b>, not a computed statistic — treat it as a
starting suggestion, not ground truth, and always check it against what
you know about your panel and biology. The approach is adapted from
ScType (see reference below), the same idea underlying the "What's that
cluster?" posts linked below: if your panel doesn't include the markers
needed to distinguish two cell types, the suggestion can't distinguish
them either, and uncorrected autofluorescence, unmixing errors, or
under-clustering (multiple biologically distinct populations lumped
into one cluster) will all degrade the match.</li>
</ol>
<p>Click <b>Compute Cluster ID Suggestions</b> to run both. Any channel
used for scoring must have its <i>Antigen</i> field filled in on the
Spectral Process tab (a blank Antigen blocks the computation); an
Antigen that doesn't match anything in the marker database contributes
nothing to cell-type scoring but doesn't block MEM.</p>

<h4>Marker Summary sub-tab</h4>
<ul>
<li><b>Heatmap of median MFI per cluster</b> (transformed): a quick
overview of which clusters express which markers, useful alongside or
instead of the Cluster Labels table when you'd rather read the raw
expression pattern yourself.</li>
<li><b>Marker Ridgeline Grid</b>: per-cluster expression histograms for
each marker, stacked for easy comparison.</li>
</ul>
<p>Click <b>Recompute Marker Summary</b> after changing the clustering
run or channel selection.</p>

<h4>Further Reading and Background</h4>
<p>Documentation:</p>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/what-s-that-cluster">What's that cluster?</a> — Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/what-s-that-cluster-part-ii">What's that cluster? Part II</a> — Colibri Cytometry blog</li>
</ul>
<p>References:<br/>
[1]<a href="https://doi.org/10.1038/nmeth.4149">Diggins et al. 2017, Nat Methods (MEM)</a><br/>
[2]<a href="https://doi.org/10.1038/s41467-022-28803-w">Ianevski, Giri and Aittokallio 2022, Nat Commun (ScType)</a><br/>
[3]<a href="https://doi.org/10.1002/cyto.a.22625">Van Gassen et al. 2015, Cytometry A (FlowSOM)</a><br/>
</p>
'''

stats_tab_help_text = '''
<h3>Stats — group comparisons, differential testing, and PCA</h3>
<p><b>Prerequisite:</b> a clustering run must exist (Configuration tab)
before you can run Frequency/Counts/MFI statistics, Confusion Matrix, or
Composition-by-group. DR-only runs are shown in the Run selector but
statistics stay disabled for them.</p>

<h4>1. Comparison Groups and Sample Assignment</h4>
<ul>
<li>Add one or more named <b>Comparison Groups</b>, then assign each
sample to a group in the table below — either by hand, by
<b>Auto-assign by pattern</b> (regex against sample names), or by
<b>Import CSV</b> / <b>Export CSV</b>. <b>Suggest Groupings…</b> offers
candidate groupings based on sample names.</li>
<li>You need at least 3 samples per group for the differential
statistics below to run.</li>
</ul>

<h4>2. Setting up a comparison</h4>
<ul>
<li>Pick the clustering <b>Run</b> to test, then check the groups to
include under <b>Groups to Test</b>.</li>
<li><b>Contrast mode:</b> compare every checked group against one
<i>Reference group</i>, or run <i>All pairwise</i> comparisons.</li>
<li><b>Paired design:</b> tick this if your samples are matched across
groups (e.g. before/after the same subject) — see "Using Paired design"
below for how this changes the model and what it requires.</li>
<li><b>Tests</b> — Cluster Frequencies (limma), Cluster Counts
(negative-binomial GLM), and Cluster MFIs (limma) can each be ticked
independently; see the dedicated sections below for what each is
actually doing and how to read its result.</li>
<li>Set the <i>p-value</i> and <i>|log₂FC|</i> thresholds used to flag
significant hits in the volcano plot and heatmap, then click
<b>Run Statistics</b>.</li>
</ul>

<h4>What is limma, and what does its p-value actually mean?</h4>
<p><b>limma</b> ("linear models for microarray data") was originally
built for detecting differentially-expressed genes, and is repurposed
here to treat each cluster's frequency, or each cluster's MFI for a
given channel, the way limma would treat one gene: it fits a linear
model per cluster (comparable to a t-test or one-way ANOVA, but
generalised via a design matrix so it can also express a reference-group
contrast, all-pairwise contrasts, or the paired blocking term described
below) and reports a fold-change and a p-value for the group effect.</p>
<p>The reason to use limma rather than a plain per-cluster t-test is its
<b>empirical Bayes moderation</b> step. With a modest number of samples,
a per-cluster variance estimated from that cluster alone is noisy — some
clusters will look "significant" purely because their observed variance
happened to be unusually small by chance, not because the group effect
is real. limma addresses this by borrowing information across every
cluster (or every channel, for MFIs) being tested in the same run: it
estimates the typical spread of variances across all of them and shrinks
each individual cluster's variance estimate toward that common trend,
producing a "moderated t-statistic" that is markedly more stable, and
better calibrated, than treating each cluster in isolation. This
moderation step is the "Bayes" part of limma's name — it uses a prior
built from the data itself (an <i>empirical</i> Bayes prior, rather than
one specified in advance) purely to stabilise the variance term. The
hypothesis test and p-value that come out the other end are still
ordinary frequentist ones — see below.</p>

<h4>Frequentist vs Bayesian statistics (and where limma sits between them)</h4>
<p>These are two different ways of assigning meaning to "probability" in
a statistical test, and it matters for how you should read a result:</p>
<ul>
<li><b>Frequentist</b> — the framework behind limma's p-values (and
most classical statistics: t-tests, ANOVA, linear regression). A p-value
is the probability of seeing data at least this extreme, in a
hypothetical infinite series of repeated experiments, <i>if the null
hypothesis were exactly true</i>. It is <b>not</b> the probability that
the null hypothesis is true, and it is not the probability that your
observed effect is real (see reference [3] below).</li>
<li><b>Bayesian</b> — assigns a probability distribution directly to
the hypothesis or parameter itself, and updates that distribution from a
starting ("prior") belief using the observed data via Bayes' theorem, to
produce a posterior probability — a number that really does answer "how
likely is it, given this data, that there is a real difference between
groups?" The cost is that a genuinely Bayesian analysis requires
specifying a prior, which is itself a modelling choice open to
disagreement (reference [5] gives a practical introduction to this style
 of analysis and how it compares to the frequentist "New Statistics").</li>

<h4>What does the negative-binomial GLM do? (Cluster Counts)</h4>
<p>Raw event counts per cluster per sample are <i>count</i> data, not
continuous measurements — small clusters can have very few or zero
events, and the variance of a count generally grows with its mean (a
cluster with a mean count of 5,000 across samples will naturally vary by
more, in absolute terms, than one with a mean count of 50) — a property
called <b>overdispersion</b> that a plain linear model/t-test's
constant-variance assumption doesn't accommodate. Cluster Counts instead
fits a <b>negative-binomial generalized linear model (GLM)</b>, run
alongside (not instead of) the limma Frequency test: it models the count
directly with a log link, using each sample's total event count as an
offset (so it compares <i>rates</i>, not raw totals, across samples with
different numbers of events), and fits a separate dispersion term per
cluster to absorb the extra variance rather than mistaking it for a real
group difference. This is the same broad approach used for RNA-seq
read-count data by tools like edgeR and DESeq2. Because Frequencies
(limma) and Counts (GLM) make different statistical assumptions about
the same underlying event counts, running both and comparing the results
is a useful cross-check: broad agreement between the two is stronger
evidence than either alone, and a disagreement is itself worth a closer
look (it often points to a cluster with very few total events, where the
two methods' different assumptions matter most).</p>

<h4>Using Paired design</h4>
<p>Tick <b>Paired design</b> when your samples have a genuine matched
structure across the groups being compared — for example the same
subject sampled before and after treatment, sequential timepoints from
one animal, or a single specimen split and processed under two different
conditions. Choose the <b>Pairing variable</b>: a column in your Sample
Group Assignment table (e.g. "PatientID" or "Subject") whose value
identifies which samples belong to the same pair or block.</p>
<ul>
<li><b>What it does:</b> pairing adds a fixed-effect blocking
term to the underlying design (<tt>+ C(pair_id)</tt> in the model
formula) — in effect, it fits and removes a separate baseline offset per
pair before testing the group effect, so the comparison is driven by
<i>within-pair</i> differences rather than being diluted by
between-subject variability that has nothing to do with the group effect
you actually care about.</li>
<li><b>What it is not:</b> this is a fixed-effect approach, not a true
random-effects/repeated-measures design of the kind limma's own
<tt>duplicateCorrelation()</tt> provides for correlated technical
replicates — InMoose does not currently implement that style of
random-effect blocking, so each pair consumes one degree of freedom of
its own. This works well with a moderate number of pairs, but can leave
you underpowered if you have many pairs each with very few samples.</li>
<li><b>Requirement:</b> every sample in every checked group needs a
value for the chosen pairing variable, and that value must correctly
identify its partner sample(s) in the other group(s) — if the pairing
column doesn't line up across groups, the design matrix can't be built
correctly.</li>
<li><b>When to skip it:</b> if your samples are genuinely independent
(unrelated subjects, no shared batch/individual across groups), leave
Paired design unticked — forcing a pairing structure onto unrelated
samples doesn't help and can distort the result.</li>
</ul>

<h4>4. Confusion Matrix, Composition Barplot, and Export</h4>
<p>Despite the name (borrowed from CyCONDOR, the R package this feature
is modelled on), the <b>Confusion Matrix</b> here is <i>not</i> a
classifier's predicted-vs-actual accuracy table — it's a
per-group-normalized cluster composition heatmap. For each checked
group, all of that group's samples' events are pooled and the group's
total event count is rescaled to a fixed reference total (1000 events),
so that groups with very different total cell counts can be compared on
equal footing. Each cell then shows, for one cluster (row) and one group
(column), how many of that group's rescaled 1000 events fall into that
cluster.</p>
<ul>
<li><b>Reading a column</b> tells you that group's overall cluster
composition — do most of its cells fall into a small number of clusters,
or are they spread out?</li>
<li><b>Reading a row</b> for one cluster lets you directly compare,
cluster by cluster, whether one group contributes disproportionately
more or fewer (rescaled) events to it than another — a large difference
across columns in the same row is the same underlying signal that the
Cluster Frequency test above assesses formally, but shown here as raw
normalized counts rather than a p-value or fold-change, which makes it a
good quick visual companion to that test, and a useful sanity check even
before you've run any statistics.</li>
</ul>
<p>The <b>Composition Barplot</b> (with percent / by-group toggles) shows
the same kind of information as bars rather than a heatmap — some people
find bars easier to compare across many clusters at a glance, while the
heatmap is often quicker for spotting one standout cluster/group
combination. <b>Export Results CSV</b> writes out the full statistics
table for the currently viewed comparison.</p>

<h4>5. Sample PCA</h4>
<p>PCA (Principal Component Analysis) takes the per-sample summary table
you build from whichever combination of Frequencies, Counts, and/or MFIs
you check, and finds a small number of new axes ("principal components",
PC1, PC2, …) that best capture how the samples differ from one another
as a whole. PC1 is the single direction of largest variation between
samples; PC2 is the next-largest direction that is uncorrelated with
PC1; and so on. Each sample becomes one point on the resulting 2-D
scatter (PC1 vs PC2 by default).</p>
<ul>
<li><b>Reading the scatter:</b> samples close together are similar
across the checked summary statistics taken as a whole; samples far
apart differ substantially. If your comparison groups separate cleanly
along an axis, that's supporting evidence the groups genuinely differ —
a useful sanity check that should broadly agree with (or, if it
disagrees, prompt you to double-check) what the differential statistics
above found. A sample sitting apart from everything else is a candidate
outlier or QC failure worth investigating (a failed stain, contaminated
sample, or unusually low event count, for example).</li>
<li><b>Percent variance explained</b> (shown on the axis labels) tells
you how much of the total sample-to-sample variation each PC captures.
A PC1 explaining, say, 60% of the variance is a dominant, trustworthy
axis; if PC1 and PC2 together only explain a small fraction of the total
variance, the 2-D plot is a much less complete picture of how the
samples actually differ, and should be read more cautiously.</li>
<li><b>Loadings</b> (tick "Show loadings"): each arrow shows how
strongly, and in which direction, one input variable (one cluster's
frequency/count/MFI) contributes to the PCs shown. An arrow pointing
toward a group of samples means those samples tend to have relatively
high values for that variable; two arrows pointing in opposite
directions indicate an inverse relationship between those two variables
across your samples. "Top N" limits the display to the N
largest-magnitude loadings, since a full set of arrows across many
clusters or channels quickly becomes unreadable.</li>
</ul>
<p>PCA here is descriptive/exploratory, not inferential — it produces no
p-value, and a visible separation between groups on the plot is not by
itself a statistical test of that separation. Use it to explore
structure, spot outliers or batch effects, and build intuition, then
confirm any specific hypothesis with the Frequency/Counts/MFI tests
above.</p>

<h4>Further Reading and Background</h4>
<p>References:<br/>
[1]<a href="https://doi.org/10.1093/nar/gkv007">Ritchie et al. 2015, Nucleic Acids Research (limma software)</a><br/>
[2]<a href="https://doi.org/10.2202/1544-6115.1027">Smyth 2004, Stat Appl Genet Mol Biol (limma's empirical Bayes moderation)</a><br/>
[3]<a href="https://doi.org/10.1080/00031305.2016.1154108">Wasserstein and Lazar 2016, The American Statistician (ASA statement on p-values)</a><br/>
[4]<a href="https://doi.org/10.1038/506150a">Nuzzo 2014, Nature (statistical errors and the misuse of p-values)</a><br/>
[5]<a href="https://doi.org/10.3758/s13423-016-1221-4">Kruschke and Liddell 2018, Psychon Bull Rev (a Bayesian alternative framework)</a><br/>
[6]<a href="https://doi.org/10.1038/s42003-019-0415-5">Weber et al. 2019, Communications Biology (diffcyt)</a><br/>
[7]<a href="https://doi.org/10.1093/bioinformatics/btp616">Robinson, McCarthy and Smyth 2009, Bioinformatics (edgeR)</a><br/>
[8]<a href="https://doi.org/10.1038/s41598-025-03376-y">Colange et al. 2025, Scientific Reports (InMoose)</a><br/>
</p>
'''

workspace_tab_help_text = '''
<h3>Workspace — build a scatter-plot canvas for your DR embeddings</h3>
<p><b>Prerequisite:</b> train at least one DR run in the Configuration
tab before adding plots here. Colouring by Clusters, or viewing the
FlowSOM Tree plot type, additionally requires a clustering run.</p>
<ul>
<li>Click <b>+ Add Plot</b> to add a new plot card to the grid; use
<b>Columns</b> and <b>Theme</b> (Auto/Light/Dark, applies to every plot
in the plugin) to control the layout.</li>
<li>Each plot card lets you choose:
  <ul>
  <li><b>Plot type</b> — Scatter (per-event points on the chosen DR
  embedding), or FlowSOM Tree (minimum-spanning-tree view over SOM
  nodes, only available when a FlowSOM clustering run is selected as the
  overlay).</li>
  <li><b>DR run</b> and <b>Sample</b> — pool all samples together or
  view one individually.</li>
  <li><b>Colour mode</b> — Clusters (needs a clustering run selected as
  the overlay), Marker (colour by a channel's intensity, pick the
  channel from the marker selector), or Group (colour by comparison
  group, or by T-REX enrichment score when a T-REX run is selected).</li>
  </ul>
</li>
<li>Right-click a cluster or group's legend colour swatch to recolour
it — the choice is shared with the Comparison Groups table in the Stats
tab.</li>
<li>Use the <b>magic wand</b> to copy a card's full display
configuration (colour mode, appearance settings, etc.) and the
<b>paste</b> button on another card to apply it — a fast way to keep
several plots visually consistent.</li>
<li>Each card can export its own <b>PNG</b>; use <b>Export PDF</b> in
the toolbar to export every currently-open card as a single multi-page
PDF.</li>
<li>The <b>Appearance</b> panel on each card controls grid lines,
tick/axis labels, legend and axis font size, and plot width/height.</li>
</ul>
<p>Plots you leave open here are exactly what the Report tab's Workspace
section will offer to include when you generate a report.</p>
'''

report_tab_help_text = '''
<h3>Report — export what's currently on screen</h3>
<p>The Report tab takes information from the other tabs' current
state. It doesn't recompute anything itself, and there is no separate
"reporting run" selector. It reports on whichever clustering run is
currently selected in the <b>Cluster Annotation</b> tab, since that's
the one place in the plugin where a clustering run selection is
already required.</p>
<h4>Before you generate a report</h4>
<ol>
<li>Open and arrange whatever you want included in the <b>Workspace</b>
tab (plot cards), <b>Cluster Annotation</b> tab (violin plots, cluster
map, marker summary), and <b>Stats</b> tab (result tabs) — these are
exactly what will be offered below.</li>
<li>Switch to this tab and click <b>Refresh Items</b> to (re)populate
the three tick-lists (Workspace / Cluster Annotation / Stats) from
whatever those tabs currently have open or computed.</li>
<li>Use <b>All</b> / <b>None</b> per section, or tick items
individually, to choose what to include.</li>
</ol>
<h4>Generate Report</h4>
<p>Click <b>Generate Report</b> to write, into a timestamped folder
under <tt>DR_Clustering_Reports</tt> in the experiment directory:</p>
<ul>
<li>One PNG (for figure items) and/or CSV (for table items) per ticked
item, organised into a sub-folder per source tab.</li>
<li>One combined PDF with a title page, a section divider per source
tab, and every ticked figure and table (tables paginated as needed).</li>
<li>A <tt>settings.txt</tt> file, always generated regardless of what's
ticked, documenting how to reproduce the analysis (gates, channels,
training samples, algorithm parameters, group assignments, etc.) so a
report can always be traced back to the settings that produced it.</li>
</ul>
<p>Because items reflect whatever is currently rendered elsewhere,
re-running <b>Refresh Items</b> after changing anything in Workspace,
Cluster Annotation, or Stats will keep this tab in sync.</p>
'''