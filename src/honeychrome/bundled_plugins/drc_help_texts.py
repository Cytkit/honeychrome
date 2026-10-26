"""
drc_help_texts.py — Per-tab help text for the DR / Clustering / Statistics plugin
===================================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does NOT
end in ``_tab.py``, so it is not picked up as a separate plugin tab — same
convention as drc_report.py, drc_stats.py, etc.).

One HTML help string per tab, shown/hidden via the core app's
HelpToggleWidget (honeychrome.view_components.help_toggle_widget), the same
mechanism used for the Spectral Process help and the AutoSpectral AF help.
Each tab's help starts with what the tab is for and what to do on it, then
gives the details and background; every tab ends with where it fits in the
workflow and a shared glossary.
"""

# Tabs in workflow order: (tab name, what it is for).
_TABS = (
    ('Transforms', 'check the axis scales the events are loaded in (usually fine as '
                   'they are)'),
    ('Configuration', 'choose the cells, markers and samples, then make a map (DR) and '
                      'clusters. <i>Start here.</i>'),
    ('Cluster Annotation', 'see what each cluster expresses and name it'),
    ('Stats', 'compare groups of samples cluster by cluster'),
    ('Layout', 'lay out figures on pages for export'),
    ('Report', 'write a folder of figures, tables and settings'),
)


def _workflow(tab: str) -> str:
    """The workflow as a numbered list, with *tab* marked."""
    items = []
    for name, purpose in _TABS:
        if name == tab:
            items.append(f'<li><b>{name}</b>: {purpose} <b>◀ this tab</b></li>')
        else:
            items.append(f'<li>{name}: {purpose}</li>')
    return ('<h4>Where this tab fits</h4>\n<ol>\n' + '\n'.join(items) + '\n</ol>\n')


_GLOSSARY = '''
<h4>Glossary</h4>
<ul>
<li><b>Training pool</b>: the events of the checked gates, a set number from
each training sample, that DR and clustering are fitted on.</li>
<li><b>DR</b> (dimensionality reduction): a 2-D map of the cells
(<b>UMAP</b>, <b>tSNE</b> or <b>PaCMAP</b>) in which similar cells sit
together. Distances between islands on the map are not a quantitative
measure.</li>
<li><b>Embedding</b>: the map positions a DR run gives the cells.
<b>Apply to All Samples</b> projects the samples that were not in the
training pool onto the same map.</li>
<li><b>Run</b>: one DR fit or one clustering fit, with the settings it was
made with. Every run is kept under Archived Runs and offered in the run
selectors of the other tabs.</li>
<li><b>Cluster</b>: a group of cells with similar marker expression.
<b>FlowSOM</b> first sorts cells onto a grid of nodes (the <b>SOM</b>) and
then merges the nodes into <b>metaclusters</b>, which are the clusters you
see. <b>Leiden</b> finds communities in a nearest-neighbour graph.
<b>HDBSCAN</b> finds dense regions and marks sparse cells as
<b>noise</b> (label −1, grey).</li>
<li><b>MFI</b>: here, a cluster's median intensity for one marker, in
transformed units.</li>
<li><b>Transformed units</b>: positions on a transformed (logicle) axis,
the same scale as the main cytometry plots.</li>
<li><b>MEM label</b>: a description of a cluster's markers relative to the
rest of the cells, for example "CD4+6 CD8−5". It describes the data, so it is
safe to adopt.</li>
<li><b>Cell-type suggestion</b>: a match of a cluster's markers against a
database of known cell types. A suggestion to check, not a fact.</li>
<li><b>Frequency</b>: a cluster's share of a sample's labelled events.
<b>Counts</b>: its number of events.</li>
<li><b>Effect</b> (logFC): the size of a difference between groups: a log₂
fold change for frequencies and counts, a difference in transformed units
for MFIs.</li>
<li><b>FDR</b>: the false discovery rate, the expected share of false
positives among the results called significant.</li>
<li><b>TREAT</b>: tests whether an effect is larger than a set threshold,
not just different from zero.</li>
<li><b>Reference group</b> and <b>All pairwise</b>: compare every group with
one reference, or every pair of groups.</li>
<li><b>Paired design</b>: samples matched across groups (for example the same
donor before and after treatment), compared within each pair.</li>
<li><b>Covariate</b>: a per-sample value (for example age or batch) that the
tests can <b>adjust for</b>.</li>
<li><b>Cluster-first MFI testing</b>: asks first which clusters change, then
which markers drive each of them.</li>
<li><b>Marker roles</b>: <i>type</i> markers define the clusters,
<i>state</i> markers are tested within them.</li>
</ul>
'''


transforms_tab_help_text = '''
<h3>Transforms: the scales Honeycluster loads the events in</h3>
<p><b>In short:</b> DR, clustering and the plots in this plugin use
transformed (logicle) values, so the scale of a channel changes how cells
spread out on it. The tab starts with the transforms of the Unmixed Data
tab, which are usually right. Look here when a negative population is
squashed against zero or split into two peaks. Changes made here stay in
this plugin; they are not written back to the experiment.</p>

<h4>What to do here</h4>
<ol>
<li>Check one or more <b>gates</b> in the gate tree. This is the same gate
selection as on the Configuration tab.</li>
<li>Select a channel in the <b>Channels</b> list on the right to see its
histogram.</li>
<li>If the negative population looks wrong, drag the lower half of the
histogram's x-axis to change its width (W), or click <b>Auto-transform (all
channels)</b>.</li>
<li>Check the scales before training DR or clustering: changing them later
means training again.</li>
</ol>

<h4>What good looks like</h4>
<p>The negative population is one symmetric peak around zero, and the
positive population sits clearly to the right of it.</p>

<h4>Details</h4>
<ul>
<li>Dragging an axis works as on the main cytometry plots: the lower half
changes the logicle width W, the upper half zooms the display range.</li>
<li><b>Auto-transform (all channels)</b> estimates W for every channel from
the selected training samples pooled: about 30,000 events of the checked
gates in all, an equal share from each sample, read without loading the
whole file.</li>
<li>Add or remove <b>biplots</b> below the histogram to look at two channels
together; drag either axis to adjust that channel.</li>
<li>Your adjustments are kept, and saved between sessions, until you click
<b>Reload from experiment</b>. That returns every channel to the transform
defined in Unmixed Data. Changes made later in Unmixed Data are not picked
up unless you reload.</li>
<li><b>Save CSV</b> and <b>Load CSV</b> export and re-import these local
transforms, for example to reuse a fine-tuned set.</li>
<li>To change the transforms used elsewhere in Honeychrome, drag the
channel's axis on the main cytometry plots.</li>
</ul>
''' + _workflow('Transforms') + _GLOSSARY


configuration_tab_help_text = '''
<h3>Configuration: choose the data, make a map and find clusters</h3>
<p><b>In short:</b> pick the cells (gates), the markers and the samples to
learn from. Then train a DR map to look at the cells in 2-D, and cluster
them into groups of similar cells. Every run is archived and used by the
other tabs.</p>
''' + _workflow('Configuration') + '''
<h4>What to do here</h4>
<ol>
<li><b>Data Selection</b>: check the gate(s) to analyse (for example live
single CD45+ cells), the markers to use, the training samples and the number
of training events per sample.</li>
<li><b>Dimensionality Reduction</b>: choose UMAP, tSNE or PaCMAP and click
<b>Train DR Model</b>, then <b>Apply to All Samples</b>.</li>
<li><b>Clustering</b>: choose FlowSOM, Leiden or HDBSCAN and click <b>Run
Clustering</b>.</li>
<li>Go to <b>Cluster Annotation</b> to see what the clusters are.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>The markers used are the lineage and phenotype markers you want to
separate cells by. Scatter, viability and dump channels are usually
left out.</li>
<li>On the map, known populations form separate islands, and the clusters
follow them.</li>
<li>No run is shown in orange under Archived Runs (orange means its gating
or unmixing has changed since it was made).</li>
</ul>
<p>The four sections open one at a time. While a DR or clustering run is in
progress, its section and Data Selection are locked (Cancel stays
available), so the run is archived with the settings it was trained
on.</p>

<h4>Details: data selection</h4>
<ul>
<li><b>Gate(s)</b>: the events of the checked gates are combined before
loading. This selection is shared with the Transforms tab.</li>
<li><b>Parameters to use for DR and clustering</b>: scatter channels start
unchecked, and <tt>Time</tt> and <tt>event_id</tt> are always left
out.</li>
<li><b>Training events per sample</b> and <b>Training samples</b>: how many
events each sample gives, and which samples train. Unstained samples,
single-stained controls and bead samples are left out of the picker unless
the matching <i>Include</i> box above it is ticked.</li>
</ul>

<h4>Details: dimensionality reduction</h4>
<p>Each algorithm balances local structure (cells within one island)
against global structure (distances between islands):</p>
<ul>
<li><b>UMAP</b> [1]: fast and widely used, with strong local
structure.</li>
<li><b>tSNE</b> [2], through openTSNE [4], which implements the fast FIt-SNE
algorithm [3]: excellent local structure; the white space between islands
means nothing.</li>
<li><b>PaCMAP</b> [5]: aims to balance local and global structure.</li>
</ul>
<p>None of them keeps the distances between clusters in an absolute sense.
Use the map to see groupings and get a feel for the data, not as a
measurement.</p>
<ul>
<li><b>Train DR Model</b> fits the algorithm on the training pool.
<b>Apply to All Samples</b> projects every sample, including those not used
for training, through the trained model. Events in the training pool keep
their fitted positions; only the others are projected.</li>
<li>UMAP finds each projected event's neighbours in the training pool with
an hnswlib index, built once per model before the first projection.</li>
<li>Defaults: UMAP 200 epochs (umap-learn's own default above 10,000
events); tSNE 500 iterations after 250 of early exaggeration (openTSNE's
defaults).</li>
<li>UMAP's code is compiled in the background the first time this tab opens
in a session, so the first UMAP run does not wait for it.</li>
<li>Two related methods are not run here but inform these options: opt-SNE
[11] tunes tSNE's iteration count, early exaggeration and learning rate for
large cytometry data, and EmbedSOM [10] places cells on a map guided by a
self-organising map, the structure FlowSOM uses for clustering.</li>
</ul>

<h4>Details: clustering</h4>
<ul>
<li><b>FlowSOM</b> [6]: a self-organising map followed by metaclustering.
Set the SOM grid size and the number of metaclusters. Training is batched,
so millions of events are handled quickly.</li>
<li><b>Leiden</b> [7]: community detection on its own nearest-neighbour
graph of the training events. Leiden was designed to fix a defect in the
Louvain-based algorithms behind classic Phenograph clustering [9], which can
produce badly connected or even disconnected clusters and can depend on the
number of cells and the neighbour-search method used. Leiden guarantees
well-connected clusters and gives the same result for a given random
seed.</li>
<li><b>HDBSCAN</b> [8]: density-based clustering. Sparse events are marked
as noise (label −1, grey) instead of being forced into a cluster.</li>
<li>Choose whether to cluster on the original markers or on a trained DR
map. FlowSOM always clusters on the original markers, HDBSCAN always on a
map.</li>
<li>You can downsample the training data for speed, and assign the cluster
labels to every sample or to the training set only.</li>
</ul>

<h4>Details: archived runs</h4>
<ul>
<li>Every DR and clustering run is kept here. Double-click a run's label to
rename it, or any other cell to see its full configuration. Renaming or
deleting a run updates every run selector in the plugin at once.</li>
<li>Each run records which events of each file it holds, so it stays usable
when the gating or unmixing changes afterwards: it is matched to the current
gating event by event.
<ul>
<li>Cells that have left its gates drop out.</li>
<li>Cells that have entered them are labelled from the run's SOM codebook
or cluster centroids (when it clustered on the markers), or left unlabelled
and reported otherwise.</li>
<li>A run whose gating or unmixing has changed is shown in orange, with the
details in its tooltip.</li>
</ul></li>
<li>Runs made before event keys were recorded can be matched only while
nothing has changed. <b>Use These Settings</b> loads a run's gates,
samples, markers and parameters so it can be run again on the current
data.</li>
<li><b>Run time</b>: before a run starts, its time is estimated from
measurements of each algorithm on this computer, which improve after every
run. A run expected to take 30 seconds or more asks first. The first time an
algorithm runs on a computer, a short timing test may be run to calibrate
it.</li>
</ul>

<h4>Further reading</h4>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/data-analysis-dimensionality-reduction">Data Analysis: Dimensionality Reduction</a>: Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/data-analysis-comparing-dimensionality-reductions">Data Analysis: Comparing Dimensionality Reductions</a>: Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/the-peculiarities-of-phenograph">The peculiarities of Phenograph</a>: Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/references">Useful Publications</a>: curated reference library, Colibri Cytometry</li>
</ul>
<p>References:<br/>
[1]<a href="https://arxiv.org/abs/1802.03426">McInnes, Healy and Melville 2018 (UMAP)</a><br/>
[2]<a href="https://jmlr.org/papers/v9/vandermaaten08a.html">van der Maaten and Hinton 2008, JMLR (tSNE)</a><br/>
[3]<a href="https://doi.org/10.1038/s41592-018-0308-4">Linderman et al. 2019, Nature Methods (FIt-SNE)</a><br/>
[4]<a href="https://doi.org/10.18637/jss.v109.i03">Poličar, Stražar and Zupan 2024, J Stat Softw (openTSNE)</a><br/>
[5]<a href="https://arxiv.org/abs/2012.04456">Wang, Huang, Rudin and Shaposhnik 2021, JMLR (PaCMAP)</a><br/>
[6]<a href="https://doi.org/10.1002/cyto.a.22625">Van Gassen et al. 2015, Cytometry A (FlowSOM)</a><br/>
[7]<a href="https://doi.org/10.1038/s41598-019-41695-z">Traag, Waltman and van Eck 2019, Sci Rep (Leiden)</a><br/>
[8]<a href="https://doi.org/10.1007/978-3-642-37456-2_14">Campello, Moulavi and Sander 2013, PAKDD (HDBSCAN)</a><br/>
[9]<a href="https://doi.org/10.1016/j.cell.2015.05.047">Levine et al. 2015, Cell (Phenograph)</a><br/>
[10]<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7255855/">Kratochvil et al. 2019, F1000Research (EmbedSOM)</a><br/>
[11]<a href="https://doi.org/10.1038/s41467-019-13055-y">Belkina et al. 2019, Nature Communications (opt-SNE)</a><br/>
</p>
''' + _GLOSSARY


cluster_annotation_tab_help_text = '''
<h3>Cluster Annotation: see what each cluster is and name it</h3>
<p><b>In short:</b> pick a clustering run, look at how each cluster
expresses each marker, and give the clusters names. Suggestions are
offered, but the names are yours.</p>

<h4>What to do here</h4>
<ol>
<li>Pick the run in the <b>Clustering run</b> selector at the top. Make one
on the Configuration tab first if there is none.</li>
<li>On the <b>Marker Summary</b> sub-tab, click <b>Recompute Marker
Summary</b> for a heatmap of each cluster's median expression and a
ridgeline of each marker.</li>
<li>On the <b>Annotation</b> sub-tab, choose the species and click
<b>Compute Cluster ID Suggestions</b>.</li>
<li>Click <b>Adopt All MEM Labels</b>, or type names into the <b>Cluster
Labels</b> table yourself.</li>
<li>Check doubtful clusters with <b>Per-Marker Violin Plots</b>: select
channels and click <b>Recompute Violins</b>.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>Each cluster is clearly positive or negative for the markers that define
it (for example CD3 and CD4 for CD4 T cells).</li>
<li>No cluster mixes two populations you know to be different. If one does,
cluster again with more clusters.</li>
<li>Every cluster you will report has a name.</li>
</ul>

<h4>Details: the Annotation sub-tab</h4>
<ul>
<li><b>Per-Marker Violin Plots</b>: the expression of each selected channel
in each cluster. Use them to check that clusters separate on the markers you
expect.</li>
<li><b>Cluster Map</b>: the DR map coloured by cluster.</li>
<li><b>Cluster Labels</b>: the cluster names, with two independent kinds of
suggestion. You can always rename any cluster by hand.</li>
</ul>

<h4>Details: the two kinds of suggestion</h4>
<ol>
<li><b>MEM (Marker Enrichment Modeling)</b> [1] describes a cluster's
transformed marker expression relative to a background reference, giving
labels such as "CD4+6 CD8−5". It only describes the cluster's own data, so it
is safe to adopt. <b>MEM threshold</b> sets how large a score must be before
a marker appears in the label.</li>
<li><b>Cell-type scoring</b> compares each cluster's median expression,
relative to each channel's positivity threshold, with a database of expected
marker signatures for named cell types (the species selector and
<tt>drc_cell_type_database.csv</tt>). The score is the one published for
ScType [2]. This is a <b>biological claim</b>, not a computed statistic:
treat it as a starting suggestion and check it against your panel and
biology.
<ul>
<li>If the panel lacks the markers that tell two cell types apart, the
suggestion cannot tell them apart either.</li>
<li>Uncorrected autofluorescence, unmixing errors and under-clustering
(several populations in one cluster) all degrade the match.</li>
</ul></li>
</ol>
<p><b>Compute Cluster ID Suggestions</b> runs both. Every channel used for
scoring needs its <i>Antigen</i> field filled in on the Spectral Process tab
(a blank Antigen blocks the computation). An Antigen that matches nothing in
the marker database adds nothing to cell-type scoring but does not block
MEM.</p>

<h4>Details: the Marker Summary sub-tab</h4>
<ul>
<li><b>Median MFI per Cluster (Transformed)</b>: a heatmap showing which
clusters express which markers. Useful alongside, or instead of, the
suggestions when you would rather read the expression pattern
yourself.</li>
<li><b>Marker Ridgeline Grid</b>: each cluster's histogram for each marker,
stacked for comparison.</li>
<li>Each has a <b>Size</b> slider (20 to 200%). The heatmap's cluster and
marker names keep their size; the ridgeline grid is redrawn when you release
the slider.</li>
<li>Click <b>Recompute Marker Summary</b> after changing the clustering run
or the channel selection.</li>
<li>The violins, heatmap and ridgelines use up to 20,000 events per cluster,
spread evenly over the run's samples, taken from the marker values stored
with the run. Cluster ID suggestions use the stored values too, so neither
reads the sample files again for runs that have them.</li>
<li>The marker-distribution views and editable cluster names follow the
marker-distribution and relabelling tools of the MARMOT pipeline
[4].</li>
</ul>

<h4>Further reading</h4>
<ul>
<li><a href="https://www.colibri-cytometry.com/post/what-s-that-cluster">What's that cluster?</a>: Colibri Cytometry blog</li>
<li><a href="https://www.colibri-cytometry.com/post/what-s-that-cluster-part-ii">What's that cluster? Part II</a>: Colibri Cytometry blog</li>
</ul>
<p>References:<br/>
[1]<a href="https://doi.org/10.1038/nmeth.4149">Diggins et al. 2017, Nat Methods (MEM)</a><br/>
[2]<a href="https://doi.org/10.1038/s41467-022-28803-w">Ianevski, Giri and Aittokallio 2022, Nat Commun (ScType)</a><br/>
[3]<a href="https://doi.org/10.1002/cyto.a.22625">Van Gassen et al. 2015, Cytometry A (FlowSOM)</a><br/>
[4]<a href="https://doi.org/10.1016/j.jim.2025.113854">Kirsche et al. 2025, J Immunol Methods (MARMOT)</a><br/>
</p>
''' + _workflow('Cluster Annotation') + _GLOSSARY


stats_tab_help_text = '''
<h3>Stats: compare groups of samples cluster by cluster</h3>
<p><b>In short:</b> assign each sample to a group, then test which clusters
change in size (frequencies or counts), and which markers change within
them (MFIs), between the groups. It needs a clustering run; DR-only runs are
listed but cannot be tested.</p>

<h4>What to do here</h4>
<ol>
<li>Under <b>Comparison Groups</b>, add the groups. Assign each sample in the
<b>Sample Group Assignment</b> table, by hand, with <b>Auto-assign by
pattern</b>, with <b>Suggest Groupings…</b> or with <b>Import CSV</b>. Each
group tested needs at least 3 samples.</li>
<li>Under <b>Differential Statistics</b>, pick the clustering <b>Run</b> and
tick the <b>Groups to Test</b>. Choose <b>Reference group</b> or <b>All
pairwise</b>.</li>
<li>If samples are matched (for example the same donor in each group), tick
<b>Paired design</b> and choose the <b>Pairing variable</b>. Tick covariates
under <b>Adjust for</b> if needed.</li>
<li>Tick the tests (Cluster Frequencies, Counts, MFIs), keep or change the
thresholds, and click <b>Run Statistics</b>.</li>
<li>Read the result tabs: volcano plots, heatmaps and tables. Right-click a
heatmap row, a volcano point or an MFI Clusters cell to see that feature's
value in each sample as a dot and violin plot, in a window or as a results
tab (which the Report lists).</li>
<li>Optional: <b>Composition Barplot</b> (any time), <b>Show PCA</b> and
<b>Run Predictive Analysis</b>.</li>
<li>Export with <b>Export Results CSV</b> or <b>Export Tables…</b>.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>Every sample is in the group you expect.</li>
<li>Significant clusters rest on a reasonable number of cells (check the
Composition Barplot) and are significant in both Frequencies and Counts when
you run both.</li>
<li>The Sample PCA shows no single sample far from all the others; one that
is may be a failed stain or a low event count.</li>
</ul>

<h4>Details: groups and samples</h4>
<ul>
<li><b>Auto-assign by pattern</b> matches each group's regular expression
against the sample names. <b>Suggest Groupings…</b> offers groupings found
in the sample names. <b>Import CSV</b> and <b>Export CSV</b> read and write
the assignment table.</li>
<li>Extra columns in the table hold covariates, such as donor, age, sex or
batch. Tick <b>Numeric</b> under a column's title to make it a number with a
straight-line effect (for example age or CRP); every sample then needs a
number. Leave codes such as batch 1, 2, 3 unticked so they are treated as
categories.</li>
</ul>

<h4>Details: the tests</h4>
<ul>
<li><b>Cluster Frequencies</b> and <b>Cluster MFIs</b> use a moderated linear
model; <b>Cluster Counts</b> uses a negative-binomial model. Each can be
ticked on its own. See Background below.</li>
<li>MFIs (here and in the MFI heatmap) are in each channel's Transforms-tab
scale, the same scale as the main cytometry plots. A cluster with no events
in a sample has no MFI there, and is left out of that sample rather than
counted as zero.</li>
<li>Set the <b>FDR</b> level, and the smallest effect worth reporting:
<i>|log₂FC|</i> for Frequencies and Counts, <i>|Δ|</i> (in transformed units)
for MFIs. Choose whether to <b>Test against thresholds (TREAT)</b> and use
<b>Cluster-first MFI testing</b>, and the FDR scope.</li>
<li>Changing the thresholds and clicking Run Statistics again recomputes the
p-values from the stored fits without refitting.</li>
<li><b>Marker Roles — MFI Testing</b>: every channel starts as a
<i>state</i> marker and is tested. Untick a channel to make it a
<i>type</i> marker and leave it out of MFI testing. Do this for a channel
that drove the clustering, so it does not decide both the cluster and its
own significance. <b>Reset all to Activation</b> sets every channel back to
<i>state</i>.</li>
<li>The volcano plot draws a thin horizontal bar through each significant
point: the 95% confidence interval of its effect. A wider bar means a less
precise estimate (fewer events, more variation, or fewer samples).</li>
</ul>

<h4>Details: composition and export</h4>
<ul>
<li>The <b>Composition Barplot</b> shows how each sample's events divide
among the clusters, as stacked counts or percentages (<b>%</b>), per sample
or summed per group (<b>By group</b>). It needs no statistics, so it is a
quick check before testing: does a cluster obviously skew toward one group,
and does a percentage rest on a reasonable number of cells? These plots
follow the equivalent summaries in cyCONDOR [12].</li>
<li><b>Export Results CSV</b> writes the full statistics table of the
comparison shown.</li>
<li><b>Export Tables…</b> writes the last statistics run into a folder, one
CSV per table, each with a single header row so R, Python, Prism or a
spreadsheet can read it directly.
<ul>
<li>The sample tables (<i>cluster_frequencies</i>, <i>cluster_counts</i>,
<i>cluster_mfis</i>) have one row per sample, with its group and covariates,
then one column per cluster. Frequencies and counts also give
<i>n_events</i>, the number of labelled events the percentages are taken
from.</li>
<li><i>clusters</i> lists each cluster's id, name and event count, with the
MEM label and suggested cell type when Cluster ID suggestions were
computed.</li>
<li>The <i>differential_*</i> files hold the results of every
comparison.</li>
</ul></li>
</ul>

<h4>Details: Sample PCA</h4>
<ul>
<li><b>Show PCA</b> takes the per-sample table built from the checked
Frequencies, Counts and MFIs and finds the directions (principal
components, PC1, PC2 and so on) along which the samples differ most. Each
sample is one point.</li>
<li>Samples close together are similar overall. Groups that separate along
an axis support a real difference, and should broadly agree with the tests.
A sample far from everything else is a candidate outlier or QC
failure.</li>
<li>The axis labels give the <b>percent variance explained</b>. When PC1 and
PC2 together explain only a small share, the plot is an incomplete picture
of how the samples differ.</li>
<li><b>Show loadings</b> draws arrows for the input variables: an arrow
pointing toward some samples means they tend to have high values for that
variable. <b>Top N</b> limits the arrows to the N largest.</li>
<li>PCA is exploratory: it gives no p-value. Confirm any hypothesis with the
tests.</li>
</ul>

<h4>Details: Predictive Analysis (optional)</h4>
<p>The tests ask, one feature at a time, "does this differ between groups?".
<b>Predictive Analysis</b> asks "which small set of features best tells two
groups apart, and how well?", the approach used in several
systems-immunology studies of flow cytometry cohorts [10, 11]. It uses the
Frequency and MFI tables of the last Run Statistics, for the comparison you
pick.</p>
<ul>
<li><b>Models</b>: lasso (L1-penalised) logistic regression, which keeps a
sparse set of features, and random forest, which can capture non-linear
effects and interactions.</li>
<li><b>Nested cross-validation</b>: the samples are split into 5 folds. Each
fold is predicted by models trained on the other four, with every tuning
choice made on the training folds only. The split is repeated
(<i>Repeats</i>) and averaged, so every number reflects samples the model
never saw.</li>
<li><b>ROC</b> tab: the cross-validated ROC curve and AUC (0.5 chance, 1
perfect) with its SD over repeats, and the balanced accuracy.</li>
<li><b>Features</b> tab: how often the lasso selected each feature, and the
forest's mean importance. The bar colour shows which group has the higher
value.</li>
<li><b>Panel Size</b> tab: the AUC using only the top N features, for
increasing N. The <i>minimal panel</i> is the smallest N reaching 95% of
the best AUC.</li>
<li><b>Sample size</b>: at least 10 samples per group are required, and 30
or more are recommended. With small groups the AUC varies a lot between
splits, even when there is no real difference.</li>
<li>Correlated features (for example a population and its largest subset)
share importance: the lasso tends to keep one of them, so a feature it drops
may still differ. Use the tests, not the selected features, for claims
about individual features.</li>
</ul>

<h4>Background: the moderated linear model (Frequencies and MFIs)</h4>
<p>Each cluster's frequency, or each cluster's MFI for one marker, is fitted
with its own linear model. This is comparable to a t-test or one-way ANOVA,
but written as a design matrix so it can also express reference or pairwise
contrasts, a pairing term and covariates. The reported effect (logFC) is
the group coefficient: for Frequencies, the difference in log₂ percentage (a
log₂ fold change); for MFIs, the difference in mean transformed
intensity.</p>
<p>The model uses <b>empirical Bayes moderation</b> of the variances (Smyth
2004 [2], the method behind limma [1]). With few samples, a cluster's
variance estimated from that cluster alone is noisy, and some clusters look
significant only because their variance happened to come out small.
Moderation estimates the typical spread of variances across every feature
and shrinks each feature's variance towards it. The moderated t-statistic
is more stable and better calibrated. The prior comes from the data itself;
the p-values are ordinary frequentist ones.</p>

<h4>Background: testing against a threshold (TREAT)</h4>
<p>With many clusters and markers, very small but consistent shifts can
reach significance, and filtering on the estimated fold change afterwards
does not control the error rate for "changed by at least this much". With
<b>Test against thresholds (TREAT)</b> ticked (the default), each p-value
tests whether the true effect is <i>larger</i> than the threshold (McCarthy
and Smyth 2009 [8]). Set the thresholds to the smallest change you would
consider biologically meaningful. With TREAT off, p-values test for any
non-zero effect and the thresholds only filter, as in a classical volcano
plot.</p>

<h4>Background: cluster-first MFI testing</h4>
<p>Testing every cluster × marker combination gives a long, flat list that is
hard to read and, with hundreds of tests, costs power. With <b>Cluster-first
MFI testing</b> ticked (the default), the question is asked in two stages
(Benjamini and Bogomolov 2014 [9]):</p>
<ol>
<li><b>Which clusters change?</b> Each cluster gets one screening p-value
combining all its markers (Simes' method), and clusters are selected by
Benjamini–Hochberg FDR across clusters.</li>
<li><b>Which markers drive each selected cluster?</b> Markers are tested only
inside the selected clusters, at an FDR level scaled down by the fraction of
clusters selected. This keeps the average error rate across the selected
clusters at your FDR level.</li>
</ol>
<p>The MFI Volcano then plots the stage-wise adjusted p-value (markers in
unselected clusters sit at the bottom). The <b>MFI Clusters</b> tab shows a
cluster × marker grid of the changes in each selected cluster, with the
significant markers marked. The exported table carries each cluster's
screening p-value (<tt>cluster.P.Value</tt>), its FDR
(<tt>cluster.adj.P.Val</tt>) and each marker's stage-wise value.</p>

<h4>Background: FDR scope</h4>
<p>Every comparison gets its own Benjamini–Hochberg correction
(<tt>adj.P.Val</tt>). The <b>FDR</b> selector chooses which correction
decides the "significant" flag:</p>
<ul>
<li><b>Pooled (all comparisons)</b> (the default): one correction across
every p-value of every comparison in the run (<tt>adj.P.Val.global</tt>).
The more conservative choice when several comparisons are reported together
(for example All pairwise with several groups). For cluster-first MFI
testing, the clusters of all comparisons are screened together.</li>
<li><b>Per comparison</b>: each comparison's own correction, as if each were
a separate analysis.</li>
</ul>
<p>Both corrected values are always in the exported CSV.</p>

<h4>Background: adjusting for covariates</h4>
<p>A column ticked under <b>Adjust for</b> is added to every test's model, so
group differences are estimated with that variable held fixed: for example
disease against control at the same age and sex. Numeric columns enter as a
straight-line effect; other columns enter as categories, compared with their
first level. Every tested sample needs a value. A covariate completely
confounded with group (for example all controls in one batch and all cases
in another) cannot be separated from the group effect, and the run stops
with a message naming it. Each covariate uses up degrees of freedom, so with
few samples adjust only for variables you have reason to think
matter.</p>

<h4>Background: the negative-binomial model (Cluster Counts)</h4>
<p>Event counts per cluster per sample are count data. Small clusters can
have very few or no events, and the variance of a count usually grows with
its mean (<b>overdispersion</b>), which a constant-variance linear model does
not allow for. Cluster Counts fits a <b>negative-binomial generalised linear
model</b>: it models the count with a log link, uses each sample's total
event count as an offset (so it compares rates, not raw totals), and fits a
dispersion per cluster to absorb the extra variance. RNA-seq tools such as
edgeR [7] and DESeq2 take the same broad approach. It uses the same design,
TREAT and FDR settings as the other tests. Frequencies and Counts make
different assumptions about the same events, so running both is a useful
cross-check: agreement is stronger evidence than either alone, and
disagreement often points to a cluster with very few events.</p>

<h4>Background: paired design</h4>
<p>Tick <b>Paired design</b> when samples are matched across the groups
compared: the same subject before and after treatment, sequential
timepoints from one animal, or one specimen split and processed two ways.
The <b>Pairing variable</b> is the column (for example "PatientID") whose
value says which samples belong together.</p>
<ul>
<li><b>What it does</b>: adds a fixed-effect blocking term (a separate
baseline per pair), so the comparison is driven by within-pair differences
rather than diluted by differences between subjects.</li>
<li><b>What it is not</b>: a random-effects or repeated-measures model. Each
pair uses one degree of freedom, which works well with a moderate number of
pairs but can leave you underpowered with many pairs of very few
samples.</li>
<li><b>Requirement</b>: every sample in every checked group needs a pairing
value, and each value must appear in more than one group.</li>
<li>If your samples are independent, leave Paired design unticked.</li>
</ul>

<h4>Background: frequentist and Bayesian statistics</h4>
<ul>
<li><b>Frequentist</b>: the framework behind these p-values (and t-tests,
ANOVA and linear regression). A p-value is the probability of data at least
this extreme in repeated experiments <i>if the null hypothesis were exactly
true</i>. It is <b>not</b> the probability that the null hypothesis is true,
nor that the effect is real [3, 4].</li>
<li><b>Bayesian</b>: assigns a probability distribution to the hypothesis or
parameter itself and updates a prior belief with the data to give a
posterior probability, which does answer "how likely is a real difference,
given these data?". The cost is choosing a prior, itself a modelling choice
open to disagreement; [5] gives a practical introduction. Empirical Bayes
moderation (above) borrows the prior idea for the variances only, and still
gives frequentist p-values.</li>
</ul>

<p>References:<br/>
[1]<a href="https://doi.org/10.1093/nar/gkv007">Ritchie et al. 2015, Nucleic Acids Research (limma software)</a><br/>
[2]<a href="https://doi.org/10.2202/1544-6115.1027">Smyth 2004, Stat Appl Genet Mol Biol (empirical Bayes moderation)</a><br/>
[3]<a href="https://doi.org/10.1080/00031305.2016.1154108">Wasserstein and Lazar 2016, The American Statistician (ASA statement on p-values)</a><br/>
[4]<a href="https://doi.org/10.1038/506150a">Nuzzo 2014, Nature (statistical errors and the misuse of p-values)</a><br/>
[5]<a href="https://doi.org/10.3758/s13423-016-1221-4">Kruschke and Liddell 2018, Psychon Bull Rev (a Bayesian alternative framework)</a><br/>
[6]<a href="https://doi.org/10.1038/s42003-019-0415-5">Weber et al. 2019, Communications Biology (diffcyt)</a><br/>
[7]<a href="https://doi.org/10.1093/bioinformatics/btp616">Robinson, McCarthy and Smyth 2009, Bioinformatics (edgeR)</a><br/>
[8]<a href="https://doi.org/10.1093/bioinformatics/btp053">McCarthy and Smyth 2009, Bioinformatics (TREAT)</a><br/>
[9]<a href="https://doi.org/10.1111/rssb.12028">Benjamini and Bogomolov 2014, J R Stat Soc B (selective inference on families of hypotheses)</a><br/>
[10]<a href="https://doi.org/10.1002/alz.70952">Humblet-Baron et al. 2025, Alzheimer's &amp; Dementia (systems immunology of Alzheimer's disease)</a><br/>
[11]<a href="https://doi.org/10.1038/s41467-026-70895-1">Veiga et al. 2026, Nature Communications (autoinflammation of unknown origin)</a><br/>
[12]<a href="https://lorenzobonaguro.github.io/cyCONDOR">Bonaguro et al. 2025, R package version 0.3.1 (cyCONDOR)</a><br/>
</p>
''' + _workflow('Stats') + _GLOSSARY


workspace_tab_help_text = '''
<h3>Layout: lay out figures on pages</h3>
<p><b>In short:</b> place plots, legends, text and panel letters on pages at
their printed size, then export the pages as PDF, SVG, PNG or TIFF. It needs
at least one DR run; colouring by cluster, and the Cluster Tree plot, also
need a clustering run.</p>

<h4>What to do here</h4>
<ol>
<li>Click <b>＋ Add Plot</b>, then set its type, run, sample and colour in
the panel on the right.</li>
<li>Drag plots into place and resize them by their handles.</li>
<li>Copy figures from Cluster Annotation or Stats with <b>＋ From Other
Tabs</b>.</li>
<li>Add <b>＋ Text</b> and <b>＋ Panel Letter</b>.</li>
<li>Click <b>Export…</b>.</li>
</ol>

<h4>What good looks like</h4>
<p>Items sit on the grid, fonts are the same size across panels, and the
panel letters run A, B, C in reading order.</p>

<h4>Details</h4>
<ul>
<li>Pages are A4 by default. <b>＋ Page</b> and <b>− Page</b> add or remove
pages. With nothing selected, the panel on the right sets the current page's
size and orientation.</li>
<li><b>＋ Add Plot</b> places a plot in the first free space. Drag to move;
drag the handles on the right edge, bottom edge or corner to resize. With
<b>Lock aspect ratio</b> ticked (the default) a plot keeps its proportions;
untick it to stretch freely.</li>
<li>Items snap to the grid (<b>Snap</b>, grid spacing). Hold Alt/⌥ to place
them freely. Arrow keys nudge the selection by one grid step.</li>
<li><b>Legend: Separate</b> (the default) puts the cluster or group legend
in its own box beside the plot. It moves with the plot and can be dragged
anywhere; dragging its edges reflows the entries into more or fewer columns
(the font size stays as set). Right-click it to put it back beside the plot
or hide it. <b>Right</b> and <b>Below</b> draw it inside the plot's figure
instead.</li>
<li>Right-click a legend entry to change its colour or rename it (cluster
colours are shared with the Cluster Annotation tab, group colours with the
Comparison Groups table). Double-click a cluster name to rename it.</li>
<li><b>＋ Text</b> adds a text box where you last clicked on a page (further
ones go below it). <b>＋ Panel Letter</b> adds a bold letter at the top-left
of the selected plot or group, or where you last clicked. Right-click an
empty part of a page to add text, a letter, a plot or a figure exactly
there. Letters are given A, B, C in reading order on each page; type a
letter yourself to fix it.</li>
<li>Select several items (drag a box, or Shift-click) and use
<b>Arrange</b> to align or distribute them, to <b>Group</b> them (Ctrl/⌘+G)
into one figure panel, or to change their stacking order. A group moves as
one; drag its orange corner to scale it (plots are redrawn at the new size,
text keeps its size). Double-click an item in a group to edit it
alone.</li>
<li><b>＋ From Other Tabs</b> copies a figure shown in the Cluster
Annotation or Stats tab (cluster map, violins, marker heatmap and
ridgelines; heatmaps, volcanoes, sample PCA, composition) onto the page.
Resize it like a plot (aspect ratio locked by default; fonts keep their
size). It is a copy: right-click it and choose <b>Update from source</b>
after changing it there. When the source is not available (for example after
reopening the experiment before the statistics are recomputed), a
placeholder holds its place until it can be updated.</li>
<li><b>Undo</b> and <b>Redo</b> (Ctrl/⌘+Z, Ctrl/⌘+Shift+Z) cover moving,
resizing, arranging, grouping, scaling and adding or removing items; not
changes to a plot's settings or to pages.</li>
<li>Select a plot to edit it in the panel on the right: plot type, DR run,
sample, colour (Clusters with a choice of <b>Clustering run</b>, Marker or
Group) and appearance (size in mm, legend, fonts in points, grid, labels;
<b>Label colour</b> Auto follows the plot theme).</li>
<li><b>Export…</b> writes the pages as PDF (vector), SVG (one file per page,
with editable text if you like), PNG or TIFF (300 or 600 dpi).</li>
<li>The Report tab's Layout section offers these pages for the
report.</li>
</ul>
''' + _workflow('Layout') + _GLOSSARY


report_tab_help_text = '''
<h3>Report: write the figures, tables and settings to a folder</h3>
<p><b>In short:</b> the report collects what the other tabs currently show.
It recomputes nothing. It reports on the clustering run selected on the
<b>Cluster Annotation</b> tab.</p>

<h4>What to do here</h4>
<ol>
<li>Open and arrange what you want to include on the <b>Layout</b>,
<b>Cluster Annotation</b> and <b>Stats</b> tabs.</li>
<li>Click <b>Refresh Items</b> to fill the three tick-lists (Layout,
Cluster Annotation, Stats) from what those tabs show.</li>
<li>Tick the items to include (<b>All</b> or <b>None</b> per section, or one
at a time).</li>
<li>Click <b>Generate Report</b>.</li>
</ol>

<h4>What good looks like</h4>
<p>Every figure you need is in the tick-lists. If one is missing, open it on
its tab and click <b>Refresh Items</b> again.</p>

<h4>Details</h4>
<p><b>Generate Report</b> writes a timestamped folder under
<tt>Honeycluster_Reports</tt> in the experiment folder:</p>
<ul>
<li>one PNG (figures) or CSV (tables) per ticked item, in a sub-folder per
source tab;</li>
<li>one combined PDF with a title page, a divider per source tab, and every
ticked figure and table (tables paginated as needed);</li>
<li>a <tt>settings.txt</tt> file, always written whatever is ticked, that
records how to reproduce the analysis (gates, channels, training samples,
algorithm parameters, group assignments), so a report can always be traced
back to the settings that produced it.</li>
</ul>
<p>The items reflect what is currently shown elsewhere, so click
<b>Refresh Items</b> after changing anything on the Layout, Cluster
Annotation or Stats tabs.</p>
''' + _workflow('Report') + _GLOSSARY
