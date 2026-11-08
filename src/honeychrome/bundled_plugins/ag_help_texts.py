"""
ag_help_texts.py — per-page help text for the Automated Gating plugin
=====================================================================
Companion module to ``ag_builder.py`` and ``automated_gating_tab.py`` (the
file name does not end in ``_tab.py``, so the plugin loader does not treat
it as a tab).

One HTML string per page, shown and hidden with the core app's
HelpToggleWidget (honeychrome.view_components.help_toggle_widget). Each
page's help starts with what the page is for and what to do on it, then
gives the details; every page ends with a shared glossary.
"""

_GLOSSARY = '''
<h4>Glossary</h4>
<ul>
<li><b>Gating model</b>: a gate hierarchy, the rule (algorithm) that places
each gate, and the boundaries trained for it. Saved as a <b>.agmodel</b>
file.</li>
<li><b>Parent population</b>: the population a gate is drawn on. Every gate
is calculated from its parent's events only.</li>
<li><b>Imported gate</b> (shapes <b>as drawn</b>): a gate copied from the main
plots or from FlowJo that still applies the shape that was drawn. Shown in
red.</li>
<li><b>Trainable gate</b>: a gate whose boundary an algorithm calculates from
the data. <b>Convert to trainable</b> turns an imported gate into one. Shown
in amber until trained.</li>
<li><b>Trained</b>: a trainable gate whose boundary has been calculated on
the training samples. Shown in green.</li>
<li><b>Review sample</b> (Hierarchy page): the sample the imported and
trainable gates are compared on, the one selected in the sample panel.
<b>Training samples</b> (Train page): the
samples pooled to calculate the boundaries. <b>Validation samples</b> (Run
page): gated samples whose gates you look at one by one.</li>
<li><b>1dsep</b>: one threshold on one channel, the <b>separation
parameter</b>, plotted against a <b>view parameter</b> or as a
histogram.</li>
<li><b>2dsep</b>: a threshold on each of two channels, giving four
quadrants. A population is one quadrant or several joined.</li>
<li><b>singlets</b>: a band around the diagonal of a scatter area against
height plot. Events piled at the top of either scatter axis (saturated) are
left out of the fit; they stay in the gate when they fall inside the
band.</li>
<li><b>free</b>: a shape on two scatter channels. The <b>cells</b> algorithm
cuts the debris off and trims the edges where the density falls away;
<b>all</b> takes the convex hull of the parent population.</li>
<li><b>replicate</b> (gate type): reuses another gate's boundary on a
different parent.</li>
<li><b>tail</b>: threshold just above the negative population, which is
mirrored around zero (negatives are symmetric on a logicle axis).
<b>Tail fraction</b> is the share of that mirrored population left above
the threshold: smaller is stricter.</li>
<li><b>mixture</b>: threshold where two (or three) Gaussian components
cross. Good for two clear peaks.</li>
<li><b>kde_min</b> and <b>bimodal</b>: threshold at the lowest density
between two peaks. <b>otsu</b>: the split that best separates two
classes.</li>
<li><b>replicate</b>: the threshold is taken from elsewhere: copied from
another gate on the same channel, or calculated on a <b>reference
population</b>.</li>
<li><b>Reference population</b>: a plentiful population whose biology is
known (for example CD3+ T cells, which do not express CD19), used to set a
threshold for a population that is rare or has no clear boundary.</li>
<li><b>Anchor population</b>: the population counts are related to by
default, normally the viable cells. Marked <i>[anchor]</i> in the
tree.</li>
<li><b>Factor</b> (threshold factor): multiplies a calculated threshold in
raw intensity units. 2 is stricter, 0.5 more lenient, 1 leaves it as
calculated.</li>
<li><b>Same-named gates</b>: gates imported under one name (for example
<i>TIGIT+</i> drawn on many subsets). They are kept on one algorithm
unless you set them differently.</li>
<li><b>Fixed</b> and <b>Recalculate</b> (Run page): apply the trained
boundary to every sample as it is, or calculate the gate again on each
sample. The <b>drift limit</b> flags a recalculated boundary that moves too
far.</li>
<li><b>Transformed units</b>: positions on a transformed axis, which runs
from about 0 to 1. <b>W</b> is the width basis of a logicle axis: it sets
how much room the values around zero get.</li>
</ul>
'''


BUILDER_TRANSFORMS = '''
<h3>Transforms: the axis scales the model works in</h3>
<p><b>In short:</b> gates are calculated on transformed (logicle) axes, so
the scale of a channel decides where its thresholds fall. The page starts
with this experiment's own display scales, which are usually right. Look
here when a negative population is squashed against zero or split into two
peaks.</p>

<h4>What to do here</h4>
<ol>
<li>Pick a <b>population</b> (Singlets or Cells to start with).</li>
<li>Select a channel on the right to see its histogram and biplots.</li>
<li>If the negative population looks wrong, drag the lower half of the axis
to change its width (W), or use <b>Auto-transform</b> for every
channel.</li>
<li>If you change a scale after training, train again.</li>
</ol>

<h4>What good looks like</h4>
<p>The negative population is one symmetric peak around zero, and the
positive population sits clearly to the right of it.</p>

<h4>Details</h4>
<ul>
<li>The scales belong to the plugin. Changing them here does not change the
main window's plots.</li>
<li>The model is trained in these scales, saves them, and the Run page
gates samples in them.</li>
<li>The population's events are pooled from the training samples (or, with
none chosen, the samples ticked on the Run page). About 30,000 events are
used in all, an equal share from each sample. Each sample is unmixed and
gated as the main window does.</li>
<li>The upper half of an axis changes the displayed range. Axes dragged on
the Import/Setup and Hierarchy plots change the same scales.</li>
<li><b>Auto-transform</b> sets W for every logicle channel from the events
shown (Parks, Roederer and Moore 2006).</li>
<li><b>Reset to the experiment's transforms</b> returns to the main
window's scales.</li>
<li><b>Apply the model's transforms</b> uses the scales a saved model was
trained in. It is available after <b>Load saved model…</b> on the
Import/Setup page, when the model stores them.</li>
<li>Imported shapes and trained boundaries move with a scale, so they stay
on the same events.</li>
</ul>
''' + _GLOSSARY


BUILDER_HIERARCHY = '''
<h3>Import/Setup: bring in the gating strategy the model will learn</h3>
<p><b>In short:</b> an automated gating model is your gating strategy,
calculated afresh from the data instead of drawn by hand on each sample.
On this page you bring the gates in. You then check them on the
<b>Hierarchy</b> page, calculate them on the <b>Train</b> page and apply
them on the <b>Run</b> page. Automated gating suits well-characterised
populations in a fixed panel. To discover populations you do not know in
advance, use clustering instead [1, 3].</p>
<h4>What to do here</h4>
<ol>
<li>Bring in a gating strategy in one of three ways:
<ul>
<li><b>Import from Unmixed Data</b>: the gates drawn on the Unmixed Data
plots.</li>
<li><b>Import FlowJo…</b>: a FlowJo workspace (.wsp) or template
(.wspt).</li>
<li><b>Load saved model…</b>: a model trained before, in this experiment or
another.</li>
</ul></li>
<li>Check the gate list. Each row is one gate, coloured by its status:
<span style="color:#c0392b">red</span> imported (applies its shape as
drawn), <span style="color:#946100">amber</span> converted but not trained,
<span style="color:#0a7a23">green</span> trained.</li>
<li>Go to the <b>Hierarchy</b> page to compare each gate with the version
that can be calculated, and convert it.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>Every gate is listed under the parent you expect.</li>
<li>The X and Y columns show the markers you expect.</li>
<li>No gate is marked ⚠ Overlap (a population defined twice).</li>
</ul>

<h4>Details: importing from the main plots</h4>
<ul>
<li>Each gate keeps the shape you drew and the population names you
gave.</li>
<li>The forward-scatter area against height (or width) gate becomes a
<i>singlets</i> gate and moves to the top of the hierarchy. Gates drawn
above it are placed directly below it. A side-scatter area against height
gate is a <i>singlets</i> gate too, left where it is.</li>
<li>Ellipses become 10-corner polygons of the same area.</li>
<li>Gates that share a name (for example <i>TIGIT+</i> drawn on many T cell
subsets) are named after the population they are drawn on: <i>TIGIT+
(Tem)</i>, or <i>TIGIT+ (CD4+ › Tem)</i> when that is needed to tell them
apart. They keep the shared name as a link (see Same-named gates in the
glossary).</li>
<li>The side of a fluorescence axis a population is on comes from the drawn
coordinates. Entirely above the channel's zero is the positive side. Reaching
zero or below is the negative side.</li>
<li>How each gate is converted to a trainable gate is described in the
Hierarchy page's help.</li>
<li><b>Refresh from Unmixed Data</b> adds gates drawn since the last import and
keeps the settings you have changed. Gates no longer on the main plots are
marked stale rather than deleted.</li>
</ul>

<h4>Details: importing from FlowJo</h4>
<ul>
<li>A section opens under the buttons. Choose the group or sample whose
gating to import.</li>
<li>Check how each FlowJo parameter maps to an experiment channel. Matching
is by fluorophore name where possible. Assign or drop any parameter that
could not be matched. The search box and <b>Flagged or unmatched</b> narrow
the list to the rows that need a decision.</li>
<li>Populations that use a dropped parameter, and those below them, are not
imported.</li>
<li><b>Import all — add</b> adds the gates to the model. <b>Import all —
replace</b> replaces its gates.</li>
<li>FlowJo coordinates are raw instrument units, converted with this
experiment's scales. The experiment's own gating is never changed.</li>
</ul>

<h4>Details: loading a saved model</h4>
<ul>
<li>A section opens under the buttons. It summarises where and how the model
was trained, including the autofluorescence profiles of its training
samples.</li>
<li>Every channel the model's gates use is matched to this experiment by
antigen, so a panel that moved an antigen to another detector still lines
up. Synonyms such as CD197 and CCR7 are recognised.</li>
<li>Green rows matched by antigen. Amber rows need checking: the same
detector carries a different antigen, the match is only by a similar name,
or the antigen is on more than one detector. Red rows have no match: choose
a channel.</li>
<li><b>Import model — replace</b> replaces the gates with the model's. They
keep their trained boundaries and their Fixed or Recalculate modes (Run
page), so the model can be applied straight away, or reviewed and trained
again. Training samples from another experiment are dropped.</li>
<li>The model's own scales are applied at once when the Transforms page has
not been changed by hand since its last reset. Otherwise the boundaries are
converted to the current scales, and a note under the gate list says
<b>Apply the model's transforms</b> is available on the Transforms
page.</li>
</ul>

<h4>Details: finding and editing gates</h4>
<ul>
<li>The X and Y columns show each axis by its marker (by its fluorophore
when the channel has no marker).</li>
<li>The search box keeps the gates whose name, populations, markers,
channels or algorithm match. The populations above a match stay
visible.</li>
<li><b>Show</b> narrows the list to <i>Unconverted</i> gates,
<i>Conflicting algorithms</i> (same-named gates set differently by hand),
<i>Flagged conversions</i> or <i>Overlapping gates</i> (see the Hierarchy
page). Flagged gates are marked ⚠, with the reason in the tooltip.</li>
<li>Double-click or right-click a gate to change its type, channels,
populations or algorithm. <b>Advanced…</b> opens the algorithm's settings:
tail fraction, number of mixture components, quantile trimming, the part of
the axis used, the threshold factor and, for singlets gates, the band width
in standard deviations.</li>
<li>Changing a gate between <i>1dsep</i> and <i>2dsep</i> keeps its
population names, so the gates below stay attached. Going to 2dsep, each
side becomes two quadrants; tick the quadrants each population covers in
the <b>Quadrants</b> column (e.g. X+ Y− only). Changing a gate's type
returns its mixture components to 2. A change that would leave gates below
without their parent population is refused, naming them.</li>
<li>For a <i>1dsep</i> gate, the <b>Side</b> column sets which side of the
threshold each population takes: <b>Below</b> or <b>Above</b>. A population
you add takes the side the others leave free, so adding <i>Non-B</i> beside
<i>B cells</i> (Above) gives it the events below. <b>Position</b> only places
the label.</li>
<li>Renaming a population in the editor renames it for the gates below
too.</li>
<li><b>Add gate manually</b> creates a gate that is not on the main
plots.</li>
<li><b>1dsep separation axis</b> sets which plot axis carries the separation
parameter of gates you add or split.</li>
</ul>

<h4>Further reading</h4>
<ul>
<li><a href="https://github.com/DrCytometer/AutoGating">AutoGating</a>: the
R package implementing the gating algorithms (GitHub)</li>
<li><a href="https://www.colibri-cytometry.com/references">Useful
Publications</a>: curated reference library, Colibri Cytometry</li>
</ul>
<p>References:<br/>
[1]<a href="https://doi.org/10.1002/cyto.a.24320">Cheung et al. 2021, Cytometry A (current trends in automated analysis software)</a><br/>
[2]<a href="https://doi.org/10.1093/bib/bbae633">Liu et al. 2025, Briefings in Bioinformatics (evaluation of gating methods)</a><br/>
[3]<a href="https://doi.org/10.1038/nri.2016.56">Saeys, Van Gassen and Lambrecht 2016, Nature Reviews Immunology (computational flow cytometry)</a><br/>
</p>
''' + _GLOSSARY


BUILDER_REVIEW = '''
<h3>Hierarchy: check each gate and convert it to a calculated gate</h3>
<p><b>In short:</b> an imported gate applies the shape that was drawn. To be
placed afresh on new samples, it has to be converted to a trainable gate,
which uses an algorithm (for example <i>tail</i> or <i>mixture</i>) to find
its boundary. This page shows the two side by side on a review sample, so
you can choose, for each gate, the algorithm that reproduces your
gating.</p>

<h4>What to do here</h4>
<ol>
<li>Select a typical sample in the sample panel: it is the <b>review
sample</b>. Select another there to check the gates on it.</li>
<li>Pick a population in the tree on the left (or in the
<b>Population</b> list). Each of its gates is shown as drawn (left) and in
its trainable form (right), level with each other.</li>
<li>Where the right-hand boundary sits where you drew it, click <b>Convert
to trainable</b>. Where it does not, choose another algorithm, change the
<b>Tail fraction</b> or <b>Factor</b>, or choose <b>replicate</b> with a
reference population.</li>
<li>Or click <b>Convert all to trainable</b> to accept every proposal at
once, then look through the populations that still need attention.</li>
<li>Click <b>Flag conversions</b> and <b>Flag overlaps</b> to mark the gates
to look at (⚠).</li>
<li>When every population in the tree is amber or green, go to the
<b>Train</b> page.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>The trainable threshold sits in the same gap between populations as the
drawn one.</li>
<li>The fraction of parent next to each population label is close to the
fraction recorded at import. When the review sample is the sample the
counts came from, it is shown in red when
the two differ by more than 2 percentage points or by more than 5% of the
imported fraction.</li>
<li>No ⚠ flags remain, or you have looked at each one.</li>
</ul>
<p>Tree colours: <span style="color:#c0392b">red</span> imported (shape as
drawn), <span style="color:#946100">amber</span> converted but not trained,
<span style="color:#0a7a23">green</span> trained. Changing a gate clears the
training of that gate and of the gates below it.</p>

<h4>Details: the plots and controls</h4>
<ul>
<li>A population with no gates below it shows the plot it is gated
on.</li>
<li>The trainable form is calculated on the review sample's parent
population with the algorithm chosen above it, or is the trained boundary
once trained.</li>
<li>A 2dsep gate has an algorithm per axis. A 2dsep threshold that separates
populations on one side only is drawn on that side only.</li>
<li>Each algorithm has a colour, used for its name and its threshold:
<span style="color:#8e44ad">mixture</span>,
<span style="color:#dc6e00">tail</span>,
<span style="color:#1e64dc">replicate</span>,
<span style="color:#008c8c">otsu</span>,
<span style="color:#c82878">bimodal</span>,
<span style="color:#8c5f28">kde_min</span> and
<span style="color:#1ea032">imported</span>.</li>
<li><b>Tail fraction</b> starts at 0.0001. The arrows step by factors of
ten, and any other value can be typed. <b>Factor</b> multiplies the
calculated threshold in raw units (for example 2 for a stricter
plasmablast gate).</li>
<li>The tail algorithm uses the whole axis. The others ignore the lowest 1%
of it.</li>
<li>A <i>mixture</i> axis shows its number of <b>components</b> and its
<b>cut</b>: which two components, ordered by mean, cross at the threshold.
With three components (a negative, a dim and a bright population) the cut
can be between the lowest two or the highest two.</li>
<li>A <i>cells</i> gate shows its fall-off per axis beside the
algorithm.</li>
<li><b>Update</b> calculates the shown gates again.</li>
<li><b>Convert to trainable</b> makes Train calculate the gate. <b>Use shapes
as drawn</b> reverses it. <b>Split…</b> replaces a 2dsep gate with
sequential 1dsep gates. The red <b>−</b> deletes a gate and the gates below
it, after asking.</li>
<li><b>Convert all to trainable</b> first checks every gate on the review
samples. Click it again to stop.</li>
</ul>

<h4>Details: replicated thresholds</h4>
<ul>
<li>Choosing <b>replicate</b> shows a source: another gate on the same
channel (its threshold is copied), or a reference population (the
threshold is calculated there, by default with <i>tail</i>).</li>
<li>Until a source gives a threshold, the imported boundary is shown in its
place, with a caption.</li>
<li>Once a source is chosen, a plot below the pair shows the population the
threshold is calculated on, with the gate drawn on it and each population's
share of it.</li>
<li>Example: CD3+ T cells, which do not express CD19, set the CD19
threshold; Foxp3− CD4 T cells set the CD25 threshold used for Tregs.</li>
<li>Mark a population as a source with <b>Use as reference population</b>
in the tree's right-click menu.</li>
</ul>

<h4>Details: how a proposal is chosen</h4>
<p>With the review sample loaded, an imported gate's proposed algorithm comes
from its shape as drawn on it. An algorithm you choose for an axis is
kept. The algorithm's tooltip gives the reason.</p>
<ul>
<li>Before the review sample is loaded, each threshold axis is proposed as
<i>mixture</i>; a histogram gate uses the counts recorded at import
(<i>mixture</i> when more than 25% of the parent is positive, else
<i>tail</i>).</li>
<li><b>1dsep</b>, from the parent population on the separation channel:
<ul>
<li>At least 40% of the parent below zero: <i>tail</i>.</li>
<li>At least 5% below zero: whichever of <i>tail</i>, <i>mixture</i> and a
three-component <i>mixture</i> reproduces the drawn shape best.</li>
<li>Less than 5% below zero and more than 40% drawn positive: the parent's
own events cannot place the threshold, so it is <i>replicate</i>d from the
anchor population. A red line then asks you to choose a reference
population; <b>Keep</b> accepts the anchor. Gates drawn on the anchor or
above it are not replicated.</li>
<li>Otherwise: <i>mixture</i> when more than 25% of the parent is on the
positive side, else <i>tail</i>.</li>
</ul></li>
<li><b>2dsep</b>: the combination of these candidates on the two axes that
reproduces the drawn shapes best.</li>
<li>Sibling gates drawn on the same two parameters become one gate with a
population each: <i>2dsep</i>, or <i>1dsep</i> on the fluorescence
parameter when the other is scatter. They take the quadrants that best
match their shapes: a gate spanning both sides of a threshold keeps both
(for example <i>CD14−</i> drawn over the whole height of the plot).</li>
<li>A single shape on two fluorescence parameters whose edges on one
parameter leave out at most 1% of the parent's events converts to
<i>1dsep</i> on the other parameter (for example a CD3 rectangle spanning
the whole CD20 range).</li>
<li>A 2dsep gate with one population on a whole side of one parameter and
one population in each quadrant of the other side is <b>split</b> into
sequential 1dsep gates when converted: that parameter first (its other side
named after the marker, for example <i>CD38+</i>), then the other parameter
on that side.</li>
<li>A histogram gate becomes a <i>1dsep</i> gate with a histogram
view.</li>
<li>Shapes drawn only on scatter parameters become <i>free</i> gates with
the <i>cells</i> algorithm. A scatter shape drawn on the debris converts to
a <i>cells</i> gate that keeps the debris side.</li>
</ul>

<h4>Details: the anchor population</h4>
<p>The anchor is the population marked with <b>Use as anchor population</b>
in the tree's right-click menu. Until one is marked it is the viable cells
(the negative side of the deepest gate using a viability dye, recognised
from the fluorophore database), else single cells, else <i>Cells</i>. It is
marked <i>[anchor]</i> in the tree. The Enrichment page uses the same
population as its default anchor.</p>

<h4>Details: same-named gates</h4>
<ul>
<li>Gates that share an imported name (for example every <i>CD80+</i>) are
kept on one algorithm. Converting one, or changing a converted one's
algorithm, converts the others still as drawn to the same algorithm and
settings.</li>
<li>Once two of them have been set differently by hand, the others are left
alone and a message at the bottom says so. When you set them alike again,
the others follow.</li>
<li><b>⇊</b> applies a converted gate's settings to every gate of the same
name below the population it is drawn on, set by hand or not. For example,
set up <i>CD25+</i> on one T cell subset, then pass it to the subsets below
it.</li>
<li><b>⇉</b> does the same for every gate of that name on the same channels
anywhere in the model.</li>
<li><b>⇶</b> lets other gates copy a converted gate's threshold instead of
calculating their own. Set up <i>CD28+</i> on a clean population such as
<i>CD4 Tconv</i>, press ⇶ and tick the gates that should use its boundary:
gates of the same name start ticked, and any gate on the same channel can be
added; Shift- or Ctrl-click selects several rows, and ticking one ticks them
all. With <i>Fixed</i> application they use its trained threshold; with
<i>Recalculate</i>, the threshold it gets on each sample.</li>
</ul>

<h4>Details: editing shapes and axes</h4>
<ul>
<li>Tick <b>Edit gates</b> to drag the drawn shapes of gates that still
apply them (left; long polygons are thinned to 32 corners) and the
boundaries of converted gates (right). A boundary moved on the right is
kept as an adjustment until the gate is trained again.</li>
<li><b>Reset to import</b> restores the shapes shown to their imported
state.</li>
<li>Drag an axis to rescale it, as on the main plots. The gates move with
the axis, so they stay on the same events.</li>
<li>Right-click a population on any plot to open it: the page moves to that
population and shows the gates drawn on it.</li>
</ul>

<h4>Details: finding and checking gates</h4>
<ul>
<li>The search box above the tree keeps the populations whose gates match
by name, marker, channel or algorithm. <b>Show</b> narrows it as on the
Import/Setup page.</li>
<li><b>Flag conversions</b> compares each gate's imported shape with its
trainable form on the review sample. It flags the gate when a threshold
moves more than the set distance (in transformed units) or a population's
share of its parent changes by more than the set number of percentage
points.</li>
<li><b>Flag overlaps</b> looks for populations defined twice: two
populations, neither below the other, that share at least the set share of
their events. A branch defined twice is reported once. The Import/Setup
table marks these gates ⚠ Overlap.</li>
</ul>

<h4>Details: the tree's right-click menu</h4>
<ul>
<li><b>Edit gate…</b> and <b>Advanced parameters…</b> of the gate that makes
the population.</li>
<li><b>Copy gate</b> or <b>Copy gating branch</b> (the gate and every gate
below it), then <b>Paste</b> onto other populations. Ctrl- or Shift-click
selects several. Copies are named after the population they are pasted on
and follow their same-named gates.</li>
<li><b>Delete gate…</b> and <b>Delete population…</b> remove the selection
with what is below it.</li>
<li><b>Use as reference population</b> and <b>Use as anchor
population</b>.</li>
</ul>

<h4>Details: the algorithms</h4>
<p><b>tail</b> takes the upper tail of a symmetric negative population
(negatives are symmetric on a logicle axis [1]). <b>kde_min</b> and
<b>bimodal</b> take the density minimum between two peaks. <b>mixture</b>
takes the point where two Gaussian components cross. <b>otsu</b> takes the
split that best separates two classes [2]. A threshold population is
open-ended: every event beyond the threshold belongs to it, however far
outside the plotted range. Shapes are tested where the plots draw the
events: an event beyond an axis sits on that axis edge, so a shape drawn up
to the edge includes the events past it.</p>
<p>Thresholds are calculated once, on the pooled training samples. OpenCyto
[3] and flowDensity [4] instead recalculate them for each sample; the Run
page offers that as the <b>Recalculate</b> mode.</p>
<p>References:<br/>
[1]<a href="https://doi.org/10.1002/cyto.a.20258">Parks, Roederer and Moore 2006, Cytometry A (Logicle display)</a><br/>
[2]<a href="https://doi.org/10.1109/TSMC.1979.4310076">Otsu 1979, IEEE Trans Syst Man Cybern (threshold selection)</a><br/>
[3]<a href="https://doi.org/10.1371/journal.pcbi.1003806">Finak et al. 2014, PLOS Computational Biology (OpenCyto)</a><br/>
[4]<a href="https://doi.org/10.1093/bioinformatics/btu677">Malek et al. 2015, Bioinformatics (flowDensity)</a><br/>
</p>
''' + _GLOSSARY


BUILDER_TRAIN = '''
<h3>Train: calculate every gate from training samples</h3>
<p><b>In short:</b> training places the boundary of every converted gate,
working down the hierarchy, on the events of the training samples pooled
together. The trained boundaries are what the Run page applies to your
samples.</p>

<h4>What to do here</h4>
<ol>
<li>Tick the <b>training samples</b>: samples that cover the conditions in
your experiment. Controls are hidden unless <b>Show controls</b> is
ticked.</li>
<li>Set <b>Events per sample</b>. Each sample gives the same number of
events, so a large file cannot dominate. <i>All</i> uses every event.</li>
<li>Click <b>Calculate Gates</b>.</li>
<li>Open <b>Gate boundaries</b> and look through the tiles, with the
hierarchy on their left. To move a boundary, click a tile's
<b>Pop out</b>, move the threshold in the window that opens and
accept.</li>
<li>When every population in the tree is green or blue, go to the
<b>Run</b> page. You can save the model on the <b>Export</b> page, the
last page, at any time.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>Each threshold sits in the gap between the populations it
separates.</li>
<li>The tree is green (trained) or blue (adjusted by hand), with nothing
amber (not trained) or red (failed).</li>
<li>The status line reports no gate without enough events.</li>
</ul>

<h4>Details</h4>
<ul>
<li>Samples are unmixed in the background, each with its own
autofluorescence profile assignment, exactly as the main window unmixes
them. Time-QC keep-masks are applied when present.</li>
<li>Gates are calculated in hierarchy order on the pooled events, in the
Transforms page's scales.</li>
<li>Tiles are shown for the gates on the populations open in the tree: the
first three levels to begin with. Expand a population to show the gates
drawn on it, or select it to jump to them. Threshold lines take the colour
of their algorithm.</li>
<li>A gate whose parent population has too few events is reported in the
status line, and the gates below it are skipped.</li>
<li>The tree's search box and <b>Show</b> work as on the Import/Setup
page.</li>
<li><b>Pause</b> stops after the gate being calculated and keeps every gate
trained so far. <b>Calculate Gates</b> then offers to <b>Resume</b> (train
the rest on the same pooled events) or <b>Overwrite</b> (train every gate
again). A paused run can be resumed only while the training samples, events
per sample, gates and scales are unchanged.</li>
<li><b>Abort</b> stops and keeps nothing from the run: the boundaries from
before stay.</li>
<li>Training results, and a paused run's progress, are kept with the
experiment and restored when you come back to it.</li>
</ul>
''' + _GLOSSARY


BUILDER_EXPORT = '''
<h3>Export: save the gating model</h3>
<p><b>In short:</b> saving is optional and can be done any time after
training; this page is last in the list for that reason. The Run page
applies the model as it stands on the model pages. Save it to reuse it in another experiment or
to keep a record of the model your results came from.</p>

<h4>What to do here</h4>
<ol>
<li>Check the gate summary: every gate should be trained.</li>
<li>Click <b>Save Model…</b> and choose a file name.</li>
<li>In another experiment, open it with <b>Load saved model…</b> on the
Import/Setup page.</li>
</ol>

<h4>What the file holds</h4>
<p>A <b>.agmodel</b> file holds the gate hierarchy, the trained boundaries,
the scale of every channel (Transforms page), the training samples with
their event counts and autofluorescence assignments, time-QC details, and
each gate's Fixed or Recalculate mode with the drift settings (Run
page).</p>
''' + _GLOSSARY


AG_RUN = '''
<h3>Run: gate your samples with the model</h3>
<p><b>In short:</b> this page applies the model to the samples you tick and
records each population's counts and marker medians. It uses the model as
it stands on the model pages, in the Transforms page's scales, once every
gate is trained.</p>

<h4>What to do here</h4>
<ol>
<li>Under <b>Gate application</b>, leave each gate <b>Fixed</b> (the usual
choice) or set some to <b>Recalculate</b> (see below).</li>
<li>Under <b>Samples and gating</b>, tick the samples to gate in the
<b>Gate</b> column. Tick <b>Validate</b> for a few samples whose gates you
want to look at one by one.</li>
<li>Click <b>Run Gating</b>.</li>
<li>Pick a population in the tree to see its gates on the pooled samples, on
the sample selected in the sample panel and on each validation sample.</li>
<li>Go to the <b>Results</b> page to compare groups.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>The gates sit on the populations in the pooled view and in each
validation sample.</li>
<li>The table's <i>Flags</i> column is empty, or you have checked each
flagged sample.</li>
<li>No warning about samples whose autofluorescence assignment differs from
the training samples'.</li>
</ul>

<h4>Details: Fixed and Recalculate</h4>
<ul>
<li><b>Fixed</b> (default): the trained boundary is applied to every sample
as it is.</li>
<li><b>Recalculate</b>: the gate's algorithm is run again on each sample's
own parent population, with the model's settings (the per-sample approach
of <a href="https://doi.org/10.1371/journal.pcbi.1003806">OpenCyto</a> and
<a href="https://doi.org/10.1093/bioinformatics/btu677">flowDensity</a>).</li>
<li>When a recalculated boundary moves further than the <b>drift limit</b>
(in transformed units, on an axis of about 0 to 1), the sample is flagged.
It then keeps its recalculated boundary or uses the stored one, as
chosen.</li>
<li>When the calculation fails (too few events), the stored boundary is
used and the sample is flagged.</li>
<li>Double-click a gate to switch its mode, or right-click to set every
selected gate. The modes and drift settings are saved with the model.</li>
</ul>

<h4>Details: gating</h4>
<ul>
<li>Controls are hidden unless <b>Show controls</b> is ticked.</li>
<li>A warning lists the samples whose autofluorescence profile assignment
none of the training samples had: their unmixing differs from the training
data.</li>
<li>Validation samples are gated too, whether or not <b>Gate</b> is
ticked.</li>
<li>Each sample is unmixed with its own autofluorescence profiles,
restricted to its time-QC keep-mask when present, and gated through the
whole hierarchy using every event.</li>
<li><b>Pause</b> stops at the next gate and keeps every sample gated so far
(the sample being gated is dropped). <b>Run Gating</b> then offers to
<b>Resume</b> (gate only the ticked samples without results) or
<b>Restart</b>. A paused run can be restarted only once the model, the gate
settings or the scales have changed.</li>
<li><b>Cancel</b> stops the same way but discards the run, keeping the
results from before it.</li>
</ul>

<h4>Details: inspecting the result</h4>
<ul>
<li>The gated views show a row for all gated samples pooled (<b>Pooled over
all gated samples</b>, with the stored boundaries), a row for the sample
selected in the sample panel (<b>Selected sample</b>, once it has been
gated) and a row for each validation sample, each with the boundaries it
was gated with.</li>
<li>Plots use a random subsample of each sample (<b>Display events per
sample</b>). Counts and statistics always use every event.</li>
<li>The table lists each sample's event counts, autofluorescence profiles
and flags: gates that could not be calculated, recalculated gates that
drifted, and events removed by time QC.</li>
<li>Gated results are saved with the experiment and restored when you open
it again. Each run's counts and medians are also written as CSV files to
<i>Automated_Gating_Results</i> in the experiment folder.</li>
</ul>
''' + _GLOSSARY


# Validation figures quoted in the Results and Enrichment help: the human
# cohort is UCL IMMPROVE (Automated Gating cache, 159 samples), the inbred
# mice the BATF chimera experiment.
AG_RESULTS = '''
<h3>Results: compare groups of samples</h3>
<p><b>In short:</b> this page tests which populations, and which markers
within them, differ between groups of gated samples, and draws the figures
and report.</p>

<h4>What to do here</h4>
<ol>
<li>Open <b>Groups</b> and check each sample's group. When no sample has a
group yet, groups found in the sample names are filled in for you (with
<b>Undo</b>). Otherwise use <b>Add group</b> and <b>Assign by pattern</b>,
<b>Suggest groupings…</b>, or choose each sample's group by hand. Every
group compared needs at least 3 samples.</li>
<li>Add covariates if samples are paired (for example by donor) or you want
to adjust for something (for example age or batch).</li>
<li>Open <b>Statistics</b>, choose the tests and thresholds, and click
<b>Run Statistics</b>.</li>
<li>Open <b>Figures</b> and <b>Populations</b> to see the differences.</li>
<li>Open <b>Export</b> and click <b>Generate Report</b>.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>Each group has the samples you expect, with no sample left
ungrouped.</li>
<li>The significant populations (★) make sense when you look at their plots
on the Populations section.</li>
</ul>

<h4>Details: groups and covariates</h4>
<ul>
<li>A group's pattern is a regular expression matched against the sample
name.</li>
<li>A covariate can <b>pair</b> samples (for example donor) or be
<b>adjusted</b> for (for example age, sex, batch).</li>
<li>Tick <b>Numeric</b> under a covariate's title to enter it as a number
(for example age or CRP), with a value for every sample. Unticked
covariates are categories, so leave codes such as batch 1, 2, 3
unticked.</li>
<li><b>Import table…</b> reads groups and covariates from a CSV or
tab-separated file with a <i>sample</i> column and one column per covariate.
A row matches a sample when its name is the file name (with or without its
extension) or appears inside the file name between spaces, underscores or
dashes, so a study key naming <i>POMS_008_FB_D</i> matches
<i>A1 POMS_008_FB_D_Plate 1</i>. Names matched inside the file name need at
least three characters including a digit. Before anything changes, a review
shows which rows matched which samples, rows that matched nothing and samples
left without a row; choose the group column there (or keep the current
groups) and the covariates to import. <b>Export table…</b> writes the same
format, naming samples as the table shows them.</li>
</ul>

<h4>Details: the tests</h4>
<ul>
<li><b>Population frequencies</b>: each population's share of its parent (or
of its stats parent) as log odds, with half an event added so 0% and 100%
stay finite. Moderated linear model [1, 2]. Effects are log2 odds
ratios. A sample's frequency is measured only when the parent has at least
<b>Min. parent events</b> there (20 by default), because a fraction of a
handful of events mostly reflects gating noise; other samples are left out
of that population's test.</li>
<li><b>Population counts</b> (under <i>Counts (advanced)</i>), against the
<b>anchor population</b> (normally the viable cells; see the Hierarchy
page), ask a different question: for this many viable cells, are there
more or fewer of this population? A population can rise as a share of its
parent while falling per viable cell, because its parent fell. A large
change in one abundant population (for example neutrophils) moves every
other population per viable cell too. Against the <b>parent</b>, counts ask
the frequency test's question, and a counts hit that is not also a
frequency hit is marked ⚠.</li>
<li>The counts <b>model</b>: <i>moderated log odds</i> (the default) fits the
frequency test's model to each population's share of the reference, so its
effects are log2 odds ratios. <i>Quasi-likelihood F</i> fits a
negative-binomial model of the event counts [5], with log2 fold changes.
Random splits of the samples of one group, where nothing differs, show how
often each test finds a difference by chance (5% expected):
<ul>
<li>Human cohort (159 samples): moderated log odds 4–6%;
quasi-likelihood F 7–8% against the parent and 9–10% against the anchor for
populations under 100 events.</li>
<li>Inbred mice (BATF chimeras, 48 samples in six tissues; six populations
of 64 to 100,000 events): moderated log odds 4–5%; quasi-likelihood F
3–6%.</li>
</ul>
The negative-binomial Wald test is not offered: it gave 10–13% on the human
cohort and 21–31% on the mice. Use moderated log odds for human or
otherwise varied donors.
Quasi-likelihood F suits homogeneous designs. It fits each comparison on
the samples of its two groups and takes each population's dispersion from
the populations with similar counts, which keeps it near 5% on mouse
samples with very few events (about 5% on samples cut down to 100 cells).
On human data it stays above 5%, because donors vary more, and more
unevenly, than the model allows for.</li>
<li><b>Marker medians</b>: each population's median of each marker (and of
AF Abundance, the per-cell autofluorescence), in transformed units.
Populations are screened first, and markers are tested only within the
populations that pass (stage-wise FDR [4]). By default a population is not
tested on the markers of its own and its ancestors' gates, which differ by
construction.</li>
<li>With <b>TREAT</b> [3] on, p-values test whether the effect exceeds the
threshold. With it off, they test for any difference and the threshold only
filters.</li>
<li>FDR is pooled over all comparisons or taken within each. Changing a
threshold updates significance without refitting.</li>
<li><b>Restore defaults</b>, at the top of the settings, puts the tests and
their options, the thresholds, TREAT, the FDR correction and <i>Leave out
flagged samples</i> back to their defaults. Comparisons, reference group,
pairing and covariates are kept.</li>
<li>The same framework has been used in systems-immunology studies of flow
cytometry cohorts [6, 7].</li>
</ul>

<h4>Details: trends with a numeric variable</h4>
<ul>
<li>Set <b>Test</b> to <i>Trend with a numeric variable</i> to ask which
populations change with a number such as age or days after vaccination, instead of comparing groups. Tick <b>Numeric</b> under
that covariate on the Groups page first.</li>
<li>Two tests are run on each population, on the same moderated linear model
as the group comparisons [1, 2]:
<ul>
<li>the <b>curve</b> test asks whether the population changes with the
variable at all, whatever the shape (a rise, a fall, a rise that levels
off, a peak). It is an F test of a smooth curve (a natural spline [8]):
one p-value for the variable as a whole, as in an ANOVA;</li>
<li>the <b>slope</b> test fits a straight line and gives the direction and
rate of change, in log2 odds per unit of the variable (per week, per
day).</li>
</ul></li>
<li><b>Partial R²</b> is the share of the sample-to-sample variation the
variable explains once groups and adjustment covariates are accounted for
(0 = none, 1 = all).</li>
<li>By default a population is called when either test is significant, each
at half the FDR so the combined error stays at the FDR set, and its partial
R² is at least 0.1 (<b>Min. partial R²</b>). In large studies very small
trends become significant; the minimum keeps them out.</li>
<li><b>Shape</b>: <i>Smooth curve</i> (default, 2 df: one bend), with its
ends fixed between the 5th and 95th percentiles of the variable so a single
youngest or oldest sample cannot bend it; <i>Straight line</i>; or
<i>Time points as groups</i> for designed time courses (day 0, 3, 7,
14), where each value is its own level. The curve gets at most one df per
10 samples. <b>Scale</b> <i>Log2</i> suits time courses sampled on a
doubling schedule; the slope is then per doubling of time.</li>
<li><b>Groups</b>: the samples of the ticked groups are tested together,
with the groups as adjustment terms. <i>Does the change differ between two
groups?</i> asks whether two groups change differently (for example two
tissues across age). With pairing (each donor's two tissues paired) this
tests the within-pair difference against the variable.
A variable shared within every pair cannot be tested for a common trend
with pairing, so the common trend is then fitted without it.</li>
<li>Counts are tested as log odds (moderated), whichever counts model is
chosen for group comparisons: with a curve, the quasi-likelihood test found
trends in about 18% of populations when age was shuffled among the samples
of one group of a human cohort (5% expected); the moderated models gave
4.5–5%.</li>
<li>The design check under the settings warns when another covariate, or
the number of events per sample, follows the variable closely (correlation
r, or η for categories, of 0.5 or more): a trend could then reflect it, and
adjusting for it is advised.</li>
<li>Marker medians are tested as one family (no population-first
screening) in a trend test.</li>
<li><b>Trend figures</b>:
<ul>
<li>the <b>trend volcano</b> places each population by its slope (left: falls,
right: rises) and by how strongly the curve test supports a change (up:
stronger). Called populations are red (rise) or blue (fall); an open diamond is
called by the curve but not the slope: a rise and fall, or a change that
levels off, which a straight line misses. The dashed line is the FDR the
curve test is called at;</li>
<li>the <b>trajectories</b> heatmap shows the fitted curve of every called
population, rescaled to mean 0 and SD 1 so the shape, not the size, is
compared, ordered by where each curve is highest: early peaks at the top,
late rises at the bottom;</li>
<li><b>Patterns</b>: the called populations are grouped by the shape of their
fitted curve, whatever the size of the change (hierarchical clustering of
the curves scaled to mean 0 and SD 1), and each group is named from its
mean curve: <i>rises</i>, <i>falls then levels off</i>, <i>peaks at day
7</i>. <i>Auto</i> uses the largest number of patterns that come back
the same (adjusted Rand index of at least 0.75) when the test is repeated
on 80% of the samples; on a human cohort we tested only the rise/fall
split did, as finer shapes there formed a continuum. The trajectories heatmap is split
by pattern and the <b>patterns</b> figure shows each pattern's members
and mean curve;</li>
<li>double-click a point or row (or right-click it) for its <b>trend
plot</b>: each sample at its value of the variable, the fitted curve with
its 95% confidence band (the range the true curve is likely in, given the
spread of the samples), and the values of the variable along the bottom.
With <i>Does the change differ between two groups?</i> and no pairing, each
group gets its own curve and band in its group colour on the same axes:
where the bands separate the groups differ at that value, and where the
curves run apart the change itself differs. With pairing each pair's level
cancels out, so only the difference can be estimated: the upper panel shows
the samples and the lower panel the fitted difference between the groups
with each pair's own difference;</li>
<li>in trend mode the <b>Populations</b> page lists the populations called
as changing (★) and shows each one's trend plot beside its pooled biplots,
twice: on the tested scale (log2 odds) and as the gated frequencies (% of
parent) with the fitted curve and band drawn back on that scale (also on
the volcano's right-click menu);
a population ticked for the report is reported the same way, and the
report also holds the trend volcanoes, trajectories and the trend
settings;</li>
<li><b>Make-up over the variable</b>: how the populations of one gate
share out their parent as the variable changes. For example, with a gate
that splits CD4 T cells into naive, central memory, effector memory and
TEMRA: does the naive share shrink with age while effector memory grows?
Open it with the <i>Make-up over the variable</i> button on the
Populations page (shown in trend mode for a population whose gate has two
or more populations), or from a population's right-click menu on the
figures.
<ul>
<li>Top: the fitted share of each of the gate's populations, plus "Other"
(the parent's events in none of them), stacked so they fill 100% of the
parent at every value of the variable.</li>
<li>Bottom: each sample's measured shares, one bar per sample, samples
ordered by the variable, to check the fit against the data.</li>
<li>Why one gate: a gate's populations never overlap, so their shares add
up to at most 100% of the parent. Populations from different gates on the
same parent usually overlap (a cell can be both naive and Ki-67+), so
stacking them would count cells twice.</li>
<li>How the curves are made: each part's counts are compared with the
average of all the parts on a log scale (centred log-ratio, Aitchison
1986), the trend test's curve (same shape, scale and df) is fitted to each,
and the fitted values are turned back into shares. The result is the
make-up of a typical sample at each value, so it never goes below 0% or
above 100%; with skewed data it can sit a few points from the plain
average of the samples' percentages.</li>
<li>It is a picture of how the make-up shifts. Whether a population changes
significantly is the trend test's answer (volcano and tables).</li>
<li>A group × trend run gets one panel per group. The values behind the
figure (fitted shares and each sample's shares) are offered as CSV beside
the figure export.</li>
</ul></li>
<li><b>Immune age model</b> (<i>Sample PCA and prediction</i> section, after a trend run; figures on the Figures page): how well does the immune profile
predict a continuous variable, such as age, days after vaccination or any
number used as the trend variable? A model learns to guess the variable
from the population frequencies and/or marker medians of the last trend run and is always judged on samples it
was not trained on (nested cross-validation: an outer split for judging,
an inner one for tuning, every donor or pair kept whole, so the same donor
is never in both). The model is an elastic net (Zou &amp;
Hastie 2005), a penalised regression between the lasso, which keeps few
features, and ridge, which shares weight between correlated ones such as a
parent and its largest subset. Read the <i>mean absolute error</i> (how far
off a typical guess is, in the variable's units) against <i>predicting the
mean</i> (the error of guessing every sample at the average): a useful
model is well below it. The <i>chance check</i> reruns the whole procedure
with the variable shuffled among donors or pairs; chance p is the share of
shuffles that did as well. The features figure shows which features the
model kept most often and their direction (red: higher at larger values
of the variable, e.g. in older donors). The <i>gap</i> is the guess minus
the actual value (with age: a sample's "immune age" minus its age); any
such model pulls guesses towards the middle (the lowest values guessed too
high, the highest too low), so the gap is shown <i>corrected</i> for that
(de Lange &amp; Cole 2020): positive means the sample's profile looks like
that of a higher value than it has (with age: older than it is). Train the
model within one group. With <i>Predict the other groups</i> ticked, the
final model also predicts every sample of the other groups, whether or not
they were in the trend run and whether or not they have a value of the
variable (open markers, not cross-validated). The table lists each group's
predicted values (mean ± SD) and, for samples that have a value, how close
the predictions are; the <i>Predicted by group</i> panel shows them against
the range the model was trained on. A prediction outside that range, or for
samples of a different kind (a different tissue or time of life), says the
profile differs from the training samples, not what their value is. Needs
at least 20 samples; 40 or more is recommended;</li>
<li>in trend mode the <b>sample PCA</b> uses the trend run's values and
colours each sample by the variable (shape = group).</li>
</ul></li>
</ul>

<h4>Details: sample PCA and prediction</h4>
<ul>
<li>Both use the per-sample values of the last statistics run.</li>
<li><b>Show Sample PCA</b> places every sample on the first two principal
components of the chosen tables (each column scaled to unit variance),
coloured by group, with the most strongly loading features as
arrows.</li>
<li><b>Run Predictive Analysis</b> asks which features best separate two
groups, and how well. Lasso logistic regression and a random forest are
assessed by repeated nested cross-validation. It gives the ROC curve with
its AUC (mean ± SD over repeats), the balanced accuracy, the most
informative features and the AUC reached with the top 1, 2, 3 features and
so on [6, 7]. It needs at least 10 samples per group; 30 or more is
recommended.</li>
</ul>

<h4>Details: figures and populations</h4>
<ul>
<li>Heatmaps, volcano plots and a population × marker summary are drawn for
the comparison chosen at the top. Large heatmaps open at half size; the
size slider enlarges them.</li>
<li><b>Pop Out</b> opens a figure in its own window.</li>
<li><b>⚙ Options</b> beside a figure opens a side panel that changes how
it is drawn, not what it shows: grid lines (off by default), the run label
in the top left corner, the title (as drawn, hidden or your own text), the
titles of individual panels, the font sizes of the title, the axis and
panel labels, the tick labels, the legend and other text (each "As drawn"
until set), and the background (the theme's: black in dark mode, white in
light mode; or white, black or transparent). Threshold lines, such as the
FDR and effect-size lines of a volcano plot, are part of the figure and
always drawn. The options are kept with the experiment, per figure, and
used for Pop Out, Export and the report as well. <i>Use for all</i> makes
the panel's settings (all but the title) the default of every figure of
this plugin; <i>Reset</i> returns the figure to those defaults.</li>
<li>Double-click a heatmap row or a volcano point to show that population in
the <b>Populations</b> section and tick it for the report. Right-click it
for more choices.</li>
<li>The Populations section lists the significant populations of the
comparison (★), or all of them. For the population picked it shows each
group's events pooled (up to 100,000 per group) on the population's gate,
beside a dot and violin plot of every sample's frequency with the test's
p-value and FDR.</li>
</ul>

<h4>Details: export</h4>
<ul>
<li><b>Generate Report</b> writes a folder under
<i>Automated_Gating_Reports</i> in the experiment folder: the tables as
CSV, a settings document, a PNG or CSV per ticked item, and one PDF with
all of them. <b>Cancel</b> on the progress window stops the report and
removes its half-written folder.</li>
<li>Every statistics run is also written to <i>Automated_Gating_Results</i>
as CSV files, and kept with the experiment. <b>Export Gating Results…</b>
and <b>Export Statistics…</b> write the same files to a folder of your
choice.</li>
</ul>

<p>References:<br/>
[1]<a href="https://doi.org/10.1093/nar/gkv007">Ritchie et al. 2015, Nucleic Acids Research (limma software)</a><br/>
[2]<a href="https://doi.org/10.2202/1544-6115.1027">Smyth 2004, Stat Appl Genet Mol Biol (empirical Bayes moderation)</a><br/>
[3]<a href="https://doi.org/10.1093/bioinformatics/btp053">McCarthy and Smyth 2009, Bioinformatics (TREAT)</a><br/>
[4]<a href="https://doi.org/10.1111/rssb.12028">Benjamini and Bogomolov 2014, J R Stat Soc B (selective inference on families of hypotheses)</a><br/>
[5]<a href="https://doi.org/10.1093/bioinformatics/btp616">Robinson, McCarthy and Smyth 2009, Bioinformatics (edgeR)</a><br/>
[6]<a href="https://doi.org/10.1002/alz.70952">Humblet-Baron et al. 2025, Alzheimer's &amp; Dementia (systems immunology of Alzheimer's disease)</a><br/>
[7]<a href="https://doi.org/10.1038/s41467-026-70895-1">Veiga et al. 2026, Nature Communications (autoinflammation of unknown origin)</a><br/>
[8]<a href="https://doi.org/10.1007/978-0-387-84858-7">Hastie, Tibshirani and Friedman 2009, The Elements of Statistical Learning, 2nd edition (natural cubic splines)</a><br/>
</p>
''' + _GLOSSARY


AG_ENRICHMENT = '''
<h3>Enrichment: immune states between groups</h3>
<p><b>In short:</b> Flow Set Enrichment Analysis asks whether the features of
a curated <b>immune state</b> shift together between two groups, in the
manner of gene set enrichment analysis [1]. For example, CD8 T-cell
exhaustion: PD-1, TIGIT, LAG3 and CD39 up and TCF1 and CD127 down in CD8 T
cells. It uses the samples, groups, pairing, covariates and comparisons of
the Results page.</p>

<h4>What to do here</h4>
<ol>
<li>Make sure the Results page has your groups. A comparison needs at least
<b>30 samples</b> in its two groups and <b>10 in each</b>.</li>
<li>Keep the bundled library, or click <b>Choose library…</b> for your
own.</li>
<li>Click <b>Check matching</b> to see how each population maps to a
library cell type, and correct any wrong mapping.</li>
<li>Click <b>Run enrichment</b>.</li>
<li>Read the dot plot, then select a state in the table for its running sum
and members.</li>
</ol>

<h4>What good looks like</h4>
<ul>
<li>The populations you expect are matched to the right cell types.</li>
<li>States with an FDR below the threshold (ringed in the dot plot) have
leading-edge members that make biological sense.</li>
</ul>

<h4>Details: samples</h4>
<p>The p-values come from permuting the samples' group labels. With fewer
samples there are too few distinct labellings, and the correlation between
features cannot be estimated, so state-level results are not reliable.</p>

<h4>Details: the library</h4>
<ul>
<li>The bundled library (<i>ag_fsea_library.csv</i>) holds draft immune
states. Each is a list of rows: a cell type with either its
<b>abundance</b> (Abund) or a <b>marker</b> read out in it, and the
direction (UP or DOWN) the feature moves when the state is gained.</li>
<li>Competing phenotypes are separate states that may share markers (PD-1 is
in exhaustion and in acute activation); the data decide which one
moves.</li>
<li>Each row carries a citation. The draft entries need review before
results are published.</li>
<li>Your own CSV needs the same columns.</li>
</ul>

<h4>Details: features and cell types</h4>
<ul>
<li><b>Positivity</b>: the fraction of a population positive for a marker
(gates such as PD1+ drawn on CD8 TEM), as log odds within the parent.</li>
<li><b>Subsets</b>: each phenotype population within its parent, as log
odds.</li>
<li><b>Abundance</b>: each phenotype population within the <b>anchor</b>, so
samples with different numbers of events, debris or dead cells compare on
the same footing. The default is the model's anchor population (see the
Hierarchy page).</li>
<li><b>Marker medians</b>: medians of phenotype populations only. Medians
inside positivity gates repeat their parent.</li>
<li>Exact duplicates and the two halves of a two-way split are counted
once.</li>
<li>Panels that re-use one fluorochrome for markers of different lineages
are handled by reading the marker from the positivity gates drawn on that
channel nearest to the population. Channels whose marker cannot be told
apart are tested but match no library row.</li>
<li>A population matches a library cell type by name, accepted only when its
lineage agrees with the population's ancestors. A library cell type matches
its own population only, not the populations below it.</li>
<li>A state needs at least the set number of distinct matched features (5
by default) to be tested.</li>
</ul>

<h4>Details: the statistics</h4>
<ul>
<li>Each feature is ranked by sign(logFC) × −log10 p from the same moderated
linear model as the Results page [2], without the TREAT threshold.</li>
<li>A state's enrichment score (ES) is the weighted running sum over that
ranking, with the DOWN members negated. ES above zero means the state is
<b>gained</b> in the second group, below zero <b>lost</b>.</li>
<li>The group labels of the two groups compared are permuted and every
feature refitted to give the null distribution, which keeps the correlation
between features. Samples of other groups keep their labels. With pairing or
covariates, the residuals of the model without the group terms are permuted
(within pairs) [3].</li>
<li>NES is the ES divided by the mean null ES of the same sign. p is never
zero [4]. FDR is Benjamini–Hochberg over the states of a comparison, and
FWER is the chance that the largest null NES of any state reaches the
NES.</li>
<li>Read the FDR. The FWER was liberal when calibrated on 15 samples (10% of
null labellings had a state below 0.05, against 5% expected).</li>
<li>The enrichment score only records <i>where</i> a state's features rank,
not <i>how much</i> they changed, and it cannot exceed 1. When groups differ
greatly, shuffled labels reproduce much of the real ranking, so a state can
change a lot yet miss the threshold. Two statistics that use effect sizes
are shown beside it, each with its own FDR:
<ul>
<li><b>Self-contained</b> (<i>Mean t</i>): did this state's features change?
The mean of the members' moderated t, with DOWN members negated, against
the same label permutations. When groups differ greatly, most states
pass.</li>
<li><b>Competitive</b> (CAMERA [5]): did they change more than the other
features? The members' scores against all others', allowing for how
strongly the members move together (correlated features are not counted as
independent evidence).</li>
</ul>
On random splits of one group of a human cohort, the enrichment
score and the self-contained statistic each gave 4–7% of states below 0.05
(5% expected).</li>
<li>The dot plot shows every tested state in every comparison: colour NES,
size the share of members in the leading edge, ringed when FDR is below the
threshold. The leading edge is the set of members that drive the
score.</li>
</ul>

<h4>Details: enrichment with a numeric variable (age, time)</h4>
<ul>
<li>When a numeric variable is set for a trend on the Results page, the
<b>Test</b> list at the top offers <i>Trend with ‹variable›</i> beside the
group comparisons. Choose it and press Run to ask which immune states change
with the variable, on the samples, groups and settings of the trend test.
Group and trend results are kept side by side: switch between them with the
Test list.</li>
<li>Features are ranked by their slope (the straight-line trend), so the
enrichment score, mean t and competitive test say whether a state is
<b>gained</b> or <b>lost</b> as the variable rises.</li>
<li><b>Any-direction FDR</b>: did the state's features change with the
variable in any way, a rise and fall included? The mean, over the members,
of their curve test's p-value turned into a normal score (large when the
change is strong).</li>
<li>The null permutes the residuals of the model without the variable
[3]: within groups; within donors sampled at several times; or, when each
pair shares the value (two tissues of one donor taken at the same age),
whole pairs swap values.</li>
<li>On a human cohort with age shuffled (40 shuffles of the samples of
one group, 20 of the pairs), every statistic gave 4–6% of
states below 0.05 and the permutation tests no FDR hit.</li>
<li><b>Patterns</b> (after a frequency trend run with the same variable):
which states' populations gather in each shape pattern (hypergeometric
test against all tested populations, BH FDR). Dot colour: fold
enrichment; size: share of the state's populations in the pattern.</li>
</ul>

<h4>References</h4>
<ol>
<li>Subramanian A et al. (2005) Gene set enrichment analysis. PNAS 102:15545–15550.</li>
<li>Smyth GK (2004) Stat Appl Genet Mol Biol 3: Article 3.</li>
<li>Freedman D, Lane D (1983) J Bus Econ Stat 1:292–298.</li>
<li>Phipson B, Smyth GK (2010) Permutation p-values should never be zero.
Stat Appl Genet Mol Biol 9: Article 39.</li>
<li>Wu D, Smyth GK (2012) Camera: a competitive gene set test accounting for
inter-gene correlation. Nucleic Acids Res 40: e133.</li>
</ol>
''' + _GLOSSARY
