---
title: Plots and gates
lead: Add and arrange plots, choose channels and transforms, and draw and edit gates.
redirect_from:
  - /docs/cytometry_plots_and_gates
  - /docs/cytometry_plots_and_gates.html
---

This guide assumes you have unmixed data to look at, from [Spectral unmixing](/docs/unmixing/spectral-unmixing/) or [Conventional compensation](/docs/unmixing/conventional-compensation/). Plots and gates work the same way in the Raw Data and Unmixed Data tabs, but do your analysis in **Unmixed Data**.

## Plots

Honeychrome draws three kinds of plot:

- **1D histograms**;
- **2D histograms**, which can also be shown as dot plots coloured by gate (see [Display options](#display-options));
- **ribbon plots**, which show the histograms of every channel side by side.

Click a plot to select it. The toolbar has **+** to add a plot, **×** to delete the selected one, and a button to show or hide the gating hierarchy.

{% include figure.html src="docs/daily_streck_3.webp" alt="Unmixed Data tab with histograms and 2D plots of immunology data and their gates" width="1600" height="1571" %}

When a plot is selected, the middle of the toolbar has buttons that move it to the start, left, right or end.

{% include figure.html src="docs/move_plots.webp" alt="Toolbar buttons that move the selected plot" width="465" height="58" %}

The gating hierarchy is shown on the right when it is switched on.

{% include figure.html src="docs/gating_hierarchy2.webp" alt="Gating hierarchy panel" width="742" height="355" %}

## Channels, source gate and transforms

- Click an **axis title** to choose its channel, and the **plot title** to choose the source gate (the population the plot shows). Once gates are drawn on a plot, these choices are limited.
- Choose **Count** on the y axis to make a 1D histogram, and **All Fluorescence** on the x axis to make a ribbon plot.
- In the Unmixed Data tab with [per-cell autofluorescence](/docs/unmixing/autofluorescence/), the **AF Abundance** and **AF Index** channels can be plotted and gated like any marker.

Right-click an axis to change its transform: linear, log or logicle. On a logicle axis, drag near the top of the scale to change the full range, and near the bottom to change the width of the linear region around zero.

{% include figure.html src="docs/2dplot_with_menu_2.webp" alt="2D plot with the channel menu open on an axis title" width="1033" height="654" %}

## Pop out a plot

Double-click a plot to open it in its own window, where it is large enough to place gates precisely. Close the window to put it back.

{% include figure.html src="docs/popoutplots2.webp" alt="A plot popped out into its own window" width="1600" height="1230" %}

## Draw and edit gates

With a 2D plot selected, the toolbar offers **rectangle**, **polygon**, **ellipse** and **quadrant** gates.

{% include figure.html src="docs/new_gate_menu.webp" alt="Toolbar buttons for rectangle, polygon, ellipse and quadrant gates" width="1600" height="50" %}

With a 1D plot selected, it offers a **range** gate.

{% include figure.html src="docs/new_gate_menu1d.webp" alt="Toolbar button for a range gate on a 1D plot" width="1600" height="50" %}

Hover over a gate to see what you would select: the whole gate, a line or a vertex. With a polygon gate you can:

- click a highlighted line to add a vertex;
- right-click a highlighted vertex to remove it;
- drag the whole gate when it is highlighted.

Drag a gate's label to move it, and double-click the label to rename the gate.

{% include figure.html src="docs/polygon_gate2.webp" alt="Polygon gate with a highlighted vertex" width="1186" height="1080" %}

Right-click a gate to make a new plot of that population, delete the gate, or [customise it for the current sample](/docs/plots-and-gating/per-sample-gating/).

{% include figure.html src="docs/right_click_gate2.webp" alt="Right-click menu on a gate" width="709" height="468" %}

All samples share one gating template, so a gate you draw or move applies to every sample. To give one sample a different gate, see [Per-sample gating](/docs/plots-and-gating/per-sample-gating/).

## Display options

**Edit > App Configuration** has several options for plots, including the colour map and the size of plots. Set **2D plot type** to **Dot Plot Coloured by Gate** to show coloured dot plots; the main window reloads.

{% include figure.html src="docs/app_config.webp" alt="App Configuration dialog with the 2D plot type option" width="1022" height="760" %}

{% include figure.html src="docs/dot_plots_coloured_by_gate2.webp" alt="Dot plots coloured by gate" width="1600" height="1592" %}

## Next steps
- [Per-sample gating](/docs/plots-and-gating/per-sample-gating/)
- [Simple Stats](/docs/statistics/simple-stats/)
- [Reports and exports](/docs/outputs/reports-and-exports/)

## Further reading
- Parks DR, Roederer M, Moore WA. A new "Logicle" display method avoids deceptive effects of logarithmic scaling for low signals and compensated data. *Cytometry A* 2006;69(6):541–551. [doi:10.1002/cyto.a.20258](https://doi.org/10.1002/cyto.a.20258)
