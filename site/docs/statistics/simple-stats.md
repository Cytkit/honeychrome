---
title: Simple Stats
lead: Compare samples or groups of samples with bar charts, box plots, histogram overlays and a sample by marker heatmap.
---

The **Simple Stats** tab (called Statistics before version 1.0) draws quick comparisons of a gate across samples. For high-dimensional analysis with clustering, differential testing and group statistics, enable the Honeycluster or Automated Gating plugin in **Edit > App Configuration**.

## Make a chart

1. **Choose the samples.** Pick a flat list of samples, or a set of folders. Folders are treated as groups.
2. **Choose the chart type:** bar chart, box-and-whisker chart, 1D histogram overlay or heatmap.
3. **Choose the gate** to compare.
4. **Choose the statistic** (bar and box charts), or the channel (histogram overlay).

{% include figure.html src="docs/select_bar_chart.webp" alt="Simple Stats settings for a bar chart" width="1600" height="139" %}

## Bar chart and box-and-whisker chart

The statistics available are:

- **% Total Events**
- **% Parent**
- **Number of Events**
- **Mean Intensity**, in any channel

A bar chart shows one bar per sample.

{% include figure.html src="docs/bar_chart.webp" alt="Bar chart comparing samples" width="1587" height="936" %}

A box-and-whisker chart compares folders of samples: each folder is a group, and each sample in it a replicate.

{% include figure.html src="docs/box_and_whisker.webp" alt="Box-and-whisker chart comparing folders of samples" width="1587" height="936" %}

## 1D histogram overlay

Overlays the histograms of one channel for the chosen gate in each sample.

{% include figure.html src="docs/select_1d_hist_overlay.webp" alt="Simple Stats settings for a 1D histogram overlay" width="1600" height="496" %}

{% include figure.html src="docs/statistical_comparison_of_samples.webp" alt="1D histogram overlay comparing samples" width="1590" height="967" %}

## Heatmap

Shows the median intensity of every marker (columns) in the chosen gate for each sample (rows). There is no statistic or channel to choose.

{% include figure.html src="docs/simplestats_heatmap.webp" alt="Heatmap of median intensity in the Singlets gate for three samples (rows) across scatter channels, markers, AF Abundance and AF Index (columns)" width="810" height="611" class="figure--light" caption="A Simple Stats heatmap of the Singlets gate for three samples. With per-cell autofluorescence, AF Abundance and AF Index appear as columns too." %}

## Export the data

Charts can be exported as an image, or as a CSV file of their values to open in a spreadsheet.

{% include figure.html src="docs/csv_export.webp" alt="Chart data exported as CSV and opened in a spreadsheet" width="1212" height="1281" %}
