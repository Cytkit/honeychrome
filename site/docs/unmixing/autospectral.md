---
title: AutoSpectral in Honeychrome
lead: Clean your controls, remove autofluorescence cell by cell, and reduce spread and spillover. This page shows how the parts fit together.
redirect_from:
  - /docs/autospectral_in_honeychrome
  - /docs/autospectral_in_honeychrome.html
---

AutoSpectral is an optimised unmixing workflow. It was first written as an R package, and Honeychrome runs all of it without any programming. It gives you the best possible spectra from your single-stained controls, removes autofluorescence cell by cell, and unmixes reproducibly. The [Science page](/science/) explains how it works and what it was tested on.

## The parts and where they are

| Step | What it fixes |
|---|---|
| [Control cleaning](/docs/unmixing/control-cleaning/), Spectral Process tab | Spectra contaminated by autofluorescence: hypernegative events, skews, diagonal correlations |
| [Autofluorescence](/docs/unmixing/autofluorescence/), AutoSpectral tab | Negative spread and spikes from autofluorescence that varies between cells |
| [Noise](/docs/unmixing/noise-model/), AutoSpectral tab | Spread from noisy detectors in crowded parts of the panel |
| [Spectral Variants](/docs/unmixing/spectral-variants/), AutoSpectral tab | Spread and skews from fluorophores that vary between cells |
| [Spectral Correction](/docs/unmixing/spectral-correction/) (beta), AutoSpectral tab | Residual spillover in fully stained samples, including tandem breakdown |

## Work in this order

1. **Clean the controls** in the Spectral Process. Everything else starts from these spectra.
2. **Autofluorescence** is the basis of the AutoSpectral tab: extract a profile from each unstained sample and assign the profiles to your samples.
3. Then, optionally, choose **one** of the **noise model** or the **spectral variants**. You cannot use both at once.
4. **Spectral correction** is optional and works with whichever method you chose. It is in beta testing.

> The noise model and spectral correction are new in Honeychrome 1.0 and are not covered in the AutoSpectral manuscript (Burton et al. 2025). Control cleaning, per-cell autofluorescence and spectral variants are.
{: .note}

> Whatever you choose on the AutoSpectral tab is used for the whole experiment. The Unmixed Data tab, unmixed exports, Simple Stats, Honeycluster and Automated Gating all unmix each sample the same way.
{: .note}

## The AutoSpectral tab

The pages are listed down the left side of the tab: **Autofluorescence**, **Noise**, **Spectral Variants** and **Spectral Correction**. Every page has:

- a **Show Help** button that explains every setting, with references;
- a side-by-side **comparison** at the end. It shows the current sample (the one selected in the sample browser) unmixed two ways. The plots open on the Singlets gate, else Cells, else the root gate, and you can change the gate, the channels and the scaling on either plot.

These calculations can take a while. Only one long job runs at a time; see [Long jobs](/docs/reference/long-jobs/).

## The example data

These guides continue with the AutoSpectral Full Workflow Example used in [Spectral unmixing](/docs/unmixing/spectral-unmixing/): a 9-colour panel on mouse tissues, run on a 5-laser Cytek Aurora. To follow along, download the data from the [AutoSpectral Full Workflow Example](https://www.colibri-cytometry.com/post/autospectral-full-workflow-example) post.

## Further reading
- Burton OT et al. AutoSpectral improves spectral flow cytometry accuracy through optimised spectral unmixing and autofluorescence-matching at the cellular level. *bioRxiv* 2025. [doi:10.1101/2025.10.27.684855](https://doi.org/10.1101/2025.10.27.684855)
- [Introducing AutoSpectral](https://www.colibri-cytometry.com/post/introducing-autospectral-an-optimized-unmixing-workflow) on the Colibri Cytometry blog
- [AutoSpectral package on GitHub](https://github.com/DrCytometer/AutoSpectral)
