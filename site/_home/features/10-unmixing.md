---
eyebrow: Unmixing
title: Cleaner spectra and less spread, with AutoSpectral
screenshot: docs/af_step3_stained.webp
alt: The same sample unmixed without autofluorescence extraction (left) and with per-cell autofluorescence (right)
width: 1600
height: 1003
caption: One sample, unmixed without AF extraction (left) and with per-cell AF (right).
more: AutoSpectral in Honeychrome
more_link: /docs/unmixing/autospectral/
---
- **AutoSpectral control cleaning** picks the brightest events, matches each control to unstained cells of the same scatter and removes intrusive autofluorescent events.
- **Per-cell autofluorescence:** every cell is unmixed with the autofluorescence spectrum that fits it best. AF Abundance and AF Index are written out as channels you can gate on.
- Optional **noise-weighted unmixing**, per-cell **spectral variants**, and **spectral correction** (in beta testing) for tandem breakdown in fully stained samples.
