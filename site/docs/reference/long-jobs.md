---
title: Long jobs
lead: Honeychrome runs one long calculation at a time, so jobs do not compete for memory and processor time.
---

Long calculations are queued so that only one runs at a time. They include:

- importing FCS files;
- Clean Controls;
- AF extraction and auto-assignment, Model Noise, Compute Spectral Variants and Fix Unmix on the AutoSpectral tab;
- Simple Stats calculations;
- unmixed FCS exports;
- Honeycluster runs, Cluster ID, statistics and predictive analysis;
- Automated Gating training and gating runs.

## While a job runs

- The status bar shows **Running:** and the name of the job.
- Buttons that would start another long job, or change the unmixing (for example the Spectral Process editor and the AF profile assignments), are greyed out. Hover over one to see which job is running.
- The comparisons on the AutoSpectral pages, and the background unmixing of the sample you are viewing, wait and continue when the job has finished.
- You can still browse samples and look at your data. Reloading the experiment is refused until the job has finished.

## A job that has stopped

If a job has stopped without finishing and the buttons stay greyed out, right-click **Running:** in the status bar and choose **Release "‹job name›"…**.

> Release a job only when you are sure it is no longer running. Releasing a job that is still working lets a second one start alongside it.
{: .caution}
