---
title: Acquire data
lead: Run a Cytkit cytometer from Honeychrome.
redirect_from:
  - /docs/acquisition
  - /docs/acquisition.html
---

Honeychrome provides an interface for acquiring data on cytometry instruments:
- acquisition / flow control
- detector gains
- oscilloscope view for raw traces and to inspect live peak detection

Honeychrome currently contains the following drivers: 
- Cytkit 1S-1F (data acquisition through a Picoscope 5000 series)
- Cytkit 2S-14F (full cytometer control through the Astute Devices Cytometry Electronics Mainboard)
- Dummy Device (generation of dummy traces from an example FCS file, for testing the analysis of traces and live plotting of data)

We hope to support more instruments in the future!

{% include figure.html src="docs/acquisition.webp" alt="The acquisition view with flow control, detector gains and the oscilloscope trace" width="1600" height="1291" %}
