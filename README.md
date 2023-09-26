# bbh_gwb_time_delay_inference

This repository contains the code and necessary material to reproduce the results of ... . In this paper, we infer the
time-delay distribution and metallicity-dependent star formation rate parameters from gravitational-wave (GW) data, using
a joint approach, combining individual binary black hole mergers and upper limits on the gravitational-wave background. We
consider the GW events from LIGO-Virgo-KAGRA's third observing run, as well as current upper limits on a gravitational-wave
background. In addition, we explore LIGO-Virgo-KAGRA's future Advanced A+ design sensitivity and provide results for that scenario
as well.

The data can be downloaded from Zenodo [here](...). A pre-print of the paper can be found [here]().

## Getting started

The code relies on several packages, which can easily be downloaded in a conda environment using the provided yml file in the repository.
To create a conda environment from the yml file, execute the following command `conda env create -f bbh_gwb_time_delay_inference.yml`. To activate
the environment, simply call `conda activate bbh_gwb_time_delay_inference`.

## Structure of the repository

The repository contains several folders, each with a specific purpose:

- **code**: Contains the code to run the analysis which produced the results in this paper.
- **data**: Contains the script to download from Zenodo [here](...) the data used in this paper, as well as the results produced in this paper.
- **figures**: Contains several notebooks to reproduce the figures of the paper. Note that all the data should have been downloaded
    from Zenodo [here](...) in order to run the notebooks in this folder.