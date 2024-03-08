# Code directory

This subdirectory contains the actual analysis code used to produce the results of the [paper](https://arxiv.org/abs/2310.17625). The main script
to perform the analysis is `run_combined_pop_gwb_cbc.py`. Below, we provide some details on how to change some aspects of
the analysis.

*Note: to run the code in this subdirectory, it is likely that you first need to download the data,
as specified in the `/data` subdirectory.*

## Customizing the code

Different analyses are performed in our paper. For example, the analysis is run both using individual black hole
mergers only, as well as the combination of individual events with gravitational-wave background data. Another
example are the different cases assumed for the star formation rate, i.e. as given by Madau-Dickinson or Vangioni et al. 
Below, we point out where these can be passed should the user want to re-run the analysis themselves.

### Running with a different star formation rate

The star formation rate is read in in the `likelihoods.py` script through the `unpack_rate_file` method, defined in `util.py`.
To run the analysis using a different star formation rate, one can change the `rate_file_path` in `run_combined_pop_gwb_cbc.py`
to point to the appropriate file containing the desired star formation rate. We note, however, that the file should follow the
expected data structure as pointed out in the code.

### Running on individual mergers only

The analysis can easily be run on individual merger events only, without considering the contribution from the gravitational-wave
background. To do so, simply change the `joint_analysis` tag to `False` in the `run_combined_pop_gwb_cbc.py` script.

### Running with different stochastic gravitational-wave background results

The stochastic results are passed through a `.mat` file, with a specific data structure containing point estimate and sigma spectra, as
well as the corresponding frequencies. However, any `.mat` file can be passed, as done for e.g. for investigations at the future O5-like
sensitivity. To run on different gravitational-wave background data, change the path to the stochastic results in the `get_stochastic_dict`
method in the `run_combined_pop_gwb_cbc.py` script.
