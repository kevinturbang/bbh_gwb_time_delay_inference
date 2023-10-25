#!/bin/bash

# Run on O3 BBH data - SFR Madau+

python3 process_cdf.py -inputfile ./O3_CBC_MD.cdf -outputfile ./O3_CBC_MD.hdf

# Run on O3 BBH + GWB data - SFR Madau+

python3 process_cdf.py -inputfile ./O3_CBC_GWB_MD.cdf -outputfile ./O3_CBC_GWB_MD.hdf

# Run on A+ data with detectable GWB - SFR Madau+

python3 process_cdf.py -inputfile ./O5_CBC_GWB_detectable_MD.cdf -outputfile ./O5_CBC_GWB_detectable_MD.hdf

# Run on A+ data with undetectable GWB - SFR Madau+

python3 process_cdf.py -inputfile ./O5_CBC_GWB_undetectable_MD.cdf -outputfile ./O5_CBC_GWB_undetectable_MD.hdf

# Run on O3 BBH data - SFR Vangioni+

python3 process_cdf.py -inputfile ./O3_CBC_VA.cdf -outputfile ./O3_CBC_VA.hdf

# Run on O3 BBH + GWB data - SFR Vangioni+

python3 process_cdf.py -inputfile ./O3_CBC_GWB_VA.cdf -outputfile ./O3_CBC_GWB_VA.hdf

# Run on A+ data with detectable GWB - SFR Vangioni+

python3 process_cdf.py -inputfile ./O5_CBC_GWB_detectable_VA.cdf -outputfile ./O5_CBC_GWB_detectable_VA.hdf

# Run on A+ data with undetectable GWB - SFR Vangioni+

python3 process_cdf.py -inputfile ./O5_CBC_GWB_undetectable_VA.cdf -outputfile ./O5_CBC_GWB_undetectable_VA.hdf