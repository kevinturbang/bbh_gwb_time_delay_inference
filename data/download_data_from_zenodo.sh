#!/bin/bash

# Download and unzip
curl https://zenodo.org/record/8087858/files/autoregressive-bbh-inference-data.zip --output "bbh-gwb-time-delay-inference-data.zip"
unzip bbh-gwb-time-delay-inference-data.zip

# Move input data to ../code/input/
mv sampleDict_FAR_1_in_1_yr.pickle ../input/
mv injectionDict_FAR_1_in_1.pickle ../input/

# Remove original zip files and annoying Mac OSX files
rm bbh-gwb-time-delay-inference-data.zip
rmdir __MACOSX/