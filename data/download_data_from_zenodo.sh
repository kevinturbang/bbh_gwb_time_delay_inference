#!/bin/bash

# Download and unzip
curl https://zenodo.org/record/8087858/files/autoregressive-bbh-inference-data.zip --output "bbh-gwb-time-delay-inference-data.zip"
unzip bbh-gwb-time-delay-inference-data.zip

# Move input data to ../input/
mv sampleDict_FAR_1_in_1_yr.pickle ../input/
mv injectionDict_FAR_1_in_1.pickle ../input/
mv delayedRateDataMD.npy ../input/
mv delayedRateDataVA.npy ../input/
mv O3_GWB_measurement.mat ../input/
mv O3_PI_curve.csv ../input/
mv O5_PI_curve.csv ../input/
mv Omega_measurement_detectable_O5_MD.mat ../input
mv Omega_measurement_undetectable_O5_MD.mat ../input
mv Omega_measurement_detectable_O5_VA.mat ../input
mv Omega_measurement_undetectable_O5_VA.mat ../input

# Move input data to ../mock_gwb_O5/
mv AplusDesign.txt ../mock_gwb_O5/
mv freqs_detectable_O5.npy ../mock_gwb_O5/
mv freqs_undetectable_O5.npy ../mock_gwb_O5/
mv Omega_detectable_O5_MD.npy ../mock_gwb_O5/
mv Omega_detectable_O5_VA.npy ../mock_gwb_O5/
mv Omega_undetectable_O5_MD.npy ../mock_gwb_O5/
mv Omega_undetectable_O5_VA.npy ../mock_gwb_O5/
mv ORF_HL_freqs.npy ../mock_gwb_O5/
mv ORF_HL.npy ../mock_gwb_O5/

# Remove original zip files and annoying Mac OSX files
rm bbh-gwb-time-delay-inference-data.zip
rmdir __MACOSX/