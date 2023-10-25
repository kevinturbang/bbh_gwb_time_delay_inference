#!/bin/bash

# Download and unzip
curl https://sandbox.zenodo.org/record/1246333/files/bbh_time_delay_data.zip --output "bbh_time_delay_data.zip"
unzip bbh_time_delay_data.zip

# Move input data to ../input/
mv bbh_time_delay_data/sampleDict_FAR_1_in_1_yr.pickle ../input/
mv bbh_time_delay_data/injectionDict_FAR_1_in_1.pickle ../input/.
mv bbh_time_delay_data/delayedRateDataMD.npy ../input/
mv bbh_time_delay_data/delayedRateDataVA.npy ../input/
mv bbh_time_delay_data/O3_GWB_measurement.mat ../input/
mv bbh_time_delay_data/O3_PI_curve.csv ../input/
mv bbh_time_delay_data/O5_PI_curve.csv ../input/
mv bbh_time_delay_data/Omega_measurement_detectable_O5_MD.mat ../input/
mv bbh_time_delay_data/Omega_measurement_undetectable_O5_MD.mat ../input/
mv bbh_time_delay_data/Omega_measurement_detectable_O5_VA.mat ../input
mv bbh_time_delay_data/Omega_measurement_undetectable_O5_VA.mat ../input/

# Move input data to ../
mv bbh_time_delay_data/O3_CBC_GWB_MD.hdf .
mv bbh_time_delay_data/O3_CBC_GWB_VA.hdf .
mv bbh_time_delay_data/O3_CBC_MD.hdf .
mv bbh_time_delay_data/O3_CBC_VA.hdf .
mv bbh_time_delay_data/O5_CBC_GWB_detectable_MD.hdf .
mv bbh_time_delay_data/O5_CBC_GWB_detectable_VA.hdf .
mv bbh_time_delay_data/O5_CBC_GWB_undetectable_MD.hdf .
mv bbh_time_delay_data/O5_CBC_GWB_undetectable_VA.hdf .

# Move input data to ../mock_gwb_O5/
mv bbh_time_delay_data/AplusDesign.txt ../mock_gwb_O5/
mv bbh_time_delay_data/freqs_detectable_O5.npy ../mock_gwb_O5/
mv bbh_time_delay_data/freqs_undetectable_O5.npy ../mock_gwb_O5/
mv bbh_time_delay_data/Omega_detectable_O5_MD.npy ../mock_gwb_O5/
mv bbh_time_delay_data/Omega_detectable_O5_VA.npy ../mock_gwb_O5/
mv bbh_time_delay_data/Omega_undetectable_O5_MD.npy ../mock_gwb_O5/
mv bbh_time_delay_data/Omega_undetectable_O5_VA.npy ../mock_gwb_O5/
mv bbh_time_delay_data/ORF_HL_freqs.npy ../mock_gwb_O5/
mv bbh_time_delay_data/ORF_HL.npy ../mock_gwb_O5/

# Remove original zip files and directory
rm bbh_time_delay_data.zip
rm -r bbh_time_delay_data/