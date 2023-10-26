import arviz as az
import numpy as np
import h5py
import argparse

def process(inputfile, outputfile):
    # Load inference results
    inference_data = az.from_netcdf(inputfile)
    samps = az.extract(inference_data,var_names=["R20","td_min","lambda_td","metMin_td",\
                    "alpha","mu_m1","sig_m1","log_f_peak","mMin","mMax","log_dmMin","log_dmMax","bq",\
                    "mu_chi","logsig_chi","sig_cost","nEff_inj_per_event","min_log_neff"])
    R_ref = samps.R20.values
    td_min = samps.td_min.values
    kappa = samps.lambda_td.values
    Zmax = samps.metMin_td.values
    alpha = samps.alpha.values
    mu_m1 = samps.mu_m1.values
    sig_m1 = samps.sig_m1.values
    log_f_peak = samps.log_f_peak.values
    mMin = samps.mMin.values
    mMax = samps.mMax.values
    log_dmMin = samps.log_dmMin.values
    log_dmMax = samps.log_dmMax.values
    bq = samps.bq.values
    mu_chi = samps.mu_chi.values
    logsig_chi = samps.logsig_chi.values
    sig_cost = samps.sig_cost.values
    nEff_inj_per_event = samps.nEff_inj_per_event.values
    min_log_neff = samps.min_log_neff.values

    # Create hdf5 file and write posterior samples
    hfile = h5py.File(outputfile,'w')
    posterior = hfile.create_group('posterior')
    posterior.create_dataset('mu_chi',data=mu_chi)
    posterior.create_dataset('logsig_chi',data=logsig_chi)
    posterior.create_dataset('sig_cost',data=sig_cost)
    posterior.create_dataset('alpha',data=alpha)
    posterior.create_dataset('mu_m1',data=mu_m1)
    posterior.create_dataset('sig_m1',data=sig_m1)
    posterior.create_dataset('log_f_peak',data=log_f_peak)
    posterior.create_dataset('mMin',data=mMin)
    posterior.create_dataset('mMax',data=mMax)
    posterior.create_dataset('log_dmMin',data=log_dmMin)
    posterior.create_dataset('log_dmMax',data=log_dmMax)
    posterior.create_dataset('bq',data=bq)
    posterior.create_dataset('R_ref',data=R_ref)
    posterior.create_dataset('td_min',data=td_min)
    posterior.create_dataset('kappa',data=kappa)
    posterior.create_dataset('Zmax',data=Zmax)
    posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
    posterior.create_dataset('min_log_neff',data=min_log_neff)

    hfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputfile',help="Input CDF file",action="store", type=str)
    parser.add_argument('-outputfile',help="Output HDF5 file",action="store", type=str)
    
    args = parser.parse_args()

    process(args.inputfile, args.outputfile)